"""
Sparse Hopfield layer with dual loss: direct retrieval MSE + LM injection.

The key insight: train retrieval and injection with SEPARATE loss signals.

1. Retrieval loss (MSE): the converged state BEFORE output projection should
   be close to the correct document embedding. This directly trains Q/K/V
   projections for retrieval without going through the frozen LLM.

2. LM loss: the output AFTER Wo projection helps the LLM generate better
   answers. This trains Wo for injection.

The retrieval loss is computed in the memory embedding space (768-d), not in
the LLM hidden space (2048-d). This avoids the output projection destroying
the cosine relationship with stored patterns.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


class DualLossHopfieldLayer(nn.Module):
    """Sparse Hopfield with separate retrieval and injection outputs.

    Args:
        query_dim: Dimension of the external query (768 for BGE).
        hidden_dim: LLM hidden state dimension (2048).
        memory_dim: Memory bank dimension (768).
        num_heads: Association heads.
        head_dim: Per-head dimension.
        num_steps: Hopfield iterations.
    """

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        head_dim: int = 64,
        num_steps: int = 3,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_steps = num_steps
        self.total_head_dim = num_heads * head_dim

        # Query projection from question embedding
        self.Wq = nn.Linear(query_dim, self.total_head_dim, bias=False)

        # K/V from memory bank
        self.Wk = nn.Linear(memory_dim, self.total_head_dim, bias=False)
        self.Wv = nn.Linear(memory_dim, self.total_head_dim, bias=False)

        # Output projection for LLM injection
        self.Wo = nn.Linear(self.total_head_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.Wo.weight)

        # Projection back to memory space for retrieval loss
        # Maps from total_head_dim to memory_dim so we can compare with doc embeddings
        self.W_retrieval = nn.Linear(self.total_head_dim, memory_dim, bias=False)

        # Learned temperature
        init_beta = 1.0 / math.sqrt(head_dim)
        self.log_beta = nn.Parameter(torch.full((num_heads,), math.log(init_beta)))

        # Norms
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_stored = nn.LayerNorm(memory_dim)

        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def forward(
        self,
        hidden: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward with both injection output and retrieval output.

        Args:
            hidden: (batch, seq_len, hidden_dim) LLM hidden state.
            query_embedding: (batch, query_dim) question embedding.

        Returns:
            injection_output: (batch, seq_len, hidden_dim) to add to hidden.
            retrieval_output: (batch, memory_dim) converged state in memory space
                for MSE loss against target document embedding.
            info: sparsity diagnostics.
        """
        if self.memory_bank is None:
            return (
                torch.zeros_like(hidden),
                torch.zeros(hidden.shape[0], self.memory_dim, device=hidden.device),
                {},
            )

        batch, seq_len, _ = hidden.shape
        num_docs = self.memory_bank.shape[0]

        # Project question to query space
        Q = self.Wq(self.norm_query(query_embedding))  # (batch, total_head_dim)
        Q = Q.view(batch, self.num_heads, 1, self.head_dim)

        # Project memory
        normed_mem = self.norm_stored(self.memory_bank)
        K = self.Wk(normed_mem).view(1, num_docs, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(normed_mem).view(1, num_docs, self.num_heads, self.head_dim).transpose(1, 2)

        # Hopfield iterations with entmax
        xi = Q
        beta = self.beta.view(1, self.num_heads, 1, 1)

        for step in range(self.num_steps):
            scores = torch.matmul(xi, K.transpose(-2, -1)) * beta
            weights = entmax15(scores, dim=-1)
            xi = torch.matmul(weights, V.expand(batch, -1, -1, -1))

        # xi: (batch, heads, 1, head_dim) → (batch, total_head_dim)
        converged = xi.squeeze(2).reshape(batch, self.total_head_dim)

        # RETRIEVAL OUTPUT: project back to memory space (for MSE loss)
        retrieval_output = self.W_retrieval(converged)  # (batch, memory_dim)

        # INJECTION OUTPUT: project to hidden dim and broadcast
        injection_output = self.Wo(converged)  # (batch, hidden_dim)
        injection_output = injection_output.unsqueeze(1).expand(-1, seq_len, -1)

        # Sparsity info
        info = {}
        if weights is not None:
            num_nonzero = (weights > 0).float().sum(dim=-1).mean().item()
            info = {
                "num_nonzero": num_nonzero,
                "sparsity": 1.0 - (num_nonzero / num_docs),
                "weights": weights.mean(dim=1).squeeze(1).detach(),
            }

        return injection_output, retrieval_output, info
