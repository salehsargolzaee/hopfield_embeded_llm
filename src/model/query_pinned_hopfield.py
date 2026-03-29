"""
Query-pinned Hopfield memory layer.

The retrieval query is the question embedding (from BGE), NOT a projection
of the LLM's hidden state. This prevents the query from collapsing to
fixed attractors — every different question produces a different retrieval.

What's learned: K, V, Wo projections (how to transform and inject the
retrieved memory). What's fixed: the query (always the question embedding).

Each layer at a different depth learns different K/V/Wo projections,
so they extract different information from the same retrieval:
  Layer 8:  might learn broad topic signal
  Layer 18: might learn relational/structural info
  Layer 30: might learn specific factual details
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


class QueryPinnedHopfieldLayer(nn.Module):
    """Hopfield layer where the query is an external embedding, not from hidden states.

    Args:
        query_dim: Dimension of the external query (768 for BGE).
        hidden_dim: LLM hidden state dimension (2048 for Qwen).
        memory_dim: Memory bank dimension (768 for BGE).
        num_heads: Association heads.
        head_dim: Dimension per head.
        num_steps: Hopfield update iterations.
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

        # Query projection: from question embedding space, NOT from hidden state
        self.Wq = nn.Linear(query_dim, self.total_head_dim, bias=False)

        # Key/Value from memory bank
        self.Wk = nn.Linear(memory_dim, self.total_head_dim, bias=False)
        self.Wv = nn.Linear(memory_dim, self.total_head_dim, bias=False)

        # Output projects to hidden dim (what gets added to the LLM's hidden state)
        self.Wo = nn.Linear(self.total_head_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.Wo.weight)

        # Learned inverse temperature
        init_beta = 1.0 / math.sqrt(head_dim)
        self.log_beta = nn.Parameter(torch.full((num_heads,), math.log(init_beta)))

        # Normalization
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
    ) -> Tuple[torch.Tensor, dict]:
        """Retrieve from memory using the question embedding, inject into hidden state.

        Args:
            hidden: (batch, seq_len, hidden_dim) LLM hidden state.
            query_embedding: (batch, query_dim) question embedding from BGE.

        Returns:
            output: (batch, seq_len, hidden_dim) to add to hidden state.
            info: sparsity diagnostics.
        """
        if self.memory_bank is None:
            return torch.zeros_like(hidden), {}

        batch, seq_len, _ = hidden.shape
        num_docs = self.memory_bank.shape[0]

        # Project question embedding to query space
        # (batch, query_dim) → (batch, total_head_dim)
        Q = self.Wq(self.norm_query(query_embedding))

        # Reshape for multi-head: (batch, num_heads, 1, head_dim)
        # The "1" is because we have one query per example (the question),
        # not one per token position
        Q = Q.view(batch, self.num_heads, 1, self.head_dim)

        # Project memory bank
        normed_mem = self.norm_stored(self.memory_bank)
        K = self.Wk(normed_mem)  # (num_docs, total_head_dim)
        V = self.Wv(normed_mem)

        K = K.view(1, num_docs, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(1, num_docs, self.num_heads, self.head_dim).transpose(1, 2)

        # Hopfield update with entmax
        xi = Q
        beta = self.beta.view(1, self.num_heads, 1, 1)

        for step in range(self.num_steps):
            scores = torch.matmul(xi, K.transpose(-2, -1)) * beta
            weights = entmax15(scores, dim=-1)  # (batch, heads, 1, num_docs)
            xi = torch.matmul(weights, V.expand(batch, -1, -1, -1))

        # xi shape: (batch, num_heads, 1, head_dim)
        # Reshape to (batch, total_head_dim)
        retrieved = xi.squeeze(2).reshape(batch, self.total_head_dim)

        # Project to hidden dim: (batch, hidden_dim)
        output = self.Wo(retrieved)

        # Broadcast to all token positions: (batch, seq_len, hidden_dim)
        output = output.unsqueeze(1).expand(-1, seq_len, -1)

        # Sparsity info
        info = {}
        if weights is not None:
            num_nonzero = (weights > 0).float().sum(dim=-1).mean().item()
            avg_weights = weights.mean(dim=1).squeeze(1)  # (batch, num_docs)
            info = {
                "num_nonzero": num_nonzero,
                "sparsity": 1.0 - (num_nonzero / num_docs),
                "weights": avg_weights.detach(),
            }

        return output, info
