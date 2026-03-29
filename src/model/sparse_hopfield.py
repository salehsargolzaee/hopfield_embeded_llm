"""
Sparse Hopfield memory layer.

Replaces the softmax in the Modern Hopfield update rule with entmax-1.5,
producing exact zeros on irrelevant documents. This has two effects:

1. Sharper retrieval: only the most relevant documents contribute to the
   output, instead of a blurred average over all 1,245 documents.

2. Meta-stable states become interpretable: when the sparse attention
   spreads across k documents (k > 1), those k documents form a
   recognized cluster. The non-zero set IS the retrieval result, and
   everything outside it is provably irrelevant (exact zero weight).

Based on: "Sparse and Structured Hopfield Networks" (Santos et al., ICML 2024)
Implementation: custom Hopfield update rule with entmax from the entmax library.

We implement the update rule directly instead of using hflayers because
hflayers hardcodes softmax and doesn't support sparse alternatives.

The Hopfield update rule (sparse version):
    ξ_{t+1} = V · entmax(β · K^T · Q)

Where:
    Q = Wq(hidden_state)    — query from LLM hidden state
    K = Wk(memory_bank)     — keys from document embeddings
    V = Wv(memory_bank)     — values from document embeddings
    β = learned inverse temperature
    entmax = sparse alternative to softmax (exact zeros on low-scoring docs)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


class SparseHopfieldMemoryLayer(nn.Module):
    """Hopfield memory layer with entmax-1.5 for sparse retrieval.

    Args:
        hidden_dim: LLM hidden state dimension.
        memory_dim: Memory bank vector dimension.
        num_heads: Number of association heads.
        head_dim: Dimension per head in association space.
        num_steps: Number of Hopfield update iterations.
        dropout: Dropout on association weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        head_dim: int = 64,
        num_steps: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_steps = num_steps
        self.total_head_dim = num_heads * head_dim

        # Projections
        self.Wq = nn.Linear(hidden_dim, self.total_head_dim, bias=False)
        self.Wk = nn.Linear(memory_dim, self.total_head_dim, bias=False)
        self.Wv = nn.Linear(memory_dim, self.total_head_dim, bias=False)
        self.Wo = nn.Linear(self.total_head_dim, hidden_dim, bias=False)

        # Zero-init output so layer starts as no-op
        nn.init.zeros_(self.Wo.weight)

        # Learned inverse temperature per head (log scale for positivity)
        init_beta = 1.0 / math.sqrt(head_dim)
        self.log_beta = nn.Parameter(torch.full((num_heads,), math.log(init_beta)))

        # Layer norms for stored and state patterns
        self.norm_state = nn.LayerNorm(hidden_dim)
        self.norm_stored = nn.LayerNorm(memory_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.register_buffer("memory_bank", None)

        # Cache for projected keys/values (recomputed only when memory changes)
        self._cached_K = None
        self._cached_V = None

    def set_memory(self, memory: torch.Tensor) -> None:
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory
        # Invalidate cache
        self._cached_K = None
        self._cached_V = None

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def _get_projected_memory(self) -> tuple:
        """Project memory bank to key/value space. Cached across forward calls."""
        if self._cached_K is None or self._cached_V is None:
            normed = self.norm_stored(self.memory_bank)  # (num_docs, memory_dim)
            self._cached_K = self.Wk(normed)  # (num_docs, total_head_dim)
            self._cached_V = self.Wv(normed)  # (num_docs, total_head_dim)
        return self._cached_K, self._cached_V

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Inject sparse memory signal into hidden states.

        Args:
            hidden: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim).

        Returns:
            Same shape as input — sparse memory output to add to hidden.
        """
        result, _ = self.forward_with_sparsity(hidden)
        return result

    def forward_with_sparsity(
        self, hidden: torch.Tensor,
    ) -> tuple:
        """Forward pass that also returns sparsity information.

        Returns:
            output: Same shape as input — memory output.
            attention_info: Dict with 'weights' (batch, num_docs) averaged
                association weights, and 'num_nonzero' average number of
                documents with non-zero weight.
        """
        if self.memory_bank is None:
            return torch.zeros_like(hidden), None

        squeezed = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
            squeezed = True

        batch, seq_len, _ = hidden.shape
        num_docs = self.memory_bank.shape[0]

        # Project query
        Q = self.Wq(self.norm_state(hidden))  # (batch, seq, total_head_dim)

        # Get cached key/value projections
        K, V = self._get_projected_memory()

        # Reshape for multi-head: (batch, num_heads, seq, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_expanded = K.view(1, num_docs, self.num_heads, self.head_dim).transpose(1, 2)
        V_expanded = V.view(1, num_docs, self.num_heads, self.head_dim).transpose(1, 2)

        # Iterative Hopfield update with entmax
        xi = Q  # initial state = projected query
        beta = self.beta.view(1, self.num_heads, 1, 1)

        all_weights = None
        for step in range(self.num_steps):
            # Similarity: (batch, heads, seq, num_docs)
            scores = torch.matmul(xi, K_expanded.transpose(-2, -1)) * beta

            # SPARSE association via entmax-1.5
            # This is the key difference from standard Hopfield:
            # entmax produces exact zeros on low-scoring documents
            weights = entmax15(scores, dim=-1)

            # Update state: weighted sum of values
            xi = torch.matmul(self.dropout(weights), V_expanded.expand(batch, -1, -1, -1))

        all_weights = weights  # keep final weights for analysis

        # Reshape back: (batch, seq, total_head_dim)
        output = xi.transpose(1, 2).contiguous().view(batch, seq_len, self.total_head_dim)

        # Project to hidden dim
        output = self.Wo(output)

        # Compute sparsity info
        attention_info = None
        if all_weights is not None:
            # Average over heads and seq positions: (batch, num_docs)
            avg_weights = all_weights.mean(dim=1).mean(dim=1)
            num_nonzero = (all_weights > 0).float().sum(dim=-1).mean().item()
            attention_info = {
                "weights": avg_weights.detach(),
                "num_nonzero": num_nonzero,
                "sparsity": 1.0 - (num_nonzero / num_docs),
            }

        if squeezed:
            output = output.squeeze(0)

        return output, attention_info
