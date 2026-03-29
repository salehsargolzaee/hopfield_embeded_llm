"""
Hierarchical Sparse Hopfield Memory with Energy Initialization.

Each Hopfield layer's converged state initializes the next layer's query,
creating a cascading retrieval that narrows from broad topic to specific fact.

The mechanism: after layer 8 converges in the Hopfield energy landscape,
its converged representation (in association space, before output projection)
is passed to layer 18. Layer 18's query starts biased toward the basin of
attraction that layer 8 found, rather than starting fresh from the raw hidden
state. Layer 30 continues from layer 18's basin.

This is energy initialization: the initial point in the energy landscape
determines which attractor you converge to. By chaining the layers, we get
hierarchical retrieval:
  Layer 8:  broad search → converges to topic cluster
  Layer 18: starts in that cluster → narrows to specific documents
  Layer 30: starts in that neighborhood → pinpoints exact passage

Mathematically:
  Q₈ = Wq₈(h₈)                          # fresh query from hidden state
  ξ₈ = HopfieldConverge(Q₈, memory)      # converged state

  Q₁₈ = Wq₁₈(h₁₈) + α₁₈ · Bridge₁₈(ξ₈)  # warm-started query
  ξ₁₈ = HopfieldConverge(Q₁₈, memory)      # converges within ξ₈'s basin

  Q₃₀ = Wq₃₀(h₃₀) + α₃₀ · Bridge₃₀(ξ₁₈)  # further refined
  ξ₃₀ = HopfieldConverge(Q₃₀, memory)       # pinpoints specific pattern

The α values and Bridge projections are learned — the model discovers how
much to trust earlier layers' retrieval decisions.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


class HierarchicalHopfieldLayer(nn.Module):
    """Sparse Hopfield layer that can receive initialization from a prior layer.

    Args:
        hidden_dim: LLM hidden state dimension.
        memory_dim: Memory bank vector dimension.
        num_heads: Number of association heads.
        head_dim: Dimension per head in association space.
        num_steps: Number of Hopfield update iterations.
        has_bridge: Whether this layer receives prior state from an earlier layer.
        dropout: Dropout on association weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        head_dim: int = 64,
        num_steps: int = 3,
        has_bridge: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_steps = num_steps
        self.total_head_dim = num_heads * head_dim

        # Standard projections
        self.Wq = nn.Linear(hidden_dim, self.total_head_dim, bias=False)
        self.Wk = nn.Linear(memory_dim, self.total_head_dim, bias=False)
        self.Wv = nn.Linear(memory_dim, self.total_head_dim, bias=False)
        self.Wo = nn.Linear(self.total_head_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.Wo.weight)

        # Learned inverse temperature per head
        init_beta = 1.0 / math.sqrt(head_dim)
        self.log_beta = nn.Parameter(torch.full((num_heads,), math.log(init_beta)))

        # Layer norms
        self.norm_state = nn.LayerNorm(hidden_dim)
        self.norm_stored = nn.LayerNorm(memory_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Bridge from prior layer's converged state (if applicable)
        self.has_bridge = has_bridge
        if has_bridge:
            # Projects prior layer's converged state to bias this layer's query
            self.bridge = nn.Linear(self.total_head_dim, self.total_head_dim, bias=False)
            # Learnable mixing weight — starts small so the layer works independently first
            self.log_alpha = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ≈ 0.14

        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    @property
    def alpha(self) -> float:
        if self.has_bridge:
            return self.log_alpha.exp().item()
        return 0.0

    def forward(
        self,
        hidden: torch.Tensor,
        prior_converged: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass with optional energy initialization from prior layer.

        Args:
            hidden: (batch, seq_len, hidden_dim) from the LLM.
            prior_converged: (batch, seq_len, total_head_dim) converged state
                from the previous Hopfield layer. If provided and has_bridge=True,
                this biases the initial query toward the prior layer's basin.

        Returns:
            output: (batch, seq_len, hidden_dim) to add to hidden state.
            converged: (batch, seq_len, total_head_dim) this layer's converged
                state to pass to the next layer.
            info: dict with sparsity diagnostics.
        """
        if self.memory_bank is None:
            zeros = torch.zeros_like(hidden)
            return zeros, torch.zeros(*hidden.shape[:2], self.total_head_dim, device=hidden.device), {}

        squeezed = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
            squeezed = True

        batch, seq_len, _ = hidden.shape
        num_docs = self.memory_bank.shape[0]

        # Project query from hidden state
        Q = self.Wq(self.norm_state(hidden))  # (batch, seq, total_head_dim)

        # Apply bridge: bias query toward prior layer's basin
        if self.has_bridge and prior_converged is not None:
            alpha = self.log_alpha.exp()
            bridge_signal = self.bridge(prior_converged)  # (batch, seq, total_head_dim)
            Q = Q + alpha * bridge_signal

        # Project memory bank
        normed_mem = self.norm_stored(self.memory_bank)
        K = self.Wk(normed_mem)  # (num_docs, total_head_dim)
        V = self.Wv(normed_mem)  # (num_docs, total_head_dim)

        # Reshape for multi-head
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_exp = K.view(1, num_docs, self.num_heads, self.head_dim).transpose(1, 2)
        V_exp = V.view(1, num_docs, self.num_heads, self.head_dim).transpose(1, 2)

        # Iterative Hopfield update with entmax
        xi = Q
        beta = self.beta.view(1, self.num_heads, 1, 1)

        for step in range(self.num_steps):
            scores = torch.matmul(xi, K_exp.transpose(-2, -1)) * beta
            weights = entmax15(scores, dim=-1)
            xi = torch.matmul(self.dropout(weights), V_exp.expand(batch, -1, -1, -1))

        # Converged state (before output projection) — pass to next layer
        converged = xi.transpose(1, 2).contiguous().view(batch, seq_len, self.total_head_dim)

        # Output projection
        output = self.Wo(converged)

        # Diagnostics
        info = {}
        if weights is not None:
            num_nonzero = (weights > 0).float().sum(dim=-1).mean().item()
            info = {
                "num_nonzero": num_nonzero,
                "sparsity": 1.0 - (num_nonzero / num_docs),
                "weights": weights.mean(dim=1).mean(dim=1).detach(),  # (batch, num_docs)
            }

        if squeezed:
            output = output.squeeze(0)
            converged = converged.squeeze(0)

        return output, converged, info
