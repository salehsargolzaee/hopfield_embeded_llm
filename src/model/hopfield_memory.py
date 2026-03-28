"""
Hopfield memory layer using the official hflayers library from
"Hopfield Networks is All You Need" (Ramsauer et al., 2020).

The Hopfield module learns separate projections for:
  - State patterns (queries from LLM hidden states, dim=2048)
  - Stored patterns (keys from document embeddings, dim=768)
  - Pattern projections (values, same source as keys)

These get projected into a shared association space where the Hopfield
energy-based retrieval happens. The output is projected back to the
LLM's hidden dimension.

For training, we also expose retrieval logits — the raw scores between
projected queries and projected keys — so we can directly supervise
which documents the model attends to (auxiliary retrieval loss).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hflayers import Hopfield


class HopfieldMemoryLayer(nn.Module):
    """A single Hopfield memory injection point using the official library.

    Args:
        hidden_dim: Dimension of the LLM's hidden states (e.g. 2048).
        memory_dim: Dimension of the memory bank vectors (e.g. 768).
        num_heads: Number of association heads.
        association_dim: Internal association space dimension.
        scaling: Inverse temperature beta. None = use library default.
        dropout: Dropout on association weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        association_dim: int = 256,
        scaling: Optional[float] = None,
        update_steps: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.association_dim = association_dim
        self.num_heads = num_heads

        self.hopfield = Hopfield(
            input_size=hidden_dim,
            stored_pattern_size=memory_dim,
            pattern_projection_size=memory_dim,
            hidden_size=association_dim,
            output_size=hidden_dim,
            num_heads=num_heads,
            # Higher β for sharper retrieval. Default 1/sqrt(dim) is too soft
            # for 1245 documents — the softmax spreads weight everywhere.
            scaling=scaling if scaling is not None else 0.5,
            # Enable Hopfield iterative convergence — this is the whole point.
            # The state pattern iterates through the energy landscape to converge
            # on the nearest stored pattern instead of doing a single-pass average.
            update_steps_max=update_steps,
            update_steps_eps=1e-4,
            normalize_stored_pattern=True,
            normalize_stored_pattern_affine=True,
            normalize_state_pattern=True,
            normalize_state_pattern_affine=True,
            normalize_pattern_projection=True,
            normalize_pattern_projection_affine=True,
            batch_first=True,
            dropout=dropout,
        )

        # Zero-initialize the output projection so the layer starts as a no-op
        for name, param in self.hopfield.named_parameters():
            if "out_proj" in name and "weight" in name:
                nn.init.zeros_(param)
                break

        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        """Load the document memory bank."""
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Query memory and return the retrieved context."""
        if self.memory_bank is None:
            return torch.zeros_like(hidden)

        squeezed = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
            squeezed = True

        batch = hidden.shape[0]
        stored = self.memory_bank.unsqueeze(0).expand(batch, -1, -1)
        result = self.hopfield((stored, hidden, stored))

        if squeezed:
            result = result.squeeze(0)
        return result

    def get_retrieval_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute retrieval scores using the Hopfield module's own projections.

        Uses the same Q and K projection weights that the Hopfield association
        uses internally, so the auxiliary retrieval loss directly trains the
        parameters that control what gets retrieved.

        Args:
            hidden: (batch, seq_len, hidden_dim) LLM hidden states.

        Returns:
            (batch, seq_len, num_docs) retrieval logits before softmax.
        """
        if self.memory_bank is None:
            return torch.zeros(hidden.shape[0], hidden.shape[1], 0, device=hidden.device)

        # Get the Q and K projection weights from the Hopfield module
        core = self.hopfield.association_core
        q_weight = core.q_proj_weight  # (association_dim * num_heads, hidden_dim)
        k_weight = core.k_proj_weight  # (association_dim * num_heads, memory_dim)

        # Project hidden states to query space
        queries = F.linear(hidden, q_weight)  # (batch, seq_len, assoc_dim * heads)

        # Project memory bank to key space
        keys = F.linear(self.memory_bank, k_weight)  # (num_docs, assoc_dim * heads)

        # L2-normalize both so dot product = cosine similarity (bounded [-1, 1])
        queries = F.normalize(queries, p=2, dim=-1)
        keys = F.normalize(keys, p=2, dim=-1)

        # Scaled cosine similarity — temperature controls sharpness
        # Using sqrt(dim) scaling like standard attention
        import math
        scale = math.sqrt(queries.shape[-1])
        logits = torch.matmul(queries, keys.T) * scale

        return logits
