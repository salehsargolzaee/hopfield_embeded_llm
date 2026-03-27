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

The key difference from naive cross-attention: the Hopfield energy
function with log-sum-exp creates proper attractor dynamics, and the
learned projections + pattern normalization help separate similar
documents (reducing meta-stable states).
"""

from typing import Optional

import torch
import torch.nn as nn

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
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        self.hopfield = Hopfield(
            # State pattern (query) comes from LLM hidden states
            input_size=hidden_dim,
            # Stored pattern (key) comes from document embeddings
            stored_pattern_size=memory_dim,
            # Pattern projection (value) same source as stored patterns
            pattern_projection_size=memory_dim,
            # Internal association space — where Q and K meet
            hidden_size=association_dim,
            # Output projects back to LLM hidden dim
            output_size=hidden_dim,

            num_heads=num_heads,
            scaling=scaling,

            # Normalize all patterns to reduce meta-stable states
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

        # Memory bank stored as buffer (moves to GPU but no gradients)
        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        """Load the document memory bank.

        Args:
            memory: (num_docs, memory_dim) tensor of L2-normalized document embeddings.
        """
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Query memory and return the retrieved context.

        Args:
            hidden: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim).

        Returns:
            Same shape as input — memory-informed vector to add to hidden.
        """
        if self.memory_bank is None:
            return torch.zeros_like(hidden)

        squeezed = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
            squeezed = True

        batch = hidden.shape[0]

        # Expand memory bank: (batch, num_docs, memory_dim)
        stored = self.memory_bank.unsqueeze(0).expand(batch, -1, -1)

        # Hopfield forward: tuple of (stored_pattern, state_pattern, pattern_projection)
        # stored_pattern = keys (document embeddings)
        # state_pattern = queries (LLM hidden states)
        # pattern_projection = values (same as stored patterns)
        result = self.hopfield((stored, hidden, stored))

        if squeezed:
            result = result.squeeze(0)

        return result
