"""
Hopfield memory layer using the official hflayers library from
"Hopfield Networks is All You Need" (Ramsauer et al., 2020).

This replaces our previous naive cross-attention implementation with the
actual Hopfield association mechanism. Key differences:

1. The Hopfield module learns projections for queries, keys, AND values —
   it doesn't just attend over a frozen memory bank, it learns how to
   transform the stored patterns into useful representations.

2. The energy function is the real Hopfield energy with log-sum-exp,
   not just scaled dot-product attention.

3. Pattern normalization and learned affine transforms are built in,
   which helps separate similar patterns (reducing meta-stable states).

4. Scaling (β) is handled properly by the library, either as a fixed
   value or as 1/sqrt(head_dim).

The memory bank (document embeddings) is passed as the "stored patterns."
The LLM hidden states are the "state patterns" (queries). The library
handles all the projection, association, and retrieval internally.
"""

import math

import torch
import torch.nn as nn

from hflayers import Hopfield


class HopfieldMemoryLayer(nn.Module):
    """A single Hopfield memory injection point using the official library.

    Args:
        hidden_dim: Dimension of the LLM's hidden states (e.g. 2048).
        memory_dim: Dimension of the memory bank vectors (e.g. 768).
        num_heads: Number of association heads.
        scaling: Inverse temperature β. None = use 1/sqrt(head_dim).
        dropout: Dropout on association weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        scaling: float | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        # The Hopfield module handles everything:
        # - Projects state patterns (queries) from hidden_dim
        # - Projects stored patterns (keys) from memory_dim
        # - Projects pattern projections (values) from memory_dim
        # - Computes Hopfield association (energy-based retrieval)
        # - Output projection back to hidden_dim
        self.hopfield = Hopfield(
            # Input = LLM hidden states (state pattern / query)
            input_size=hidden_dim,

            # Stored patterns = document embeddings (keys)
            stored_pattern_size=memory_dim,

            # Pattern projection = what we retrieve (values), same as stored
            pattern_projection_size=memory_dim,

            # Internal association space dimension
            # Smaller than hidden_dim to keep param count manageable
            pattern_size=memory_dim,

            # Output projects back to hidden_dim
            output_size=hidden_dim,

            num_heads=num_heads,
            scaling=scaling,

            # Stored patterns (memory bank) are static — not trainable
            stored_pattern_as_static=True,
            # State patterns (queries from LLM) are NOT static — they get projected
            state_pattern_as_static=False,
            # Pattern projection (values) are static — we retrieve the actual embeddings
            pattern_projection_as_static=True,
            # Connect pattern projection to stored patterns (values = keys)
            pattern_projection_as_connected=True,

            # Normalize patterns to reduce meta-stable states
            normalize_stored_pattern=True,
            normalize_stored_pattern_affine=True,
            normalize_state_pattern=True,
            normalize_state_pattern_affine=True,
            normalize_pattern_projection=True,
            normalize_pattern_projection_affine=True,

            batch_first=True,
            dropout=dropout,
        )

        # Zero-initialize the output projection so the layer starts as a no-op.
        # The Hopfield module's output projection is the last linear layer.
        self._zero_init_output()

        # Memory bank stored as buffer
        self.register_buffer("memory_bank", None)

    def _zero_init_output(self) -> None:
        """Zero-initialize the output projection weights."""
        # The Hopfield module wraps a HopfieldCore which has an out_projection
        # We need to find and zero it
        for name, param in self.hopfield.named_parameters():
            if "out_proj" in name.lower() or "out_projection" in name.lower():
                if "weight" in name:
                    nn.init.zeros_(param)

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
            hidden: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim) from the LLM.

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

        # Expand memory bank to match batch size: (batch, num_docs, memory_dim)
        stored = self.memory_bank.unsqueeze(0).expand(batch, -1, -1)

        # The Hopfield module takes:
        #   input = state patterns (queries) — our LLM hidden states
        #   stored_pattern_padding_mask = optional mask
        # and uses the stored patterns passed here as keys/values
        result = self.hopfield(input=hidden, stored_pattern=stored)

        if squeezed:
            result = result.squeeze(0)

        return result
