"""
Hopfield memory layer using the official hflayers library.

Each layer does two things:
  1. Injects memory into the LLM's hidden states (the main output, added
     to the hidden state via residual connection)
  2. Produces association weights over the memory bank — a soft distribution
     showing which documents the Hopfield convergence attended to

The association weights from multiple layers can be combined by a
DocumentSelector to make explicit document retrieval decisions.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hflayers import Hopfield


class HopfieldMemoryLayer(nn.Module):
    """A single Hopfield memory injection point.

    Args:
        hidden_dim: LLM hidden state dimension (2048 for Qwen 2.5 3B).
        memory_dim: Memory bank vector dimension (768 for BGE-base).
        num_heads: Number of Hopfield association heads.
        association_dim: Internal association space dimension per head.
        scaling: Inverse temperature β.
        update_steps: Hopfield convergence iterations.
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
        self.num_heads = num_heads

        self.hopfield = Hopfield(
            input_size=hidden_dim,
            stored_pattern_size=memory_dim,
            pattern_projection_size=memory_dim,
            hidden_size=association_dim,
            output_size=hidden_dim,
            num_heads=num_heads,
            scaling=scaling if scaling is not None else 0.5,
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

        # Zero-init output projection so layer starts as no-op
        for name, param in self.hopfield.named_parameters():
            if "out_proj" in name and "weight" in name:
                nn.init.zeros_(param)
                break

        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Standard forward: returns memory output to add to hidden state."""
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

    def forward_with_association_weights(
        self, hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns the association weights.

        Uses hflayers' internal _associate() with return_raw_associations=True
        to capture the exact weights the Hopfield convergence produces —
        including the effect of all update steps and normalization.

        Args:
            hidden: (batch, seq_len, hidden_dim) from the LLM.

        Returns:
            output: (batch, seq_len, hidden_dim) memory output for injection.
            doc_weights: (batch, num_docs) soft distribution over documents,
                averaged across heads and sequence positions.
        """
        if self.memory_bank is None:
            return torch.zeros_like(hidden), None

        squeezed = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
            squeezed = True

        batch = hidden.shape[0]
        stored = self.memory_bank.unsqueeze(0).expand(batch, -1, -1)

        # Call _associate directly to get raw association weights.
        # Returns: (output, None, association_weights, None)
        # association_weights shape: (batch, num_heads, seq_len, num_docs)
        assoc_result = self.hopfield._associate(
            data=(stored, hidden, stored),
            return_raw_associations=True,
        )

        # Output needs to be transposed back if batch_first
        # _associate returns (seq, batch, dim) internally when batch_first=True
        # because it transposes before processing then transposes back in forward()
        raw_output = assoc_result[0]
        if raw_output.shape[0] != batch and raw_output.shape[1] == batch:
            raw_output = raw_output.transpose(0, 1)

        # Association weights: (batch, num_heads, seq_len, num_docs)
        raw_weights = assoc_result[2]

        # Average over heads and sequence positions to get per-document weights
        # (batch, num_heads, seq_len, num_docs) → (batch, num_docs)
        doc_weights = raw_weights.mean(dim=1).mean(dim=1)

        if squeezed:
            raw_output = raw_output.squeeze(0)

        return raw_output, doc_weights
