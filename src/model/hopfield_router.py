"""
Hopfield router using HopfieldPooling as a denoising retriever.

Based on the approach from Sargolzaei & Rueda (2024): the HopfieldPooling
layer is trained as a denoiser — given a partial/noisy input (question
embedding), it converges to the nearest stored pattern (document embedding).

Training: MSE between pooling output and the correct document embedding.
Inference: output vector → cosine similarity against memory bank → top-k.

This is simpler and more stable than cross-entropy over 1,245 classes
because MSE regression is a continuous objective — the model learns to
move toward the right embedding rather than classifying among thousands.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hflayers import HopfieldPooling


class HopfieldPoolingRouter(nn.Module):
    """Selects documents using HopfieldPooling denoising.

    Args:
        memory_dim: Dimension of memory bank vectors (768 for BGE-base).
        num_heads: Number of Hopfield association heads.
        scaling: Inverse temperature β.
        update_steps: Hopfield convergence iterations.
        top_k: How many documents to retrieve.
    """

    def __init__(
        self,
        memory_dim: int = 768,
        num_heads: int = 16,
        scaling: float = 0.25,
        update_steps: int = 5,
        top_k: int = 1,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.top_k = top_k

        # HopfieldPooling: input = memory bank, pooling weights = query
        # The pooling layer learns to converge from a question embedding
        # toward the nearest stored document embedding.
        self.pooling = HopfieldPooling(
            input_size=memory_dim,
            hidden_size=memory_dim // num_heads,
            output_size=memory_dim,
            num_heads=num_heads,
            scaling=scaling,
            update_steps_max=update_steps,
            update_steps_eps=1e-4,
            normalize_stored_pattern=True,
            normalize_stored_pattern_affine=True,
            normalize_state_pattern=True,
            normalize_state_pattern_affine=True,
            normalize_pattern_projection=True,
            normalize_pattern_projection_affine=True,
            batch_first=True,
            quantity=1,
            trainable=True,
        )

        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    def forward(
        self, query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route queries to documents.

        The trick: we replace the pooling layer's learned query with the
        actual question embedding before each forward pass. This makes the
        retrieval query-dependent (not static).

        Args:
            query_embedding: (batch, memory_dim) question embeddings.

        Returns:
            top_indices: (batch, top_k) indices into memory bank.
            pooled_output: (batch, memory_dim) the converged embedding
                (used for MSE loss during training).
        """
        if self.memory_bank is None:
            raise ValueError("Call set_memory() first")

        batch = query_embedding.shape[0]

        # Override the pooling weights with the question embedding
        # so the Hopfield convergence starts from the question
        # and walks toward the nearest stored document pattern
        self.pooling.pooling_weights.data = query_embedding.unsqueeze(1).detach().clone()

        # Feed the memory bank as input — pooling attends over all documents
        stored = self.memory_bank.unsqueeze(0).expand(batch, -1, -1)
        pooled = self.pooling(stored)  # (batch, memory_dim)

        # Find nearest documents by cosine similarity
        pooled_norm = F.normalize(pooled, p=2, dim=-1)
        memory_norm = F.normalize(self.memory_bank, p=2, dim=-1)
        similarities = torch.matmul(pooled_norm, memory_norm.T)  # (batch, num_docs)

        _, top_indices = similarities.topk(self.top_k, dim=-1)

        return top_indices, pooled
