"""
Hopfield router — selects documents to prepend to the LLM's input.

Instead of injecting a compressed vector into the LLM's hidden states,
the router uses Hopfield energy-based convergence to select the most
relevant document(s) from the memory bank. The selected documents' raw
text is prepended to the prompt, giving the LLM full token-level access.

The key difference from standard dense retrieval (cosine/dot product):
the Hopfield update rule iterates through the energy landscape, converging
to the nearest stored pattern. This produces sharper, more committed
retrieval decisions — especially useful when documents are semantically
similar (common in enterprise department knowledge bases).

Flow:
  1. Embed the question using the same model as the memory bank
  2. Use Hopfield convergence to find the nearest stored pattern(s)
  3. Map the converged pattern back to the original document text
  4. Prepend that text to the LLM's input
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hflayers import Hopfield


class HopfieldRouter(nn.Module):
    """Selects documents from memory bank using Hopfield energy minimization.

    Args:
        query_dim: Dimension of query embeddings (from the embedding model).
        memory_dim: Dimension of memory bank vectors (should match query_dim).
        num_heads: Number of Hopfield association heads.
        association_dim: Internal association space dimension.
        scaling: Inverse temperature β. Higher = sharper convergence.
        update_steps: Number of Hopfield update iterations.
        top_k: How many documents to retrieve.
    """

    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        association_dim: int = 256,
        scaling: float = 2.0,
        update_steps: int = 5,
        top_k: int = 3,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.top_k = top_k

        # The Hopfield module handles the energy-based convergence.
        # Query = question embedding, stored patterns = document embeddings.
        # After convergence, we look at which stored patterns the state
        # converged toward (the convergence weights).
        self.hopfield = Hopfield(
            input_size=query_dim,
            stored_pattern_size=memory_dim,
            pattern_projection_size=memory_dim,
            hidden_size=association_dim,
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
        )

        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        """Load the document memory bank."""
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    def forward(
        self, query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route a query to the most relevant documents.

        Args:
            query_embedding: (batch, query_dim) embedded question.

        Returns:
            top_indices: (batch, top_k) indices into the memory bank.
            convergence_scores: (batch, num_docs) the Hopfield convergence
                weights over all documents (for the retrieval loss).
        """
        if self.memory_bank is None:
            raise ValueError("Call set_memory() before routing")

        batch = query_embedding.shape[0]

        # Add sequence dim: (batch, 1, query_dim) — single query per example
        query = query_embedding.unsqueeze(1)

        # Stored patterns: (batch, num_docs, memory_dim)
        stored = self.memory_bank.unsqueeze(0).expand(batch, -1, -1)

        # Run Hopfield convergence — the state iterates toward the nearest
        # energy minimum in the pattern space
        # Output: (batch, 1, memory_dim) — the converged state
        converged = self.hopfield((stored, query, stored))
        converged = converged.squeeze(1)  # (batch, memory_dim)

        # Compute similarity between converged state and all stored patterns.
        # The converged state should be very close to one (or few) stored patterns.
        # This gives us the "convergence weights" — how much each document
        # contributed to the energy minimum the query fell into.
        converged_norm = F.normalize(converged, p=2, dim=-1)
        memory_norm = F.normalize(self.memory_bank, p=2, dim=-1)
        convergence_scores = torch.matmul(converged_norm, memory_norm.T)  # (batch, num_docs)

        # Select top-k documents
        top_scores, top_indices = convergence_scores.topk(self.top_k, dim=-1)

        return top_indices, convergence_scores
