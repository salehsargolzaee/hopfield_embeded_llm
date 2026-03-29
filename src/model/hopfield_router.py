"""
Hopfield router with learned retrieval head.

Architecture:
  1. Hopfield convergence: question embedding iterates through the energy
     landscape defined by document embeddings (stored patterns)
  2. Retrieval head: small MLP maps the converged representation to
     document selection logits via cross-entropy

The Hopfield dynamics provide a richer representation than a raw query
embedding — the converged state encodes information about which region
of the document space the query falls into. The retrieval head then
makes the discrete selection.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hflayers import Hopfield


class HopfieldPoolingRouter(nn.Module):
    """Selects documents using Hopfield convergence + retrieval head.

    Args:
        memory_dim: Dimension of memory bank vectors.
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

        # Hopfield convergence: produces a representation that encodes
        # which documents the query is near in the energy landscape
        self.hopfield = Hopfield(
            input_size=memory_dim,
            stored_pattern_size=memory_dim,
            pattern_projection_size=memory_dim,
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
        )

        # Retrieval head: maps converged representation to a query vector
        # that can be compared against memory bank for document selection
        self.retrieval_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )

        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    def forward(
        self, query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route queries to documents.

        Args:
            query_embedding: (batch, memory_dim) question embeddings.

        Returns:
            top_indices: (batch, top_k) indices into memory bank.
            logits: (batch, num_docs) retrieval logits for loss computation.
        """
        if self.memory_bank is None:
            raise ValueError("Call set_memory() first")

        batch = query_embedding.shape[0]

        # Hopfield convergence: query walks through energy landscape
        state = query_embedding.unsqueeze(1)  # (batch, 1, dim)
        stored = self.memory_bank.unsqueeze(0).expand(batch, -1, -1)  # (batch, docs, dim)
        converged = self.hopfield((stored, state, stored)).squeeze(1)  # (batch, dim)

        # Combine raw query + converged state — the retrieval head sees both
        combined = query_embedding + converged

        # Retrieval head: map to document selection space
        retrieval_query = self.retrieval_head(combined)  # (batch, dim)
        retrieval_query = F.normalize(retrieval_query, dim=-1)
        memory_norm = F.normalize(self.memory_bank, dim=-1)

        # Scaled logits for cross-entropy
        logits = (retrieval_query @ memory_norm.T) * 20.0  # (batch, num_docs)

        _, top_indices = logits.topk(self.top_k, dim=-1)

        return top_indices, logits
