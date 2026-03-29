"""
Document selector that combines association weights from multiple Hopfield layers.

Each Hopfield layer at a different depth produces a soft distribution over
documents in the memory bank. These distributions capture different aspects:
  - Early layers: broad topical relevance
  - Middle layers: blending of related documents
  - Late layers: sharp focus on specific facts

The selector learns a gating function over these distributions to produce
a final document selection. The gate is conditioned on the distributions
themselves, so the model can learn rules like "trust the late layer when
it's confident, fall back to the early layer otherwise."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DocumentSelector(nn.Module):
    """Combines per-layer document distributions into a retrieval decision.

    Args:
        num_layers: Number of Hopfield injection layers (typically 3).
        top_k: How many documents to select.
    """

    def __init__(self, num_layers: int = 3, top_k: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.top_k = top_k

        # Learned gate: which layer's opinion to trust.
        # Input: concatenation of all layer distributions (num_layers * num_docs)
        # Output: per-layer gate values
        # We use a small network that looks at the sharpness/entropy of each
        # distribution to decide which layer is most informative.
        self.gate_net = nn.Sequential(
            nn.Linear(num_layers, num_layers),
            nn.ReLU(),
            nn.Linear(num_layers, num_layers),
        )

        # Learned temperature for the final logits
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        layer_weights: list[torch.Tensor],
        target_chunk_idxs: Optional[torch.Tensor] = None,
    ) -> dict:
        """Combine layer-wise document distributions.

        Args:
            layer_weights: List of (batch, num_docs) tensors, one per Hopfield layer.
                Each tensor is a soft distribution over documents (sums to 1).
            target_chunk_idxs: (batch,) ground truth indices for training.

        Returns:
            Dict with 'top_indices', 'logits', and optionally 'loss'.
        """
        batch = layer_weights[0].shape[0]
        num_docs = layer_weights[0].shape[1]

        # Stack: (batch, num_layers, num_docs)
        stacked = torch.stack(layer_weights, dim=1)

        # Compute per-layer entropy as a signal for the gate.
        # Low entropy = layer is confident about specific documents.
        # (batch, num_layers)
        entropies = -(stacked * (stacked + 1e-10).log()).sum(dim=-1)

        # Gate: (batch, num_layers) — learned combination weights
        gate_logits = self.gate_net(entropies)
        gate = F.softmax(gate_logits, dim=-1)  # (batch, num_layers)

        # Weighted combination: (batch, num_docs)
        # gate: (batch, num_layers, 1) * stacked: (batch, num_layers, num_docs) → sum
        combined = (gate.unsqueeze(-1) * stacked).sum(dim=1)

        # Apply learned temperature and convert to logits
        temperature = self.log_temperature.exp()
        logits = combined.log().clamp(min=-20) * temperature

        # Select top-k documents
        _, top_indices = logits.topk(self.top_k, dim=-1)

        result = {
            "top_indices": top_indices,
            "logits": logits,
            "gate_values": gate,
        }

        # Compute retrieval loss if targets provided
        if target_chunk_idxs is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            result["loss"] = loss_fn(logits, target_chunk_idxs)

        return result
