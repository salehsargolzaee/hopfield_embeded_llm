"""
Document selector that combines association weights from multiple Hopfield layers.

Each Hopfield layer produces a soft distribution over documents. The selector
learns to combine them using an entropy-conditioned gate: it observes how
confident each layer is (via entropy) and weights them accordingly.

The gate is constrained to avoid collapse — all layers maintain a minimum
contribution, preventing the model from ignoring any layer's signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DocumentSelector(nn.Module):
    """Combines per-layer document distributions into a retrieval decision.

    Args:
        num_layers: Number of Hopfield injection layers.
        top_k: How many documents to select.
        min_gate: Minimum gate value per layer to prevent collapse.
    """

    def __init__(self, num_layers: int = 3, top_k: int = 1, min_gate: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.top_k = top_k
        self.min_gate = min_gate

        # Gate network: entropy of each layer's distribution → gate weights
        self.gate_net = nn.Sequential(
            nn.Linear(num_layers, num_layers * 4),
            nn.ReLU(),
            nn.Linear(num_layers * 4, num_layers),
        )

    def forward(
        self,
        layer_weights: list[torch.Tensor],
        target_chunk_idxs: Optional[torch.Tensor] = None,
    ) -> dict:
        """Combine layer-wise document distributions.

        Args:
            layer_weights: List of (batch, num_docs) tensors. Each is a soft
                distribution over documents (sums to 1).
            target_chunk_idxs: (batch,) ground truth for training.
        """
        # Stack: (batch, num_layers, num_docs)
        stacked = torch.stack(layer_weights, dim=1)

        # Per-layer entropy: low = confident, high = uncertain
        # (batch, num_layers)
        entropies = -(stacked * (stacked + 1e-10).log()).sum(dim=-1)

        # Gate with minimum floor to prevent collapse
        gate_logits = self.gate_net(entropies)
        gate_raw = F.softmax(gate_logits, dim=-1)  # (batch, num_layers)

        # Apply minimum gate: each layer gets at least min_gate weight
        # Rescale the remainder proportionally
        gate = gate_raw * (1 - self.num_layers * self.min_gate) + self.min_gate

        # Weighted combination: (batch, num_docs)
        combined = (gate.unsqueeze(-1) * stacked).sum(dim=1)

        # Use combined distribution directly as logits (already positive, sums to ~1)
        # Scale by a fixed factor for cross-entropy (not learned, to avoid instability)
        logits = combined * 100.0  # scale up for sharper cross-entropy

        _, top_indices = logits.topk(self.top_k, dim=-1)

        result = {
            "top_indices": top_indices,
            "logits": logits,
            "gate_values": gate,
        }

        if target_chunk_idxs is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            result["loss"] = loss_fn(logits, target_chunk_idxs)

        return result
