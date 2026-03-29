"""
Logs training metrics to a JSON file for plotting.

Saves per-step: loss, per-layer sparsity, per-layer beta values.
After training, use scripts/plot_training.py to generate figures.
"""

import json
from pathlib import Path
from typing import Optional


class MetricsLogger:
    """Accumulates metrics during training and saves to disk."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []

    def log(
        self,
        step: int,
        epoch: int,
        loss: float,
        sparsity_stats: Optional[dict] = None,
        beta_values: Optional[dict] = None,
    ) -> None:
        record = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
        }

        if sparsity_stats:
            for layer_idx, info in sparsity_stats.items():
                record[f"layer_{layer_idx}_nonzero"] = info["num_nonzero"]
                record[f"layer_{layer_idx}_sparsity"] = info["sparsity"]

        if beta_values:
            for layer_idx, betas in beta_values.items():
                record[f"layer_{layer_idx}_beta_mean"] = sum(betas) / len(betas)
                record[f"layer_{layer_idx}_beta_max"] = max(betas)

        self.records.append(record)

    def save(self) -> str:
        path = self.output_dir / "training_metrics.json"
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        return str(path)
