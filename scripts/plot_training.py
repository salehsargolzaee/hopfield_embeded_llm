"""
Generate training analysis plots from metrics JSON.

Usage:
    python scripts/plot_training.py --metrics data/results/training_metrics.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(values, window=50):
    """Running average for noisy loss curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/results/")
    args = parser.parse_args()

    with open(args.metrics) as f:
        records = json.load(f)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]

    # Find layer indices from the keys
    layer_indices = set()
    for r in records:
        for key in r:
            if key.startswith("layer_") and key.endswith("_sparsity"):
                idx = int(key.split("_")[1])
                layer_indices.add(idx)
    layer_indices = sorted(layer_indices)

    # --- Plot 1: Loss curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, alpha=0.2, color="blue", linewidth=0.5)
    if len(losses) > 50:
        smoothed = smooth(losses, window=50)
        ax.plot(steps[24:-25], smoothed, color="blue", linewidth=2, label="Smoothed (window=50)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    path = out / "loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 2: Sparsity evolution per layer ---
    if layer_indices:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2E86AB", "#A23B72", "#F18F01"]

        for i, layer_idx in enumerate(layer_indices):
            key = f"layer_{layer_idx}_nonzero"
            values = [r.get(key, None) for r in records]
            valid_steps = [s for s, v in zip(steps, values) if v is not None]
            valid_values = [v for v in values if v is not None]

            if valid_values:
                color = colors[i % len(colors)]
                ax.plot(valid_steps, valid_values, marker="o", markersize=3,
                        label=f"Layer {layer_idx}", color=color, linewidth=1.5)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Non-zero Documents")
        ax.set_title("Sparsity Evolution: Non-zero Documents per Layer")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=1245, color="gray", linestyle="--", alpha=0.5, label="Total docs (1245)")
        path = out / "sparsity_evolution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # --- Plot 3: Beta evolution per layer ---
    if layer_indices:
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, layer_idx in enumerate(layer_indices):
            key = f"layer_{layer_idx}_beta_mean"
            values = [r.get(key, None) for r in records]
            valid_steps = [s for s, v in zip(steps, values) if v is not None]
            valid_values = [v for v in values if v is not None]

            if valid_values:
                color = colors[i % len(colors)]
                ax.plot(valid_steps, valid_values, marker="o", markersize=3,
                        label=f"Layer {layer_idx}", color=color, linewidth=1.5)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("β (inverse temperature)")
        ax.set_title("Learned β Evolution per Layer")
        ax.legend()
        ax.grid(alpha=0.3)
        path = out / "beta_evolution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # --- Plot 4: Sparsity bar chart (final state) ---
    if layer_indices:
        last = records[-1]
        fig, ax = plt.subplots(figsize=(8, 5))

        layer_labels = [f"Layer {idx}" for idx in layer_indices]
        nonzero_vals = [last.get(f"layer_{idx}_nonzero", 0) for idx in layer_indices]
        sparsity_vals = [last.get(f"layer_{idx}_sparsity", 0) for idx in layer_indices]

        bars = ax.bar(layer_labels, nonzero_vals, color=colors[:len(layer_indices)],
                       edgecolor="white", linewidth=0.5)

        for bar, nz, sp in zip(bars, nonzero_vals, sparsity_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{nz:.0f}\n({sp:.1%} sparse)",
                    ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("Non-zero Documents (out of 1,245)")
        ax.set_title("Final Sparsity Pattern Across Layers")
        ax.set_ylim(0, 1400)
        ax.axhline(y=1245, color="gray", linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        path = out / "final_sparsity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    print("Done")


if __name__ == "__main__":
    main()
