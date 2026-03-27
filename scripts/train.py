"""
Train the memory-augmented model.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.model.memory_injected_model import MemoryInjectedModel
from src.training.trainer import train
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train memory-augmented model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load memory bank
    memory_path = Path(config.memory.output_dir) / "memory_bank.pt"
    if not memory_path.exists():
        logger.error(f"Memory bank not found at {memory_path}. Run build_memory.py first.")
        return

    memory_bank = torch.load(memory_path, weights_only=True)
    logger.info(f"Loaded memory bank: {memory_bank.shape}")

    # Build model
    model = MemoryInjectedModel(config)
    model.set_memory(memory_bank.to(next(model.parameters()).device))

    # Train
    train(model, config)


if __name__ == "__main__":
    main()
