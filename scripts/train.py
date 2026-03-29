"""
Train the memory-augmented model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --sparse
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.training.trainer import train
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train memory-augmented model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse Hopfield (entmax) instead of dense (softmax)")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Use hierarchical sparse Hopfield with energy initialization cascade")
    parser.add_argument("--query-pinned", action="store_true",
                        help="Use query-pinned Hopfield (question embedding drives retrieval)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    memory_path = Path(config.memory.output_dir) / "memory_bank.pt"
    if not memory_path.exists():
        logger.error(f"Memory bank not found at {memory_path}. Run build_memory.py first.")
        return

    memory_bank = torch.load(memory_path, weights_only=True)
    logger.info(f"Loaded memory bank: {memory_bank.shape}")

    if args.query_pinned:
        from src.model.query_pinned_model import QueryPinnedModel
        logger.info("Using QUERY-PINNED Hopfield (question embedding drives retrieval)")
        model = QueryPinnedModel(config)
    elif args.hierarchical:
        from src.model.hierarchical_model import HierarchicalSparseModel
        logger.info("Using HIERARCHICAL sparse Hopfield (energy initialization cascade)")
        model = HierarchicalSparseModel(config)
    elif args.sparse:
        from src.model.sparse_injected_model import SparseInjectedModel
        logger.info("Using SPARSE Hopfield (entmax-1.5)")
        model = SparseInjectedModel(config)
    else:
        from src.model.memory_injected_model import MemoryInjectedModel
        logger.info("Using DENSE Hopfield (softmax)")
        model = MemoryInjectedModel(config)

    model.set_memory(memory_bank.to(next(model.parameters()).device))

    train(model, config)


if __name__ == "__main__":
    main()
