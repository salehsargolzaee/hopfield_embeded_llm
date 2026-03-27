"""
Build the memory bank from document chunks.

Takes SQuAD contexts, chunks them, embeds them with a sentence-transformer,
L2-normalizes, and saves as a .pt file ready to load into the Hopfield layers.

Usage:
    python scripts/build_memory.py --config configs/default.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.embedding.embedder import Embedder
from src.ingestion.registry import get_source
from src.ingestion.chunker import chunk_documents
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build memory bank")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load and chunk documents
    source = get_source(config)
    logger.info(f"Loading from: {source.name()}")
    documents = list(source.load_documents())
    chunks = chunk_documents(documents, config)
    logger.info(f"Created {len(chunks)} chunks")

    # Embed and normalize
    embedder = Embedder(config)
    embeddings = embedder.embed_chunks(chunks)  # already L2-normalized

    # Convert to torch tensor
    memory_bank = torch.from_numpy(embeddings).float()
    logger.info(f"Memory bank shape: {memory_bank.shape}")

    # Save memory bank
    output_dir = Path(config.memory.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "memory_bank.pt"
    torch.save(memory_bank, output_path)
    logger.info(f"Saved memory bank to {output_path}")

    # Save chunks for retrieval supervision (trainer needs text for context mapping)
    import pickle
    chunks_path = Path("data/processed/chunks.pkl")
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")


if __name__ == "__main__":
    main()
