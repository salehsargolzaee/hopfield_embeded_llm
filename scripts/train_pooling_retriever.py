"""
Train HopfieldPooling as a standalone retriever (no LLM).

Based on Sargolzaei & Rueda (2024): train HopfieldPooling as a denoiser.
Input = question embedding (partial/noisy view of the answer).
Target = correct document embedding (clean pattern).
Loss = MSE between output and target.

After training, use as retriever: question → pooling → nearest doc.

Then combine with internal Hopfield injection layers for the full system.

Usage:
    python scripts/train_pooling_retriever.py --config configs/default.yaml
"""

import argparse
import pickle
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from datasets import load_dataset
from omegaconf import OmegaConf
from hflayers import HopfieldPooling

from src.embedding.embedder import Embedder
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


class PoolingRetriever(torch.nn.Module):
    """HopfieldPooling retriever — exactly like Sargolzaei & Rueda (2024)."""

    def __init__(self, dim=768, num_heads=8, hidden_size=8, scaling=0.25, update_steps=5):
        super().__init__()
        self.dim = dim
        self.pooling = HopfieldPooling(
            input_size=dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            update_steps_max=update_steps,
            scaling=scaling,
            quantity=1,
            batch_first=True,
        )

    def forward(self, x):
        """x: (batch, dim) → out: (batch, dim)"""
        return self.pooling(x.unsqueeze(1))  # (batch, 1, dim) → (batch, dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="Association space dimension per head")
    parser.add_argument("--num-heads", type=int, default=16,
                        help="Number of Hopfield association heads")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Gaussian noise std added to inputs (0 = use question embeddings as-is)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load memory bank
    memory_bank = torch.load(Path(config.memory.output_dir) / "memory_bank.pt", weights_only=True)
    with open("data/processed/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    chunk_texts = [c.text for c in chunks]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build question → document mapping
    logger.info("Loading SQuAD and building question → document mapping...")
    dataset = load_dataset("squad_v2", split="train")
    embedder = Embedder(config)

    unique_contexts = list(set(row["context"] for row in dataset if row["answers"]["text"]))
    logger.info(f"Embedding {len(unique_contexts)} contexts...")
    context_embeddings = embedder.embed_texts(unique_contexts)
    memory_np = memory_bank.cpu().numpy()
    sims = context_embeddings @ memory_np.T
    context_to_chunk = {}
    for i, ctx in enumerate(unique_contexts):
        best = int(sims[i].argmax())
        if float(sims[i, best]) > 0.5:
            context_to_chunk[ctx] = best

    # Build training examples: (question_embedding, target_doc_embedding)
    logger.info("Embedding questions...")
    questions = []
    target_idxs = []
    for row in dataset:
        if not row["answers"]["text"]:
            continue
        if row["context"] not in context_to_chunk:
            continue
        questions.append(row["question"])
        target_idxs.append(context_to_chunk[row["context"]])

    logger.info(f"Embedding {len(questions)} questions...")
    question_embeddings = embedder.embed_texts(questions)
    question_embeddings = torch.from_numpy(question_embeddings).float()
    target_idxs = torch.tensor(target_idxs, dtype=torch.long)
    target_embeddings = memory_bank[target_idxs]

    logger.info(f"Training set: {len(questions)} question-document pairs")

    # Build retriever
    retriever = PoolingRetriever(
        dim=config.memory.embedding_dim,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        scaling=0.25,
        update_steps=5,
    ).to(device)

    params = sum(p.numel() for p in retriever.parameters())
    logger.info(f"Retriever params: {params:,}")

    optimizer = AdamW(retriever.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    n = len(questions)
    batch_size = args.batch_size

    logger.info("=" * 60)
    logger.info(f"Training pooling retriever: {args.epochs} epochs, lr={args.lr}")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_steps = 0
        start = time.time()

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            q_emb = question_embeddings[idx].to(device)
            t_emb = target_embeddings[idx].to(device)

            # Optional noise (like your paper's denoising task)
            if args.noise > 0:
                q_emb = q_emb + args.noise * torch.randn_like(q_emb)

            out = retriever(q_emb)
            loss = F.mse_loss(out, t_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        avg_loss = epoch_loss / epoch_steps
        elapsed = time.time() - start

        # Check retrieval accuracy
        retriever.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for i in range(0, min(n, 2000), batch_size):
                q_emb = question_embeddings[i:i+batch_size].to(device)
                targets = target_idxs[i:i+batch_size]

                out = retriever(q_emb)
                out_norm = F.normalize(out, dim=-1)
                mem_norm = F.normalize(memory_bank.to(device), dim=-1)
                sims = out_norm @ mem_norm.T
                top5 = sims.topk(5, dim=-1).indices.cpu()

                for j in range(len(targets)):
                    if targets[j] == top5[j, 0]:
                        correct_top1 += 1
                    if targets[j] in top5[j]:
                        correct_top5 += 1
                    total += 1

        retriever.train()
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | MSE: {avg_loss:.6f} | "
            f"Top-1: {correct_top1}/{total} ({100*correct_top1/total:.1f}%) | "
            f"Top-5: {correct_top5}/{total} ({100*correct_top5/total:.1f}%) | "
            f"{elapsed:.1f}s"
        )

    # Save
    save_path = output_dir / "pooling_retriever.pt"
    torch.save(retriever.state_dict(), save_path)
    logger.info(f"Saved retriever to {save_path}")


if __name__ == "__main__":
    main()
