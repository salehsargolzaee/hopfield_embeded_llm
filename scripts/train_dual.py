"""
Train the dual-loss Hopfield model.

Two losses trained simultaneously:
  MSE retrieval loss: trains Q/K/V directly (no LLM gradient path)
  LM loss: trains Wo (how to inject retrieved memory)

Usage:
    python scripts/train_dual.py --config configs/default.yaml
"""

import argparse
import pickle
import time
from pathlib import Path

import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.model.dual_loss_model import DualLossModel
from src.embedding.embedder import Embedder
from src.training.squad_dataset import SQuADMemoryDataset, collate_fn
from src.training.metrics_logger import MetricsLogger
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load memory + chunks
    memory_bank = torch.load(Path(config.memory.output_dir) / "memory_bank.pt", weights_only=True)
    with open("data/processed/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    chunk_texts = [c.text for c in chunks]

    # Build model
    model = DualLossModel(config)
    device = next(model.llm.parameters()).device
    model.set_memory(memory_bank.to(device))

    trainable = model.count_trainable_params()
    total = model.count_total_params()
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")

    optimizer = AdamW(
        model.get_trainable_params(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Embedder for question embeddings
    embedder = Embedder(config)

    # Dataset
    train_dataset = SQuADMemoryDataset(
        tokenizer=model.tokenizer,
        config=config,
        split="train",
        memory_bank=memory_bank,
        chunk_texts=chunk_texts,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    num_epochs = config.training.num_epochs
    retrieval_weight = config.training.get("retrieval_loss_weight", 1.0)
    log_every = config.training.get("log_every", 10)
    save_every = config.training.get("save_every", 500)
    diag_every = config.training.get("diag_every", 100)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsLogger(config.evaluation.get("output_dir", "data/results"))
    has_sparsity = hasattr(model, 'log_sparsity')

    global_step = 0
    model.train()

    logger.info("=" * 60)
    logger.info(f"DUAL LOSS TRAINING: {num_epochs} epochs")
    logger.info(f"  LM loss weight: 1.0")
    logger.info(f"  Retrieval MSE weight: {retrieval_weight}")
    logger.info("=" * 60)

    for epoch in range(num_epochs):
        epoch_lm = 0.0
        epoch_ret = 0.0
        epoch_steps = 0
        start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            target_chunks = batch["target_chunk_idxs"].to(device)

            # Embed questions
            q_texts = batch["question_texts"]
            q_embs = torch.from_numpy(embedder.embed_texts(q_texts)).float().to(device)

            track = has_sparsity and (global_step % diag_every == 0)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                question_embedding=q_embs,
                target_chunk_idxs=target_chunks,
                track_sparsity=track,
            )

            lm_loss = outputs["loss"]
            ret_loss = outputs["retrieval_loss"]
            total_loss = lm_loss + retrieval_weight * ret_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
            optimizer.step()

            epoch_lm += lm_loss.item()
            epoch_ret += ret_loss.item()
            epoch_steps += 1
            global_step += 1

            metrics.log(
                step=global_step, epoch=epoch + 1, loss=lm_loss.item(),
                sparsity_stats=outputs.get("sparsity_stats") if track else None,
            )

            if global_step % log_every == 0:
                elapsed = time.time() - start
                logger.info(
                    f"Ep{epoch+1} | Step {global_step} | "
                    f"LM: {lm_loss.item():.4f} (avg {epoch_lm/epoch_steps:.4f}) | "
                    f"Ret MSE: {ret_loss.item():.6f} (avg {epoch_ret/epoch_steps:.6f}) | "
                    f"{elapsed:.0f}s"
                )

            if track and has_sparsity:
                model.log_sparsity()

            if global_step % save_every == 0:
                torch.save({
                    "step": global_step,
                    "hopfield_state_dict": {
                        k: v.cpu() for k, v in model.hopfield_layers.state_dict().items()
                    },
                }, output_dir / f"checkpoint_step_{global_step}.pt")
                logger.info(f"Saved step {global_step}")

        logger.info(
            f"Epoch {epoch+1} done | LM: {epoch_lm/epoch_steps:.4f} | "
            f"Ret MSE: {epoch_ret/epoch_steps:.6f}"
        )

    # Final save
    torch.save({
        "step": global_step,
        "hopfield_state_dict": {
            k: v.cpu() for k, v in model.hopfield_layers.state_dict().items()
        },
    }, output_dir / f"checkpoint_step_{global_step}.pt")
    metrics.save()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
