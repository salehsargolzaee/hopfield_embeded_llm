"""
Train the Hopfield-routed model.

The router learns to select documents via Hopfield convergence.
Selected documents are prepended as text to the LLM input.

Usage:
    python scripts/train_routed.py --config configs/default.yaml
"""

import argparse
import pickle
import time
from pathlib import Path

import torch
import numpy as np
from torch.optim import AdamW
from datasets import load_dataset
from omegaconf import OmegaConf

from src.model.routed_model import RoutedModel
from src.embedding.embedder import Embedder
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Hopfield-routed model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load memory bank and chunk texts
    memory_path = Path(config.memory.output_dir) / "memory_bank.pt"
    chunks_path = Path("data/processed/chunks.pkl")
    memory_bank = torch.load(memory_path, weights_only=True)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    chunk_texts = [c.text for c in chunks]

    # Build model
    model = RoutedModel(config)
    model.set_memory(memory_bank.to(next(model.llm.parameters()).device))
    model.set_chunk_texts(chunk_texts)

    trainable = model.count_trainable_params()
    total = model.count_total_params()
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")

    optimizer = AdamW(
        model.get_trainable_params(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Load training data
    logger.info("Loading SQuAD v2 training data")
    dataset = load_dataset("squad_v2", split="train")

    # Build context → chunk mapping for retrieval targets
    logger.info("Building context → chunk embedding mapping...")
    embedder = Embedder(config)

    # Get unique contexts and their target chunk indices
    unique_contexts = list(set(row["context"] for row in dataset if row["answers"]["text"]))
    context_embeddings = embedder.embed_texts(unique_contexts)
    memory_np = memory_bank.cpu().numpy()
    similarities = context_embeddings @ memory_np.T
    context_to_chunk = {}
    for i, ctx in enumerate(unique_contexts):
        best_idx = int(similarities[i].argmax())
        if float(similarities[i, best_idx]) > 0.5:
            context_to_chunk[ctx] = best_idx

    logger.info(f"Mapped {len(context_to_chunk)}/{len(unique_contexts)} contexts to chunks")

    # Prepare training examples
    examples = []
    for row in dataset:
        if not row["answers"]["text"]:
            continue
        ctx = row["context"]
        if ctx not in context_to_chunk:
            continue
        examples.append({
            "question": row["question"],
            "answer": row["answers"]["text"][0],
            "target_chunk_idx": context_to_chunk[ctx],
        })

    logger.info(f"Prepared {len(examples)} training examples")

    # Training loop
    batch_size = config.training.batch_size
    num_epochs = config.training.num_epochs
    retrieval_loss_weight = config.training.get("retrieval_loss_weight", 1.0)
    log_every = config.training.get("log_every", 10)
    save_every = config.training.get("save_every", 500)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        # Shuffle examples each epoch
        np.random.shuffle(examples)

        epoch_lm_loss = 0.0
        epoch_ret_loss = 0.0
        epoch_steps = 0
        start_time = time.time()

        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start:batch_start + batch_size]
            if len(batch) < 2:
                continue

            questions = [ex["question"] for ex in batch]
            answers = [ex["answer"] for ex in batch]
            target_idxs = torch.tensor(
                [ex["target_chunk_idx"] for ex in batch], dtype=torch.long,
            ).to(device)

            # Embed questions for the router
            query_embs = torch.from_numpy(
                embedder.embed_texts(questions)
            ).float().to(device)

            outputs = model(
                query_embeddings=query_embs,
                questions=questions,
                answers=answers,
                target_chunk_idxs=target_idxs,
            )

            total_loss = outputs["loss"] + retrieval_loss_weight * outputs["retrieval_loss"]

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), config.training.max_grad_norm)
            optimizer.step()

            epoch_lm_loss += outputs["loss"].item()
            epoch_ret_loss += outputs["retrieval_loss"].item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                avg_lm = epoch_lm_loss / epoch_steps
                avg_ret = epoch_ret_loss / epoch_steps
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | "
                    f"LM: {outputs['loss'].item():.4f} (avg {avg_lm:.4f}) | "
                    f"Ret: {outputs['retrieval_loss'].item():.4f} (avg {avg_ret:.4f}) | "
                    f"Time: {elapsed:.1f}s"
                )

            if global_step % save_every == 0:
                ckpt = {
                    "step": global_step,
                    "router_state_dict": {
                        k: v.cpu() for k, v in model.router.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                path = output_dir / f"routed_checkpoint_step_{global_step}.pt"
                torch.save(ckpt, path)
                logger.info(f"Saved checkpoint to {path}")

        logger.info(f"Epoch {epoch+1} done | LM: {epoch_lm_loss/max(epoch_steps,1):.4f} | Ret: {epoch_ret_loss/max(epoch_steps,1):.4f}")

    # Final save
    path = output_dir / f"routed_checkpoint_step_{global_step}.pt"
    torch.save({
        "step": global_step,
        "router_state_dict": {k: v.cpu() for k, v in model.router.state_dict().items()},
    }, path)
    logger.info(f"Training complete. Final checkpoint: {path}")


if __name__ == "__main__":
    main()
