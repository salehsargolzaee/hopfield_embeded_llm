"""
Train the combined model: HopfieldPooling router + internal memory layers.

Phase 1: Train router only (MSE loss, no LLM forward pass).
  Fast — just teach the pooling layer to map question → document embeddings.

Phase 2: Joint training (MSE router loss + LM loss with internal layers).
  Router picks documents to prepend, internal layers inject memory signal.

Usage:
    python scripts/train_routed.py --config configs/default.yaml
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

from src.model.routed_model import RoutedModel
from src.embedding.embedder import Embedder
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load memory bank and chunks
    memory_bank = torch.load(Path(config.memory.output_dir) / "memory_bank.pt", weights_only=True)
    with open("data/processed/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    chunk_texts = [c.text for c in chunks]

    # Build model
    model = RoutedModel(config)
    model.set_memory(memory_bank.to(next(model.llm.parameters()).device))
    model.set_chunk_texts(chunk_texts)

    logger.info(f"Trainable params: {model.count_trainable_params():,} / {model.count_total_params():,}")

    # Load data + build context mapping
    logger.info("Loading SQuAD v2")
    dataset = load_dataset("squad_v2", split="train")
    embedder = Embedder(config)

    unique_contexts = list(set(row["context"] for row in dataset if row["answers"]["text"]))
    logger.info(f"Embedding {len(unique_contexts)} contexts for mapping...")
    context_embeddings = embedder.embed_texts(unique_contexts)
    memory_np = memory_bank.cpu().numpy()
    sims = context_embeddings @ memory_np.T
    context_to_chunk = {}
    for i, ctx in enumerate(unique_contexts):
        best = int(sims[i].argmax())
        if float(sims[i, best]) > 0.5:
            context_to_chunk[ctx] = best
    logger.info(f"Mapped {len(context_to_chunk)}/{len(unique_contexts)} contexts")

    examples = []
    for row in dataset:
        if not row["answers"]["text"] or row["context"] not in context_to_chunk:
            continue
        examples.append({
            "question": row["question"],
            "answer": row["answers"]["text"][0],
            "target_chunk_idx": context_to_chunk[row["context"]],
        })
    logger.info(f"Prepared {len(examples)} examples")

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = config.training.batch_size
    log_every = config.training.get("log_every", 10)

    # ============================================================
    # PHASE 1: Router warm-up (MSE loss only, no LLM)
    # ============================================================
    phase1_steps = config.training.get("phase1_steps", 3000)
    phase1_lr = config.training.get("phase1_lr", 1e-3)

    logger.info("=" * 60)
    logger.info(f"PHASE 1: Router warm-up ({phase1_steps} steps, CE loss)")
    logger.info("=" * 60)

    # Only optimize router params in phase 1
    router_params = list(model.router.parameters())
    optimizer = AdamW(router_params, lr=phase1_lr, weight_decay=0.01)
    model.train()
    np.random.shuffle(examples)

    step = 0
    idx = 0
    start = time.time()

    while step < phase1_steps:
        if idx + batch_size > len(examples):
            np.random.shuffle(examples)
            idx = 0

        batch = examples[idx:idx + batch_size]
        idx += batch_size

        questions = [ex["question"] for ex in batch]
        target_idxs = torch.tensor([ex["target_chunk_idx"] for ex in batch], dtype=torch.long).to(device)

        # Embed questions
        query_embs = torch.from_numpy(embedder.embed_texts(questions)).float().to(device)

        # Router forward — get retrieval logits
        _, logits = model.router(query_embs)

        # Cross-entropy loss on retrieval logits
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fn(logits, target_idxs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        optimizer.step()

        step += 1
        if step % log_every == 0:
            logger.info(f"Phase 1 | Step {step}/{phase1_steps} | Ret loss: {loss.item():.4f} | Time: {time.time()-start:.1f}s")

    # Check retrieval accuracy
    _check_accuracy(model, embedder, examples[:500], device)

    # Save phase 1
    torch.save({"router": {k: v.cpu() for k, v in model.router.state_dict().items()}},
               output_dir / "phase1_router.pt")
    logger.info("Phase 1 complete")

    # ============================================================
    # PHASE 2: Joint training (router MSE + LM loss)
    # ============================================================
    num_epochs = config.training.num_epochs
    phase2_lr = config.training.lr
    router_loss_weight = config.training.get("retrieval_loss_weight", 1.0)
    save_every = config.training.get("save_every", 500)

    logger.info("=" * 60)
    logger.info(f"PHASE 2: Joint training ({num_epochs} epochs)")
    logger.info("=" * 60)

    # Optimize all trainable params (router + internal layers)
    optimizer = AdamW(model.get_trainable_params(), lr=phase2_lr, weight_decay=0.01)
    global_step = 0

    for epoch in range(num_epochs):
        np.random.shuffle(examples)
        epoch_lm = 0.0
        epoch_router = 0.0
        epoch_steps = 0
        start = time.time()

        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start:batch_start + batch_size]
            if len(batch) < 2:
                continue

            questions = [ex["question"] for ex in batch]
            answers = [ex["answer"] for ex in batch]
            target_idxs = torch.tensor([ex["target_chunk_idx"] for ex in batch], dtype=torch.long).to(device)
            query_embs = torch.from_numpy(embedder.embed_texts(questions)).float().to(device)

            outputs = model(
                query_embeddings=query_embs,
                questions=questions,
                answers=answers,
                target_chunk_idxs=target_idxs,
            )

            total_loss = outputs["loss"] + router_loss_weight * outputs["router_loss"]

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
            optimizer.step()

            epoch_lm += outputs["loss"].item()
            epoch_router += outputs["router_loss"].item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                logger.info(
                    f"Phase 2 Ep{epoch+1} | Step {global_step} | "
                    f"LM: {outputs['loss'].item():.4f} (avg {epoch_lm/epoch_steps:.4f}) | "
                    f"Router MSE: {outputs['router_loss'].item():.6f} (avg {epoch_router/epoch_steps:.6f}) | "
                    f"Time: {time.time()-start:.1f}s"
                )

            if global_step % save_every == 0:
                _save(model, global_step, output_dir)

        logger.info(f"Epoch {epoch+1} done | LM: {epoch_lm/max(epoch_steps,1):.4f} | Router: {epoch_router/max(epoch_steps,1):.6f}")
        _check_accuracy(model, embedder, examples[:500], device)

    _save(model, global_step, output_dir)
    logger.info("Training complete")


def _check_accuracy(model, embedder, examples, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(examples), 32):
            batch = examples[i:i+32]
            questions = [ex["question"] for ex in batch]
            targets = [ex["target_chunk_idx"] for ex in batch]
            query_embs = torch.from_numpy(embedder.embed_texts(questions)).float().to(device)
            top_indices, _ = model.router(query_embs)
            for j, target in enumerate(targets):
                if target in top_indices[j].cpu().tolist():
                    correct += 1
                total += 1
    logger.info(f"Retrieval accuracy (top-{model.router.top_k}): {correct}/{total} = {correct/max(total,1):.3f}")
    model.train()


def _save(model, step, output_dir):
    torch.save({
        "step": step,
        "router": {k: v.cpu() for k, v in model.router.state_dict().items()},
        "hopfield_layers": {k: v.cpu() for k, v in model.hopfield_layers.state_dict().items()},
    }, output_dir / f"combined_step_{step}.pt")
    logger.info(f"Saved checkpoint step {step}")


if __name__ == "__main__":
    main()
