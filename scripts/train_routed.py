"""
Train the Hopfield-routed model in two phases.

Phase 1 (warm-up): Train the router on retrieval loss ONLY.
  No LLM forward pass, just teach the Hopfield module to route questions
  to the correct document chunk. Fast because we skip the LLM entirely.

Phase 2 (joint): Train with both retrieval loss AND LM loss.
  The router already knows roughly which documents to pick, so the LLM
  gets useful context from the start. Teacher forcing is used for a
  portion of examples (use the known-correct document instead of the
  router's choice) to stabilize early training.

Usage:
    python scripts/train_routed.py --config configs/default.yaml
"""

import argparse
import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn
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

    # Load training data
    logger.info("Loading SQuAD v2 training data")
    dataset = load_dataset("squad_v2", split="train")

    # Build context → chunk mapping
    logger.info("Building context → chunk embedding mapping...")
    embedder = Embedder(config)

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

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # PHASE 1: Warm-up — retrieval loss only, no LLM forward pass
    # ============================================================
    phase1_steps = config.training.get("phase1_steps", 3000)
    phase1_lr = config.training.get("phase1_lr", 1e-3)
    batch_size = config.training.batch_size
    log_every = config.training.get("log_every", 10)

    logger.info("=" * 60)
    logger.info(f"PHASE 1: Router warm-up ({phase1_steps} steps, lr={phase1_lr})")
    logger.info("=" * 60)

    optimizer = AdamW(
        model.get_trainable_params(),
        lr=phase1_lr,
        weight_decay=config.training.weight_decay,
    )

    ret_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    model.train()
    np.random.shuffle(examples)

    step = 0
    epoch = 0
    idx = 0
    phase1_start = time.time()

    while step < phase1_steps:
        # Get a batch
        if idx + batch_size > len(examples):
            np.random.shuffle(examples)
            idx = 0
            epoch += 1

        batch = examples[idx:idx + batch_size]
        idx += batch_size

        questions = [ex["question"] for ex in batch]
        target_idxs = torch.tensor(
            [ex["target_chunk_idx"] for ex in batch], dtype=torch.long,
        ).to(device)

        # Embed questions and run router only (no LLM)
        query_embs = torch.from_numpy(
            embedder.embed_texts(questions)
        ).float().to(device)

        _, convergence_scores = model.router(query_embs)
        loss = ret_loss_fn(convergence_scores, target_idxs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), config.training.max_grad_norm)
        optimizer.step()

        step += 1
        if step % log_every == 0:
            elapsed = time.time() - phase1_start
            logger.info(
                f"Phase 1 | Step {step}/{phase1_steps} | "
                f"Ret loss: {loss.item():.4f} | Time: {elapsed:.1f}s"
            )

    # Check retrieval accuracy after phase 1
    _check_retrieval_accuracy(model, embedder, examples[:500], device)

    # Save phase 1 checkpoint
    phase1_path = output_dir / "routed_phase1.pt"
    torch.save({
        "step": step,
        "router_state_dict": {k: v.cpu() for k, v in model.router.state_dict().items()},
    }, phase1_path)
    logger.info(f"Phase 1 complete. Saved to {phase1_path}")

    # ============================================================
    # PHASE 2: Joint training — retrieval + LM loss
    # ============================================================
    num_epochs = config.training.num_epochs
    retrieval_loss_weight = config.training.get("retrieval_loss_weight", 1.0)
    teacher_forcing_ratio = config.training.get("teacher_forcing_ratio", 0.5)
    save_every = config.training.get("save_every", 500)

    # Reset optimizer with potentially different lr for phase 2
    phase2_lr = config.training.lr
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=phase2_lr,
        weight_decay=config.training.weight_decay,
    )

    logger.info("=" * 60)
    logger.info(f"PHASE 2: Joint training ({num_epochs} epochs, lr={phase2_lr})")
    logger.info(f"  Teacher forcing ratio: {teacher_forcing_ratio}")
    logger.info("=" * 60)

    global_step = 0

    for epoch in range(num_epochs):
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

            query_embs = torch.from_numpy(
                embedder.embed_texts(questions)
            ).float().to(device)

            # Teacher forcing: sometimes use the correct document instead of
            # the router's choice. This stabilizes early phase 2 training.
            use_teacher = np.random.random() < teacher_forcing_ratio

            if use_teacher:
                # Force the correct documents into the prompt
                outputs = _teacher_forced_forward(
                    model, query_embs, questions, answers, target_idxs, device,
                )
            else:
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
                tf_label = "TF" if use_teacher else "RT"
                logger.info(
                    f"Phase 2 Ep{epoch+1} | Step {global_step} [{tf_label}] | "
                    f"LM: {outputs['loss'].item():.4f} (avg {avg_lm:.4f}) | "
                    f"Ret: {outputs['retrieval_loss'].item():.4f} (avg {avg_ret:.4f}) | "
                    f"Time: {elapsed:.1f}s"
                )

            if global_step % save_every == 0:
                ckpt_path = output_dir / f"routed_checkpoint_step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "router_state_dict": {k: v.cpu() for k, v in model.router.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        # Decay teacher forcing each epoch
        teacher_forcing_ratio *= 0.5
        logger.info(
            f"Epoch {epoch+1} done | LM: {epoch_lm_loss/max(epoch_steps,1):.4f} | "
            f"Ret: {epoch_ret_loss/max(epoch_steps,1):.4f} | "
            f"Next TF ratio: {teacher_forcing_ratio:.3f}"
        )

    # Final save
    final_path = output_dir / f"routed_final_step_{global_step}.pt"
    torch.save({
        "step": global_step,
        "router_state_dict": {k: v.cpu() for k, v in model.router.state_dict().items()},
    }, final_path)
    logger.info(f"Training complete. Final checkpoint: {final_path}")


def _teacher_forced_forward(model, query_embs, questions, answers, target_idxs, device):
    """Run forward pass but force the correct document into the prompt."""
    # Still run the router to get convergence scores for retrieval loss
    _, convergence_scores = model.router(query_embs)
    ret_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    retrieval_loss = ret_loss_fn(convergence_scores, target_idxs)

    # Build prompts with the CORRECT document prepended
    batch_size = len(questions)
    lm_loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        correct_idx = target_idxs[i].item()
        if correct_idx < 0 or correct_idx >= len(model.chunk_texts):
            continue

        context = model.chunk_texts[correct_idx]
        prompt = f"Context: {context}\nQuestion: {questions[i]}\nAnswer:"
        full_text = prompt + " " + answers[i]

        full_enc = model.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)
        prompt_enc = model.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)

        input_ids = full_enc["input_ids"]
        labels = input_ids.clone()
        labels[0, :prompt_enc["input_ids"].shape[1]] = -100

        outputs = model.llm(input_ids=input_ids, attention_mask=full_enc["attention_mask"])
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_loss + loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    lm_loss = lm_loss / max(batch_size, 1)

    return {
        "loss": lm_loss,
        "retrieval_loss": retrieval_loss,
        "convergence_scores": convergence_scores,
    }


def _check_retrieval_accuracy(model, embedder, examples, device):
    """Quick check: what fraction of questions route to the correct chunk?"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(examples), 32):
            batch = examples[i:i+32]
            questions = [ex["question"] for ex in batch]
            targets = [ex["target_chunk_idx"] for ex in batch]

            query_embs = torch.from_numpy(
                embedder.embed_texts(questions)
            ).float().to(device)

            top_indices, _ = model.router(query_embs)
            # Check if correct chunk is in top-k
            for j, target in enumerate(targets):
                if target in top_indices[j].cpu().tolist():
                    correct += 1
                total += 1

    accuracy = correct / max(total, 1)
    top_k = model.router.top_k
    logger.info(f"Retrieval accuracy (top-{top_k}): {correct}/{total} = {accuracy:.3f}")
    model.train()
    return accuracy


if __name__ == "__main__":
    main()
