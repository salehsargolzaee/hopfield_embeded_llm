"""
Hyperparameter sweep for Hopfield memory layers.

Trains multiple configurations back-to-back and evaluates each one.
Results are saved to a summary file for comparison.

Usage:
    python scripts/sweep.py --config configs/default.yaml
"""

import argparse
import json
import time
from pathlib import Path
from copy import deepcopy

import torch
from omegaconf import OmegaConf

from src.model.memory_injected_model import MemoryInjectedModel
from src.training.trainer import train
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)

# Each config is: (name, overrides)
SWEEP_CONFIGS = [
    ("lr1e-3_beta0.5_steps3_dim256", {
        "training.lr": 1e-3,
        "model.scaling": 0.5,
        "model.update_steps": 3,
        "model.association_dim": 256,
    }),
    ("lr1e-3_beta2.0_steps3_dim256", {
        "training.lr": 1e-3,
        "model.scaling": 2.0,
        "model.update_steps": 3,
        "model.association_dim": 256,
    }),
    ("lr1e-3_beta0.5_steps5_dim256", {
        "training.lr": 1e-3,
        "model.scaling": 0.5,
        "model.update_steps": 5,
        "model.association_dim": 256,
    }),
    ("lr1e-3_beta2.0_steps5_dim512", {
        "training.lr": 1e-3,
        "model.scaling": 2.0,
        "model.update_steps": 5,
        "model.association_dim": 512,
    }),
    ("lr5e-4_beta1.0_steps5_dim512", {
        "training.lr": 5e-4,
        "model.scaling": 1.0,
        "model.update_steps": 5,
        "model.association_dim": 512,
    }),
]


def evaluate_quick(model, config):
    """Quick perplexity eval on 200 samples — returns the key numbers."""
    import math
    import torch.nn as nn
    from datasets import load_dataset

    device = next(model.parameters()).device
    dataset = load_dataset("squad_v2", split="validation")

    questions, answers, contexts = [], [], []
    for row in dataset:
        if not row["answers"]["text"]:
            continue
        questions.append(row["question"])
        answers.append(row["answers"]["text"][0])
        contexts.append(row["context"])
        if len(questions) >= 200:
            break

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    tokenizer = model.tokenizer
    model.eval()

    results = {}

    for condition, use_memory, ctx in [
        ("no_context", False, None),
        ("hopfield", True, None),
        ("rag", False, contexts),
    ]:
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i in range(len(questions)):
                if ctx is not None:
                    prompt = f"Context: {ctx[i]}\nQuestion: {questions[i]}\nAnswer:"
                else:
                    prompt = f"Question: {questions[i]}\nAnswer:"

                full_text = prompt + " " + answers[i]
                full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
                prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

                input_ids = full_enc["input_ids"].to(device)
                attention_mask = full_enc["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[0, :prompt_enc["input_ids"].shape[1]] = -100

                if use_memory:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs["logits"]
                else:
                    outputs = model.llm(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                token_losses = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                mask = shift_labels.view(-1) != -100
                if mask.sum() > 0:
                    total_loss += token_losses[mask].sum().item()
                    total_tokens += mask.sum().item()

        avg_loss = total_loss / max(total_tokens, 1)
        results[condition] = {"loss": avg_loss, "perplexity": math.exp(avg_loss)}

    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    base_config = OmegaConf.load(args.config)
    sweep_dir = Path("sweep_results")
    sweep_dir.mkdir(exist_ok=True)

    all_results = []

    for run_name, overrides in SWEEP_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"SWEEP RUN: {run_name}")
        logger.info(f"{'='*60}")

        # Apply overrides
        config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        for key, value in overrides.items():
            OmegaConf.update(config, key, value)

        # Set checkpoint dir per run
        config.training.output_dir = f"sweep_results/{run_name}/checkpoints"

        seed_everything(config.seed)
        start_time = time.time()

        # Build model fresh each run
        model = MemoryInjectedModel(config)

        # Load memory bank
        memory_path = Path(config.memory.output_dir) / "memory_bank.pt"
        memory_bank = torch.load(memory_path, weights_only=True)
        model.set_memory(memory_bank.to(next(model.parameters()).device))

        # Train
        train(model, config)
        train_time = time.time() - start_time

        # Evaluate
        logger.info(f"Evaluating {run_name}...")
        eval_results = evaluate_quick(model, config)

        result = {
            "name": run_name,
            "overrides": overrides,
            "train_time_sec": train_time,
            **{f"{k}_ppl": v["perplexity"] for k, v in eval_results.items()},
            **{f"{k}_loss": v["loss"] for k, v in eval_results.items()},
        }
        all_results.append(result)

        logger.info(f"  No context PPL:  {eval_results['no_context']['perplexity']:.2f}")
        logger.info(f"  Hopfield PPL:    {eval_results['hopfield']['perplexity']:.2f}")
        logger.info(f"  RAG PPL:         {eval_results['rag']['perplexity']:.2f}")
        logger.info(f"  Train time:      {train_time:.0f}s")

        # Free GPU memory before next run
        del model
        torch.cuda.empty_cache()

    # Print final comparison
    logger.info(f"\n{'='*60}")
    logger.info("SWEEP SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Config':<40} {'No ctx':>8} {'Hopfield':>10} {'RAG':>8} {'Time':>6}")
    logger.info(f"  {'-'*40} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
    for r in sorted(all_results, key=lambda x: x["hopfield_ppl"]):
        logger.info(
            f"  {r['name']:<40} {r['no_context_ppl']:>8.1f} {r['hopfield_ppl']:>10.2f} "
            f"{r['rag_ppl']:>8.2f} {r['train_time_sec']:>5.0f}s"
        )

    # Save results
    results_path = sweep_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
