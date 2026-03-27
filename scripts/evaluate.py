"""
Evaluate the memory-augmented model on SQuAD validation set.

Compares:
  1. Base model (no memory) — how does Qwen do on its own?
  2. Memory-augmented model — does Hopfield memory help?

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_step_500.pt
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from src.model.memory_injected_model import MemoryInjectedModel
from src.evaluation.qa_metrics import compute_squad_metrics
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def evaluate_model(
    model: MemoryInjectedModel,
    questions: list[str],
    ground_truths: list[list[str]],
    max_new_tokens: int = 50,
) -> dict[str, float]:
    """Generate answers and compute metrics."""
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    with torch.no_grad():
        for question in tqdm(questions, desc="Evaluating"):
            prompt = f"Question: {question}\nAnswer:"
            inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
            )

            # Decode only the generated part (after the prompt)
            prompt_len = inputs["input_ids"].shape[1]
            answer_ids = output_ids[0][prompt_len:]
            answer = model.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            predictions.append(answer)

    metrics = compute_squad_metrics(predictions, ground_truths)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory-augmented model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Hopfield checkpoint")
    parser.add_argument("--max-samples", type=int, default=500, help="Max questions to evaluate")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load validation questions
    dataset = load_dataset("squad_v2", split="validation", trust_remote_code=True)

    questions = []
    ground_truths = []
    for row in dataset:
        if not row["answers"]["text"]:
            continue
        questions.append(row["question"])
        ground_truths.append(row["answers"]["text"])

    # Limit for speed
    if args.max_samples and len(questions) > args.max_samples:
        questions = questions[:args.max_samples]
        ground_truths = ground_truths[:args.max_samples]

    logger.info(f"Evaluating on {len(questions)} questions")

    # Build model
    model = MemoryInjectedModel(config)

    # Load memory bank
    memory_path = Path(config.memory.output_dir) / "memory_bank.pt"
    if memory_path.exists():
        memory_bank = torch.load(memory_path, weights_only=True)
        model.set_memory(memory_bank.to(next(model.parameters()).device))

    # Evaluate baseline (no training, Wo=0 so memory has no effect)
    logger.info("=== Baseline (no memory, Wo=zeros) ===")
    baseline_metrics = evaluate_model(model, questions, ground_truths)
    logger.info(f"Baseline EM: {baseline_metrics['exact_match']:.4f}")
    logger.info(f"Baseline F1: {baseline_metrics['f1']:.4f}")

    # Load trained checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.hopfield_layers.load_state_dict(checkpoint["hopfield_state_dict"])
        logger.info(f"Loaded checkpoint from step {checkpoint['step']}")

        logger.info("=== Memory-augmented (trained) ===")
        trained_metrics = evaluate_model(model, questions, ground_truths)
        logger.info(f"Trained EM: {trained_metrics['exact_match']:.4f}")
        logger.info(f"Trained F1: {trained_metrics['f1']:.4f}")

        # Comparison
        em_diff = trained_metrics["exact_match"] - baseline_metrics["exact_match"]
        f1_diff = trained_metrics["f1"] - baseline_metrics["f1"]
        logger.info(f"EM improvement: {em_diff:+.4f}")
        logger.info(f"F1 improvement: {f1_diff:+.4f}")


if __name__ == "__main__":
    main()
