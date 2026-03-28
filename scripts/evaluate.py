"""
Evaluate the memory-augmented model on SQuAD validation set.

Three conditions compared:
  1. No context    — question only, no help (parametric knowledge baseline)
  2. RAG context   — question + source paragraph in prompt (standard RAG)
  3. Hopfield mem  — question only, but Hopfield layers retrieve from memory bank

Metric: perplexity on the answer tokens. Lower = the model assigns higher
probability to the correct answer. No generation needed — we just measure
how well the model "knows" the answer under each condition.

We also generate a few sample answers for qualitative inspection.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_step_65118.pt
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from src.model.memory_injected_model import MemoryInjectedModel
from src.evaluation.qa_metrics import compute_squad_metrics
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def compute_answer_perplexity(
    model: MemoryInjectedModel,
    questions: list[str],
    answers: list[str],
    contexts: list[str] | None = None,
    use_memory: bool = False,
    desc: str = "Evaluating",
) -> dict:
    """Compute perplexity on answer tokens under a given condition.

    Args:
        model: The model to evaluate.
        questions: List of questions.
        answers: List of gold answers.
        contexts: If provided, prepend context paragraph to each prompt (RAG mode).
        use_memory: If True, Hopfield hooks are active (memory mode).
        desc: Progress bar description.

    Returns:
        Dict with avg perplexity, avg loss, and per-example losses.
    """
    model.eval()
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    tokenizer = model.tokenizer

    total_loss = 0.0
    total_tokens = 0
    per_example_loss = []

    with torch.no_grad():
        for i in tqdm(range(len(questions)), desc=desc):
            # Build the prompt depending on condition
            if contexts is not None:
                prompt = f"Context: {contexts[i]}\nQuestion: {questions[i]}\nAnswer:"
            else:
                prompt = f"Question: {questions[i]}\nAnswer:"

            full_text = prompt + " " + answers[i]

            # Tokenize
            full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            input_ids = full_enc["input_ids"].to(device)
            attention_mask = full_enc["attention_mask"].to(device)
            prompt_len = prompt_enc["input_ids"].shape[1]

            # Labels: -100 for prompt tokens, actual IDs for answer tokens
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            # Forward pass — with or without memory hooks
            if use_memory:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs["logits"]
            else:
                # Bypass Hopfield hooks — run the LLM directly
                outputs = model.llm(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # Compute per-token loss on answer tokens only
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            token_losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Only count non-ignored tokens
            answer_mask = shift_labels.view(-1) != -100
            if answer_mask.sum() > 0:
                answer_loss = token_losses[answer_mask].sum().item()
                answer_count = answer_mask.sum().item()
                total_loss += answer_loss
                total_tokens += answer_count
                per_example_loss.append(answer_loss / answer_count)

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "per_example_losses": per_example_loss,
    }


def generate_samples(
    model: MemoryInjectedModel,
    questions: list[str],
    gold_answers: list[str],
    use_memory: bool = False,
    n: int = 10,
) -> None:
    """Generate and print sample answers for qualitative inspection."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(min(n, len(questions))):
            messages = [
                {"role": "system", "content": "Answer in as few words as possible. Only the answer, no explanation."},
                {"role": "user", "content": questions[i]},
            ]
            prompt = model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

            if use_memory:
                output_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            else:
                output_ids = model.llm.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            answer = model.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
            logger.info(f"  Q: {questions[i]}")
            logger.info(f"  Pred: {answer}")
            logger.info(f"  Gold: {gold_answers[i]}")
            logger.info("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory-augmented model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load validation data
    dataset = load_dataset("squad_v2", split="validation")

    questions = []
    answers = []
    contexts = []
    for row in dataset:
        if not row["answers"]["text"]:
            continue
        questions.append(row["question"])
        answers.append(row["answers"]["text"][0])
        contexts.append(row["context"])

    if args.max_samples and len(questions) > args.max_samples:
        questions = questions[:args.max_samples]
        answers = answers[:args.max_samples]
        contexts = contexts[:args.max_samples]

    logger.info(f"Evaluating on {len(questions)} questions")

    # Build model
    model = MemoryInjectedModel(config)

    # Load memory bank
    memory_path = Path(config.memory.output_dir) / "memory_bank.pt"
    if memory_path.exists():
        memory_bank = torch.load(memory_path, weights_only=True)
        model.set_memory(memory_bank.to(next(model.parameters()).device))

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.hopfield_layers.load_state_dict(checkpoint["hopfield_state_dict"])
        logger.info(f"Loaded checkpoint from step {checkpoint['step']}")

    # === Condition 1: No context (parametric knowledge only) ===
    logger.info("=" * 60)
    logger.info("Condition 1: NO CONTEXT (question only, no memory)")
    logger.info("=" * 60)
    no_ctx = compute_answer_perplexity(
        model, questions, answers,
        contexts=None, use_memory=False,
        desc="No context",
    )
    logger.info(f"  Loss: {no_ctx['avg_loss']:.4f} | Perplexity: {no_ctx['perplexity']:.2f}")

    logger.info("\nSample generations (no context):")
    generate_samples(model, questions, answers, use_memory=False, n=5)

    # === Condition 2: RAG context (paragraph in prompt) ===
    logger.info("=" * 60)
    logger.info("Condition 2: RAG CONTEXT (source paragraph in prompt)")
    logger.info("=" * 60)
    rag_ctx = compute_answer_perplexity(
        model, questions, answers,
        contexts=contexts, use_memory=False,
        desc="RAG context",
    )
    logger.info(f"  Loss: {rag_ctx['avg_loss']:.4f} | Perplexity: {rag_ctx['perplexity']:.2f}")

    # === Condition 3: Hopfield memory (correct memory) ===
    logger.info("=" * 60)
    logger.info("Condition 3: HOPFIELD MEMORY (question only, correct memory)")
    logger.info("=" * 60)
    hopfield = compute_answer_perplexity(
        model, questions, answers,
        contexts=None, use_memory=True,
        desc="Hopfield memory",
    )
    logger.info(f"  Loss: {hopfield['avg_loss']:.4f} | Perplexity: {hopfield['perplexity']:.2f}")

    logger.info("\nSample generations (with memory):")
    generate_samples(model, questions, answers, use_memory=True, n=5)

    # === Condition 4: Half memory (remove half the chunks) ===
    logger.info("=" * 60)
    logger.info("Condition 4: HALF MEMORY (only keep every other chunk)")
    logger.info("=" * 60)
    # If the model is selectively retrieving, removing chunks should hurt
    # performance on questions whose answers were in the removed chunks.
    device = next(model.parameters()).device
    half_memory = memory_bank[::2].to(device)  # keep even-indexed chunks only
    model.set_memory(half_memory)

    half = compute_answer_perplexity(
        model, questions, answers,
        contexts=None, use_memory=True,
        desc="Half memory",
    )
    logger.info(f"  Loss: {half['avg_loss']:.4f} | Perplexity: {half['perplexity']:.2f}")

    # Restore full memory before next test
    model.set_memory(memory_bank.to(device))

    # === Condition 5: Random memory (pure noise) ===
    logger.info("=" * 60)
    logger.info("Condition 5: RANDOM MEMORY (noise vectors, no real documents)")
    logger.info("=" * 60)
    # Replace memory with random unit vectors. If the model's perplexity
    # gets worse, it means the Hopfield layers are actively using memory
    # content — noise hurts because the model trusts what it retrieves.
    random_memory = torch.randn_like(memory_bank)
    random_memory = (random_memory / random_memory.norm(dim=1, keepdim=True)).to(device)
    model.set_memory(random_memory)

    random_mem = compute_answer_perplexity(
        model, questions, answers,
        contexts=None, use_memory=True,
        desc="Random memory",
    )
    logger.info(f"  Loss: {random_mem['avg_loss']:.4f} | Perplexity: {random_mem['perplexity']:.2f}")

    # Restore correct memory for any follow-up
    model.set_memory(memory_bank.to(device))

    # === Summary ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Condition':<25} {'Loss':>8} {'Perplexity':>12}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*12}")
    logger.info(f"  {'No context':<25} {no_ctx['avg_loss']:>8.4f} {no_ctx['perplexity']:>12.2f}")
    logger.info(f"  {'RAG context':<25} {rag_ctx['avg_loss']:>8.4f} {rag_ctx['perplexity']:>12.2f}")
    logger.info(f"  {'Hopfield memory':<25} {hopfield['avg_loss']:>8.4f} {hopfield['perplexity']:>12.2f}")
    logger.info(f"  {'Half memory':<25} {half['avg_loss']:>8.4f} {half['perplexity']:>12.2f}")
    logger.info(f"  {'Random memory':<25} {random_mem['avg_loss']:>8.4f} {random_mem['perplexity']:>12.2f}")
    logger.info("")
    logger.info("How to read this:")
    logger.info("  - Lower perplexity = model assigns higher probability to correct answer")
    logger.info("  - RAG context is the upper bound (model sees the answer paragraph directly)")
    logger.info("  - No context is the lower bound (model relies on parametric knowledge only)")
    logger.info("  - Hopfield memory should be between these two")
    logger.info("  - If Half memory > Hopfield: model uses specific chunks, not just general signal (good)")
    logger.info("  - If Random >> Hopfield: model relies on real document content (good)")


if __name__ == "__main__":
    main()
