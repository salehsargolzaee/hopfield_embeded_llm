"""
Hallucination test: find concrete examples where memory prevents wrong answers.

For each question:
  1. Generate answer WITHOUT memory (baseline — model's parametric knowledge)
  2. Generate answer WITH Hopfield memory
  3. Generate answer WITH RAG context (paragraph in prompt)
  4. Compare all three against gold answer

Focus on cases where baseline is WRONG but memory/RAG is RIGHT.
These are direct evidence of hallucination reduction.

Usage:
    python scripts/hallucination_test.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_step_64476.pt --sparse --max-samples 100
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf

from src.evaluation.qa_metrics import f1_score, exact_match
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def generate_answer(model, question, context=None, use_memory=True):
    """Generate a short answer using the model's chat template."""
    device = next(model.llm.parameters()).device

    if context:
        user_msg = f"Context: {context}\n\nQuestion: {question}"
    else:
        user_msg = question

    messages = [
        {"role": "system", "content": "Answer in as few words as possible. Only the answer, no explanation."},
        {"role": "user", "content": user_msg},
    ]
    prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

    if use_memory and hasattr(model, 'generate'):
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
    return model.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load model
    if args.sparse:
        from src.model.sparse_injected_model import SparseInjectedModel
        model = SparseInjectedModel(config)
    else:
        from src.model.memory_injected_model import MemoryInjectedModel
        model = MemoryInjectedModel(config)

    device = next(model.llm.parameters()).device

    memory_path = Path(config.memory.output_dir) / "memory_bank.pt"
    if memory_path.exists():
        memory_bank = torch.load(memory_path, weights_only=True)
        model.set_memory(memory_bank.to(device))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.hopfield_layers.load_state_dict(checkpoint["hopfield_state_dict"])
        logger.info(f"Loaded checkpoint from step {checkpoint['step']}")

    # Load questions
    dataset = load_dataset("squad_v2", split="validation")
    examples = []
    for row in dataset:
        if not row["answers"]["text"]:
            continue
        examples.append({
            "question": row["question"],
            "gold": row["answers"]["text"],
            "context": row["context"],
        })
        if len(examples) >= args.max_samples:
            break

    logger.info(f"Testing {len(examples)} questions")

    # Counters
    baseline_correct = 0
    memory_correct = 0
    rag_correct = 0
    memory_saved = []  # cases where baseline wrong, memory right
    rag_saved = []
    both_wrong = []
    total = 0

    model.eval()
    with torch.no_grad():
        for i, ex in enumerate(examples):
            question = ex["question"]
            golds = ex["gold"]
            context = ex["context"]

            # Generate under three conditions
            ans_baseline = generate_answer(model, question, context=None, use_memory=False)
            ans_memory = generate_answer(model, question, context=None, use_memory=True)
            ans_rag = generate_answer(model, question, context=context, use_memory=False)

            # Score each
            f1_base = max(f1_score(ans_baseline, g) for g in golds)
            f1_mem = max(f1_score(ans_memory, g) for g in golds)
            f1_rag = max(f1_score(ans_rag, g) for g in golds)

            base_ok = f1_base > 0.5
            mem_ok = f1_mem > 0.5
            rag_ok = f1_rag > 0.5

            if base_ok:
                baseline_correct += 1
            if mem_ok:
                memory_correct += 1
            if rag_ok:
                rag_correct += 1

            # Hallucination cases: baseline wrong, memory right
            if not base_ok and mem_ok:
                memory_saved.append({
                    "question": question,
                    "gold": golds[0],
                    "baseline": ans_baseline,
                    "memory": ans_memory,
                    "rag": ans_rag,
                    "f1_base": f1_base,
                    "f1_mem": f1_mem,
                })

            if not base_ok and rag_ok:
                rag_saved.append({
                    "question": question,
                    "gold": golds[0],
                    "baseline": ans_baseline,
                    "rag": ans_rag,
                })

            if not base_ok and not mem_ok:
                both_wrong.append({
                    "question": question,
                    "gold": golds[0],
                    "baseline": ans_baseline,
                    "memory": ans_memory,
                })

            total += 1

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(examples)}")

    # Results
    logger.info("")
    logger.info("=" * 70)
    logger.info("HALLUCINATION TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Total questions: {total}")
    logger.info(f"Baseline correct (F1>0.5): {baseline_correct}/{total} = {baseline_correct/total:.1%}")
    logger.info(f"Hopfield correct (F1>0.5): {memory_correct}/{total} = {memory_correct/total:.1%}")
    logger.info(f"RAG correct     (F1>0.5): {rag_correct}/{total} = {rag_correct/total:.1%}")
    logger.info("")
    logger.info(f"Cases where baseline WRONG but Hopfield RIGHT: {len(memory_saved)}")
    logger.info(f"Cases where baseline WRONG but RAG RIGHT:      {len(rag_saved)}")
    logger.info(f"Cases where both baseline and Hopfield WRONG:  {len(both_wrong)}")

    # Show concrete examples
    if memory_saved:
        logger.info("")
        logger.info("=" * 70)
        logger.info("HOPFIELD MEMORY PREVENTED HALLUCINATION (baseline wrong, memory right)")
        logger.info("=" * 70)
        for j, case in enumerate(memory_saved[:10]):
            logger.info(f"")
            logger.info(f"  Example {j+1}:")
            logger.info(f"    Question:  {case['question']}")
            logger.info(f"    Gold:      {case['gold']}")
            logger.info(f"    Baseline:  {case['baseline']}  (F1={case['f1_base']:.2f} WRONG)")
            logger.info(f"    Hopfield:  {case['memory']}  (F1={case['f1_mem']:.2f} CORRECT)")
            logger.info(f"    RAG:       {case['rag']}")

    if both_wrong:
        logger.info("")
        logger.info("=" * 70)
        logger.info("BOTH WRONG (hallucination not prevented)")
        logger.info("=" * 70)
        for j, case in enumerate(both_wrong[:5]):
            logger.info(f"")
            logger.info(f"  Example {j+1}:")
            logger.info(f"    Question:  {case['question']}")
            logger.info(f"    Gold:      {case['gold']}")
            logger.info(f"    Baseline:  {case['baseline']}")
            logger.info(f"    Hopfield:  {case['memory']}")


if __name__ == "__main__":
    main()
