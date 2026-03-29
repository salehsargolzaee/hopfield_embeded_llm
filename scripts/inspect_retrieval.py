"""
Inspect what each Hopfield layer retrieves for specific questions.

Shows the cascade: which document(s) each layer attends to,
how the retrieval narrows across layers, and what text is being retrieved.

Usage:
    python scripts/inspect_retrieval.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_step_64476.pt --hierarchical
"""

import argparse
import pickle
from pathlib import Path

import torch
from datasets import load_dataset
from omegaconf import OmegaConf

from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--query-pinned", action="store_true")
    parser.add_argument("--max-questions", type=int, default=20)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load chunks
    with open("data/processed/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    chunk_texts = [c.text for c in chunks]
    chunk_sources = [c.source_doc for c in chunks]

    # Load model
    if args.query_pinned:
        from src.model.query_pinned_model import QueryPinnedModel
        model = QueryPinnedModel(config)
    elif args.hierarchical:
        from src.model.hierarchical_model import HierarchicalSparseModel
        model = HierarchicalSparseModel(config)
    elif args.sparse:
        from src.model.sparse_injected_model import SparseInjectedModel
        model = SparseInjectedModel(config)
    else:
        from src.model.memory_injected_model import MemoryInjectedModel
        model = MemoryInjectedModel(config)

    device = next(model.llm.parameters()).device
    memory_bank = torch.load(Path(config.memory.output_dir) / "memory_bank.pt", weights_only=True)
    model.set_memory(memory_bank.to(device))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.hopfield_layers.load_state_dict(checkpoint["hopfield_state_dict"])
        logger.info(f"Loaded checkpoint from step {checkpoint['step']}")

    # Load questions
    dataset = load_dataset("squad_v2", split="validation")
    questions = []
    for row in dataset:
        if not row["answers"]["text"]:
            continue
        questions.append({
            "question": row["question"],
            "gold": row["answers"]["text"][0],
            "context": row["context"][:100] + "...",
            "title": row["title"],
        })
        if len(questions) >= args.max_questions:
            break

    # Load embedder if query-pinned
    question_embedder = None
    if args.query_pinned:
        from src.embedding.embedder import Embedder
        question_embedder = Embedder(config)

    model.eval()
    with torch.no_grad():
        for i, q in enumerate(questions):
            prompt = f"Question: {q['question']}\nAnswer:"
            inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

            # Forward with sparsity tracking
            fwd_kwargs = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                track_sparsity=True,
            )
            if question_embedder is not None and hasattr(model, '_question_embedding'):
                q_emb = torch.from_numpy(
                    question_embedder.embed_texts([q['question']])
                ).float().to(device)
                fwd_kwargs["question_embedding"] = q_emb

            outputs = model(**fwd_kwargs)

            logger.info(f"\n{'='*70}")
            logger.info(f"Q{i+1}: {q['question']}")
            logger.info(f"Gold: {q['gold']}")
            logger.info(f"Source article: {q['title']}")
            logger.info(f"Retrieval cascade:")

            for layer_idx in model.injection_layers if hasattr(model, 'injection_layers') else []:
                stats = model._sparsity_stats.get(layer_idx, {})
                weights = stats.get("weights", None)

                if weights is None:
                    logger.info(f"  Layer {layer_idx}: no weights captured")
                    continue

                # Get top documents for this layer
                w = weights[0]  # first (only) batch item
                nonzero_mask = w > 0
                nonzero_count = nonzero_mask.sum().item()

                # Top 3 documents by weight
                top_vals, top_idxs = w.topk(min(3, len(w)))

                alpha_str = ""
                if hasattr(model, 'hopfield_layers'):
                    hopfield = model.hopfield_layers[str(layer_idx)]
                    if hasattr(hopfield, 'alpha'):
                        alpha_str = f", α={hopfield.alpha:.3f}"

                logger.info(f"  Layer {layer_idx} ({nonzero_count} non-zero docs{alpha_str}):")

                for rank, (val, idx) in enumerate(zip(top_vals, top_idxs)):
                    idx = idx.item()
                    val = val.item()
                    if val == 0:
                        break
                    source = chunk_sources[idx]
                    text_preview = chunk_texts[idx][:120].replace('\n', ' ')
                    match = "MATCH" if source == q["title"] else ""
                    logger.info(f"    #{rank+1} (w={val:.4f}) [{source}] {match}")
                    logger.info(f"       \"{text_preview}...\"")


if __name__ == "__main__":
    main()
