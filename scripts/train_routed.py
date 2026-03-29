"""
Train the model with Hopfield memory injection + document selection.

The Hopfield layers learn to inject useful memory AND produce association
weights that the DocumentSelector combines for document retrieval.

Both the LM loss (answer quality) and the selector loss (retrieval accuracy)
train the Hopfield layers jointly — the layers must both inject useful
information AND attend to the right documents.

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
from src.training.squad_dataset import SQuADMemoryDataset, collate_fn, PROMPT_TEMPLATE
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    # Load memory bank and chunks
    memory_bank = torch.load(Path(config.memory.output_dir) / "memory_bank.pt", weights_only=True)
    with open("data/processed/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    chunk_texts = [c.text for c in chunks]

    # Build model
    model = RoutedModel(config)
    device = next(model.llm.parameters()).device
    model.set_memory(memory_bank.to(device))
    model.set_chunk_texts(chunk_texts)

    logger.info(f"Trainable: {model.count_trainable_params():,} / {model.count_total_params():,}")

    # Build dataset with retrieval targets
    logger.info("Building context → chunk mapping...")
    embedder = Embedder(config)

    dataset = load_dataset("squad_v2", split="train")
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
    logger.info(f"Mapped {len(context_to_chunk)}/{len(unique_contexts)} contexts")

    # Build training dataset
    train_dataset = SQuADMemoryDataset(
        tokenizer=model.tokenizer,
        config=config,
        split="train",
        memory_bank=memory_bank,
        chunk_texts=chunk_texts,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Training setup
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    num_epochs = config.training.num_epochs
    selector_loss_weight = config.training.get("retrieval_loss_weight", 1.0)
    log_every = config.training.get("log_every", 10)
    save_every = config.training.get("save_every", 500)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()

    logger.info("=" * 60)
    logger.info(f"Training: {num_epochs} epochs, lr={config.training.lr}")
    logger.info(f"Selector loss weight: {selector_loss_weight}")
    logger.info("=" * 60)

    for epoch in range(num_epochs):
        epoch_lm = 0.0
        epoch_sel = 0.0
        epoch_steps = 0
        start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            target_chunks = batch["target_chunk_idxs"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                target_chunk_idxs=target_chunks,
                capture_weights=True,
            )

            lm_loss = outputs["lm_loss"]
            total_loss = lm_loss

            sel_loss = torch.tensor(0.0, device=device)
            if outputs["selector_result"] is not None and "loss" in outputs["selector_result"]:
                sel_loss = outputs["selector_result"]["loss"]
                total_loss = lm_loss + selector_loss_weight * sel_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
            optimizer.step()

            epoch_lm += lm_loss.item()
            epoch_sel += sel_loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                elapsed = time.time() - start

                # Show gate values if available
                gate_str = ""
                if outputs["selector_result"] is not None:
                    gate = outputs["selector_result"]["gate_values"].mean(dim=0)
                    gate_str = f" | Gate: [{', '.join(f'{g:.2f}' for g in gate.tolist())}]"

                logger.info(
                    f"Ep{epoch+1} | Step {global_step} | "
                    f"LM: {lm_loss.item():.4f} (avg {epoch_lm/epoch_steps:.4f}) | "
                    f"Sel: {sel_loss.item():.4f} (avg {epoch_sel/epoch_steps:.4f})"
                    f"{gate_str} | {elapsed:.0f}s"
                )

            if global_step % save_every == 0:
                _save(model, global_step, output_dir)

        # End of epoch: check retrieval accuracy
        logger.info(f"Epoch {epoch+1} | LM: {epoch_lm/epoch_steps:.4f} | Sel: {epoch_sel/epoch_steps:.4f}")
        _check_accuracy(model, train_dataset, device)

    _save(model, global_step, output_dir)
    logger.info("Training complete")


def _check_accuracy(model, dataset, device):
    """Check document selection accuracy on a subset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        indices = list(range(min(200, len(dataset))))
        for i in indices:
            ex = dataset[i]
            target = ex.target_chunk_idx
            if target < 0:
                continue

            input_ids = ex.input_ids.unsqueeze(0).to(device)
            attention_mask = ex.attention_mask.unsqueeze(0).to(device)
            target_t = torch.tensor([target], device=device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_chunk_idxs=target_t,
                capture_weights=True,
            )

            if outputs["selector_result"] is not None:
                selected = outputs["selector_result"]["top_indices"][0].cpu().tolist()
                if target in selected:
                    correct += 1
            total += 1

    top_k = model.selector.top_k
    logger.info(f"Retrieval accuracy (top-{top_k}): {correct}/{total} = {correct/max(total,1):.3f}")
    model.train()


def _save(model, step, output_dir):
    torch.save({
        "step": step,
        "hopfield_layers": {k: v.cpu() for k, v in model.hopfield_layers.state_dict().items()},
        "selector": {k: v.cpu() for k, v in model.selector.state_dict().items()},
    }, output_dir / f"checkpoint_step_{step}.pt")
    logger.info(f"Saved step {step}")


if __name__ == "__main__":
    main()
