"""
Training loop for the memory-augmented model.

Two losses:
  1. LM loss: cross-entropy on answer tokens (teaches the model to generate answers)
  2. Retrieval loss: cross-entropy on Hopfield attention targets (teaches WHICH docs to retrieve)

The retrieval loss gives direct gradient signal to the Hopfield Q/K projections,
solving the problem where the LM loss gradient is too diluted after passing
through dozens of frozen LLM layers.
"""

import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.model.memory_injected_model import MemoryInjectedModel
from src.training.squad_dataset import SQuADMemoryDataset, collate_fn
from src.utils.logging import get_logger

logger = get_logger(__name__)


def train(model: MemoryInjectedModel, config: DictConfig) -> None:
    device = next(model.parameters()).device
    logger.info(f"Training on {device}")

    trainable_params = model.get_trainable_params()
    total = model.count_total_params()
    trainable = model.count_trainable_params()
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)")

    optimizer = AdamW(
        trainable_params,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Load chunk texts and memory bank for context → chunk mapping
    chunks_path = Path("data/processed/chunks.pkl")
    memory_path = Path(config.memory.output_dir) / "memory_bank.pt"
    chunk_texts = None
    memory_bank_cpu = None
    if chunks_path.exists():
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        chunk_texts = [c.text for c in chunks]
        logger.info(f"Loaded {len(chunk_texts)} chunk texts for retrieval supervision")
    if memory_path.exists():
        memory_bank_cpu = torch.load(memory_path, weights_only=True)

    train_dataset = SQuADMemoryDataset(
        tokenizer=model.tokenizer,
        config=config,
        split="train",
        memory_bank=memory_bank_cpu,
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

    retrieval_loss_weight = config.training.get("retrieval_loss_weight", 1.0)
    num_epochs = config.training.num_epochs
    log_every = config.training.get("log_every", 10)
    save_every = config.training.get("save_every", 500)
    diag_every = config.training.get("diag_every", 50)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        epoch_lm_loss = 0.0
        epoch_ret_loss = 0.0
        epoch_steps = 0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            target_chunks = batch["target_chunk_idxs"].to(device)

            has_retrieval_targets = (target_chunks >= 0).any()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                compute_retrieval_logits=has_retrieval_targets,
            )

            lm_loss = outputs["loss"]
            total_loss = lm_loss

            # Auxiliary retrieval loss: teach Hopfield layers which docs to attend to
            ret_loss = torch.tensor(0.0, device=device)
            if has_retrieval_targets and outputs["retrieval_logits"]:
                ret_loss = _compute_retrieval_loss(
                    outputs["retrieval_logits"],
                    target_chunks,
                )
                total_loss = lm_loss + retrieval_loss_weight * ret_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.training.max_grad_norm)
            optimizer.step()

            epoch_lm_loss += lm_loss.item()
            epoch_ret_loss += ret_loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                avg_lm = epoch_lm_loss / epoch_steps
                avg_ret = epoch_ret_loss / epoch_steps
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | "
                    f"LM loss: {lm_loss.item():.4f} (avg {avg_lm:.4f}) | "
                    f"Ret loss: {ret_loss.item():.4f} (avg {avg_ret:.4f}) | "
                    f"Time: {elapsed:.1f}s"
                )

            if global_step % diag_every == 0:
                _run_diagnostic(model)

            if global_step % save_every == 0:
                _save_checkpoint(model, optimizer, global_step, output_dir)

        avg_lm = epoch_lm_loss / max(epoch_steps, 1)
        avg_ret = epoch_ret_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch+1} done | LM loss: {avg_lm:.4f} | Ret loss: {avg_ret:.4f}")

    _save_checkpoint(model, optimizer, global_step, output_dir)
    logger.info("Training complete")


def _compute_retrieval_loss(
    retrieval_logits: dict,
    target_chunks: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss on retrieval targets.

    For each Hopfield layer, take the retrieval logits at the first token position
    (the question start) and compute cross-entropy against the correct chunk index.
    Using the first token because it has the broadest view of the question.

    Args:
        retrieval_logits: {layer_idx: (batch, seq_len, num_docs)} from each Hopfield layer
        target_chunks: (batch,) correct chunk indices

    Returns:
        Scalar loss averaged across layers and valid examples.
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    losses = []

    for layer_idx, logits in retrieval_logits.items():
        # Take logits at the first token position: (batch, num_docs)
        first_token_logits = logits[:, 0, :]
        layer_loss = loss_fn(first_token_logits, target_chunks)
        losses.append(layer_loss)

    if not losses:
        return torch.tensor(0.0, device=target_chunks.device)

    return torch.stack(losses).mean()


def _run_diagnostic(model: MemoryInjectedModel) -> None:
    for layer_idx, hopfield in model.hopfield_layers.items():
        for name, param in hopfield.hopfield.named_parameters():
            if "out_proj" in name and "weight" in name:
                norm = param.data.norm().item()
                logger.info(f"  Layer {layer_idx} | out_proj norm: {norm:.4f}")
                break


def _save_checkpoint(model, optimizer, step, output_dir):
    checkpoint = {
        "step": step,
        "hopfield_state_dict": {
            name: param.cpu() for name, param in model.hopfield_layers.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path = output_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
