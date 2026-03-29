"""
Training loop for the memory-augmented model.

Works with both dense (softmax) and sparse (entmax) Hopfield layers.
Only the Hopfield parameters are optimized. The LLM stays frozen.
"""

import pickle
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.training.squad_dataset import SQuADMemoryDataset, collate_fn
from src.utils.logging import get_logger

logger = get_logger(__name__)


def train(model, config: DictConfig) -> None:
    # The model is already on GPU via device_map="auto"
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

    # Load chunk texts and memory bank for retrieval supervision
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

    num_epochs = config.training.num_epochs
    log_every = config.training.get("log_every", 10)
    save_every = config.training.get("save_every", 500)
    diag_every = config.training.get("diag_every", 200)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if model supports sparsity tracking
    has_sparsity = hasattr(model, 'log_sparsity')

    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass — works with both dense and sparse models
            track = has_sparsity and (global_step % diag_every == 0)
            if has_sparsity:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    track_sparsity=track,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.training.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} (avg {avg_loss:.4f}) | "
                    f"Time: {elapsed:.1f}s"
                )

            # Log sparsity stats if available
            if track and has_sparsity:
                model.log_sparsity()

            if global_step % save_every == 0:
                _save_checkpoint(model, optimizer, global_step, output_dir)

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch+1} done | Avg loss: {avg_epoch_loss:.4f}")

    _save_checkpoint(model, optimizer, global_step, output_dir)
    logger.info("Training complete")


def _save_checkpoint(model, optimizer, step, output_dir):
    hopfield_dict = {}
    if hasattr(model, 'hopfield_layers'):
        hopfield_dict = {
            name: param.cpu() for name, param in model.hopfield_layers.state_dict().items()
        }

    checkpoint = {
        "step": step,
        "hopfield_state_dict": hopfield_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path = output_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
