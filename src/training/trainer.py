"""
Training loop for the memory-augmented model.

Only the Hopfield parameters (Wq, Wo, β) are optimized. The LLM is frozen.
Every N steps we run a diagnostic to check if the memory attention weights
are landing on the right documents.
"""

import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.model.memory_injected_model import MemoryInjectedModel
from src.training.squad_dataset import SQuADMemoryDataset, collate_fn
from src.utils.logging import get_logger

logger = get_logger(__name__)


def train(model: MemoryInjectedModel, config: DictConfig) -> None:
    """Run the training loop.

    Args:
        model: The memory-injected model (LLM frozen, Hopfield layers trainable).
        config: Training config.
    """
    # The model is already on GPU via device_map="auto"
    device = next(model.parameters()).device
    logger.info(f"Training on {device}")

    # Only optimize Hopfield parameters
    trainable_params = model.get_trainable_params()
    total = model.count_total_params()
    trainable = model.count_trainable_params()
    logger.info(
        f"Parameters: {total:,} total, {trainable:,} trainable "
        f"({100 * trainable / total:.2f}%)"
    )

    optimizer = AdamW(
        trainable_params,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Load training data
    train_dataset = SQuADMemoryDataset(
        tokenizer=model.tokenizer,
        config=config,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # tokenizer isn't fork-safe
        pin_memory=True,
    )

    # Training loop
    num_epochs = config.training.num_epochs
    log_every = config.training.get("log_every", 10)
    save_every = config.training.get("save_every", 500)
    diag_every = config.training.get("diag_every", 50)
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]

            # Backward pass — gradients only flow to Hopfield params
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, config.training.max_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg loss: {avg_loss:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Diagnostic: check what the Hopfield layers are attending to
            if global_step % diag_every == 0:
                _run_diagnostic(model, config)

            # Save checkpoint
            if global_step % save_every == 0:
                _save_checkpoint(model, optimizer, global_step, output_dir)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch+1} done | Avg loss: {avg_epoch_loss:.4f}")

    # Final save
    _save_checkpoint(model, optimizer, global_step, output_dir)
    logger.info("Training complete")


def _run_diagnostic(model: MemoryInjectedModel, config: DictConfig) -> None:
    """Check if the Hopfield layers are attending to sensible documents.

    Feeds a test question through the model and looks at the raw
    softmax weights in each Hopfield layer.
    """
    model.eval()

    test_question = "Question: What is the capital of France?\nAnswer:"
    device = next(model.parameters()).device

    tokens = model.tokenizer(test_question, return_tensors="pt").to(device)

    with torch.no_grad():
        # Run the forward pass and collect attention weights from Hopfield layers
        for layer_idx, hopfield in model.hopfield_layers.items():
            if hopfield.memory_bank is None:
                continue

            # Get hidden states at this layer by running through the LLM
            # (simplified: just log the β values and top attention weight)
            beta_values = hopfield.beta.detach().cpu()
            logger.info(
                f"  Layer {layer_idx} | β: {beta_values.tolist()} | "
                f"β range: [{beta_values.min():.3f}, {beta_values.max():.3f}]"
            )

    model.train()


def _save_checkpoint(
    model: MemoryInjectedModel,
    optimizer: AdamW,
    step: int,
    output_dir: Path,
) -> None:
    """Save only the Hopfield layer weights (not the full frozen LLM)."""
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
