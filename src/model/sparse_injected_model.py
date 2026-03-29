"""
LLM with sparse Hopfield memory injection.

Same architecture as MemoryInjectedModel but using SparseHopfieldMemoryLayer
(entmax-1.5 instead of softmax). This produces exact zeros on irrelevant
documents, making the retrieval pattern interpretable:

- When few documents have non-zero weight → sharp single-document retrieval
- When many documents have non-zero weight → meta-stable cluster → group knowledge

We log sparsity statistics during training and evaluation to understand
what the model is doing at each layer depth.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from typing import Optional

from src.model.sparse_hopfield import SparseHopfieldMemoryLayer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SparseInjectedModel(nn.Module):
    """Qwen with sparse Hopfield memory layers."""

    def __init__(self, config: DictConfig):
        super().__init__()
        model_name = config.model.name

        logger.info(f"Loading {model_name} (frozen)")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        for param in self.llm.parameters():
            param.requires_grad = False

        self.hidden_dim = self.llm.config.hidden_size
        self.num_layers = self.llm.config.num_hidden_layers

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device = next(self.llm.parameters()).device
        memory_dim = config.memory.embedding_dim

        # Sparse Hopfield layers at injection points
        injection_layers = list(config.model.get("injection_layers", []))
        self.injection_layers = sorted(injection_layers)
        self.hopfield_layers = nn.ModuleDict()

        num_heads = config.model.get("num_heads", 4)
        head_dim = config.model.get("head_dim", 64)
        num_steps = config.model.get("update_steps", 3)

        for layer_idx in self.injection_layers:
            self.hopfield_layers[str(layer_idx)] = SparseHopfieldMemoryLayer(
                hidden_dim=self.hidden_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_steps=num_steps,
            )
            logger.info(f"Created sparse Hopfield layer at {layer_idx}")

        self.hopfield_layers = self.hopfield_layers.to(device)

        # Track sparsity stats during forward passes
        self._sparsity_stats: dict[int, dict] = {}

        logger.info(f"Model: {self.num_layers} LLM layers, "
                     f"{len(self.injection_layers)} sparse Hopfield layers")

    def set_memory(self, memory: torch.Tensor) -> None:
        for hopfield in self.hopfield_layers.values():
            hopfield.set_memory(memory)
        logger.info(f"Loaded memory bank: {memory.shape[0]} vectors, {memory.shape[1]}-d")

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = []
        for hopfield in self.hopfield_layers.values():
            params.extend(hopfield.parameters())
        return params

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _make_hook(self, layer_idx: int, track_sparsity: bool = False):
        hopfield = self.hopfield_layers[str(layer_idx)]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                hidden = output[0]

            if track_sparsity:
                memory_out, info = hopfield.forward_with_sparsity(hidden.float())
                if info is not None:
                    self._sparsity_stats[layer_idx] = info
            else:
                memory_out = hopfield(hidden.float())

            modified = hidden + memory_out.to(hidden.dtype)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return modified
            else:
                output[0] = modified
                return output

        return hook_fn

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        track_sparsity: bool = False,
    ) -> dict:
        self._sparsity_stats = {}

        hooks = []
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, track_sparsity=track_sparsity)
            )
            hooks.append(h)

        try:
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        finally:
            for h in hooks:
                h.remove()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
            "sparsity_stats": dict(self._sparsity_stats),
        }

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        hooks = []
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, track_sparsity=False)
            )
            hooks.append(h)

        try:
            output = self.llm.generate(input_ids=input_ids, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        return output

    def log_sparsity(self) -> None:
        """Log sparsity statistics from the last forward pass."""
        for layer_idx in self.injection_layers:
            info = self._sparsity_stats.get(layer_idx)
            if info is None:
                continue
            logger.info(
                f"  Layer {layer_idx}: "
                f"avg non-zero docs={info['num_nonzero']:.1f}, "
                f"sparsity={info['sparsity']:.3f}"
            )
