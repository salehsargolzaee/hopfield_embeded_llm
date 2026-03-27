"""
Wraps a frozen Qwen model with Hopfield memory layers injected at specific points.

The idea: the LLM processes tokens normally, but at certain layers we pause,
let the hidden state query a memory bank of document embeddings, and add the
retrieved context back into the hidden state. Then the LLM continues.

Different layers serve different purposes:
  - Early layer:  broad topic context ("what domain are we in?")
  - Middle layer: blending relevant documents ("which docs matter?")
  - Late layer:   sharp fact retrieval ("what's the exact answer?")

Only the Hopfield layer parameters (Wq, Wo, β) are trainable.
The entire LLM stays frozen.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig

from src.model.hopfield_memory import HopfieldMemoryLayer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryInjectedModel(nn.Module):
    """Qwen model with Hopfield memory layers injected at configurable points."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        model_name = config.model.name
        injection_layers = list(config.model.injection_layers)
        memory_dim = config.memory.embedding_dim
        num_heads = config.model.get("num_heads", 4)

        # Load the frozen LLM
        logger.info(f"Loading {model_name} (frozen)")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )

        # Freeze everything in the LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # Figure out the hidden dimension and total layers from the model config
        self.hidden_dim = self.llm.config.hidden_size
        self.num_layers = self.llm.config.num_hidden_layers

        logger.info(f"Model has {self.num_layers} layers, hidden_dim={self.hidden_dim}")

        # Validate injection points
        for layer_idx in injection_layers:
            if layer_idx >= self.num_layers:
                raise ValueError(
                    f"Injection layer {layer_idx} is out of range "
                    f"(model has {self.num_layers} layers, 0-indexed)"
                )

        # Create Hopfield memory layers at each injection point
        self.injection_layers = sorted(injection_layers)
        self.hopfield_layers = nn.ModuleDict()
        for layer_idx in self.injection_layers:
            self.hopfield_layers[str(layer_idx)] = HopfieldMemoryLayer(
                hidden_dim=self.hidden_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
            )
            logger.info(f"Created Hopfield memory layer at layer {layer_idx}")

        # Move Hopfield layers to the same device as the LLM
        device = next(self.llm.parameters()).device
        self.hopfield_layers = self.hopfield_layers.to(device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def set_memory(self, memory: torch.Tensor) -> None:
        """Load the same memory bank into all Hopfield layers."""
        for layer_idx in self.injection_layers:
            self.hopfield_layers[str(layer_idx)].set_memory(memory)
        logger.info(f"Loaded memory bank: {memory.shape[0]} vectors, {memory.shape[1]}-d")

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return only the Hopfield layer parameters (the ones we train)."""
        params = []
        for hopfield in self.hopfield_layers.values():
            params.extend(hopfield.parameters())
        return params

    def count_trainable_params(self) -> int:
        """Count trainable parameters for logging."""
        return sum(p.numel() for p in self.get_trainable_params())

    def count_total_params(self) -> int:
        """Count all parameters (frozen + trainable)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass with memory injection via hooks.

        Instead of manually iterating through layers (which breaks across
        different transformers versions), we register forward hooks on the
        injection layers. The hooks add the Hopfield memory output to the
        hidden states as they pass through. This lets HuggingFace handle
        all the internal plumbing (position embeddings, caching, etc.).
        """
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = output[0]
                memory_out = self.hopfield_layers[str(layer_idx)](hidden.float())
                modified = hidden + memory_out.to(hidden.dtype)
                return (modified,) + output[1:]
            return hook_fn

        # Register hooks at injection points
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            # Let HuggingFace handle the full forward pass
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
        finally:
            for h in hooks:
                h.remove()

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate text using the memory-augmented model.

        For generation we use a hook-based approach instead of manual layer
        iteration, since HuggingFace's generate() handles caching and sampling.
        """
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = output[0]
                memory_out = self.hopfield_layers[str(layer_idx)](hidden.float())
                modified = hidden + memory_out.to(hidden.dtype)
                return (modified,) + output[1:]
            return hook_fn

        # Register hooks at injection points
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            output = self.llm.generate(input_ids=input_ids, **kwargs)
        finally:
            # Always remove hooks
            for h in hooks:
                h.remove()

        return output
