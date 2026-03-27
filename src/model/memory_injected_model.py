"""
Wraps a frozen Qwen model with Hopfield memory layers injected at specific points.

Different layers serve different purposes:
  - Early layer:  broad topic context ("what domain are we in?")
  - Middle layer: blending relevant documents ("which docs matter?")
  - Late layer:   sharp fact retrieval ("what's the exact answer?")

Only the Hopfield layer parameters are trainable. The entire LLM stays frozen.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from typing import Optional

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
        association_dim = config.model.get("association_dim", 256)

        # Load the frozen LLM
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

        logger.info(f"Model has {self.num_layers} layers, hidden_dim={self.hidden_dim}")

        for layer_idx in injection_layers:
            if layer_idx >= self.num_layers:
                raise ValueError(
                    f"Injection layer {layer_idx} out of range (model has {self.num_layers} layers)"
                )

        self.injection_layers = sorted(injection_layers)
        self.hopfield_layers = nn.ModuleDict()
        for layer_idx in self.injection_layers:
            self.hopfield_layers[str(layer_idx)] = HopfieldMemoryLayer(
                hidden_dim=self.hidden_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
                association_dim=association_dim,
            )
            logger.info(f"Created Hopfield memory layer at layer {layer_idx}")

        device = next(self.llm.parameters()).device
        self.hopfield_layers = self.hopfield_layers.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Storage for hidden states captured during forward pass (for retrieval loss)
        self._captured_hidden_states = {}

    def set_memory(self, memory: torch.Tensor) -> None:
        for layer_idx in self.injection_layers:
            self.hopfield_layers[str(layer_idx)].set_memory(memory)
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

    def _make_hook(self, layer_idx: int, capture_hidden: bool = False):
        """Create a forward hook that injects memory and optionally captures hidden states."""
        hopfield_layer = self.hopfield_layers[str(layer_idx)]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                hidden = output[0]

            # Capture hidden states BEFORE memory injection (for retrieval loss)
            if capture_hidden:
                self._captured_hidden_states[layer_idx] = hidden.detach().float()

            memory_out = hopfield_layer(hidden.float())
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
        compute_retrieval_logits: bool = False,
    ) -> dict:
        """Forward pass with memory injection.

        Args:
            compute_retrieval_logits: If True, also return retrieval logits
                from each Hopfield layer for the auxiliary retrieval loss.
        """
        self._captured_hidden_states = {}

        hooks = []
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, capture_hidden=compute_retrieval_logits)
            )
            hooks.append(h)

        try:
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        finally:
            for h in hooks:
                h.remove()

        # Compute LM loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # Compute retrieval logits from captured hidden states
        retrieval_logits = {}
        if compute_retrieval_logits and self._captured_hidden_states:
            for layer_idx, hidden in self._captured_hidden_states.items():
                hopfield = self.hopfield_layers[str(layer_idx)]
                retrieval_logits[layer_idx] = hopfield.get_retrieval_logits(hidden)

        return {
            "loss": loss,
            "logits": logits,
            "retrieval_logits": retrieval_logits,
        }

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        hooks = []
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, capture_hidden=False)
            )
            hooks.append(h)

        try:
            output = self.llm.generate(input_ids=input_ids, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        return output
