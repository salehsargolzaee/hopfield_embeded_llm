"""
LLM with hierarchical sparse Hopfield memory injection.

Three Hopfield layers form a cascade: each layer's converged state
initializes the next layer's query, creating a coarse-to-fine retrieval
through the document memory bank.

The hooks chain the converged states: layer 8 passes its converged state
to layer 18 via model._chain_state, layer 18 passes to layer 30.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from typing import Optional

from src.model.hierarchical_hopfield import HierarchicalHopfieldLayer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class HierarchicalSparseModel(nn.Module):
    """LLM with cascading sparse Hopfield layers."""

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
        num_heads = config.model.get("num_heads", 4)
        head_dim = config.model.get("head_dim", 64)
        num_steps = config.model.get("update_steps", 3)

        # Create hierarchical Hopfield layers
        # First layer has no bridge (starts fresh), subsequent layers have bridges
        injection_layers = list(config.model.get("injection_layers", []))
        self.injection_layers = sorted(injection_layers)
        self.hopfield_layers = nn.ModuleDict()

        for i, layer_idx in enumerate(self.injection_layers):
            has_bridge = (i > 0)  # first layer starts fresh, rest receive prior state
            self.hopfield_layers[str(layer_idx)] = HierarchicalHopfieldLayer(
                hidden_dim=self.hidden_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_steps=num_steps,
                has_bridge=has_bridge,
            )
            bridge_str = " (with bridge from prior layer)" if has_bridge else " (first layer, fresh start)"
            logger.info(f"Created hierarchical Hopfield at layer {layer_idx}{bridge_str}")

        self.hopfield_layers = self.hopfield_layers.to(device)

        # Chain state: stores converged states between hooks
        self._chain_state: dict[int, torch.Tensor] = {}
        self._sparsity_stats: dict[int, dict] = {}

        logger.info(f"Hierarchical model: {len(self.injection_layers)} layers, "
                     f"cascade: {' → '.join(str(l) for l in self.injection_layers)}")

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

    def _make_hook(self, layer_idx: int, layer_position: int, track_sparsity: bool = False):
        """Create hook that injects memory with hierarchical chaining.

        Args:
            layer_idx: The LLM layer index (8, 18, or 30).
            layer_position: Position in the cascade (0, 1, or 2).
            track_sparsity: Whether to log sparsity info.
        """
        hopfield = self.hopfield_layers[str(layer_idx)]

        # Find the prior layer's index (to read its converged state)
        prior_layer_idx = self.injection_layers[layer_position - 1] if layer_position > 0 else None

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                hidden = output[0]

            # Get prior layer's converged state (if this isn't the first layer)
            prior_converged = None
            if prior_layer_idx is not None and prior_layer_idx in self._chain_state:
                prior_converged = self._chain_state[prior_layer_idx]

            # Run hierarchical Hopfield
            memory_out, converged, info = hopfield(
                hidden.float(),
                prior_converged=prior_converged,
            )

            # Store converged state for the next layer in the cascade
            self._chain_state[layer_idx] = converged.detach()

            if track_sparsity and info:
                self._sparsity_stats[layer_idx] = info

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
        self._chain_state = {}
        self._sparsity_stats = {}

        hooks = []
        model_layers = self.llm.model.layers
        for position, layer_idx in enumerate(self.injection_layers):
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, position, track_sparsity=track_sparsity)
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
        self._chain_state = {}
        hooks = []
        model_layers = self.llm.model.layers
        for position, layer_idx in enumerate(self.injection_layers):
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, position, track_sparsity=False)
            )
            hooks.append(h)
        try:
            output = self.llm.generate(input_ids=input_ids, **kwargs)
        finally:
            for h in hooks:
                h.remove()
        return output

    def log_sparsity(self) -> None:
        for layer_idx in self.injection_layers:
            info = self._sparsity_stats.get(layer_idx)
            if info is None:
                continue
            hopfield = self.hopfield_layers[str(layer_idx)]
            alpha_str = f", α={hopfield.alpha:.3f}" if hopfield.has_bridge else ""
            logger.info(
                f"  Layer {layer_idx}: "
                f"non-zero={info['num_nonzero']:.1f}, "
                f"sparsity={info['sparsity']:.3f}"
                f"{alpha_str}"
            )
