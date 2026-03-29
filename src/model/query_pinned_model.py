"""
LLM with query-pinned Hopfield memory injection.

The question embedding (from the same model used for the memory bank)
drives retrieval at every injection layer. The LLM hidden states determine
what the model does with the retrieved memory, but the retrieval itself
is always question-dependent.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from typing import Optional

from src.model.query_pinned_hopfield import QueryPinnedHopfieldLayer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QueryPinnedModel(nn.Module):
    """LLM with question-embedding-driven Hopfield memory layers."""

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

        injection_layers = list(config.model.get("injection_layers", []))
        self.injection_layers = sorted(injection_layers)
        self.hopfield_layers = nn.ModuleDict()

        for layer_idx in self.injection_layers:
            self.hopfield_layers[str(layer_idx)] = QueryPinnedHopfieldLayer(
                query_dim=memory_dim,
                hidden_dim=self.hidden_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_steps=num_steps,
            )
            logger.info(f"Created query-pinned Hopfield at layer {layer_idx}")

        self.hopfield_layers = self.hopfield_layers.to(device)

        # The question embedding is stored here so hooks can access it
        self._question_embedding: Optional[torch.Tensor] = None
        self._sparsity_stats: dict[int, dict] = {}

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
            if self._question_embedding is None:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                hidden = output[0]

            memory_out, info = hopfield(hidden.float(), self._question_embedding)

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
        question_embedding: Optional[torch.Tensor] = None,
        track_sparsity: bool = False,
    ) -> dict:
        """Forward pass with question-driven memory retrieval.

        Args:
            question_embedding: (batch, memory_dim) the question embedded
                with the same model as the memory bank.
        """
        self._question_embedding = question_embedding
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
            self._question_embedding = None

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

    def generate(
        self, input_ids: torch.Tensor,
        question_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self._question_embedding = question_embedding
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
            self._question_embedding = None
        return output

    def log_sparsity(self) -> None:
        for layer_idx in self.injection_layers:
            info = self._sparsity_stats.get(layer_idx)
            if info is None:
                continue
            logger.info(
                f"  Layer {layer_idx}: non-zero={info['num_nonzero']:.1f}, "
                f"sparsity={info['sparsity']:.3f}"
            )
