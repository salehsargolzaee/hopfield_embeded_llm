"""
LLM with dual-loss Hopfield: direct retrieval MSE + LM injection.

Each Hopfield layer produces two outputs:
  1. Injection signal → added to LLM hidden state → trained by LM loss
  2. Retrieval vector → compared to correct doc embedding → trained by MSE loss

The MSE loss trains Q/K/V directly without going through the LLM.
The LM loss trains Wo (how to use the retrieved memory).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from typing import Optional

from src.model.dual_loss_hopfield import DualLossHopfieldLayer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DualLossModel(nn.Module):

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
            self.hopfield_layers[str(layer_idx)] = DualLossHopfieldLayer(
                query_dim=memory_dim,
                hidden_dim=self.hidden_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_steps=num_steps,
            )
            logger.info(f"Created dual-loss Hopfield at layer {layer_idx}")

        self.hopfield_layers = self.hopfield_layers.to(device)

        self._question_embedding: Optional[torch.Tensor] = None
        self._retrieval_outputs: dict[int, torch.Tensor] = {}
        self._sparsity_stats: dict[int, dict] = {}

    def set_memory(self, memory: torch.Tensor) -> None:
        for hopfield in self.hopfield_layers.values():
            hopfield.set_memory(memory)
        self._memory_bank = memory
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

            injection, retrieval, info = hopfield(hidden.float(), self._question_embedding)

            # Store retrieval output for MSE loss
            self._retrieval_outputs[layer_idx] = retrieval

            if track_sparsity and info:
                self._sparsity_stats[layer_idx] = info

            modified = hidden + injection.to(hidden.dtype)

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
        target_chunk_idxs: Optional[torch.Tensor] = None,
        track_sparsity: bool = False,
    ) -> dict:
        self._question_embedding = question_embedding
        self._retrieval_outputs = {}
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

        # LM loss
        lm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # Retrieval MSE loss: direct supervision on Q/K/V
        retrieval_loss = torch.tensor(0.0, device=logits.device)
        if target_chunk_idxs is not None and self._retrieval_outputs:
            target_embs = self._memory_bank[target_chunk_idxs]  # (batch, memory_dim)
            for layer_idx, retrieval_out in self._retrieval_outputs.items():
                retrieval_loss = retrieval_loss + F.mse_loss(retrieval_out, target_embs)
            retrieval_loss = retrieval_loss / len(self._retrieval_outputs)

        return {
            "loss": lm_loss,
            "retrieval_loss": retrieval_loss,
            "logits": logits,
            "sparsity_stats": dict(self._sparsity_stats),
        }

    def generate(
        self, input_ids: torch.Tensor,
        question_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        self._question_embedding = question_embedding
        self._retrieval_outputs = {}
        hooks = []
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
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
