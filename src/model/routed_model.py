"""
LLM with Hopfield memory injection + document selection from association weights.

Architecture:
  1. Forward pass through frozen LLM with Hopfield layers at 8/18/30
  2. Each Hopfield layer produces:
     - Memory output added to hidden states (improves generation)
     - Association weights over documents (captures retrieval signal)
  3. DocumentSelector combines the 3 layers' association weights
     into a final document selection
  4. Selected document text is used for evaluation / next-turn context

The key insight: the Hopfield layers learn to retrieve relevant documents
as part of the LM objective. The DocumentSelector just combines their
opinions — no separate retrieval module needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from typing import Optional

from src.model.hopfield_memory import HopfieldMemoryLayer
from src.model.document_selector import DocumentSelector
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RoutedModel(nn.Module):
    """LLM with Hopfield memory injection and document selection."""

    def __init__(self, config: DictConfig):
        super().__init__()
        model_name = config.model.name

        # Load frozen LLM
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

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device = next(self.llm.parameters()).device
        memory_dim = config.memory.embedding_dim

        # Hopfield memory layers at injection points
        injection_layers = list(config.model.get("injection_layers", []))
        self.injection_layers = sorted(injection_layers)
        self.hopfield_layers = nn.ModuleDict()

        for layer_idx in self.injection_layers:
            self.hopfield_layers[str(layer_idx)] = HopfieldMemoryLayer(
                hidden_dim=self.hidden_dim,
                memory_dim=memory_dim,
                num_heads=config.model.get("num_heads", 4),
                association_dim=config.model.get("association_dim", 256),
                scaling=config.model.get("scaling", 0.5),
                update_steps=config.model.get("update_steps", 3),
            )
            logger.info(f"Created Hopfield memory layer at layer {layer_idx}")

        self.hopfield_layers = self.hopfield_layers.to(device)

        # Document selector: combines association weights from all layers
        self.selector = DocumentSelector(
            num_layers=len(self.injection_layers),
            top_k=config.model.get("top_k", 1),
        ).to(device)

        logger.info(f"Model has {self.num_layers} LLM layers, "
                     f"{len(self.injection_layers)} Hopfield layers, "
                     f"DocumentSelector with {len(self.injection_layers)}-layer gating")

        # Chunk texts for document prepending
        self.chunk_texts: list[str] = []

        # Storage for association weights captured during forward pass
        self._captured_weights: dict[int, torch.Tensor] = {}

    def set_memory(self, memory: torch.Tensor) -> None:
        for hopfield in self.hopfield_layers.values():
            hopfield.set_memory(memory)
        logger.info(f"Loaded memory bank: {memory.shape[0]} vectors, {memory.shape[1]}-d")

    def set_chunk_texts(self, texts: list[str]) -> None:
        self.chunk_texts = texts
        logger.info(f"Loaded {len(texts)} chunk texts")

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = []
        for hopfield in self.hopfield_layers.values():
            params.extend(hopfield.parameters())
        params.extend(self.selector.parameters())
        return params

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _make_hook(self, layer_idx: int, capture_weights: bool = False):
        """Create hook that injects memory and optionally captures association weights."""
        hopfield_layer = self.hopfield_layers[str(layer_idx)]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                hidden = output[0]

            if capture_weights:
                # Get both the injection output and the association weights
                memory_out, doc_weights = hopfield_layer.forward_with_association_weights(
                    hidden.float()
                )
                if doc_weights is not None:
                    self._captured_weights[layer_idx] = doc_weights
            else:
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
        target_chunk_idxs: Optional[torch.Tensor] = None,
        capture_weights: bool = True,
    ) -> dict:
        """Forward pass with memory injection and document selection.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) attention mask.
            labels: (batch, seq_len) labels with -100 for ignored positions.
            target_chunk_idxs: (batch,) ground truth document indices.
            capture_weights: Whether to capture association weights for selection.
        """
        self._captured_weights = {}

        # Register hooks
        hooks = []
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, capture_weights=capture_weights)
            )
            hooks.append(h)

        try:
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        finally:
            for h in hooks:
                h.remove()

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

        # Document selection from captured association weights
        selector_result = None
        if capture_weights and len(self._captured_weights) == len(self.injection_layers):
            # Collect weights in layer order
            layer_weights = [
                self._captured_weights[idx] for idx in self.injection_layers
            ]
            selector_result = self.selector(
                layer_weights,
                target_chunk_idxs=target_chunk_idxs,
            )

        return {
            "logits": logits,
            "lm_loss": lm_loss,
            "selector_result": selector_result,
        }

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate with memory injection (no weight capture needed)."""
        hooks = []
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            h = model_layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, capture_weights=False)
            )
            hooks.append(h)

        try:
            output = self.llm.generate(input_ids=input_ids, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        return output

    def select_documents(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a forward pass and return selected document indices."""
        with torch.no_grad():
            result = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                capture_weights=True,
            )
        if result["selector_result"] is not None:
            return result["selector_result"]["top_indices"]
        return torch.tensor([])
