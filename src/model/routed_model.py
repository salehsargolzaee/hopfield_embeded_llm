"""
LLM with both Hopfield-routed document prepending AND internal memory injection.

Two memory mechanisms working together:
  1. HopfieldPooling router selects documents to prepend as text (explicit context)
  2. Hopfield memory layers at layers 8/18/30 inject memory into hidden states (implicit context)

The router uses the denoising approach: question embedding → HopfieldPooling
convergence → nearest document embedding → prepend document text.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from typing import Optional

from src.model.hopfield_router import HopfieldPoolingRouter
from src.model.hopfield_memory import HopfieldMemoryLayer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _make_memory_hook(hopfield_layer):
    """Hook for internal memory injection."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        elif isinstance(output, torch.Tensor):
            hidden = output
        else:
            hidden = output[0]

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


class RoutedModel(nn.Module):
    """LLM with Hopfield routing + internal memory injection."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
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

        # --- Router (HopfieldPooling-based) ---
        self.router = HopfieldPoolingRouter(
            memory_dim=memory_dim,
            num_heads=config.model.get("router_num_heads", 16),
            scaling=config.model.get("router_scaling", 0.25),
            update_steps=config.model.get("router_update_steps", 5),
            top_k=config.model.get("top_k", 1),
        ).to(device)

        # --- Internal Hopfield memory layers ---
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
            logger.info(f"Created internal Hopfield memory layer at layer {layer_idx}")

        self.hopfield_layers = self.hopfield_layers.to(device)

        # Chunk texts for document prepending
        self.chunk_texts: list[str] = []

    def set_memory(self, memory: torch.Tensor) -> None:
        self.router.set_memory(memory)
        for hopfield in self.hopfield_layers.values():
            hopfield.set_memory(memory)
        logger.info(f"Loaded memory bank: {memory.shape[0]} vectors, {memory.shape[1]}-d")

    def set_chunk_texts(self, texts: list[str]) -> None:
        self.chunk_texts = texts
        logger.info(f"Loaded {len(texts)} chunk texts for document prepending")

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = list(self.router.parameters())
        for hopfield in self.hopfield_layers.values():
            params.extend(hopfield.parameters())
        return params

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _register_hooks(self) -> list:
        hooks = []
        model_layers = self.llm.model.layers
        for layer_idx in self.injection_layers:
            hopfield = self.hopfield_layers[str(layer_idx)]
            h = model_layers[layer_idx].register_forward_hook(_make_memory_hook(hopfield))
            hooks.append(h)
        return hooks

    def forward(
        self,
        query_embeddings: torch.Tensor,
        questions: list[str],
        answers: list[str],
        target_chunk_idxs: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass with routing + internal memory injection.

        Args:
            query_embeddings: (batch, memory_dim) question embeddings.
            questions: Raw question strings.
            answers: Raw answer strings.
            target_chunk_idxs: (batch,) ground truth chunk indices.
        """
        device = next(self.llm.parameters()).device
        batch_size = len(questions)

        # Step 1: Route — get top documents and pooled embedding
        top_indices, pooled_output = self.router(query_embeddings)

        # Step 2: Build prompts with retrieved document prepended
        lm_loss = torch.tensor(0.0, device=device)

        # Register internal memory hooks
        hooks = self._register_hooks()

        try:
            for i in range(batch_size):
                doc_indices = top_indices[i].cpu().tolist()
                retrieved_texts = [self.chunk_texts[idx] for idx in doc_indices]
                context = "\n".join(retrieved_texts)

                prompt = f"Context: {context}\nQuestion: {questions[i]}\nAnswer:"
                full_text = prompt + " " + answers[i]

                full_enc = self.tokenizer(
                    full_text, return_tensors="pt", truncation=True, max_length=512,
                ).to(device)
                prompt_enc = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512,
                ).to(device)

                input_ids = full_enc["input_ids"]
                labels = input_ids.clone()
                labels[0, :prompt_enc["input_ids"].shape[1]] = -100

                outputs = self.llm(input_ids=input_ids, attention_mask=full_enc["attention_mask"])
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                lm_loss = lm_loss + loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
        finally:
            for h in hooks:
                h.remove()

        lm_loss = lm_loss / max(batch_size, 1)

        # Step 3: Router MSE loss — denoising objective
        # pooled_output should converge toward the correct document embedding
        router_loss = torch.tensor(0.0, device=device)
        if target_chunk_idxs is not None:
            target_embeddings = self.router.memory_bank[target_chunk_idxs]  # (batch, memory_dim)
            router_loss = F.mse_loss(pooled_output, target_embeddings)

        return {
            "loss": lm_loss,
            "router_loss": router_loss,
            "top_indices": top_indices,
            "pooled_output": pooled_output,
        }

    def generate(self, query_embedding: torch.Tensor, question: str, max_new_tokens: int = 50) -> str:
        device = next(self.llm.parameters()).device

        # Route
        top_indices, _ = self.router(query_embedding.unsqueeze(0))
        doc_indices = top_indices[0].cpu().tolist()
        retrieved_texts = [self.chunk_texts[idx] for idx in doc_indices]
        context = "\n".join(retrieved_texts)

        # Build prompt
        messages = [
            {"role": "system", "content": "Answer in as few words as possible. Only the answer, no explanation."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        # Generate with internal memory hooks active
        hooks = self._register_hooks()
        try:
            output_ids = self.llm.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            for h in hooks:
                h.remove()

        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
