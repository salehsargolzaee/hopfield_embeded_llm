"""
LLM with Hopfield-routed document retrieval.

The Hopfield router selects the most relevant document(s) from the memory bank,
their raw text is prepended to the question, and the LLM processes everything
with full self-attention. This gives the LLM the same quality of context as RAG,
but with a learned retriever trained end-to-end.

Two loss signals:
  1. LM loss on answer tokens — did prepending this document lead to a correct answer?
  2. Retrieval loss on convergence weights — did the router pick the right document?
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from typing import Optional

from src.model.hopfield_router import HopfieldRouter
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RoutedModel(nn.Module):
    """LLM with Hopfield-routed document prepending."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        model_name = config.model.name

        # Load the frozen LLM
        logger.info(f"Loading {model_name} (frozen)")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        for param in self.llm.parameters():
            param.requires_grad = False

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Hopfield router
        memory_dim = config.memory.embedding_dim
        self.router = HopfieldRouter(
            query_dim=memory_dim,
            memory_dim=memory_dim,
            num_heads=config.model.get("num_heads", 4),
            association_dim=config.model.get("association_dim", 256),
            scaling=config.model.get("scaling", 2.0),
            update_steps=config.model.get("update_steps", 5),
            top_k=config.model.get("top_k", 1),
        )

        # Move router to same device as LLM
        device = next(self.llm.parameters()).device
        self.router = self.router.to(device)

        # Chunk texts — loaded separately for prepending
        self.chunk_texts: list[str] = []

    def set_memory(self, memory: torch.Tensor) -> None:
        self.router.set_memory(memory)
        logger.info(f"Loaded memory bank: {memory.shape[0]} vectors, {memory.shape[1]}-d")

    def set_chunk_texts(self, texts: list[str]) -> None:
        """Store the raw text for each chunk so we can prepend it."""
        self.chunk_texts = texts
        logger.info(f"Loaded {len(texts)} chunk texts for document prepending")

    def get_trainable_params(self) -> list[nn.Parameter]:
        return list(self.router.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        query_embeddings: torch.Tensor,
        questions: list[str],
        answers: list[str],
        target_chunk_idxs: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass: route, prepend, generate loss.

        Args:
            query_embeddings: (batch, memory_dim) question embeddings.
            questions: Raw question strings.
            answers: Raw answer strings.
            target_chunk_idxs: (batch,) ground truth chunk indices for retrieval loss.
        """
        device = next(self.llm.parameters()).device

        # Step 1: Hopfield routing — select documents
        top_indices, convergence_scores = self.router(query_embeddings)
        # top_indices: (batch, top_k)

        # Step 2: Build prompts with retrieved documents prepended
        batch_size = len(questions)
        prompts = []
        for i in range(batch_size):
            # Get the text of retrieved documents
            doc_indices = top_indices[i].cpu().tolist()
            retrieved_texts = [self.chunk_texts[idx] for idx in doc_indices]
            context = "\n".join(retrieved_texts)

            prompt = f"Context: {context}\nQuestion: {questions[i]}\nAnswer:"
            full_text = prompt + " " + answers[i]
            prompts.append(full_text)

        # Step 3: Tokenize and compute LM loss
        # We need to know where the answer starts to mask the loss
        lm_loss = torch.tensor(0.0, device=device)
        for i in range(batch_size):
            doc_texts = [self.chunk_texts[idx] for idx in top_indices[i].cpu().tolist()]
            context_text = "\n".join(doc_texts)
            prompt_only = f"Context: {context_text}\nQuestion: {questions[i]}\nAnswer:"

            full_enc = self.tokenizer(
                prompts[i], return_tensors="pt", truncation=True, max_length=512,
            ).to(device)
            prompt_enc = self.tokenizer(
                prompt_only, return_tensors="pt", truncation=True, max_length=512,
            ).to(device)

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            labels = input_ids.clone()
            labels[0, :prompt_enc["input_ids"].shape[1]] = -100

            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = lm_loss + loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        lm_loss = lm_loss / batch_size

        # Step 4: Retrieval loss on convergence weights
        retrieval_loss = torch.tensor(0.0, device=device)
        if target_chunk_idxs is not None:
            ret_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            retrieval_loss = ret_loss_fn(convergence_scores, target_chunk_idxs)

        return {
            "loss": lm_loss,
            "retrieval_loss": retrieval_loss,
            "top_indices": top_indices,
            "convergence_scores": convergence_scores,
        }

    def generate(
        self,
        query_embedding: torch.Tensor,
        question: str,
        max_new_tokens: int = 50,
    ) -> str:
        """Generate an answer using Hopfield-routed context."""
        device = next(self.llm.parameters()).device

        # Route
        top_indices, _ = self.router(query_embedding.unsqueeze(0))
        doc_indices = top_indices[0].cpu().tolist()
        retrieved_texts = [self.chunk_texts[idx] for idx in doc_indices]
        context = "\n".join(retrieved_texts)

        # Build prompt with retrieved context
        messages = [
            {"role": "system", "content": "Answer in as few words as possible. Only the answer, no explanation."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = self.llm.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        return answer
