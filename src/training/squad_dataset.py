"""
SQuAD dataset prepared for memory-augmented LLM training.

Each training example has:
  - input_ids / attention_mask: tokenized "Question: X\nAnswer: Y"
  - labels: -100 on question tokens, real IDs on answer tokens
  - target_chunk_idx: which memory bank chunk this question's context maps to
    (used for the auxiliary retrieval loss)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


@dataclass
class SQuADExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    question_length: int
    target_chunk_idx: int
    question_text: str = ""


class SQuADMemoryDataset(Dataset):
    """SQuAD v2 with ground truth memory bank mapping for retrieval supervision."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        split: str = "train",
        memory_bank: Optional[torch.Tensor] = None,
        chunk_texts: Optional[list] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = config.training.max_seq_length
        self.examples = []

        logger.info(f"Loading SQuAD v2 ({split})")
        dataset = load_dataset("squad_v2", split=split)

        # Build context → chunk index mapping
        context_to_chunk = {}
        if chunk_texts is not None and memory_bank is not None:
            context_to_chunk = self._build_context_mapping_embedding(
                dataset, chunk_texts, memory_bank, config
            )
            logger.info(f"Mapped {len(context_to_chunk)} unique contexts to memory chunks")

        skipped = 0
        no_chunk = 0
        for row in dataset:
            if not row["answers"]["text"]:
                skipped += 1
                continue

            question = row["question"]
            answer = row["answers"]["text"][0]
            context = row["context"]

            # Find the matching memory bank chunk
            target_idx = context_to_chunk.get(context, -1)
            if target_idx == -1 and chunk_texts is not None:
                no_chunk += 1
                continue

            prompt = PROMPT_TEMPLATE.format(question=question)
            full_text = prompt + " " + answer

            encoded = tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

            prompt_encoded = tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            prompt_length = len(prompt_encoded["input_ids"])

            input_ids = encoded["input_ids"]
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            labels = labels[:len(input_ids)]

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
                "question_length": prompt_length,
                "target_chunk_idx": target_idx,
                "question_text": question,
            })

        logger.info(
            f"Prepared {len(self.examples)} examples "
            f"(skipped {skipped} unanswerable, {no_chunk} with no matching chunk)"
        )

    def _build_context_mapping_embedding(
        self, dataset, chunk_texts: list, memory_bank: torch.Tensor, config: DictConfig,
    ) -> dict:
        """Map each SQuAD context to the nearest memory bank chunk via embedding similarity.

        Embeds each unique context with the same model used for the memory bank,
        then finds the closest chunk by cosine similarity. Much more robust than
        string matching since chunking changes the text boundaries.
        """
        from src.embedding.embedder import Embedder

        unique_contexts = list(set(row["context"] for row in dataset))
        logger.info(f"Embedding {len(unique_contexts)} unique contexts for mapping...")

        embedder = Embedder(config)
        context_embeddings = embedder.embed_texts(unique_contexts)  # (num_contexts, 768)

        # memory_bank is already (num_chunks, 768) and L2-normalized
        # context_embeddings are also L2-normalized by the embedder
        # cosine similarity = dot product
        memory_np = memory_bank.cpu().numpy()
        similarities = context_embeddings @ memory_np.T  # (num_contexts, num_chunks)

        mapping = {}
        for i, context in enumerate(unique_contexts):
            best_chunk_idx = int(similarities[i].argmax())
            best_sim = float(similarities[i, best_chunk_idx])
            # Only map if similarity is reasonable (> 0.5)
            if best_sim > 0.5:
                mapping[context] = best_chunk_idx

        logger.info(f"Mapped {len(mapping)}/{len(unique_contexts)} contexts (sim > 0.5)")
        return mapping

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SQuADExample:
        ex = self.examples[idx]
        return SQuADExample(
            input_ids=torch.tensor(ex["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(ex["attention_mask"], dtype=torch.long),
            labels=torch.tensor(ex["labels"], dtype=torch.long),
            question_length=ex["question_length"],
            target_chunk_idx=ex["target_chunk_idx"],
            question_text=ex.get("question_text", ""),
        )


def collate_fn(batch: list[SQuADExample]) -> dict:
    """Pad a batch to the same length."""
    max_len = max(ex.input_ids.size(0) for ex in batch)

    input_ids = []
    attention_mask = []
    labels = []
    target_chunk_idxs = []

    for ex in batch:
        pad_len = max_len - ex.input_ids.size(0)
        input_ids.append(F.pad(ex.input_ids, (0, pad_len), value=0))
        attention_mask.append(F.pad(ex.attention_mask, (0, pad_len), value=0))
        labels.append(F.pad(ex.labels, (0, pad_len), value=-100))
        target_chunk_idxs.append(ex.target_chunk_idx)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "target_chunk_idxs": torch.tensor(target_chunk_idxs, dtype=torch.long),
        "question_texts": [ex.question_text for ex in batch],
    }
