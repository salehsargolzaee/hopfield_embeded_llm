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
    target_chunk_idx: int  # index into memory bank for retrieval loss


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

        # Build context → chunk index mapping if memory bank info is provided
        context_to_chunk = {}
        if chunk_texts is not None:
            context_to_chunk = self._build_context_mapping(dataset, chunk_texts)
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
            })

        logger.info(
            f"Prepared {len(self.examples)} examples "
            f"(skipped {skipped} unanswerable, {no_chunk} with no matching chunk)"
        )

    def _build_context_mapping(self, dataset, chunk_texts: list) -> dict:
        """Map each SQuAD context paragraph to its nearest memory bank chunk.

        For each unique context in SQuAD, find which chunk in our memory bank
        contains it (or is contained by it). Uses string matching, not embeddings.
        """
        mapping = {}
        unique_contexts = set()
        for row in dataset:
            unique_contexts.add(row["context"])

        for context in unique_contexts:
            context_clean = " ".join(context.split())
            best_idx = -1
            best_overlap = 0

            for i, chunk_text in enumerate(chunk_texts):
                chunk_clean = " ".join(chunk_text.split())
                # Check containment both ways
                if chunk_clean in context_clean or context_clean in chunk_clean:
                    overlap = min(len(chunk_clean), len(context_clean))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = i

            if best_idx >= 0:
                mapping[context] = best_idx

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
    }
