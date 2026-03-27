"""
SQuAD dataset prepared for memory-augmented LLM training.

Each training example is:
  - Input:  a question (formatted as a prompt)
  - Label:  the answer text
  - Memory: the document chunks are stored in the memory bank (shared, not per-example)

The loss is only computed on answer tokens. Question tokens are masked with -100
so the model isn't penalized for "generating" the question — it only learns to
produce the right answer given the question + memory access.
"""

from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Simple prompt template — question goes in, model generates the answer
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


@dataclass
class SQuADExample:
    """A single training example."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    question_length: int  # for diagnostics


class SQuADMemoryDataset(Dataset):
    """SQuAD v2 formatted for training a memory-augmented LLM.

    Each item returns tokenized input where:
    - The prompt ("Question: ... Answer:") tokens have labels = -100 (ignored in loss)
    - The answer tokens have their actual token IDs as labels (model learns these)
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DictConfig, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = config.training.max_seq_length
        self.examples = []

        logger.info(f"Loading SQuAD v2 ({split})")
        dataset = load_dataset("squad_v2", split=split, )

        skipped = 0
        for row in dataset:
            # Skip unanswerable questions
            if not row["answers"]["text"]:
                skipped += 1
                continue

            question = row["question"]
            # Take the first answer (SQuAD can have multiple valid answers)
            answer = row["answers"]["text"][0]

            prompt = PROMPT_TEMPLATE.format(question=question)
            full_text = prompt + " " + answer

            # Tokenize the full sequence
            encoded = tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

            # Tokenize just the prompt to find where the answer starts
            prompt_encoded = tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            prompt_length = len(prompt_encoded["input_ids"])

            # Build labels: -100 for prompt tokens, actual IDs for answer tokens
            input_ids = encoded["input_ids"]
            labels = [-100] * prompt_length + input_ids[prompt_length:]

            # Make sure labels and input_ids are the same length
            labels = labels[:len(input_ids)]

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
                "question_length": prompt_length,
            })

        logger.info(
            f"Prepared {len(self.examples)} examples "
            f"(skipped {skipped} unanswerable questions)"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SQuADExample:
        ex = self.examples[idx]
        return SQuADExample(
            input_ids=torch.tensor(ex["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(ex["attention_mask"], dtype=torch.long),
            labels=torch.tensor(ex["labels"], dtype=torch.long),
            question_length=ex["question_length"],
        )


def collate_fn(batch: list[SQuADExample]) -> dict:
    """Pad a batch of examples to the same length.

    Padding goes on the right. Input IDs are padded with pad_token_id,
    labels are padded with -100 (so padding doesn't affect the loss).
    """
    max_len = max(ex.input_ids.size(0) for ex in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for ex in batch:
        pad_len = max_len - ex.input_ids.size(0)
        input_ids.append(F.pad(ex.input_ids, (0, pad_len), value=0))
        attention_mask.append(F.pad(ex.attention_mask, (0, pad_len), value=0))
        labels.append(F.pad(ex.labels, (0, pad_len), value=-100))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


# Need this import for collate_fn
import torch.nn.functional as F
