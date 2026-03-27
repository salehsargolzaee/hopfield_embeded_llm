"""
Question answering evaluation metrics.

Standard SQuAD metrics:
  - Exact Match (EM): did the generated answer exactly match the gold answer?
  - F1: token-level overlap between generated and gold answer.

EM is strict — one wrong word and it's 0. F1 is forgiving — it gives partial
credit for getting some of the answer right.
"""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation/whitespace.

    Same normalization SQuAD uses so our numbers are comparable to published results.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized prediction matches normalized ground truth, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth.

    Treats both strings as bags of words after normalization.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_squad_metrics(
    predictions: list[str],
    ground_truths: list[list[str]],
) -> dict[str, float]:
    """Compute aggregate EM and F1 over a set of predictions.

    Args:
        predictions: Generated answers, one per question.
        ground_truths: For each question, a list of acceptable answers.
            (SQuAD questions can have multiple valid answers.)

    Returns:
        Dict with "exact_match" and "f1" averaged over all questions.
    """
    total_em = 0.0
    total_f1 = 0.0

    for pred, golds in zip(predictions, ground_truths):
        # Take the best score across all acceptable answers
        best_em = max(exact_match(pred, gold) for gold in golds)
        best_f1 = max(f1_score(pred, gold) for gold in golds)
        total_em += best_em
        total_f1 += best_f1

    n = len(predictions)
    return {
        "exact_match": total_em / n if n > 0 else 0.0,
        "f1": total_f1 / n if n > 0 else 0.0,
    }
