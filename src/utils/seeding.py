"""
Pin random seeds everywhere so experiments are reproducible.

This matters more than people think. Without it, two runs of the same config
can give different retrieval scores because embeddings get batched differently,
or torch operations use different random states internally.
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch.

    Call this once at the start of any script before doing anything else.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For GPU reproducibility (if we ever move to CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Slower but deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
