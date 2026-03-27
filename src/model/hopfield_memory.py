"""
Hopfield memory layer that injects into an LLM's forward pass.

This module does one thing: given a hidden state from the LLM, query a memory
bank of document embeddings using the Hopfield association rule, and return
a memory-informed vector to add back to the hidden state.

The math:
    1. Project the hidden state into query space:  q = Wq @ hidden
    2. Compute attention over memory:  weights = softmax(β * q @ memory^T)
    3. Retrieve:  retrieved = weights @ memory
    4. Project back to hidden dim:  output = Wo @ retrieved

Wq, Wo, and β are the only trainable parameters. Everything else is frozen.

Wo is zero-initialized so the layer starts as a no-op — the LLM behaves
exactly as before on step 1 of training, and gradually learns to use memory.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HopfieldMemoryLayer(nn.Module):
    """A single Hopfield memory injection point.

    Args:
        hidden_dim: Dimension of the LLM's hidden states (e.g. 2048 for Qwen 3.5 4B).
        memory_dim: Dimension of the memory bank vectors (e.g. 768 for bge-base).
        num_heads: Number of attention heads for multi-head retrieval.
    """

    def __init__(self, hidden_dim: int, memory_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads

        assert memory_dim % num_heads == 0, "memory_dim must be divisible by num_heads"

        # Projects LLM hidden states into memory query space
        self.Wq = nn.Linear(hidden_dim, memory_dim, bias=False)

        # Projects retrieved memory back to LLM hidden dimension.
        # Zero-initialized: at training start, this layer outputs zeros,
        # so the LLM runs as if the memory layer isn't there.
        self.Wo = nn.Linear(memory_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.Wo.weight)

        # Learnable inverse temperature, one per head.
        # Initialized to 1/sqrt(head_dim) which is the standard scaled dot-product value.
        init_beta = 1.0 / math.sqrt(self.head_dim)
        self.log_beta = nn.Parameter(torch.full((num_heads,), math.log(init_beta)))

        # Memory bank — registered as buffer so it moves to GPU with the model
        # but doesn't get gradients
        self.register_buffer("memory_bank", None)

    def set_memory(self, memory: torch.Tensor) -> None:
        """Load the document memory bank.

        Args:
            memory: (num_docs, memory_dim) tensor of L2-normalized document embeddings.
        """
        assert memory.dim() == 2 and memory.shape[1] == self.memory_dim
        self.memory_bank = memory

    @property
    def beta(self) -> torch.Tensor:
        """Actual β values (always positive via exp)."""
        return self.log_beta.exp()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Query memory and return the retrieved context.

        Args:
            hidden: (batch, seq_len, hidden_dim) from the LLM.

        Returns:
            (batch, seq_len, hidden_dim) memory-informed vector to add to hidden.
        """
        if self.memory_bank is None:
            return torch.zeros_like(hidden)

        # Handle both 2D (seq_len, hidden_dim) and 3D (batch, seq_len, hidden_dim)
        squeezed = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
            squeezed = True

        batch, seq_len, _ = hidden.shape
        num_docs = self.memory_bank.shape[0]

        # Project hidden states to query space: (batch, seq_len, memory_dim)
        queries = self.Wq(hidden)

        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        queries = queries.view(batch, seq_len, self.num_heads, self.head_dim)
        # (batch, num_heads, seq_len, head_dim)
        queries = queries.transpose(1, 2)

        # Reshape memory bank for multi-head: (1, num_heads, num_docs, head_dim)
        memory = self.memory_bank.view(1, num_docs, self.num_heads, self.head_dim)
        memory = memory.transpose(1, 2)

        # Scaled dot-product attention with learned β per head
        # queries: (batch, num_heads, seq_len, head_dim)
        # memory:  (1, num_heads, num_docs, head_dim)
        # scores:  (batch, num_heads, seq_len, num_docs)
        scores = torch.matmul(queries, memory.transpose(-2, -1))

        # Scale by learned β (one per head)
        beta = self.beta.view(1, self.num_heads, 1, 1)
        scores = scores * beta

        # Softmax over documents
        weights = F.softmax(scores, dim=-1)

        # Retrieve: weighted sum of memory vectors
        # (batch, num_heads, seq_len, head_dim)
        retrieved = torch.matmul(weights, memory.expand(batch, -1, -1, -1))

        # Reshape back: (batch, seq_len, memory_dim)
        retrieved = retrieved.transpose(1, 2).contiguous().view(batch, seq_len, self.memory_dim)

        # Project back to hidden dim
        result = self.Wo(retrieved)

        if squeezed:
            result = result.squeeze(0)

        return result
