"""
Text embedding using sentence-transformers.

Takes text strings, returns fixed-dimensional vectors. These vectors are the
"language" that the Hopfield network and cosine retriever speak — they don't
see raw text, they see points in 384-dimensional space (for MiniLM-L6-v2).

The key property of these embeddings: semantically similar texts land close
together in vector space. "How do I reset my password?" and "I forgot my
credentials" get similar vectors, even though they share almost no words.

We cache embeddings to disk after computing them. Embedding 10k chunks takes
a few minutes — you don't want to redo that every time you tweak a retriever.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig

from src.ingestion.base import DocumentChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    """Wraps a sentence-transformer model for encoding text to vectors."""

    def __init__(self, config: DictConfig) -> None:
        self.model_name = config.embedding.model_name
        self.batch_size = config.embedding.batch_size
        self.device = config.embedding.device
        self.dimension = config.embedding.dimension
        self.cache_dir = Path(config.embedding.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Sanity check: make sure the configured dimension matches the model
        test_dim = self.model.get_sentence_embedding_dimension()
        if test_dim != self.dimension:
            raise ValueError(
                f"Config says dimension={self.dimension}, but {self.model_name} "
                f"outputs {test_dim}-d vectors. Update config.embedding.dimension."
            )

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
        use_cache: bool = True,
    ) -> np.ndarray:
        """Embed a list of document chunks, returning an (N, D) array.

        Args:
            chunks: Document chunks to embed.
            use_cache: If True, try to load from disk first.

        Returns:
            NumPy array of shape (len(chunks), self.dimension).
        """
        cache_path = self._cache_path(chunks)

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            embeddings = np.load(cache_path)
            if embeddings.shape[0] == len(chunks):
                return embeddings
            logger.warning("Cache size mismatch — recomputing embeddings")

        texts = [chunk.text for chunk in chunks]
        logger.info(f"Embedding {len(texts)} chunks (batch_size={self.batch_size})")

        # sentence-transformers handles batching internally, but we pass our
        # configured batch_size to control memory usage
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize so dot product = cosine sim
        )

        if use_cache:
            np.save(cache_path, embeddings)
            logger.info(f"Cached embeddings to {cache_path}")

        return embeddings

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed raw text strings (for queries, not chunks).

        No caching here — queries are small and change every run.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a 1-D vector."""
        return self.embed_texts([text])[0]

    def _cache_path(self, chunks: list[DocumentChunk]) -> Path:
        """Generate a cache filename based on chunk content and model.

        If the chunks change (different data, different chunking config),
        the hash changes and we recompute. If only the retriever config
        changes, we reuse cached embeddings.
        """
        # Hash the chunk texts + model name to detect changes
        hasher = hashlib.md5()
        hasher.update(self.model_name.encode())
        for chunk in chunks:
            hasher.update(chunk.text.encode())
        content_hash = hasher.hexdigest()[:12]

        return self.cache_dir / f"embeddings_{content_hash}.npy"
