"""
Semantic chunking.

The goal: split documents into pieces that are small enough to embed meaningfully
but large enough to contain a complete thought. Arbitrary splits at fixed token
counts are bad because they cut mid-sentence or mid-paragraph, which hurts
retrieval — the embedding of a half-sentence doesn't represent anything useful.

Strategy:
1. Split text into sentences using NLTK's sentence tokenizer
2. Group sentences into chunks that don't exceed max_chunk_tokens
3. Add overlap between consecutive chunks so we don't lose context at boundaries

The overlap is important: if a key fact spans two chunks, the overlap ensures
at least one chunk contains the full fact.
"""

from dataclasses import replace

import nltk
from omegaconf import DictConfig

from src.ingestion.base import DocumentChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Download sentence tokenizer data if we don't have it
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _count_tokens(text: str) -> int:
    """Rough token count by whitespace splitting.

    Not exact (real tokenizers handle subwords), but good enough for chunking.
    We don't want to load a full tokenizer just to count chunk sizes.
    """
    return len(text.split())


def chunk_document(
    text: str,
    source_doc: str,
    config: DictConfig,
    base_chunk_id: str = "",
    metadata: dict | None = None,
) -> list[DocumentChunk]:
    """Split a document into semantically meaningful chunks.

    Args:
        text: Full document text.
        source_doc: Identifier for the source document.
        config: Must contain chunking.max_chunk_tokens, overlap_tokens, min_chunk_tokens.
        base_chunk_id: Prefix for chunk IDs (e.g. "squad_article_3").
        metadata: Extra metadata to attach to every chunk.

    Returns:
        List of DocumentChunk objects, each with a unique chunk_id.
    """
    max_tokens = config.chunking.max_chunk_tokens
    overlap_tokens = config.chunking.overlap_tokens
    min_tokens = config.chunking.min_chunk_tokens
    strategy = config.chunking.get("strategy", "semantic")

    if strategy == "semantic":
        chunks = _semantic_chunk(text, max_tokens, overlap_tokens)
    else:
        chunks = _fixed_chunk(text, max_tokens, overlap_tokens)

    result = []
    meta = metadata or {}

    for i, chunk_text in enumerate(chunks):
        if _count_tokens(chunk_text) < min_tokens:
            continue

        chunk_id = f"{base_chunk_id}_chunk_{i}" if base_chunk_id else f"chunk_{i}"
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=chunk_text.strip(),
            source_doc=source_doc,
            metadata={**meta, "chunk_index": i},
        )
        result.append(chunk)

    return result


def _semantic_chunk(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Group sentences into chunks respecting sentence boundaries."""
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sent_tokens = _count_tokens(sentence)

        # If a single sentence exceeds max_tokens, it becomes its own chunk.
        # This is rare but happens with long run-on sentences or tables.
        if sent_tokens > max_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_length = 0
            chunks.append(sentence)
            continue

        # Would adding this sentence exceed the limit?
        if current_length + sent_tokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Overlap: keep the last few sentences whose total is <= overlap_tokens
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                s_len = _count_tokens(s)
                if overlap_len + s_len > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_len += s_len

            current_sentences = overlap_sents
            current_length = overlap_len

        current_sentences.append(sentence)
        current_length += sent_tokens

    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _fixed_chunk(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Fallback: split on token count boundaries (no semantic awareness)."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Step forward by (max_tokens - overlap) so chunks overlap
        start += max_tokens - overlap_tokens

    return chunks


def chunk_documents(
    documents: list[dict],
    config: DictConfig,
) -> list[DocumentChunk]:
    """Chunk a batch of raw documents.

    Args:
        documents: List of dicts with 'text', 'source', and optional 'metadata'.
        config: Chunking configuration.

    Returns:
        Flat list of all chunks from all documents.
    """
    all_chunks = []

    for i, doc in enumerate(documents):
        base_id = f"doc_{i}"
        chunks = chunk_document(
            text=doc["text"],
            source_doc=doc["source"],
            config=config,
            base_chunk_id=base_id,
            metadata=doc.get("metadata", {}),
        )
        all_chunks.extend(chunks)

    logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks
