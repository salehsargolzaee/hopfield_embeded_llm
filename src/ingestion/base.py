"""
Abstract base class for all document sources.

This is the security boundary. Everything downstream of this module receives
the same DocumentChunk dataclass regardless of where the data came from.
When we switch from SQuAD to confidential internal documents, only the
ingestion module changes. The rest of the codebase never touches raw data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from omegaconf import DictConfig


@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk of text from a document.

    frozen=True makes these immutable — once created, they can't be modified.
    This is intentional: chunks are the atomic unit of retrieval, and we don't
    want anything downstream accidentally mutating them.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        text: The actual text content.
        source_doc: Which document this came from (filename, article title, etc.).
        metadata: Anything else worth keeping — page number, section title, etc.
    """
    chunk_id: str
    text: str
    source_doc: str
    metadata: dict = field(default_factory=dict)


class DocumentSource(ABC):
    """Interface that all data sources must implement.

    To add a new data source (e.g. internal PDFs, Confluence pages, Notion exports):
    1. Subclass this
    2. Implement load_documents()
    3. Register it in registry.py

    That's it. No other module needs to change.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    @abstractmethod
    def load_documents(self) -> Iterator[dict]:
        """Yield raw documents as dicts with at least 'text' and 'source' keys.

        Each dict represents one logical document (an article, a PDF, a page).
        Chunking happens separately — this method just loads the raw material.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this source, used in logs."""
        ...
