"""
PDF data source.

Reads all PDFs from a directory and yields pages as documents.
This is the source you'd swap in for internal enterprise documents —
point it at a folder of department PDFs and the rest of the pipeline
stays identical.

Uses PyMuPDF (imported as fitz) because it's fast and handles most
PDF layouts well. It won't do OCR on scanned documents — for that
you'd need to add pytesseract or a similar OCR step.
"""

from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from omegaconf import DictConfig

from src.ingestion.base import DocumentSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PDFSource(DocumentSource):
    """Loads text from PDF files in a directory."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.path = Path(config.source.path)

    def load_documents(self) -> Iterator[dict]:
        """Yield one dict per page of each PDF found in the source directory."""
        if not self.path.exists():
            logger.warning(f"PDF source path does not exist: {self.path}")
            return

        pdf_files = sorted(self.path.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.path}")

        for pdf_path in pdf_files:
            try:
                doc = fitz.open(pdf_path)
            except Exception:
                logger.exception(f"Failed to open {pdf_path}")
                continue

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                # Skip empty pages (cover pages, images-only, etc.)
                if not text:
                    continue

                yield {
                    "text": text,
                    "source": pdf_path.stem,
                    "metadata": {
                        "file_path": str(pdf_path),
                        "page_number": page_num + 1,  # 1-indexed for humans
                        "total_pages": len(doc),
                    },
                }

            doc.close()

    def name(self) -> str:
        return f"PDF ({self.path})"
