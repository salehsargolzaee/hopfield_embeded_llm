"""
SQuAD v2 data source.

SQuAD (Stanford Question Answering Dataset) has ~150k questions over ~500
Wikipedia articles. Each article is split into paragraphs, and each paragraph
has associated questions. We use it because:

1. It has ground-truth question→paragraph mappings (built-in eval labels)
2. The paragraphs are real Wikipedia text (not synthetic)
3. It's a standard benchmark everyone knows

We load articles and yield their paragraphs as documents. The questions
are handled separately by the evaluation module — we don't load them here
because this module's job is just "give me text to chunk and embed."
"""

from typing import Iterator

from datasets import load_dataset
from omegaconf import DictConfig

from src.ingestion.base import DocumentSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SQuADSource(DocumentSource):
    """Loads articles from SQuAD v2 via HuggingFace datasets."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.split = config.source.get("squad_split", "validation")
        self.max_articles = config.source.get("max_articles", None)

    def load_documents(self) -> Iterator[dict]:
        """Yield one dict per unique article context in SQuAD.

        SQuAD's structure is: each row is a question, and the 'context' field
        is the paragraph that answers it. Multiple questions share the same context.
        We deduplicate by context text to get unique paragraphs.
        """
        logger.info(f"Loading SQuAD v2 ({self.split} split)")
        dataset = load_dataset("squad_v2", split=self.split, )

        # SQuAD rows are question-level, but we want document-level.
        # Group by article title, collect unique contexts (paragraphs).
        articles: dict[str, list[str]] = {}
        for row in dataset:
            title = row["title"]
            context = row["context"]
            if title not in articles:
                articles[title] = []
            # Same paragraph appears multiple times (once per question) — skip dupes
            if context not in articles[title]:
                articles[title].append(context)

        article_titles = list(articles.keys())
        if self.max_articles is not None:
            article_titles = article_titles[: self.max_articles]

        logger.info(f"Found {len(article_titles)} articles (max_articles={self.max_articles})")

        for title in article_titles:
            for i, paragraph in enumerate(articles[title]):
                yield {
                    "text": paragraph,
                    "source": title,
                    "metadata": {
                        "paragraph_index": i,
                        "dataset": "squad_v2",
                        "split": self.split,
                    },
                }

    def name(self) -> str:
        return f"SQuAD v2 ({self.split})"
