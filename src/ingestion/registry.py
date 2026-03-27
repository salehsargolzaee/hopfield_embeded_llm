"""
Source registry — maps config strings to source classes.

This is how data-model separation works in practice. The config says
`source.type: "squad"`, and the registry resolves that to SQuADSource.
Adding a new data source means: write the class, add one line here.
"""

from omegaconf import DictConfig

from src.ingestion.base import DocumentSource
from src.ingestion.pdf_source import PDFSource
from src.ingestion.squad_source import SQuADSource

# Add new sources here. The key matches config.source.type.
_REGISTRY: dict[str, type[DocumentSource]] = {
    "squad": SQuADSource,
    "pdf": PDFSource,
}


def get_source(config: DictConfig) -> DocumentSource:
    """Look up and instantiate the data source specified in config."""
    source_type = config.source.type

    if source_type not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown source type '{source_type}'. Available: {available}"
        )

    return _REGISTRY[source_type](config)
