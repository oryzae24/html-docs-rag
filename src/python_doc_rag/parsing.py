"""Backward-compatible facade for configurable ingestion APIs."""

from python_doc_rag.ingestion.orchestration import ingest_documents
from python_doc_rag.ingestion.protocols import (
    DocumentParseResult,
    HtmlDocumentParser,
    IngestionFailure,
    IngestionResult,
)
from python_doc_rag.parsers.generic_html import GenericHtmlParser


def __getattr__(name: str) -> object:
    """Resolve the historical Python adapter lazily from its site module."""
    if name == "PythonSphinxHtmlParserAdapter":
        from python_doc_rag.sites.python_docs.parser import (
            PythonSphinxHtmlParserAdapter,
        )

        return PythonSphinxHtmlParserAdapter
    raise AttributeError(name)


__all__ = [
    "DocumentParseResult",
    "GenericHtmlParser",
    "HtmlDocumentParser",
    "IngestionFailure",
    "IngestionResult",
    "PythonSphinxHtmlParserAdapter",  # noqa: F822 - resolved by __getattr__
    "ingest_documents",
]
