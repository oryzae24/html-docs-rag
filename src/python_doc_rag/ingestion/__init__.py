"""Site-neutral document ingestion contracts and orchestration."""

from python_doc_rag.ingestion.orchestration import ingest_documents
from python_doc_rag.ingestion.protocols import (
    DocumentParseResult,
    HtmlDocumentParser,
    IngestionFailure,
    IngestionResult,
)

__all__ = [
    "DocumentParseResult",
    "HtmlDocumentParser",
    "IngestionFailure",
    "IngestionResult",
    "ingest_documents",
]
