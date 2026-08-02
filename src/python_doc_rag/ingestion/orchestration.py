"""Site-neutral parsing and chunking orchestration."""

from __future__ import annotations

from collections.abc import Iterable

from python_doc_rag.chunking import chunk_sections
from python_doc_rag.config import ChunkingConfig
from python_doc_rag.ingestion.protocols import (
    HtmlDocumentParser,
    IngestionFailure,
    IngestionResult,
)
from python_doc_rag.models import DocumentSection, SearchChunk, SourceDocument


def ingest_documents(
    documents: Iterable[SourceDocument],
    parser: HtmlDocumentParser,
    *,
    chunking_config: ChunkingConfig | None = None,
) -> IngestionResult:
    """Parse and chunk documents deterministically while isolating page failures."""
    settings = chunking_config or ChunkingConfig()
    retained_documents: list[SourceDocument] = []
    sections: list[DocumentSection] = []
    chunks: list[SearchChunk] = []
    failures: list[IngestionFailure] = []
    excluded_sections = 0
    excluded_nodes = 0
    code_blocks = 0
    tables = 0
    fallback_pages = 0
    for document in documents:
        try:
            result = parser.parse(document)
            page_chunks = chunk_sections(result.sections, settings)
        except Exception as error:  # noqa: BLE001 - page failure must remain local
            failures.append(
                IngestionFailure(
                    logical_path=document.logical_path,
                    reason=f"{type(error).__name__}: {error}",
                )
            )
            continue
        retained_documents.append(document)
        sections.extend(result.sections)
        chunks.extend(page_chunks)
        excluded_sections += result.excluded_section_count
        excluded_nodes += result.excluded_node_count
        code_blocks += result.code_block_count
        tables += result.table_count
        fallback_pages += int(result.used_fallback)
    return IngestionResult(
        documents=tuple(retained_documents),
        sections=tuple(sections),
        chunks=tuple(chunks),
        failures=tuple(failures),
        excluded_section_count=excluded_sections,
        excluded_node_count=excluded_nodes,
        code_block_count=code_blocks,
        table_count=tables,
        fallback_page_count=fallback_pages,
    )
