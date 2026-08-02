"""Split long document sections without discarding citation metadata."""

from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from python_doc_rag.config import ChunkingConfig
from python_doc_rag.models import DocumentSection, SearchChunk


def chunk_sections(
    sections: Iterable[DocumentSection],
    config: ChunkingConfig | None = None,
) -> list[SearchChunk]:
    """Convert sections to chunks, splitting only text above the size limit."""
    settings = config or ChunkingConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    chunks: list[SearchChunk] = []

    for section in sections:
        section_text = section.text.strip()
        if not section_text:
            continue
        if len(section_text) <= settings.chunk_size:
            pieces = [(section_text, 0)]
        else:
            documents = splitter.create_documents([section_text])
            pieces = [
                (document.page_content, int(document.metadata["start_index"]))
                for document in documents
                if document.page_content.strip()
            ]

        chunks.extend(
            SearchChunk(
                text=text,
                page_title=section.page_title,
                section_title=section.section_title,
                source_url=section.source_url,
                category=section.category,
                chunk_index=chunk_index,
                start_index=start_index,
            )
            for chunk_index, (text, start_index) in enumerate(pieces)
        )

    return chunks
