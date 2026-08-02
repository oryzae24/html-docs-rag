from python_doc_rag.chunking import chunk_sections
from python_doc_rag.config import ChunkingConfig
from python_doc_rag.models import DocumentSection


def make_section(text: str) -> DocumentSection:
    return DocumentSection(
        text=text,
        page_title="制御フロー",
        section_title="if 文",
        source_path="tutorial/controlflow.html",
        source_url="https://docs.python.org/ja/3.13/tutorial/controlflow.html#if",
        anchor="if",
        category="tutorial",
        python_version="3.13",
    )


def test_short_section_remains_one_chunk() -> None:
    chunks = chunk_sections(
        [make_section("短いセクションです。")],
        ChunkingConfig(chunk_size=80, chunk_overlap=10),
    )

    assert len(chunks) == 1
    assert chunks[0].text == "短いセクションです。"
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_index == 0
    assert chunks[0].section_title == "if 文"


def test_long_section_is_split_with_metadata_and_offsets() -> None:
    text = "\n\n".join(f"段落{i}。Pythonの制御フローを説明します。" for i in range(12))
    chunks = chunk_sections(
        [make_section(text)],
        ChunkingConfig(chunk_size=80, chunk_overlap=15),
    )

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.page_title == "制御フロー" for chunk in chunks)
    assert all(chunk.source_url.endswith("#if") for chunk in chunks)
    assert all(text[chunk.start_index :].startswith(chunk.text) for chunk in chunks)


def test_empty_section_does_not_create_a_chunk() -> None:
    assert chunk_sections([make_section(" \n\t ")]) == []
