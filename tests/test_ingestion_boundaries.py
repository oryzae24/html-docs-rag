import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from python_doc_rag.chunking import chunk_sections
from python_doc_rag.dataset_layout import (
    DATASET_MANIFEST_REVISION,
    LEGACY_DATASET_MANIFEST_REVISION,
    DatasetManifest,
    generic_dataset_manifest,
    load_dataset_manifest,
    resolve_dataset_artifacts,
    resolve_manifest_path,
    safe_relative_path,
    write_dataset_manifest_atomic,
)
from python_doc_rag.html_parser import parse_python_doc_html_result
from python_doc_rag.models import SourceDocument
from python_doc_rag.parsing import (
    PythonSphinxHtmlParserAdapter,
    ingest_documents,
)
from python_doc_rag.site_config import (
    BoundedHttpLoaderSettings,
    LoaderSettings,
    LocalLoaderSettings,
    PinnedLocalArchiveLoaderSettings,
    load_site_config,
)
from python_doc_rag.sites.python_docs.local_compat import LocalHtmlTreeLoader


def test_loading_module_preserves_lazy_historical_exports() -> None:
    import python_doc_rag.loading as loading
    from python_doc_rag.ingestion.protocols import DocumentLoader

    assert loading.DocumentLoader is DocumentLoader
    assert loading.LocalHtmlTreeLoader is LocalHtmlTreeLoader


def test_loader_settings_preserves_runtime_union_compatibility() -> None:
    settings = LocalLoaderSettings(
        type="local-html-tree",
        categories=("tutorial",),
    )

    assert isinstance(settings, LoaderSettings)


FIXTURES = Path(__file__).parent / "fixtures"


def _source_document(
    *,
    content: str = "<html><main><h1 id='x'>X</h1><p>long enough text</p></main></html>",
    logical_path: str = "tutorial/example.html",
) -> SourceDocument:
    return SourceDocument(
        source_url="https://docs.python.org/ja/3.13/tutorial/example.html",
        canonical_url="https://docs.python.org/ja/3.13/tutorial/example.html",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        source_kind="local-html-tree",
        logical_path=logical_path,
        category="tutorial",
        metadata={"nested": {"values": [1, True, None]}},
    )


def _http_toml(*, extra_loader: str = "", extra_top: str = "") -> str:
    return f"""
[dataset]
name = "Example docs"
slug = "example-docs"
language = "en"
description = "Fixture"

[loader]
type = "bounded-http"
start_urls = ["https://example.com/docs/", "https://other.example/guide/"]
max_pages = 40
timeout_seconds = 15
request_delay_seconds = 0.5
max_response_bytes = 3000000
user_agent = "fixture/1.0"
respect_robots_txt = true
retain_query = false
max_retries = 2
{extra_loader}

[parser]
type = "generic-html"
content_selectors = ["article", "main"]
exclude_selectors = ["nav", "footer"]
title_selectors = ["h1", "title"]
heading_levels = [1, 2, 3]
minimum_section_text_length = 20
fallback_to_body = false
include_lead_text = true

[chunking]
chunk_size = 1000
chunk_overlap = 150

[index]
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_batch_size = 64

[profile]
prepare = "recommended-v2"
{extra_top}
"""


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "site.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _manifest() -> DatasetManifest:
    return generic_dataset_manifest(
        dataset_name="Example docs",
        dataset_slug="example-docs",
        loader_type="bounded-http",
        parser_type="generic-html",
        site_config_sha256="a" * 64,
        created_at="2026-08-01T00:00:00Z",
        source_page_count=2,
        section_count=3,
        chunk_count=4,
        source_config_sha256="b" * 64,
        processing_config_sha256="c" * 64,
        source_snapshot_sha256="d" * 64,
    )


def test_source_document_hash_identity_and_immutable_json_metadata() -> None:
    document = _source_document()

    assert (
        document.content_sha256 == hashlib.sha256(document.content.encode()).hexdigest()
    )
    assert document.canonical_url == document.source_url
    assert document.to_dict()["metadata"] == {"nested": {"values": [1, True, None]}}
    with pytest.raises(TypeError):
        document.metadata["new"] = "value"  # type: ignore[index]
    nested = document.metadata["nested"]
    assert not isinstance(nested, dict)


@pytest.mark.parametrize("logical_path", ("/tmp/page.html", "../page.html"))
def test_source_document_rejects_local_absolute_or_traversing_paths(
    logical_path: str,
) -> None:
    with pytest.raises(ValueError, match="logical_path"):
        _source_document(logical_path=logical_path)


def test_source_document_rejects_hash_mismatch_and_arbitrary_metadata() -> None:
    document = _source_document()
    with pytest.raises(ValueError, match="content_sha256"):
        replace(document, content_sha256="0" * 64)
    with pytest.raises(TypeError, match="JSON-compatible"):
        replace(document, metadata={"path": Path("local")})


def test_local_loader_reuses_protected_enumeration_and_url_order(
    tmp_path: Path,
) -> None:
    for relative in ("tutorial/b.html", "tutorial/a.html", "library/z.html"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html></html>", encoding="utf-8")
    loader = LocalHtmlTreeLoader(
        tmp_path,
        source_base_url="https://docs.python.org/ja/3.13/",
        include_path_prefixes=("tutorial/", "library/"),
    )

    documents = list(loader.load())

    assert [item.logical_path for item in documents] == [
        "library/z.html",
        "tutorial/a.html",
        "tutorial/b.html",
    ]
    assert [item.source_url for item in documents] == [
        "https://docs.python.org/ja/3.13/library/z.html",
        "https://docs.python.org/ja/3.13/tutorial/a.html",
        "https://docs.python.org/ja/3.13/tutorial/b.html",
    ]
    assert loader.enumeration is not None


def test_local_loader_does_not_follow_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "outside.html"
    target.write_text("outside", encoding="utf-8")
    category = tmp_path / "tutorial"
    category.mkdir()
    (category / "unsafe.html").symlink_to(target)

    assert (
        list(
            LocalHtmlTreeLoader(
                tmp_path,
                source_base_url="https://docs.python.org/ja/3.13/",
                include_path_prefixes=("tutorial/",),
            ).load()
        )
        == []
    )


@pytest.mark.parametrize(
    "fixture_name",
    ("sample_python_doc.html", "sample_python_doc_faq.html"),
)
def test_python_adapter_preserves_existing_parser_and_chunk_bytes(
    fixture_name: str,
) -> None:
    content = (FIXTURES / fixture_name).read_text(encoding="utf-8")
    document = _source_document(content=content)
    direct = parse_python_doc_html_result(
        content,
        source_path=document.logical_path,
        category=document.category,
    )

    adapted = PythonSphinxHtmlParserAdapter().parse(document)

    assert adapted.sections == direct.sections
    assert adapted.excluded_section_count == direct.excluded_section_count
    assert [item.to_dict() for item in chunk_sections(adapted.sections)] == [
        item.to_dict() for item in chunk_sections(direct.sections)
    ]


def test_ingestion_continues_after_one_parser_failure() -> None:
    good = _source_document()
    bad = _source_document(logical_path="tutorial/bad.html")

    class Parser:
        def parse(self, document: SourceDocument):  # type: ignore[no-untyped-def]
            if document.logical_path.endswith("bad.html"):
                raise ValueError("fixture failure")
            return PythonSphinxHtmlParserAdapter().parse(document)

    result = ingest_documents([bad, good], Parser())

    assert len(result.documents) == 1
    assert len(result.failures) == 1
    assert result.failures[0].logical_path == "tutorial/bad.html"


def test_real_python_site_config_is_valid_and_hashed() -> None:
    path = Path("configs/sites/python-docs.toml")

    config = load_site_config(path)

    assert isinstance(config.loader, PinnedLocalArchiveLoaderSettings)
    assert config.loader.archive_sha256 == (
        "1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777"
    )
    assert config.loader.include_path_prefixes == (
        "tutorial/",
        "library/",
        "reference/",
        "howto/",
        "faq/",
    )
    assert config.loader.source_base_url == "https://docs.python.org/ja/3.13/"
    assert config.parser.type == "python-sphinx"
    assert config.parser.python_version == "3.13"
    assert config.chunking.chunk_size == 1000
    assert config.chunking.chunk_overlap == 150
    assert config.config_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_http_site_config_supports_multiple_urls_and_origins(tmp_path: Path) -> None:
    config = load_site_config(_write_config(tmp_path, _http_toml()))

    assert isinstance(config.loader, BoundedHttpLoaderSettings)
    assert config.loader.start_urls == (
        "https://example.com/docs/",
        "https://other.example/guide/",
    )
    assert config.loader.max_pages == 40
    assert config.parser.content_selectors == ("article", "main")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('slug = "example-docs"', 'slug = "../unsafe"', "kebab-case"),
        ("request_delay_seconds = 0.5", "request_delay_seconds = -1", "negative"),
        ("max_pages = 40", "max_pages = 201", "must not exceed"),
        (
            'content_selectors = ["article", "main"]',
            'content_selectors = ["["]',
            "selector",
        ),
        ("chunk_overlap = 150", "chunk_overlap = 1000", "smaller"),
        ("https://example.com/docs/", "ftp://example.com/docs/", "HTTP"),
    ],
)
def test_site_config_rejects_unsafe_or_invalid_values(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        load_site_config(_write_config(tmp_path, _http_toml().replace(old, new)))


def test_site_config_rejects_unknown_keys_and_wrong_types(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown loader"):
        load_site_config(
            _write_config(tmp_path, _http_toml(extra_loader="unexpected = true"))
        )
    wrong = _http_toml().replace("max_pages = 40", 'max_pages = "40"')
    with pytest.raises(TypeError, match="max_pages"):
        load_site_config(_write_config(tmp_path, wrong))


def test_site_config_rejects_empty_start_urls_and_query_by_default(
    tmp_path: Path,
) -> None:
    empty = _http_toml().replace(
        'start_urls = ["https://example.com/docs/", "https://other.example/guide/"]',
        "start_urls = []",
    )
    with pytest.raises(ValueError, match="must not be empty"):
        load_site_config(_write_config(tmp_path, empty))
    query = _http_toml().replace(
        "https://example.com/docs/", "https://example.com/docs/?version=1"
    )
    with pytest.raises(ValueError, match="query"):
        load_site_config(_write_config(tmp_path, query))


def test_generic_dataset_layout_and_atomic_manifest_round_trip(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    path = tmp_path / "dataset_manifest.json"

    write_dataset_manifest_atomic(manifest, path)
    loaded = load_dataset_manifest(path)
    resolved = resolve_dataset_artifacts(tmp_path)

    assert loaded == manifest
    assert resolved.dataset_manifest == manifest
    assert resolved.processed_jsonl == tmp_path / "data/processed/chunks.jsonl"
    assert resolved.index_path == tmp_path / "indexes/dense.faiss"
    assert resolved.profile_artifact_root == tmp_path / "profiles"


def test_missing_dataset_manifest_uses_exact_legacy_python_layout(
    tmp_path: Path,
) -> None:
    resolved = resolve_dataset_artifacts(tmp_path)

    assert resolved.legacy_python_layout
    assert resolved.processed_jsonl == (
        tmp_path / "data/processed/python_3_13_ja_chunks.jsonl"
    )
    assert resolved.index_path == tmp_path / "indexes/python_3_13_ja.faiss"
    assert resolved.metadata_path == (
        tmp_path / "indexes/python_3_13_ja_metadata.jsonl"
    )


@pytest.mark.parametrize("value", ("/tmp/chunks.jsonl", "data/../chunks.jsonl"))
def test_dataset_manifest_rejects_absolute_and_traversal_paths(value: str) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        safe_relative_path(value, "fixture")
    with pytest.raises(ValueError):
        replace(_manifest(), processed_chunks_path=value)


def test_dataset_manifest_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        resolve_manifest_path(tmp_path, "data/chunks.jsonl")


def test_dataset_manifest_rejects_unknown_key(tmp_path: Path) -> None:
    payload = _manifest().to_dict()
    payload["absolute_output"] = "/tmp/output"
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema mismatch"):
        load_dataset_manifest(path)


def test_dataset_manifest_revision_is_fixed() -> None:
    assert _manifest().schema_revision == DATASET_MANIFEST_REVISION


def test_generic_manifest_without_fingerprints_remains_honest_legacy_v1() -> None:
    manifest = generic_dataset_manifest(
        dataset_name="Example docs",
        dataset_slug="example-docs",
        loader_type="bounded-http",
        parser_type="generic-html",
        site_config_sha256="a" * 64,
        created_at="2026-08-01T00:00:00Z",
        source_page_count=2,
        section_count=3,
        chunk_count=4,
    )

    assert manifest.schema_revision == LEGACY_DATASET_MANIFEST_REVISION
    assert manifest.source_config_sha256 is None
    assert manifest.processing_config_sha256 is None
    assert manifest.source_snapshot_sha256 is None
    assert "source_config_sha256" not in manifest.to_dict()


def test_generic_manifest_rejects_partial_v2_fingerprint_identity() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        generic_dataset_manifest(
            dataset_name="Example docs",
            dataset_slug="example-docs",
            loader_type="bounded-http",
            parser_type="generic-html",
            site_config_sha256="a" * 64,
            created_at="2026-08-01T00:00:00Z",
            source_page_count=2,
            section_count=3,
            chunk_count=4,
            source_config_sha256="b" * 64,
        )
