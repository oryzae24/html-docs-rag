import hashlib
from dataclasses import replace

from python_doc_rag.config import ChunkingConfig
from python_doc_rag.models import SourceDocument
from python_doc_rag.site_config import (
    BoundedHttpLoaderSettings,
    DatasetSettings,
    GenericHtmlParserSettings,
    IndexSettings,
    LocalCompatibilityLoaderSettings,
    ProfilePreparationSettings,
    SiteConfig,
)
from python_doc_rag.source_identity import (
    canonical_json_sha256,
    processing_config_sha256,
    source_config_sha256,
    source_snapshot_sha256,
)


def _bounded_loader() -> BoundedHttpLoaderSettings:
    return BoundedHttpLoaderSettings(
        type="bounded-http",
        start_urls=("https://example.com/docs/",),
        max_pages=10,
        timeout_seconds=30,
        request_delay_seconds=0.25,
        max_response_bytes=1_000_000,
        user_agent="fixture/1.0",
        respect_robots_txt=True,
        retain_query=False,
        max_retries=2,
    )


def _site_config(*, raw_sha: str = "a" * 64) -> SiteConfig:
    return SiteConfig(
        revision="site-config-v2",
        config_sha256=raw_sha,
        dataset=DatasetSettings(
            name="Fixture docs",
            slug="fixture-docs",
            language="en",
            description="fixture",
        ),
        loader=_bounded_loader(),
        parser=GenericHtmlParserSettings(
            type="generic-html",
            content_selectors=("main",),
            exclude_selectors=("nav",),
            title_selectors=("h1", "title"),
            heading_levels=(1, 2, 3),
            minimum_section_text_length=20,
            fallback_to_body=False,
            include_lead_text=True,
        ),
        chunking=ChunkingConfig(chunk_size=1000, chunk_overlap=150),
        index=IndexSettings(embedding_model="fixture/model", embedding_batch_size=8),
        profile=ProfilePreparationSettings(prepare="none"),
    )


def _document(*, url: str, text: str, logical_path: str) -> SourceDocument:
    return SourceDocument(
        source_url=url,
        canonical_url=url,
        content=text,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_kind="bounded-http",
        logical_path=logical_path,
        category="fixture-docs",
    )


def test_canonical_json_hash_is_independent_of_mapping_order() -> None:
    assert canonical_json_sha256({"b": [2, 1], "a": "value"}) == (
        canonical_json_sha256({"a": "value", "b": [2, 1]})
    )


def test_source_fingerprint_ignores_transport_tuning() -> None:
    settings = _bounded_loader()
    tuned = replace(
        settings,
        timeout_seconds=90,
        request_delay_seconds=1.0,
        max_retries=7,
    )

    assert source_config_sha256(settings, category="fixture-docs") == (
        source_config_sha256(tuned, category="fixture-docs")
    )


def test_bounded_source_url_and_max_pages_change_source_fingerprint() -> None:
    settings = _bounded_loader()
    original = source_config_sha256(settings, category="fixture-docs")

    assert original != source_config_sha256(
        replace(settings, start_urls=("https://example.com/other/",)),
        category="fixture-docs",
    )
    assert original != source_config_sha256(
        replace(settings, max_pages=settings.max_pages + 1),
        category="fixture-docs",
    )


def test_local_prefix_and_source_base_url_change_source_fingerprint() -> None:
    settings = LocalCompatibilityLoaderSettings(
        type="local-html-tree",
        source_base_url="https://docs.example.test/v1/",
        include_path_prefixes=("guide/",),
    )
    original = source_config_sha256(settings, category="ignored")

    assert original != source_config_sha256(
        replace(settings, source_base_url="https://docs.example.test/v2/"),
        category="ignored",
    )
    assert original != source_config_sha256(
        replace(settings, include_path_prefixes=("reference/",)),
        category="ignored",
    )


def test_comment_only_raw_config_change_affects_neither_semantic_fingerprint() -> None:
    original = _site_config(raw_sha="a" * 64)
    comment_only_change = replace(original, config_sha256="b" * 64)

    assert source_config_sha256(original.loader, category="fixture-docs") == (
        source_config_sha256(comment_only_change.loader, category="fixture-docs")
    )
    assert processing_config_sha256(original) == processing_config_sha256(
        comment_only_change
    )


def test_parser_change_only_changes_processing_fingerprint() -> None:
    original = _site_config()
    changed = replace(
        original,
        parser=replace(original.parser, minimum_section_text_length=21),
    )

    assert source_config_sha256(original.loader, category="fixture-docs") == (
        source_config_sha256(changed.loader, category="fixture-docs")
    )
    assert processing_config_sha256(original) != processing_config_sha256(changed)


def test_source_snapshot_hash_preserves_page_order_and_content_identity() -> None:
    first = _document(
        url="https://example.com/docs/a.html",
        text="alpha",
        logical_path="docs/a.html",
    )
    second = _document(
        url="https://example.com/docs/b.html",
        text="beta",
        logical_path="docs/b.html",
    )
    changed_second = _document(
        url=second.source_url,
        text="changed beta",
        logical_path=second.logical_path,
    )

    original = source_snapshot_sha256((first, second))
    assert original != source_snapshot_sha256((second, first))
    assert original != source_snapshot_sha256((first, changed_second))
