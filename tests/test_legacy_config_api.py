from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from python_doc_rag.ingestion.registry import build_document_parser
from python_doc_rag.models import SourceDocument
from python_doc_rag.parsers.generic_html import GenericHtmlParser
from python_doc_rag.site_config import (
    GenericHtmlParserSettings,
    LocalCompatibilityLoaderSettings,
    LocalLoaderSettings,
    ParserSettings,
    PythonSphinxParserSettings,
    load_site_config,
    resolve_loader_settings,
    resolve_parser_settings,
)
from python_doc_rag.sites.python_docs.local_compat import LocalHtmlTreeLoader
from python_doc_rag.sites.python_docs.parser import PythonSphinxHtmlParserAdapter
from python_doc_rag.source_identity import (
    processing_config_sha256,
    source_config_sha256,
)


def _legacy_site_toml() -> str:
    return '''[dataset]
name = "Python 3.13 documentation"
slug = "python-3-13-ja"
language = "ja"
description = "legacy fixture"

[loader]
type = "local-html-tree"
categories = ["tutorial", "library"]

[parser]
type = "python-sphinx"
minimum_section_text_length = 20

[chunking]
chunk_size = 1000
chunk_overlap = 150

[index]
embedding_model = "fixture/model"
embedding_batch_size = 8

[profile]
prepare = "none"
'''


def _legacy_parser(parser_type: str = "python-sphinx") -> ParserSettings:
    return ParserSettings(
        type=parser_type,  # type: ignore[arg-type]
        content_selectors=("main",),
        exclude_selectors=("nav",),
        title_selectors=("h1", "title"),
        heading_levels=(1, 2, 3),
        minimum_section_text_length=20,
        fallback_to_body=False,
        include_lead_text=True,
    )


def test_historical_python_constants_are_lazy_and_importable() -> None:
    script = """
import sys
import python_doc_rag.config as config
assert 'python_doc_rag.sites.python_docs.constants' not in sys.modules
assert config.PYTHON_DOC_VERSION == '3.13'
assert config.PYTHON_DOC_BASE_URL == 'https://docs.python.org/ja/3.13/'
assert config.TARGET_CATEGORIES == ('tutorial', 'library', 'reference', 'howto', 'faq')
assert 'python_doc_rag.sites.python_docs.constants' in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_historical_settings_constructors_resolve_to_current_variants() -> None:
    loader = LocalLoaderSettings(
        type="local-html-tree",
        categories=("tutorial", "library"),
    )
    python_parser = _legacy_parser()
    generic_parser = _legacy_parser("generic-html")

    assert resolve_loader_settings(loader) == LocalCompatibilityLoaderSettings(
        type="local-html-tree",
        source_base_url="https://docs.python.org/ja/3.13/",
        include_path_prefixes=("tutorial/", "library/"),
    )
    assert resolve_parser_settings(python_parser) == PythonSphinxParserSettings(
        type="python-sphinx",
        python_version="3.13",
        minimum_section_text_length=20,
    )
    assert resolve_parser_settings(generic_parser) == GenericHtmlParserSettings(
        type="generic-html",
        content_selectors=("main",),
        exclude_selectors=("nav",),
        title_selectors=("h1", "title"),
        heading_levels=(1, 2, 3),
        minimum_section_text_length=20,
        fallback_to_body=False,
        include_lead_text=True,
    )


def test_v1_local_python_toml_loads_without_implicit_source_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.toml"
    path.write_text(_legacy_site_toml(), encoding="utf-8")

    config = load_site_config(path)

    assert config.loader == LocalCompatibilityLoaderSettings(
        type="local-html-tree",
        source_base_url="https://docs.python.org/ja/3.13/",
        include_path_prefixes=("tutorial/", "library/"),
    )
    assert config.parser == PythonSphinxParserSettings(
        type="python-sphinx",
        python_version="3.13",
        minimum_section_text_length=20,
    )


def test_legacy_and_resolved_settings_have_identical_effective_fingerprints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.toml"
    path.write_text(_legacy_site_toml(), encoding="utf-8")
    resolved_config = load_site_config(path)
    legacy_loader = LocalLoaderSettings(
        type="local-html-tree",
        categories=("tutorial", "library"),
    )
    legacy_config = replace(resolved_config, parser=_legacy_parser())

    assert source_config_sha256(legacy_loader, category="ignored") == (
        source_config_sha256(resolved_config.loader, category="ignored")
    )
    assert processing_config_sha256(legacy_config) == processing_config_sha256(
        resolved_config
    )


def test_parser_registry_accepts_historical_settings_objects() -> None:
    assert isinstance(
        build_document_parser(_legacy_parser()),
        PythonSphinxHtmlParserAdapter,
    )
    assert isinstance(
        build_document_parser(_legacy_parser("generic-html")),
        GenericHtmlParser,
    )


def test_no_argument_python_parser_preserves_historical_document_semantics() -> None:
    content = Path("tests/fixtures/sample_python_doc.html").read_text(encoding="utf-8")
    document = SourceDocument(
        source_url="https://example.invalid/custom.html",
        canonical_url="https://example.invalid/custom.html",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        source_kind="local-html-tree",
        logical_path="tutorial/example.html",
        category="tutorial",
        metadata={"python_version": "3.12"},
    )

    legacy = PythonSphinxHtmlParserAdapter().parse(document)
    configured = PythonSphinxHtmlParserAdapter(
        PythonSphinxParserSettings(
            type="python-sphinx",
            python_version="3.13",
            minimum_section_text_length=20,
        )
    ).parse(document)

    assert {section.python_version for section in legacy.sections} == {"3.12"}
    assert {section.python_version for section in configured.sections} == {"3.13"}
    assert all(
        section.source_url.startswith(
            "https://docs.python.org/ja/3.13/tutorial/example.html#"
        )
        for section in legacy.sections
    )
    assert all(
        section.source_url.startswith("https://example.invalid/custom.html#")
        for section in configured.sections
    )


def test_local_loader_accepts_historical_constructor_and_metadata(
    tmp_path: Path,
) -> None:
    for relative in ("tutorial/a.html", "tutorial/b.html"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html></html>", encoding="utf-8")
    loader = LocalHtmlTreeLoader(
        tmp_path,
        categories=("tutorial", "missing"),
        test_mode=True,
        max_files_per_category=1,
    )

    documents = list(loader.load())

    assert [item.logical_path for item in documents] == ["tutorial/a.html"]
    assert documents[0].metadata == {"python_version": "3.13"}
    assert loader.enumeration is not None
    assert loader.enumeration.missing_categories == ("missing",)
