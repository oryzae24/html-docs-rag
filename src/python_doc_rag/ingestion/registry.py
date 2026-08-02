"""Explicit built-in registries for independently configured loaders/parsers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python_doc_rag.ingestion.protocols import DocumentLoader, HtmlDocumentParser
from python_doc_rag.loaders.zip_archive import (
    PinnedLocalArchiveHtmlLoader,
    SnapshotHttpArchiveHtmlLoader,
)
from python_doc_rag.loading import BoundedHttpHtmlLoader
from python_doc_rag.parsers.generic_html import GenericHtmlParser
from python_doc_rag.site_config import (
    BoundedHttpLoaderSettings,
    GenericHtmlParserSettings,
    LoaderSettings,
    LocalCompatibilityLoaderSettings,
    ParserSettings,
    PinnedLocalArchiveLoaderSettings,
    PythonSphinxParserSettings,
    ResolvedParserSettings,
    SnapshotHttpArchiveLoaderSettings,
    resolve_loader_settings,
    resolve_parser_settings,
)
from python_doc_rag.sites.python_docs.local_compat import LocalHtmlTreeLoader
from python_doc_rag.sites.python_docs.parser import PythonSphinxHtmlParserAdapter

ParserFactory = Callable[[ResolvedParserSettings], HtmlDocumentParser]


@dataclass(frozen=True, slots=True)
class LoaderRuntimeOptions:
    """Runtime-only paths, modes, and injectable transports for one loader."""

    data_root: Path
    config_directory: Path
    dataset_slug: str
    source_config_sha256: str
    source_root: Path | None = None
    offline: bool = False
    refresh: bool = False
    resume: bool = False
    http_transport: Any | None = None
    archive_transport: Any | None = None


def _build_generic_html(settings: ResolvedParserSettings) -> HtmlDocumentParser:
    if not isinstance(settings, GenericHtmlParserSettings):
        raise TypeError("generic-html requires GenericHtmlParserSettings")
    return GenericHtmlParser(settings)


def _build_python_sphinx(settings: ResolvedParserSettings) -> HtmlDocumentParser:
    if not isinstance(settings, PythonSphinxParserSettings):
        raise TypeError("python-sphinx requires PythonSphinxParserSettings")
    return PythonSphinxHtmlParserAdapter(settings)


_PARSER_FACTORIES: dict[str, ParserFactory] = {
    "generic-html": _build_generic_html,
    "python-sphinx": _build_python_sphinx,
}

_SUPPORTED_COMBINATIONS = frozenset(
    {
        ("pinned-local-archive", "python-sphinx"),
        ("snapshot-http-archive", "python-sphinx"),
        ("local-html-tree", "python-sphinx"),
        ("bounded-http", "generic-html"),
    }
)


def build_document_parser(
    settings: ParserSettings | ResolvedParserSettings,
) -> HtmlDocumentParser:
    """Build the parser selected only by the discriminated settings type ID."""
    settings = resolve_parser_settings(settings)
    try:
        factory = _PARSER_FACTORIES[settings.type]
    except KeyError as error:
        raise ValueError(f"unsupported parser.type: {settings.type}") from error
    return factory(settings)


def build_document_loader(
    settings: LoaderSettings,
    runtime: LoaderRuntimeOptions,
) -> DocumentLoader:
    """Build the loader selected only by its discriminated settings type ID."""
    settings = resolve_loader_settings(settings)
    if isinstance(settings, LocalCompatibilityLoaderSettings):
        if runtime.source_root is None:
            raise ValueError("--source-root is required for local-html-tree")
        if runtime.refresh:
            raise ValueError(
                "--refresh is not supported for local-html-tree; use --rebuild"
            )
        return LocalHtmlTreeLoader(
            runtime.source_root,
            source_base_url=settings.source_base_url,
            include_path_prefixes=settings.include_path_prefixes,
        )
    if runtime.source_root is not None:
        raise ValueError(f"--source-root is not accepted for {settings.type}")
    if isinstance(settings, BoundedHttpLoaderSettings):
        return BoundedHttpHtmlLoader(
            settings,
            runtime.data_root / "data/raw",
            category=runtime.dataset_slug,
            offline=runtime.offline,
            refresh=runtime.refresh,
            resume=runtime.resume,
            transport=runtime.http_transport,
        )
    if isinstance(settings, PinnedLocalArchiveLoaderSettings):
        return PinnedLocalArchiveHtmlLoader(
            settings,
            runtime.data_root / "data/raw",
            config_directory=runtime.config_directory,
            source_config_sha256=runtime.source_config_sha256,
            offline=runtime.offline,
            refresh=runtime.refresh,
        )
    if isinstance(settings, SnapshotHttpArchiveLoaderSettings):
        return SnapshotHttpArchiveHtmlLoader(
            settings,
            runtime.data_root / "data/raw",
            source_config_sha256=runtime.source_config_sha256,
            offline=runtime.offline,
            refresh=runtime.refresh,
            transport=runtime.archive_transport,
        )
    raise ValueError(f"unsupported loader.type: {settings.type}")


def validate_loader_parser_combination(
    loader_settings: LoaderSettings,
    parser_settings: ParserSettings | ResolvedParserSettings,
) -> None:
    """Fail closed for combinations not declared by the built-in registry."""
    loader_settings = resolve_loader_settings(loader_settings)
    parser_settings = resolve_parser_settings(parser_settings)
    combination = (loader_settings.type, parser_settings.type)
    if combination not in _SUPPORTED_COMBINATIONS:
        raise ValueError(
            "unsupported loader/parser combination: "
            f"{loader_settings.type} + {parser_settings.type}"
        )


__all__ = [
    "LoaderRuntimeOptions",
    "build_document_loader",
    "build_document_parser",
    "validate_loader_parser_combination",
]
