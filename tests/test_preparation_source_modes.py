import ast
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import pytest

import python_doc_rag.preparation as preparation
from python_doc_rag.dataset_layout import (
    LEGACY_DATASET_MANIFEST_REVISION,
    generic_dataset_manifest,
    load_dataset_manifest,
    resolve_dataset_artifacts,
)
from python_doc_rag.ingestion.protocols import DocumentParseResult
from python_doc_rag.loaders.zip_archive import (
    ArchiveDownloadResult,
    SnapshotHttpArchiveHtmlLoader,
)
from python_doc_rag.models import DocumentSection, SourceDocument
from python_doc_rag.source_identity import source_snapshot_sha256


def _write_site_config(
    path: Path,
    *,
    loader: str = "bounded-http",
    parser_minimum: int = 5,
    start_url: str = "https://example.com/docs/",
    max_pages: int = 2,
    source_base_url: str = "https://docs.example.test/v1/",
    include_prefix: str = "guide/",
) -> None:
    if loader == "bounded-http":
        loader_table = f"""type = "bounded-http"
start_urls = ["{start_url}"]
max_pages = {max_pages}
timeout_seconds = 2
request_delay_seconds = 0
max_response_bytes = 10000
user_agent = "fixture/1.0"
respect_robots_txt = true
retain_query = false
max_retries = 0"""
        parser_table = f"""type = "generic-html"
content_selectors = ["main"]
exclude_selectors = ["nav"]
title_selectors = ["h1", "title"]
heading_levels = [1, 2, 3]
minimum_section_text_length = {parser_minimum}
fallback_to_body = false
include_lead_text = true"""
    elif loader == "local-html-tree":
        loader_table = f"""type = "local-html-tree"
source_base_url = "{source_base_url}"
include_path_prefixes = ["{include_prefix}"]"""
        parser_table = f"""type = "python-sphinx"
python_version = "3.13"
minimum_section_text_length = {parser_minimum}"""
    elif loader == "pinned-local-archive":
        loader_table = f"""type = "pinned-local-archive"
archive_path = "fixture.zip"
archive_sha256 = "{"a" * 64}"
archive_format = "zip"
archive_root = "python-docs-html"
original_archive_url = "https://example.com/python-docs.zip"
source_base_url = "https://docs.example.test/v1/"
include_path_prefixes = ["guide/"]
max_archive_bytes = 1000000
max_members = 100
max_member_bytes = 100000
max_extracted_bytes = 500000"""
        parser_table = f"""type = "python-sphinx"
python_version = "3.13"
minimum_section_text_length = {parser_minimum}"""
    elif loader == "snapshot-http-archive":
        loader_table = '''type = "snapshot-http-archive"
archive_url = "https://example.com/python-docs.zip"
archive_format = "zip"
archive_root = "python-docs-html"
source_base_url = "https://docs.example.test/v1/"
include_path_prefixes = ["guide/"]
update_policy = "manual"
timeout_seconds = 2
max_retries = 0
max_archive_bytes = 1000000
max_members = 100
max_member_bytes = 100000
max_extracted_bytes = 500000
user_agent = "fixture/1.0"'''
        parser_table = f"""type = "python-sphinx"
python_version = "3.13"
minimum_section_text_length = {parser_minimum}"""
    else:  # pragma: no cover - test helper boundary
        raise AssertionError(loader)
    path.write_text(
        f"""[dataset]
name = "Fixture docs"
slug = "fixture-docs"
language = "en"
description = "fixture"

[loader]
{loader_table}

[parser]
{parser_table}

[chunking]
chunk_size = 1000
chunk_overlap = 150

[index]
embedding_model = "fixture/model"
embedding_batch_size = 2

[profile]
prepare = "none"
""",
        encoding="utf-8",
    )


def _document() -> SourceDocument:
    html = "<html><main><h1>Fixture</h1><p>fixture text</p></main></html>"
    return SourceDocument(
        source_url="https://example.com/docs/page.html",
        canonical_url="https://example.com/docs/page.html",
        content=html,
        content_sha256=hashlib.sha256(html.encode()).hexdigest(),
        source_kind="bounded-http",
        logical_path="docs/page.html",
        category="fixture-docs",
    )


class _FixtureParser:
    def parse(self, document: SourceDocument) -> DocumentParseResult:
        return DocumentParseResult(
            sections=(
                DocumentSection(
                    text="fixture text long enough to produce one searchable chunk",
                    page_title="Fixture",
                    section_title="Fixture",
                    source_path=document.logical_path,
                    source_url=f"{document.source_url}#fixture",
                    anchor="fixture",
                    category=document.category,
                    python_version="",
                ),
            )
        )


class _EmptyParser:
    def parse(self, document: SourceDocument) -> DocumentParseResult:
        del document
        return DocumentParseResult(sections=())


class _ArchiveTransport:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def download(
        self,
        url: str,
        destination: BinaryIO,
        *,
        timeout_seconds: float,
        user_agent: str,
        max_archive_bytes: int,
    ) -> ArchiveDownloadResult:
        del timeout_seconds, user_agent, max_archive_bytes
        self.calls.append(url)
        destination.write(self.payload)
        return ArchiveDownloadResult(url, url, len(self.payload))


def _snapshot_archive_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "python-docs-html/guide/page.html",
            "<html><main><h1>Fixture</h1><p>fixture text</p></main></html>",
        )
    return output.getvalue()


class _FixtureLoader:
    def __init__(
        self,
        runtime: Any,
        *,
        loader_type: str = "bounded-http",
        snapshot_sha256: str | None = None,
        replace_manifest_on_load: bool = False,
    ) -> None:
        self.runtime = runtime
        self.loader_type = loader_type
        self.source_snapshot_sha256 = snapshot_sha256
        self.summary = SimpleNamespace(cache_reused=runtime.offline)
        self.replace_manifest_on_load = replace_manifest_on_load
        self.previous_manifest: bytes | None = None
        self.rollback_called = False
        self.commit_called = False

    def load(self) -> tuple[SourceDocument, ...]:
        document = _document()
        if self.replace_manifest_on_load:
            manifest = self.runtime.data_root / "data/raw/fetch_manifest.json"
            self.previous_manifest = (
                manifest.read_bytes() if manifest.is_file() else None
            )
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text('{"candidate": true}\n', encoding="utf-8")
        elif self.loader_type == "bounded-http":
            raw_root = self.runtime.data_root / "data/raw"
            cache_relative = "pages/fixture.html"
            cache = raw_root / cache_relative
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(document.content, encoding="utf-8")
            payload = {
                "schema_revision": "bounded-http-fetch-v2",
                "complete": True,
                "loader_type": "bounded-http",
                "cache_reused": self.runtime.offline,
                "offline": self.runtime.offline,
                "source_config_fingerprint_revision": "source-config-v1",
                "source_config_sha256": self.runtime.source_config_sha256,
                "start_urls": ["https://example.com/docs/"],
                "boundaries": [],
                "retain_query": False,
                "max_pages": 2,
                "fetched_page_count": 1,
                "failure_count": 0,
                "duplicate_canonical_count": 0,
                "duplicate_content_count": 0,
                "excluded_external_url_count": 0,
                "pages": [
                    {
                        "source_url": document.source_url,
                        "canonical_url": document.canonical_url,
                        "logical_path": document.logical_path,
                        "category": document.category,
                        "fetched_at": None,
                        "content_sha256": document.content_sha256,
                        "cache_path": cache_relative,
                        "http_status": 200,
                        "content_type": "text/html; charset=utf-8",
                        "response_bytes": len(document.content.encode("utf-8")),
                    }
                ],
                "failures": [],
                "finished_at": "2026-08-01T00:00:00+00:00",
            }
            (raw_root / "fetch_manifest.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            if self.source_snapshot_sha256 is None:
                self.source_snapshot_sha256 = source_snapshot_sha256((document,))
        if self.source_snapshot_sha256 is None:
            self.source_snapshot_sha256 = "c" * 64
        return (document,)

    def rollback_source(self) -> None:
        self.rollback_called = True
        if not self.replace_manifest_on_load:
            return
        manifest = self.runtime.data_root / "data/raw/fetch_manifest.json"
        if self.previous_manifest is None:
            manifest.unlink(missing_ok=True)
        else:
            manifest.write_bytes(self.previous_manifest)

    def commit_source(self) -> None:
        self.commit_called = True


def _install_fixture_factories(
    monkeypatch: pytest.MonkeyPatch,
    loaders: list[_FixtureLoader],
    *,
    parser: object | None = None,
) -> None:
    def build_loader(settings: object, runtime: Any) -> _FixtureLoader:
        loader = _FixtureLoader(runtime, loader_type=str(settings.type))
        loaders.append(loader)
        return loader

    monkeypatch.setattr(preparation, "build_document_loader", build_loader)
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: parser or _FixtureParser(),
    )


def test_completed_dataset_reuse_does_not_rebuild_source_or_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)

    first = preparation.prepare_dataset(config, data_root, until="corpus")
    config.write_text(
        config.read_text(encoding="utf-8") + "\n# non-semantic edit\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preparation,
        "build_document_loader",
        lambda settings, runtime: pytest.fail("completed dataset must skip loader"),
    )
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: pytest.fail("completed dataset must skip parser"),
    )

    second = preparation.prepare_dataset(config, data_root, until="corpus")

    assert len(loaders) == 1
    assert not first.reused_dataset
    assert second.reused_dataset
    assert first.source_snapshot_sha256 == second.source_snapshot_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("archive_root", "wrong-root"),
        ("byte_size", 999),
        ("unexpected", "field"),
    ),
)
def test_completed_snapshot_dataset_strictly_validates_source_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config, loader="snapshot-http-archive")
    transport = _ArchiveTransport(_snapshot_archive_bytes())

    def build_loader(settings: Any, runtime: Any) -> SnapshotHttpArchiveHtmlLoader:
        return SnapshotHttpArchiveHtmlLoader(
            settings,
            runtime.data_root / "data/raw",
            source_config_sha256=runtime.source_config_sha256,
            transport=transport,
        )

    monkeypatch.setattr(preparation, "build_document_loader", build_loader)
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: _FixtureParser(),
    )
    preparation.prepare_dataset(config, data_root, until="corpus")
    lock_path = data_root / "data/raw/source.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock[field] = value
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(
        preparation,
        "build_document_loader",
        lambda settings, runtime: pytest.fail(
            "fast validation must not build a loader"
        ),
    )

    with pytest.raises(RuntimeError, match="source lock"):
        preparation.prepare_dataset(config, data_root, until="corpus")

    assert transport.calls == ["https://example.com/python-docs.zip"]


@pytest.mark.parametrize("mutation", ("archive-url", "page-url"))
def test_completed_snapshot_dataset_validates_archive_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config, loader="snapshot-http-archive")
    transport = _ArchiveTransport(_snapshot_archive_bytes())

    def build_loader(settings: Any, runtime: Any) -> SnapshotHttpArchiveHtmlLoader:
        return SnapshotHttpArchiveHtmlLoader(
            settings,
            runtime.data_root / "data/raw",
            source_config_sha256=runtime.source_config_sha256,
            transport=transport,
        )

    monkeypatch.setattr(preparation, "build_document_loader", build_loader)
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: _FixtureParser(),
    )
    preparation.prepare_dataset(config, data_root, until="corpus")
    manifest_path = data_root / "data/raw/fetch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "archive-url":
        manifest["requested_archive_url"] = "https://attacker.invalid/archive.zip"
    else:
        manifest["pages"][0]["source_url"] = "https://attacker.invalid/"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="archive source manifest"):
        preparation.prepare_dataset(config, data_root, until="corpus")

    assert transport.calls == ["https://example.com/python-docs.zip"]


def test_same_byte_snapshot_refresh_crash_before_augmentation_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config, loader="snapshot-http-archive")
    payload = _snapshot_archive_bytes()
    transport = _ArchiveTransport(payload)

    def build_loader(settings: Any, runtime: Any) -> SnapshotHttpArchiveHtmlLoader:
        return SnapshotHttpArchiveHtmlLoader(
            settings,
            runtime.data_root / "data/raw",
            source_config_sha256=runtime.source_config_sha256,
            offline=runtime.offline,
            refresh=runtime.refresh,
            transport=transport,
        )

    monkeypatch.setattr(preparation, "build_document_loader", build_loader)
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: _FixtureParser(),
    )
    first = preparation.prepare_dataset(config, data_root, until="corpus")
    source_path = data_root / "data/raw/fetch_manifest.json"
    old_source = source_path.read_bytes()
    settings = preparation.load_site_config(config).loader
    assert isinstance(settings, preparation.SnapshotHttpArchiveLoaderSettings)
    crashed_refresh = SnapshotHttpArchiveHtmlLoader(
        settings,
        data_root / "data/raw",
        source_config_sha256=first.dataset_manifest.source_config_sha256 or "",
        refresh=True,
        transport=transport,
    )

    crashed_refresh.load()

    assert (data_root / "data/raw/.source-refresh-rollback").is_dir()
    assert "parser_type" not in json.loads(source_path.read_text(encoding="utf-8"))
    recovered = preparation.prepare_dataset(config, data_root, until="corpus")
    assert recovered.reused_dataset
    assert source_path.read_bytes() == old_source
    assert not (data_root / "data/raw/.source-refresh-rollback").exists()
    assert transport.calls == [
        "https://example.com/python-docs.zip",
        "https://example.com/python-docs.zip",
    ]


@pytest.mark.parametrize("mutation", ("legacy-schema", "page-url"))
def test_completed_bounded_dataset_recomputes_source_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)
    preparation.prepare_dataset(config, data_root, until="corpus")
    manifest_path = data_root / "data/raw/fetch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "legacy-schema":
        manifest["schema_revision"] = "bounded-http-fetch-v1"
    else:
        manifest["pages"][0]["source_url"] = "https://attacker.invalid/"
        manifest["pages"][0]["canonical_url"] = "https://attacker.invalid/"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="bounded source"):
        preparation.prepare_dataset(config, data_root, until="corpus")

    assert len(loaders) == 1


def test_bounded_raw_replayability_does_not_depend_on_derived_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config_path)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)
    result = preparation.prepare_dataset(config_path, data_root, until="corpus")
    (data_root / "dataset_manifest.json").write_text("not-json", encoding="utf-8")
    config = preparation.load_site_config(config_path)

    assert preparation._loader_current_source_is_replayable(
        config,
        root=data_root,
        source_config=result.dataset_manifest.source_config_sha256 or "",
    )


def test_parser_change_requires_rebuild_without_source_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config, parser_minimum=5)
    loaders: list[_FixtureLoader] = []
    network_calls = 0

    def build_loader(settings: object, runtime: Any) -> _FixtureLoader:
        nonlocal network_calls
        del settings
        if not runtime.offline:
            network_calls += 1
        loader = _FixtureLoader(runtime)
        loaders.append(loader)
        return loader

    monkeypatch.setattr(preparation, "build_document_loader", build_loader)
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: _FixtureParser(),
    )
    first = preparation.prepare_dataset(config, data_root, until="corpus")
    _write_site_config(config, parser_minimum=6)

    with pytest.raises(RuntimeError, match="use --rebuild"):
        preparation.prepare_dataset(config, data_root, until="corpus")

    rebuilt = preparation.prepare_dataset(
        config,
        data_root,
        rebuild=True,
        until="corpus",
    )

    assert network_calls == 1
    assert len(loaders) == 2
    assert loaders[-1].runtime.offline
    assert first.source_snapshot_sha256 == rebuilt.source_snapshot_sha256
    assert not rebuilt.reused_dataset


@pytest.mark.parametrize(
    ("loader", "original", "changed", "source_root"),
    [
        (
            "bounded-http",
            {"start_url": "https://example.com/docs/"},
            {"start_url": "https://example.com/changed/"},
            None,
        ),
        (
            "bounded-http",
            {"max_pages": 2},
            {"max_pages": 3},
            None,
        ),
        (
            "local-html-tree",
            {"include_prefix": "guide/"},
            {"include_prefix": "reference/"},
            "source",
        ),
    ],
)
def test_source_setting_change_rejects_implicit_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: str,
    original: dict[str, Any],
    changed: dict[str, Any],
    source_root: str | None,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    root = None if source_root is None else tmp_path / source_root
    _write_site_config(config, loader=loader, **original)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)
    preparation.prepare_dataset(config, data_root, source_root=root, until="corpus")
    _write_site_config(config, loader=loader, **changed)

    with pytest.raises(RuntimeError, match="different source config"):
        preparation.prepare_dataset(
            config,
            data_root,
            source_root=root,
            until="corpus",
        )

    assert len(loaders) == 1


@pytest.mark.parametrize(
    ("loader", "kwargs", "message"),
    [
        (
            "bounded-http",
            {"refresh": True, "rebuild": True},
            "mutually exclusive",
        ),
        (
            "bounded-http",
            {"refresh": True, "offline": True},
            "cannot be used together",
        ),
        ("local-html-tree", {}, "--source-root is required"),
        (
            "local-html-tree",
            {"source_root": Path("source"), "refresh": True},
            "--refresh is not supported",
        ),
        (
            "bounded-http",
            {"source_root": Path("source")},
            "--source-root is not accepted",
        ),
        (
            "snapshot-http-archive",
            {"source_root": Path("source")},
            "--source-root is not accepted",
        ),
        (
            "pinned-local-archive",
            {"source_root": Path("source")},
            "--source-root is not accepted",
        ),
        (
            "pinned-local-archive",
            {"refresh": True},
            "--refresh is not supported",
        ),
    ],
)
def test_prepare_mode_and_source_root_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: str,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    config = tmp_path / "site.toml"
    _write_site_config(config, loader=loader)
    monkeypatch.setattr(
        preparation,
        "build_document_loader",
        lambda settings, runtime: pytest.fail("validation must precede loader build"),
    )

    with pytest.raises((ValueError, RuntimeError), match=message):
        preparation.prepare_dataset(
            config,
            tmp_path / "data-root",
            until="corpus",
            **kwargs,
        )


def test_refresh_failure_rolls_back_source_and_keeps_derived_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config, loader="snapshot-http-archive")
    initial_loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, initial_loaders)
    preparation.prepare_dataset(config, data_root, until="corpus")
    old_manifest = (data_root / "dataset_manifest.json").read_bytes()
    old_chunks = (data_root / "data/processed/chunks.jsonl").read_bytes()
    old_source_manifest = (data_root / "data/raw/fetch_manifest.json").read_bytes()
    refreshed: list[_FixtureLoader] = []

    def build_refresh_loader(settings: object, runtime: Any) -> _FixtureLoader:
        del settings
        loader = _FixtureLoader(
            runtime,
            snapshot_sha256="d" * 64,
            replace_manifest_on_load=True,
        )
        refreshed.append(loader)
        return loader

    monkeypatch.setattr(preparation, "build_document_loader", build_refresh_loader)
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: _EmptyParser(),
    )

    with pytest.raises(RuntimeError, match="no searchable chunks"):
        preparation.prepare_dataset(
            config,
            data_root,
            refresh=True,
            until="corpus",
        )

    assert len(refreshed) == 1
    assert refreshed[0].rollback_called
    assert not refreshed[0].commit_called
    assert (data_root / "dataset_manifest.json").read_bytes() == old_manifest
    assert (data_root / "data/processed/chunks.jsonl").read_bytes() == old_chunks
    assert (data_root / "data/raw/fetch_manifest.json").read_bytes() == (
        old_source_manifest
    )


def test_failed_rebuild_restores_exact_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config, parser_minimum=5)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)
    preparation.prepare_dataset(config, data_root, until="corpus")
    old_source_manifest = (data_root / "data/raw/fetch_manifest.json").read_bytes()
    old_dataset_manifest = (data_root / "dataset_manifest.json").read_bytes()
    _write_site_config(config, parser_minimum=6)
    _install_fixture_factories(monkeypatch, loaders, parser=_EmptyParser())

    with pytest.raises(RuntimeError, match="no searchable chunks"):
        preparation.prepare_dataset(
            config,
            data_root,
            rebuild=True,
            until="corpus",
        )

    assert (data_root / "data/raw/fetch_manifest.json").read_bytes() == (
        old_source_manifest
    )
    assert (data_root / "dataset_manifest.json").read_bytes() == old_dataset_manifest


def test_source_commit_failure_rolls_back_derived_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config, loader="snapshot-http-archive")
    initial: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, initial)
    preparation.prepare_dataset(config, data_root, until="corpus")
    old_dataset = (data_root / "dataset_manifest.json").read_bytes()
    old_chunks = (data_root / "data/processed/chunks.jsonl").read_bytes()
    old_source = (data_root / "data/raw/fetch_manifest.json").read_bytes()

    class CommitFailureLoader(_FixtureLoader):
        def commit_source(self) -> None:
            raise RuntimeError("injected source commit failure")

    def build_loader(settings: object, runtime: Any) -> CommitFailureLoader:
        del settings
        return CommitFailureLoader(
            runtime,
            snapshot_sha256="d" * 64,
            replace_manifest_on_load=True,
        )

    monkeypatch.setattr(preparation, "build_document_loader", build_loader)
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: _FixtureParser(),
    )

    with pytest.raises(RuntimeError, match="injected source commit failure"):
        preparation.prepare_dataset(
            config,
            data_root,
            refresh=True,
            until="corpus",
        )

    assert (data_root / "dataset_manifest.json").read_bytes() == old_dataset
    assert (data_root / "data/processed/chunks.jsonl").read_bytes() == old_chunks
    assert (data_root / "data/raw/fetch_manifest.json").read_bytes() == old_source
    assert not (data_root / ".prepare-publication.json").exists()


def test_v1_dataset_manifest_remains_readable(tmp_path: Path) -> None:
    current = generic_dataset_manifest(
        dataset_name="Fixture",
        dataset_slug="fixture-docs",
        loader_type="bounded-http",
        parser_type="generic-html",
        site_config_sha256="a" * 64,
        created_at="2026-08-01T00:00:00+00:00",
        source_page_count=1,
        section_count=1,
        chunk_count=1,
        source_config_sha256="b" * 64,
        processing_config_sha256="c" * 64,
        source_snapshot_sha256="d" * 64,
    ).to_dict()
    current["schema_revision"] = LEGACY_DATASET_MANIFEST_REVISION
    current.pop("source_config_sha256")
    current.pop("processing_config_sha256")
    current.pop("source_snapshot_sha256")
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(current), encoding="utf-8")

    loaded = load_dataset_manifest(path)

    assert loaded.schema_revision == LEGACY_DATASET_MANIFEST_REVISION
    assert loaded.source_config_sha256 is None
    assert loaded.processing_config_sha256 is None
    assert loaded.source_snapshot_sha256 is None


def test_preparation_does_not_import_or_construct_python_parser_directly() -> None:
    module_path = Path(preparation.__file__ or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert not any(
        name.startswith("python_doc_rag.sites.python_docs") for name in imported_modules
    )
    assert "PythonSphinxHtmlParserAdapter" not in names


def test_publication_preserves_backups_when_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    staging = root / ".prepare-staging"
    (root / "data/processed").mkdir(parents=True)
    (staging / "data/processed").mkdir(parents=True)
    (root / "data/processed/value.txt").write_text("old", encoding="utf-8")
    (staging / "data/processed/value.txt").write_text("new", encoding="utf-8")
    (root / "dataset_manifest.json").write_text("old", encoding="utf-8")
    (staging / "dataset_manifest.json").write_text("new", encoding="utf-8")
    real_rename = os.rename

    def unreliable_rename(source: Path | str, destination: Path | str) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == staging / "dataset_manifest.json":
            raise OSError("publication failure")
        if (
            source_path == root / "data/processed"
            and destination_path == staging / "data/processed"
        ):
            raise OSError("rollback failure")
        real_rename(source, destination)

    monkeypatch.setattr(preparation.os, "rename", unreliable_rename)

    with pytest.raises(RuntimeError, match="recoverable backups retained"):
        preparation._publish_derived_transaction(
            staging,
            root,
            include_index=False,
            include_profile=False,
            remove_unbuilt=False,
        )

    backup_roots = list(root.glob(".prepare-publication-backup-*"))
    assert len(backup_roots) == 1
    assert any(backup_roots[0].iterdir())
    monkeypatch.setattr(preparation.os, "rename", real_rename)

    preparation._recover_derived_publication(root)

    assert (root / "data/processed/value.txt").read_text(encoding="utf-8") == "old"
    assert (root / "dataset_manifest.json").read_text(encoding="utf-8") == "old"
    assert not list(root.glob(".prepare-publication-backup-*"))
    assert not (root / ".prepare-publication.json").exists()


def test_publication_journal_rejects_escaping_source_before_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    staging = root / ".prepare-staging"
    backup = root / ".prepare-publication-backup-fixture"
    (staging / "data/processed").mkdir(parents=True)
    backup.mkdir(parents=True)
    (root / "dataset_manifest.json").write_text("old", encoding="utf-8")
    payload = {
        "schema_revision": "prepare-publication-v1",
        "phase": "publishing",
        "staging": ".prepare-staging",
        "backup_root": backup.name,
        "entries": [
            {
                "source": ".prepare-staging/data/processed",
                "destination": "data/processed",
                "backup": f"{backup.name}/0",
                "had_destination": False,
            },
            {
                "source": ".prepare-staging/../../escaped-X",
                "destination": "dataset_manifest.json",
                "backup": f"{backup.name}/1",
                "had_destination": True,
            },
        ],
    }
    (root / ".prepare-publication.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="source path"):
        preparation._recover_derived_publication(root)

    assert (root / "dataset_manifest.json").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "escaped-X").exists()


@pytest.mark.parametrize(
    ("destinations", "expected"),
    (
        (("data/processed", "dataset_manifest.json"), "corpus"),
        (("data/processed", "indexes", "dataset_manifest.json"), "index"),
        (
            (
                "data/processed",
                "indexes",
                "profiles/recommended-v2",
                "dataset_manifest.json",
            ),
            "profile",
        ),
    ),
)
def test_publication_validation_target_covers_all_published_artifacts(
    destinations: tuple[str, ...],
    expected: str,
) -> None:
    publication = {
        "entries": [
            {"destination": destination, "source": f"staged/{destination}"}
            for destination in destinations
        ]
    }

    assert preparation._publication_validation_target(publication) == expected


def test_publication_validation_target_ignores_removed_unbuilt_artifacts() -> None:
    publication = {
        "entries": [
            {"destination": "data/processed", "source": "staged/data/processed"},
            {"destination": "indexes", "source": None},
            {"destination": "profiles/recommended-v2", "source": None},
            {
                "destination": "dataset_manifest.json",
                "source": "staged/dataset_manifest.json",
            },
        ]
    }

    assert preparation._publication_validation_target(publication) == "corpus"


def test_publication_rollback_restores_backup_when_candidate_is_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    staging = root / ".prepare-staging"
    (root / "data/processed").mkdir(parents=True)
    (staging / "data/processed").mkdir(parents=True)
    (root / "data/processed/value.txt").write_text("old", encoding="utf-8")
    (staging / "data/processed/value.txt").write_text("new", encoding="utf-8")
    (root / "dataset_manifest.json").write_text("old", encoding="utf-8")
    (staging / "dataset_manifest.json").write_text("new", encoding="utf-8")
    publication = preparation._publish_derived_transaction(
        staging,
        root,
        include_index=False,
        include_profile=False,
        remove_unbuilt=False,
    )
    shutil.rmtree(root / "data/processed")
    preparation._mark_derived_publication_for_rollback(root, publication)

    preparation._rollback_derived_publication(root, publication)

    assert (root / "data/processed/value.txt").read_text(encoding="utf-8") == "old"
    assert (root / "dataset_manifest.json").read_text(encoding="utf-8") == "old"
    assert not (root / ".prepare-publication.json").exists()


def test_partial_precommit_rollback_is_not_forward_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    staging = root / ".prepare-staging"
    for relative in ("data/processed", "indexes"):
        (root / relative).mkdir(parents=True)
        (staging / relative).mkdir(parents=True)
        (root / relative / "value.txt").write_text("old", encoding="utf-8")
        (staging / relative / "value.txt").write_text("new", encoding="utf-8")
    (root / "dataset_manifest.json").write_text("old", encoding="utf-8")
    (staging / "dataset_manifest.json").write_text("new", encoding="utf-8")
    publication = preparation._publish_derived_transaction(
        staging,
        root,
        include_index=True,
        include_profile=False,
        remove_unbuilt=False,
    )
    index_entry = next(
        entry for entry in publication["entries"] if entry["destination"] == "indexes"
    )
    index_backup = root / index_entry["backup"]
    real_rename = preparation.os.rename
    failed_once = False

    def fail_one_index_restore(source: Path | str, destination: Path | str) -> None:
        nonlocal failed_once
        if Path(source) == index_backup and not failed_once:
            failed_once = True
            raise OSError("injected index rollback failure")
        real_rename(source, destination)

    monkeypatch.setattr(preparation.os, "rename", fail_one_index_restore)

    errors = preparation._rollback_preparation_transaction(
        root=root,
        loader=SimpleNamespace(),
        publication=publication,
        source_manifest_path=root / "data/raw/fetch_manifest.json",
        previous_source_manifest=None,
        restore_source_manifest=False,
    )

    assert errors
    journal = json.loads(
        (root / ".prepare-publication.json").read_text(encoding="utf-8")
    )
    assert journal["phase"] == "publishing"
    assert (root / "indexes/value.txt").exists() is False
    assert index_backup.is_dir()
    monkeypatch.setattr(preparation.os, "rename", real_rename)

    preparation._recover_derived_publication(root)

    for relative in ("data/processed", "indexes"):
        assert (root / relative / "value.txt").read_text(encoding="utf-8") == "old"
    assert (root / "dataset_manifest.json").read_text(encoding="utf-8") == "old"
    assert not (root / ".prepare-publication.json").exists()


def test_commit_marker_interrupt_and_partial_rollback_remain_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    staging = root / ".prepare-staging"
    for relative in ("data/processed", "indexes"):
        (root / relative).mkdir(parents=True)
        (staging / relative).mkdir(parents=True)
        (root / relative / "value.txt").write_text("old", encoding="utf-8")
        (staging / relative / "value.txt").write_text("new", encoding="utf-8")
    (root / "dataset_manifest.json").write_text("old", encoding="utf-8")
    (staging / "dataset_manifest.json").write_text("new", encoding="utf-8")
    real_write_json = preparation._write_json_atomic
    real_rename = preparation.os.rename
    interrupted = False
    failed_restore = False

    def interrupt_after_commit_marker(payload: dict[str, Any], path: Path) -> None:
        nonlocal interrupted
        real_write_json(payload, path)
        if (
            path.name == ".prepare-publication.json"
            and payload.get("phase") == "committed"
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("injected post-marker interrupt")

    def fail_one_index_restore(source: Path | str, destination: Path | str) -> None:
        nonlocal failed_restore
        source_path = Path(source)
        if (
            Path(destination) == root / "indexes"
            and source_path.parent.name.startswith(".prepare-publication-backup-")
            and not failed_restore
        ):
            failed_restore = True
            raise OSError("injected index restore failure")
        real_rename(source, destination)

    monkeypatch.setattr(
        preparation, "_write_json_atomic", interrupt_after_commit_marker
    )
    monkeypatch.setattr(preparation.os, "rename", fail_one_index_restore)

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        preparation._publish_derived_transaction(
            staging,
            root,
            include_index=True,
            include_profile=False,
            remove_unbuilt=False,
        )

    journal_path = root / ".prepare-publication.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert interrupted and failed_restore
    assert journal["phase"] == "publishing"
    assert (root / journal["backup_root"]).is_dir()
    monkeypatch.setattr(preparation, "_write_json_atomic", real_write_json)
    monkeypatch.setattr(preparation.os, "rename", real_rename)

    preparation._recover_derived_publication(root)

    for relative in ("data/processed", "indexes"):
        assert (root / relative / "value.txt").read_text(encoding="utf-8") == "old"
    assert (root / "dataset_manifest.json").read_text(encoding="utf-8") == "old"
    assert not journal_path.exists()


def test_committed_derived_cleanup_interrupt_never_rolls_back_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)
    preparation.prepare_dataset(config, data_root, until="corpus")
    old_dataset = (data_root / "dataset_manifest.json").read_bytes()
    real_rmtree = preparation.shutil.rmtree

    def interrupt_backup_cleanup(path: Path | str, *args: Any, **kwargs: Any) -> None:
        candidate = Path(path)
        if candidate.name.startswith(".prepare-publication-backup-"):
            raise KeyboardInterrupt("injected committed cleanup interrupt")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(preparation.shutil, "rmtree", interrupt_backup_cleanup)

    with pytest.raises(KeyboardInterrupt, match="committed cleanup"):
        preparation.prepare_dataset(
            config,
            data_root,
            refresh=True,
            until="corpus",
        )

    committed_dataset = (data_root / "dataset_manifest.json").read_bytes()
    assert committed_dataset != old_dataset
    assert (data_root / ".prepare-publication.json").is_file()
    monkeypatch.setattr(preparation.shutil, "rmtree", real_rmtree)

    recovered = preparation.prepare_dataset(config, data_root, until="corpus")

    assert recovered.reused_dataset
    assert (data_root / "dataset_manifest.json").read_bytes() == committed_dataset
    assert not (data_root / ".prepare-publication.json").exists()


def test_rebuild_marker_recovers_committed_publication_before_gating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)
    preparation.prepare_dataset(config, data_root, until="corpus")
    real_finalize = preparation._finalize_committed_publication

    monkeypatch.setattr(
        preparation,
        "_finalize_committed_publication",
        lambda root, payload: (_ for _ in ()).throw(
            KeyboardInterrupt("injected recovery publication cleanup interrupt")
        ),
    )
    with pytest.raises(KeyboardInterrupt, match="publication cleanup"):
        preparation.prepare_dataset(
            config,
            data_root,
            refresh=True,
            until="corpus",
        )
    monkeypatch.setattr(
        preparation,
        "_finalize_committed_publication",
        real_finalize,
    )

    publication = json.loads(
        (data_root / ".prepare-publication.json").read_text(encoding="utf-8")
    )
    recovery_root = data_root / ".prepare-publication-backup-recovery-fixture"
    recovery_root.mkdir()
    retained = json.loads(json.dumps(publication))
    retained["backup_root"] = recovery_root.name
    for index, entry in enumerate(retained["entries"]):
        entry["backup"] = f"{recovery_root.name}/{index}"
    (recovery_root / "recovery-journal.json").write_text(
        json.dumps(retained),
        encoding="utf-8",
    )
    (data_root / ".prepare-rebuild-recovery.json").write_text(
        json.dumps(
            {
                "schema_revision": "prepare-rebuild-recovery-v1",
                "complete": True,
                "backup_root": recovery_root.name,
                "until": "corpus",
            }
        ),
        encoding="utf-8",
    )

    recovered = preparation.prepare_dataset(config, data_root, until="corpus")

    assert recovered.reused_dataset
    assert not (data_root / ".prepare-publication.json").exists()
    assert not (data_root / ".prepare-rebuild-recovery.json").exists()
    assert not (data_root / ".prepare-staging").exists()
    assert (recovery_root / "recovery-journal.json").is_file()


@pytest.mark.parametrize(
    "mutation",
    ("source_config_sha256", "processing_config_sha256", "legacy_revision"),
)
def test_committed_candidate_manifest_identity_mismatch_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)
    preparation.prepare_dataset(config, data_root, until="corpus")
    old_dataset = (data_root / "dataset_manifest.json").read_bytes()
    real_finalize = preparation._finalize_committed_publication

    monkeypatch.setattr(
        preparation,
        "_finalize_committed_publication",
        lambda root, payload: (_ for _ in ()).throw(
            KeyboardInterrupt("injected post-source-commit interrupt")
        ),
    )
    with pytest.raises(KeyboardInterrupt, match="post-source-commit"):
        preparation.prepare_dataset(
            config,
            data_root,
            refresh=True,
            until="corpus",
        )
    monkeypatch.setattr(
        preparation,
        "_finalize_committed_publication",
        real_finalize,
    )
    manifest_path = data_root / "dataset_manifest.json"
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "legacy_revision":
        candidate["schema_revision"] = LEGACY_DATASET_MANIFEST_REVISION
        candidate.pop("source_config_sha256")
        candidate.pop("processing_config_sha256")
        candidate.pop("source_snapshot_sha256")
    else:
        candidate[mutation] = "0" * 64
    manifest_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(RuntimeError, match="partial preparation staging"):
        preparation.prepare_dataset(config, data_root, until="corpus")

    assert manifest_path.read_bytes() == old_dataset
    assert not (data_root / ".prepare-publication.json").exists()
    assert (data_root / ".prepare-staging").is_dir()


def test_explicit_rebuild_recovers_committed_source_with_corrupt_derived_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    data_root = tmp_path / "data-root"
    _write_site_config(config)
    loaders: list[_FixtureLoader] = []
    _install_fixture_factories(monkeypatch, loaders)
    preparation.prepare_dataset(config, data_root, until="corpus")
    real_finalize = preparation._finalize_committed_publication

    def interrupt_before_cleanup(root: Path, payload: dict[str, Any]) -> None:
        del root, payload
        raise KeyboardInterrupt("injected post-source-commit interrupt")

    monkeypatch.setattr(
        preparation,
        "_finalize_committed_publication",
        interrupt_before_cleanup,
    )
    with pytest.raises(KeyboardInterrupt, match="post-source-commit"):
        preparation.prepare_dataset(
            config,
            data_root,
            refresh=True,
            until="corpus",
        )
    monkeypatch.setattr(
        preparation,
        "_finalize_committed_publication",
        real_finalize,
    )
    journal = json.loads(
        (data_root / ".prepare-publication.json").read_text(encoding="utf-8")
    )
    dataset_entry = next(
        entry
        for entry in journal["entries"]
        if entry["destination"] == "dataset_manifest.json"
    )
    previous_dataset_path = data_root / dataset_entry["backup"]
    previous_dataset = json.loads(previous_dataset_path.read_text(encoding="utf-8"))
    previous_dataset["source_snapshot_sha256"] = "0" * 64
    previous_dataset_path.write_text(
        json.dumps(previous_dataset),
        encoding="utf-8",
    )
    (data_root / "data/processed/chunks.jsonl").write_text(
        '{"corrupt": true}\n',
        encoding="utf-8",
    )
    (data_root / "data/raw/fetch_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="recoverable derived backups were retained"):
        preparation.prepare_dataset(
            config,
            data_root,
            rebuild=True,
            until="corpus",
        )

    assert not (data_root / ".prepare-rebuild-recovery.json").exists()
    assert (data_root / ".prepare-publication.json").is_file()
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: _EmptyParser(),
    )

    with pytest.raises(RuntimeError, match="no searchable chunks"):
        preparation.prepare_dataset(
            config,
            data_root,
            refresh=True,
            until="corpus",
        )

    recovery_root = data_root / journal["backup_root"]
    assert (data_root / ".prepare-rebuild-recovery.json").is_file()
    assert (data_root / ".prepare-staging").is_dir()
    assert (recovery_root / "recovery-staging").is_dir()
    monkeypatch.setattr(
        preparation,
        "build_document_parser",
        lambda settings: _FixtureParser(),
    )

    real_recovery_finalize = preparation._finalize_rebuild_recovery
    monkeypatch.setattr(
        preparation,
        "_finalize_rebuild_recovery",
        lambda root: (_ for _ in ()).throw(
            KeyboardInterrupt("injected recovery marker cleanup interrupt")
        ),
    )
    with pytest.raises(KeyboardInterrupt, match="marker cleanup"):
        preparation.prepare_dataset(
            config,
            data_root,
            refresh=True,
            resume=True,
            until="corpus",
        )
    assert (data_root / ".prepare-rebuild-recovery.json").is_file()
    assert not (data_root / ".prepare-publication.json").exists()
    assert not (data_root / ".prepare-staging").exists()
    monkeypatch.setattr(
        preparation,
        "_finalize_rebuild_recovery",
        real_recovery_finalize,
    )

    recovered = preparation.prepare_dataset(config, data_root, until="corpus")

    assert recovered.reused_dataset
    assert not (data_root / ".prepare-publication.json").exists()
    assert not (data_root / ".prepare-rebuild-recovery.json").exists()
    assert (recovery_root / "recovery-journal.json").is_file()
    assert (recovery_root / "recovery-staging").is_dir()
    resolved = resolve_dataset_artifacts(data_root)
    assert resolved.dataset_manifest is not None
    assert resolved.dataset_manifest.chunk_count == recovered.chunk_count


def test_profile_resume_rebuilds_inconsistent_staged_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "site.toml"
    _write_site_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'prepare = "none"',
            'prepare = "recommended-v2"',
        ),
        encoding="utf-8",
    )
    config = preparation.load_site_config(config_path)
    staging = tmp_path / "data-root/.prepare-staging"
    target = staging / "profiles/recommended-v2"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("stale", encoding="utf-8")
    expected = SimpleNamespace(reused_existing=False)
    calls = 0

    def prepare_profile(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            assert target.is_dir()
            raise preparation.ArtifactPreparationError("staged identity mismatch")
        assert not target.exists()
        return expected

    monkeypatch.setattr(
        preparation,
        "prepare_dataset_recommended_v2_artifacts",
        prepare_profile,
    )

    result = preparation._prepare_profile(
        config,
        staging,
        until="profile",
        device="cpu",
    )

    assert result is expected
    assert calls == 2
