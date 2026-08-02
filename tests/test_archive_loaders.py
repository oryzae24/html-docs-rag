from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import pytest

import python_doc_rag.loaders.zip_archive as zip_archive_module
from python_doc_rag.dataset_layout import (
    generic_dataset_manifest,
    write_dataset_manifest_atomic,
)
from python_doc_rag.loaders.urls import build_source_url
from python_doc_rag.loaders.zip_archive import (
    ArchiveDownloadResult,
    PinnedLocalArchiveHtmlLoader,
    SnapshotHttpArchiveHtmlLoader,
)
from python_doc_rag.site_config import (
    PinnedLocalArchiveLoaderSettings,
    SnapshotHttpArchiveLoaderSettings,
)
from python_doc_rag.source_identity import source_config_payload


def _archive_bytes(
    pages: dict[str, str] | None = None,
    *,
    archive_root: str = "docs",
) -> bytes:
    selected = pages or {"tutorial/page.html": "<html>fixture one</html>"}
    output = io.BytesIO()
    directories = {f"{archive_root}/"}
    for logical_path in selected:
        parts = logical_path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add(f"{archive_root}/{'/'.join(parts[:index])}/")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(directories):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFDIR | 0o755) << 16
            archive.writestr(info, b"")
        for logical_path, content in sorted(selected.items()):
            info = zipfile.ZipInfo(f"{archive_root}/{logical_path}")
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, content.encode("utf-8"))
    return output.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pinned_settings(
    archive_name: str,
    archive_sha256: str,
    *,
    max_archive_bytes: int = 1_000_000,
) -> PinnedLocalArchiveLoaderSettings:
    return PinnedLocalArchiveLoaderSettings(
        type="pinned-local-archive",
        archive_path=archive_name,
        archive_sha256=archive_sha256,
        archive_format="zip",
        archive_root="docs",
        original_archive_url="https://example.test/archive.zip",
        source_base_url="https://example.test/reference/",
        include_path_prefixes=("tutorial/",),
        max_archive_bytes=max_archive_bytes,
        max_members=100,
        max_member_bytes=100_000,
        max_extracted_bytes=500_000,
    )


def _snapshot_settings(
    *,
    max_archive_bytes: int = 1_000_000,
    max_retries: int = 0,
) -> SnapshotHttpArchiveLoaderSettings:
    return SnapshotHttpArchiveLoaderSettings(
        type="snapshot-http-archive",
        archive_url="https://example.test/archive.zip",
        archive_format="zip",
        archive_root="docs",
        source_base_url="https://example.test/reference/",
        include_path_prefixes=("tutorial/",),
        update_policy="manual",
        timeout_seconds=10,
        max_retries=max_retries,
        max_archive_bytes=max_archive_bytes,
        max_members=100,
        max_member_bytes=100_000,
        max_extracted_bytes=500_000,
        user_agent="fixture/1.0",
    )


@dataclass(frozen=True)
class _Response:
    payload: bytes
    final_url: str = "https://cdn.example.test/archive.zip"
    reported_size: int | None = None


type Outcome = _Response | Exception


class _FakeTransport:
    def __init__(
        self,
        outcomes: list[Outcome] | None = None,
        *,
        partial_on_error: bytes = b"partial",
    ) -> None:
        self.outcomes = list(outcomes or [])
        self.partial_on_error = partial_on_error
        self.calls: list[dict[str, object]] = []

    def download(
        self,
        url: str,
        destination: BinaryIO,
        *,
        timeout_seconds: float,
        user_agent: str,
        max_archive_bytes: int,
    ) -> ArchiveDownloadResult:
        self.calls.append(
            {
                "url": url,
                "destination_name": str(destination.name),
                "timeout_seconds": timeout_seconds,
                "user_agent": user_agent,
                "max_archive_bytes": max_archive_bytes,
            }
        )
        if not self.outcomes:
            raise AssertionError("unexpected network call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            destination.write(self.partial_on_error)
            raise outcome
        destination.write(outcome.payload)
        return ArchiveDownloadResult(
            requested_url=url,
            final_url=outcome.final_url,
            byte_size=(
                len(outcome.payload)
                if outcome.reported_size is None
                else outcome.reported_size
            ),
        )


def _fixed_now() -> datetime:
    return datetime(2026, 8, 1, 12, 30, tzinfo=UTC)


def _all_values(value: object) -> list[object]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _all_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _all_values(child)]
    return [value]


def test_pinned_archive_loads_without_network_and_emits_portable_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _archive_bytes(
        {
            "tutorial/日本 語.html": "<html>日本語</html>",
            "tutorial/a.html": "<html>A</html>",
            "library/ignored.html": "<html>ignored</html>",
        }
    )
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    archive = config_directory / "snapshot.zip"
    archive.write_bytes(payload)

    def unexpected_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("pinned archive attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_network)
    raw_root = tmp_path / "data/raw"
    loader = PinnedLocalArchiveHtmlLoader(
        _pinned_settings("snapshot.zip", _sha256(payload)),
        raw_root,
        config_directory=config_directory,
        source_config_sha256="c" * 64,
        offline=True,
    )

    documents = loader.load()

    assert [item.logical_path for item in documents] == [
        "tutorial/a.html",
        "tutorial/日本 語.html",
    ]
    assert [item.source_url for item in documents] == [
        "https://example.test/reference/tutorial/a.html",
        "https://example.test/reference/tutorial/%E6%97%A5%E6%9C%AC%20%E8%AA%9E.html",
    ]
    assert all(item.canonical_url == item.source_url for item in documents)
    assert all(item.metadata == {} for item in documents)
    assert all("python_version" not in item.to_dict()["metadata"] for item in documents)
    assert str(tmp_path) not in json.dumps(
        [item.to_dict() for item in documents],
        ensure_ascii=False,
    )
    manifest = json.loads((raw_root / "fetch_manifest.json").read_text())
    assert manifest["expected_archive_sha256"] == _sha256(payload)
    assert manifest["observed_archive_sha256"] == _sha256(payload)
    assert manifest["source_base_url"] == "https://example.test/reference/"
    assert manifest["offline"] is True
    assert str(tmp_path) not in json.dumps(manifest)
    assert loader.summary is not None
    assert loader.summary.offline is True


def test_pinned_archive_reuses_valid_extraction_offline(tmp_path: Path) -> None:
    payload = _archive_bytes()
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    (config_directory / "snapshot.zip").write_bytes(payload)
    settings = _pinned_settings("snapshot.zip", _sha256(payload))
    raw_root = tmp_path / "raw"
    first = PinnedLocalArchiveHtmlLoader(
        settings,
        raw_root,
        config_directory=config_directory,
        source_config_sha256="c" * 64,
    )
    second = PinnedLocalArchiveHtmlLoader(
        settings,
        raw_root,
        config_directory=config_directory,
        source_config_sha256="c" * 64,
        offline=True,
    )

    first.load()
    second.load()

    assert second.summary is not None
    assert second.summary.cache_reused is True
    assert second.summary.offline is True


def test_pinned_archive_sha_mismatch_fails_before_extraction(tmp_path: Path) -> None:
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    (config_directory / "snapshot.zip").write_bytes(_archive_bytes())
    raw_root = tmp_path / "raw"
    loader = PinnedLocalArchiveHtmlLoader(
        _pinned_settings("snapshot.zip", "0" * 64),
        raw_root,
        config_directory=config_directory,
        source_config_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        loader.load()

    assert not (raw_root / "fetch_manifest.json").exists()
    assert not (raw_root / "extracted").exists()


def test_pinned_archive_rejects_refresh_with_rebuild_guidance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--rebuild"):
        PinnedLocalArchiveHtmlLoader(
            _pinned_settings("snapshot.zip", "0" * 64),
            tmp_path / "raw",
            config_directory=tmp_path,
            source_config_sha256="c" * 64,
            refresh=True,
        )


def test_snapshot_first_download_creates_complete_portable_source_lock(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    transport = _FakeTransport([_Response(payload)])
    raw_root = tmp_path / "raw"
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=transport,
        now=_fixed_now,
    )

    documents = loader.load()

    assert len(transport.calls) == 1
    assert [document.logical_path for document in documents] == ["tutorial/page.html"]
    lock = json.loads((raw_root / "source.lock.json").read_text())
    assert lock["complete"] is True
    assert lock["requested_url"] == "https://example.test/archive.zip"
    assert lock["final_url"] == "https://cdn.example.test/archive.zip"
    assert lock["observed_sha256"] == _sha256(payload)
    assert lock["byte_size"] == len(payload)
    assert lock["source_config_sha256"] == "d" * 64
    assert lock["source_snapshot_sha256"] == _sha256(payload)
    assert lock["cache_relative_path"] == f"archives/{_sha256(payload)}.zip"
    assert lock["fetched_at"] == "2026-08-01T12:30:00+00:00"
    assert not any(
        isinstance(value, str) and value.startswith("/") for value in _all_values(lock)
    )
    assert str(tmp_path) not in json.dumps(lock)
    assert (raw_root / lock["cache_relative_path"]).read_bytes() == payload
    assert not list(raw_root.glob(".archive-download-*.zip"))


def test_snapshot_second_load_and_offline_replay_make_zero_network_calls(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    initial_transport = _FakeTransport([_Response(payload)])
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=initial_transport,
        now=_fixed_now,
    ).load()
    lock_before = (raw_root / "source.lock.json").read_bytes()
    no_network = _FakeTransport()

    normal = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=no_network,
    )
    normal_documents = normal.load()
    offline = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        offline=True,
        transport=no_network,
    )
    offline_documents = offline.load()

    assert no_network.calls == []
    assert normal_documents == offline_documents
    assert (raw_root / "source.lock.json").read_bytes() == lock_before
    assert normal.summary is not None and normal.summary.cache_reused
    assert offline.summary is not None and offline.summary.offline


def test_snapshot_offline_without_lock_never_calls_network(tmp_path: Path) -> None:
    transport = _FakeTransport()
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        tmp_path / "raw",
        source_config_sha256="d" * 64,
        offline=True,
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="offline"):
        loader.load()

    assert transport.calls == []


def test_snapshot_offline_with_missing_locked_cache_never_calls_network(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload)]),
    ).load()
    (raw_root / f"archives/{_sha256(payload)}.zip").unlink()
    no_network = _FakeTransport()

    with pytest.raises(RuntimeError, match="missing or invalid"):
        SnapshotHttpArchiveHtmlLoader(
            _snapshot_settings(),
            raw_root,
            source_config_sha256="d" * 64,
            offline=True,
            transport=no_network,
        ).load()

    assert no_network.calls == []


def test_snapshot_refresh_repairs_corrupt_content_addressed_archive(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    settings = _snapshot_settings()
    first = SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload)]),
    )
    first.load()
    first.commit_source()
    archive_path = raw_root / f"archives/{_sha256(payload)}.zip"
    archive_path.write_bytes(b"corrupt")

    refreshed = SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=_FakeTransport([_Response(payload)]),
    )
    documents = refreshed.load()
    refreshed.commit_source()

    assert documents[0].content == "<html>fixture one</html>"
    assert _sha256(archive_path.read_bytes()) == _sha256(payload)
    no_network = _FakeTransport()
    replay = SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        offline=True,
        transport=no_network,
    ).load()
    assert replay[0].content == documents[0].content
    assert no_network.calls == []


def test_snapshot_download_retries_only_up_to_configured_limit(tmp_path: Path) -> None:
    payload = _archive_bytes()
    transport = _FakeTransport([RuntimeError("temporary failure"), _Response(payload)])
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(max_retries=1),
        tmp_path / "raw",
        source_config_sha256="d" * 64,
        transport=transport,
    )

    documents = loader.load()

    assert len(documents) == 1
    assert len(transport.calls) == 2


def test_snapshot_retry_keeps_download_inode_after_staging_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _archive_bytes()
    victim = tmp_path / "outside.txt"
    victim.write_bytes(b"do-not-overwrite")
    temporary: Path | None = None
    real_mkstemp = zip_archive_module.tempfile.mkstemp

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal temporary
        descriptor, name = real_mkstemp(*args, **kwargs)
        if kwargs.get("prefix") == ".archive-download-":
            temporary = Path(name)
        return descriptor, name

    class SwappingTransport:
        attempts = 0

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
            self.attempts += 1
            if self.attempts == 1:
                destination.write(b"partial")
                assert temporary is not None
                temporary.unlink()
                temporary.symlink_to(victim)
                raise RuntimeError("retry")
            destination.write(payload)
            return ArchiveDownloadResult(url, url, len(payload))

    monkeypatch.setattr(zip_archive_module.tempfile, "mkstemp", tracked_mkstemp)
    transport = SwappingTransport()
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(max_retries=1),
        tmp_path / "raw",
        source_config_sha256="d" * 64,
        transport=transport,
    )

    with pytest.raises(ValueError, match="staging path changed"):
        loader.load()

    assert transport.attempts == 2
    assert victim.read_bytes() == b"do-not-overwrite"


def test_snapshot_normal_load_does_not_refresh_but_explicit_refresh_does(
    tmp_path: Path,
) -> None:
    first_payload = _archive_bytes({"tutorial/page.html": "<html>version one</html>"})
    second_payload = _archive_bytes({"tutorial/page.html": "<html>version two</html>"})
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(first_payload)]),
        now=_fixed_now,
    ).load()
    no_refresh_transport = _FakeTransport([_Response(second_payload)])

    reused = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=no_refresh_transport,
    ).load()
    refresh_transport = _FakeTransport([_Response(second_payload)])
    refreshed_loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=refresh_transport,
        now=lambda: datetime(2026, 8, 2, tzinfo=UTC),
    )
    refreshed = refreshed_loader.load()
    refreshed_loader.commit_source()

    assert no_refresh_transport.calls == []
    assert reused[0].content == "<html>version one</html>"
    assert len(refresh_transport.calls) == 1
    assert refreshed[0].content == "<html>version two</html>"
    lock = json.loads((raw_root / "source.lock.json").read_text())
    assert lock["observed_sha256"] == _sha256(second_payload)


@pytest.mark.parametrize(
    "final_url",
    (
        "http://cdn.example.test/archive.zip",
        "https:///missing-host.zip",
        "https://user:password@cdn.example.test/archive.zip",
        "https://cdn.example.test/archive.zip?X-Amz-Signature=secret",
        "https://cdn.example.test/archive.zip#access-token",
    ),
)
def test_snapshot_rejects_non_absolute_or_untrusted_https_redirect(
    tmp_path: Path,
    final_url: str,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload, final_url=final_url)]),
    )

    with pytest.raises(ValueError, match="final URL|authentication"):
        loader.load()

    assert not (raw_root / "source.lock.json").exists()
    assert not (raw_root / "archives").exists()


def test_snapshot_rejects_actual_oversize_download_and_removes_partial(
    tmp_path: Path,
) -> None:
    payload = b"x" * 101
    raw_root = tmp_path / "raw"
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(max_archive_bytes=100),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload)]),
    )

    with pytest.raises(ValueError, match="max_archive_bytes"):
        loader.load()

    assert not (raw_root / "source.lock.json").exists()
    assert not (raw_root / "archives").exists()
    assert not list(raw_root.glob(".archive-download-*.zip"))


def test_snapshot_download_failure_does_not_publish_partial_cache(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    transport = _FakeTransport([RuntimeError("connection lost")])
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="connection lost"):
        loader.load()

    assert not (raw_root / "source.lock.json").exists()
    assert not (raw_root / "archives").exists()
    assert not list(raw_root.glob(".archive-download-*.zip"))


def test_snapshot_rejects_transport_size_mismatch(tmp_path: Path) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload, reported_size=len(payload) + 1)]),
    )

    with pytest.raises(ValueError, match="transport result"):
        loader.load()

    assert not (raw_root / "source.lock.json").exists()
    assert not (raw_root / "archives").exists()


def test_snapshot_invalid_zip_is_not_published_as_completed_cache(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(b"not a ZIP archive")]),
    )

    with pytest.raises(zipfile.BadZipFile):
        loader.load()

    assert not (raw_root / "source.lock.json").exists()
    assert not (raw_root / "archives").exists()


def test_snapshot_rejects_symlinked_archive_cache_directory(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    outside = tmp_path / "outside"
    raw_root.mkdir()
    outside.mkdir()
    (raw_root / "archives").symlink_to(outside, target_is_directory=True)
    transport = _FakeTransport([_Response(payload)])

    with pytest.raises(ValueError, match="symlink"):
        SnapshotHttpArchiveHtmlLoader(
            _snapshot_settings(),
            raw_root,
            source_config_sha256="d" * 64,
            transport=transport,
        ).load()

    assert transport.calls == []
    assert list(outside.iterdir()) == []
    assert not (raw_root / "source.lock.json").exists()


def test_snapshot_rejects_symlinked_content_addressed_archive_target(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    archives = raw_root / "archives"
    outside = tmp_path / "outside.zip"
    archives.mkdir(parents=True)
    outside.write_bytes(payload)
    target = archives / f"{_sha256(payload)}.zip"
    target.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        SnapshotHttpArchiveHtmlLoader(
            _snapshot_settings(),
            raw_root,
            source_config_sha256="d" * 64,
            transport=_FakeTransport([_Response(payload)]),
        ).load()

    assert outside.read_bytes() == payload
    assert target.is_symlink()
    assert not (raw_root / "source.lock.json").exists()
    assert not list(raw_root.glob(".archive-download-*.zip"))


def test_snapshot_refresh_failure_preserves_old_lock_manifest_and_cache(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload)]),
        now=_fixed_now,
    ).load()
    lock_before = (raw_root / "source.lock.json").read_bytes()
    manifest_before = (raw_root / "fetch_manifest.json").read_bytes()
    cache_path = raw_root / f"archives/{_sha256(payload)}.zip"
    cache_before = cache_path.read_bytes()
    failing = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=_FakeTransport([RuntimeError("refresh failed")]),
    )

    with pytest.raises(RuntimeError, match="refresh failed"):
        failing.load()

    assert (raw_root / "source.lock.json").read_bytes() == lock_before
    assert (raw_root / "fetch_manifest.json").read_bytes() == manifest_before
    assert cache_path.read_bytes() == cache_before


def test_snapshot_downstream_rollback_restores_old_lock_and_manifest(
    tmp_path: Path,
) -> None:
    first_payload = _archive_bytes({"tutorial/page.html": "<html>version one</html>"})
    second_payload = _archive_bytes({"tutorial/page.html": "<html>version two</html>"})
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(first_payload)]),
        now=_fixed_now,
    ).load()
    lock_before = (raw_root / "source.lock.json").read_bytes()
    manifest_before = (raw_root / "fetch_manifest.json").read_bytes()
    refreshed = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=_FakeTransport([_Response(second_payload)]),
        now=lambda: datetime(2026, 8, 2, tzinfo=UTC),
    )

    refreshed.load()
    assert (raw_root / "source.lock.json").read_bytes() != lock_before
    refreshed.rollback_source()

    assert (raw_root / "source.lock.json").read_bytes() == lock_before
    assert (raw_root / "fetch_manifest.json").read_bytes() == manifest_before
    assert (raw_root / f"archives/{_sha256(first_payload)}.zip").is_file()


def test_snapshot_refresh_replaces_corrupt_lock_and_can_restore_opaque_bytes(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    lock_path = raw_root / "source.lock.json"
    manifest_path = raw_root / "fetch_manifest.json"
    corrupt_lock = b"{not valid JSON\n"
    opaque_manifest = b"old opaque manifest\n"
    lock_path.write_bytes(corrupt_lock)
    manifest_path.write_bytes(opaque_manifest)
    loader = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=_FakeTransport([_Response(payload)]),
    )

    documents = loader.load()

    assert len(documents) == 1
    assert json.loads(lock_path.read_text())["observed_sha256"] == _sha256(payload)
    loader.rollback_source()
    assert lock_path.read_bytes() == corrupt_lock
    assert manifest_path.read_bytes() == opaque_manifest


def test_snapshot_next_load_recovers_uncommitted_refresh_journal(
    tmp_path: Path,
) -> None:
    first_payload = _archive_bytes({"tutorial/page.html": "<html>version one</html>"})
    second_payload = _archive_bytes({"tutorial/page.html": "<html>version two</html>"})
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(first_payload)]),
    ).load()
    old_lock = (raw_root / "source.lock.json").read_bytes()
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=_FakeTransport([_Response(second_payload)]),
    ).load()
    assert (raw_root / ".source-refresh-rollback").is_dir()
    no_network = _FakeTransport()

    recovered = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=no_network,
    ).load()

    assert recovered[0].content == "<html>version one</html>"
    assert (raw_root / "source.lock.json").read_bytes() == old_lock
    assert not (raw_root / ".source-refresh-rollback").exists()
    assert no_network.calls == []


def test_snapshot_dataset_manifest_forward_commits_refresh_journal(
    tmp_path: Path,
) -> None:
    first_payload = _archive_bytes({"tutorial/page.html": "<html>version one</html>"})
    second_payload = _archive_bytes({"tutorial/page.html": "<html>version two</html>"})
    data_root = tmp_path / "dataset"
    raw_root = data_root / "data/raw"
    settings = _snapshot_settings()
    SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(first_payload)]),
    ).load()
    SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=_FakeTransport([_Response(second_payload)]),
    ).load()
    write_dataset_manifest_atomic(
        generic_dataset_manifest(
            dataset_name="Fixture",
            dataset_slug="fixture",
            loader_type="snapshot-http-archive",
            parser_type="generic-html",
            site_config_sha256="a" * 64,
            created_at="2026-08-02T00:00:00+00:00",
            source_page_count=1,
            section_count=1,
            chunk_count=1,
            source_config_sha256="d" * 64,
            processing_config_sha256="e" * 64,
            source_snapshot_sha256=_sha256(second_payload),
        ),
        data_root / "dataset_manifest.json",
    )
    source_path = raw_root / "fetch_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update(
        {
            "parser_type": "generic-html",
            "parser_settings": {"type": "generic-html"},
            "site_config_file_sha256": "a" * 64,
            "processing_config_fingerprint_revision": "processing-config-v1",
            "processing_config_sha256": "e" * 64,
            "configured_source": source_config_payload(
                settings,
                category="fixture",
            )["source"],
        }
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")
    no_network = _FakeTransport()

    recovered = SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        transport=no_network,
    ).load()

    assert recovered[0].content == "<html>version two</html>"
    assert not (raw_root / ".source-refresh-rollback").exists()
    assert no_network.calls == []


def test_snapshot_rollback_attempts_all_records_and_retains_failed_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_payload = _archive_bytes({"tutorial/page.html": "<html>version one</html>"})
    second_payload = _archive_bytes({"tutorial/page.html": "<html>version two</html>"})
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(first_payload)]),
    ).load()
    lock_path = raw_root / "source.lock.json"
    manifest_path = raw_root / "fetch_manifest.json"
    old_lock = lock_path.read_bytes()
    old_manifest = manifest_path.read_bytes()
    refreshed = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=_FakeTransport([_Response(second_payload)]),
    )
    refreshed.load()
    original_write = zip_archive_module._write_bytes_atomic

    def fail_lock_restore(payload: bytes, path: Path) -> None:
        if path == lock_path and payload == old_lock:
            raise OSError("injected lock restore failure")
        original_write(payload, path)

    monkeypatch.setattr(zip_archive_module, "_write_bytes_atomic", fail_lock_restore)
    with pytest.raises(RuntimeError, match="1 publication record"):
        refreshed.rollback_source()

    assert manifest_path.read_bytes() == old_manifest
    assert lock_path.read_bytes() != old_lock
    assert (raw_root / ".source-refresh-rollback").is_dir()
    monkeypatch.setattr(zip_archive_module, "_write_bytes_atomic", original_write)
    refreshed.rollback_source()
    assert lock_path.read_bytes() == old_lock
    assert manifest_path.read_bytes() == old_manifest
    assert not (raw_root / ".source-refresh-rollback").exists()


def test_snapshot_recovery_does_not_retire_backup_for_missing_candidate_archive(
    tmp_path: Path,
) -> None:
    first_payload = _archive_bytes({"tutorial/page.html": "<html>version one</html>"})
    second_payload = _archive_bytes({"tutorial/page.html": "<html>version two</html>"})
    data_root = tmp_path / "data-root"
    raw_root = data_root / "data/raw"
    settings = _snapshot_settings()
    SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(first_payload)]),
    ).load()
    SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        refresh=True,
        transport=_FakeTransport([_Response(second_payload)]),
    ).load()
    write_dataset_manifest_atomic(
        generic_dataset_manifest(
            dataset_name="Fixture",
            dataset_slug="fixture",
            loader_type="snapshot-http-archive",
            parser_type="generic-html",
            site_config_sha256="a" * 64,
            created_at="2026-08-02T00:00:00+00:00",
            source_page_count=1,
            section_count=1,
            chunk_count=1,
            source_config_sha256="d" * 64,
            processing_config_sha256="e" * 64,
            source_snapshot_sha256=_sha256(second_payload),
        ),
        data_root / "dataset_manifest.json",
    )
    lock = json.loads((raw_root / "source.lock.json").read_text(encoding="utf-8"))
    (raw_root / lock["cache_relative_path"]).unlink()

    with pytest.raises(RuntimeError, match="source lock cache"):
        zip_archive_module.recover_snapshot_http_archive_refresh(
            raw_root,
            settings=settings,
            source_config_sha256="d" * 64,
        )

    journal = raw_root / ".source-refresh-rollback"
    assert journal.is_dir()
    assert (journal / "source.lock.json.previous").is_file()
    zip_archive_module.recover_snapshot_http_archive_refresh(
        raw_root,
        settings=settings,
        source_config_sha256="d" * 64,
        force_rollback=True,
    )
    no_network = _FakeTransport()
    recovered = SnapshotHttpArchiveHtmlLoader(
        settings,
        raw_root,
        source_config_sha256="d" * 64,
        transport=no_network,
    ).load()
    assert recovered[0].content == "<html>version one</html>"
    assert no_network.calls == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_snapshot_rollback_journal_rejects_special_files_without_reading(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    journal = raw_root / ".source-refresh-rollback"
    journal.mkdir(parents=True)
    os.mkfifo(journal / "rollback.json")

    with pytest.raises(RuntimeError, match="rollback journal is invalid"):
        zip_archive_module.recover_snapshot_http_archive_refresh(raw_root)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_snapshot_rollback_journal_rejects_special_backup_without_reading(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    journal = raw_root / ".source-refresh-rollback"
    journal.mkdir(parents=True)
    (journal / "rollback.json").write_text(
        json.dumps(
            {
                "schema_revision": "source-refresh-rollback-v1",
                "complete": True,
                "candidate_source_snapshot_sha256": "a" * 64,
                "source_config_sha256": "b" * 64,
                "records": [
                    {
                        "target": "source.lock.json",
                        "existed": True,
                        "backup": "source.lock.json.previous",
                        "sha256": "c" * 64,
                    },
                    {
                        "target": "fetch_manifest.json",
                        "existed": False,
                        "backup": None,
                        "sha256": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    os.mkfifo(journal / "source.lock.json.previous")

    with pytest.raises(RuntimeError, match="special entry"):
        zip_archive_module.recover_snapshot_http_archive_refresh(raw_root)


def test_snapshot_source_config_mismatch_fails_without_network(tmp_path: Path) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload)]),
    ).load()
    no_network = _FakeTransport()
    mismatched = SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="e" * 64,
        transport=no_network,
    )

    with pytest.raises(RuntimeError, match="source config does not match"):
        mismatched.load()

    assert no_network.calls == []


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/tmp/archive.zip",
        "../archive.zip",
        "archives\\archive.zip",
        "C:\\archive.zip",
    ),
)
def test_snapshot_lock_rejects_unsafe_cache_path(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload)]),
    ).load()
    lock_path = raw_root / "source.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["cache_relative_path"] = unsafe_path
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    no_network = _FakeTransport()

    with pytest.raises(ValueError, match="unsafe source lock cache path"):
        SnapshotHttpArchiveHtmlLoader(
            _snapshot_settings(),
            raw_root,
            source_config_sha256="d" * 64,
            transport=no_network,
        ).load()

    assert no_network.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("requested_url", "http://example.test/archive.zip"),
        ("final_url", "https:///missing-host.zip"),
        ("archive_format", "tar"),
        ("archive_root", "other-docs"),
        ("source_base_url", "https://other.example/docs/"),
        ("include_path_prefixes", ["other/"]),
        ("byte_size", True),
        ("fetched_at", 123),
    ),
)
def test_snapshot_lock_rejects_tampered_identity_fields_without_network(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _archive_bytes()
    raw_root = tmp_path / "raw"
    SnapshotHttpArchiveHtmlLoader(
        _snapshot_settings(),
        raw_root,
        source_config_sha256="d" * 64,
        transport=_FakeTransport([_Response(payload)]),
    ).load()
    lock_path = raw_root / "source.lock.json"
    lock = json.loads(lock_path.read_text())
    lock[field] = value
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    no_network = _FakeTransport()

    with pytest.raises(RuntimeError, match="source lock"):
        SnapshotHttpArchiveHtmlLoader(
            _snapshot_settings(),
            raw_root,
            source_config_sha256="d" * 64,
            transport=no_network,
        ).load()

    assert no_network.calls == []


def test_source_url_builder_quotes_path_and_rejects_origin_escape() -> None:
    assert build_source_url(
        "https://example.test/docs/",
        "tutorial/日本 語.html",
    ) == ("https://example.test/docs/tutorial/%E6%97%A5%E6%9C%AC%20%E8%AA%9E.html")
    with pytest.raises(ValueError, match="safe relative"):
        build_source_url("https://example.test/docs/", "../escape.html")
    with pytest.raises(ValueError, match="directory URL"):
        build_source_url(
            "https://example.test/docs/?next=https://evil.test", "page.html"
        )
