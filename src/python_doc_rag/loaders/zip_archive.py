"""Pinned and source-locked ZIP loaders with fail-closed extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import unicodedata
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol
from urllib.parse import urlsplit

from python_doc_rag.dataset_layout import (
    DATASET_MANIFEST_REVISION,
    DatasetManifest,
)
from python_doc_rag.loaders.urls import build_source_url
from python_doc_rag.models import SourceDocument
from python_doc_rag.site_config import (
    PinnedLocalArchiveLoaderSettings,
    SnapshotHttpArchiveLoaderSettings,
)
from python_doc_rag.source_identity import source_config_payload, source_snapshot_sha256

ARCHIVE_SOURCE_MANIFEST_REVISION = "zip-html-source-v1"
SOURCE_LOCK_REVISION = "source-lock-v1"
EXTRACTION_MANIFEST_REVISION = "safe-zip-extraction-v1"
SOURCE_ROLLBACK_REVISION = "source-refresh-rollback-v1"
_MAX_SOURCE_ROLLBACK_JSON_BYTES = 64 * 1024
_MAX_SOURCE_ROLLBACK_BACKUP_BYTES = 64 * 1024 * 1024
_MAX_EXTRACTION_MANIFEST_BYTES = 64 * 1024 * 1024
_COMMITTED_ARCHIVE_SOURCE_KEYS = frozenset(
    {
        "schema_revision",
        "complete",
        "loader_type",
        "requested_archive_url",
        "final_archive_url",
        "expected_archive_sha256",
        "observed_archive_sha256",
        "archive_sha256_origin",
        "archive_byte_size",
        "archive_format",
        "archive_root",
        "source_base_url",
        "include_path_prefixes",
        "source_page_count",
        "source_config_fingerprint_revision",
        "source_config_sha256",
        "source_snapshot_sha256",
        "document_snapshot_sha256",
        "acquired_at",
        "cache_reused",
        "offline",
        "archive_cache_path",
        "pages",
        "parser_type",
        "parser_settings",
        "site_config_file_sha256",
        "processing_config_fingerprint_revision",
        "processing_config_sha256",
        "configured_source",
    }
)
_ARCHIVE_PAGE_KEYS = frozenset(
    {"source_url", "canonical_url", "logical_path", "category", "content_sha256"}
)
_EXTRACTION_MANIFEST_KEYS = frozenset(
    {
        "schema_revision",
        "complete",
        "archive_sha256",
        "archive_byte_size",
        "archive_format",
        "archive_root",
        "member_count",
        "extracted_byte_size",
        "members",
    }
)
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
        *(f"com{index}" for index in ("¹", "²", "³")),
        *(f"lpt{index}" for index in ("¹", "²", "³")),
    }
)


@dataclass(frozen=True, slots=True)
class ArchiveDownloadResult:
    """Portable result of streaming one HTTP archive into a temporary file."""

    requested_url: str
    final_url: str
    byte_size: int


class ArchiveTransport(Protocol):
    """Injectable bounded streaming transport used by archive source tests."""

    def download(
        self,
        url: str,
        destination: BinaryIO,
        *,
        timeout_seconds: float,
        user_agent: str,
        max_archive_bytes: int,
    ) -> ArchiveDownloadResult:
        """Write a complete candidate archive or raise without publishing it."""
        ...


class UrllibArchiveTransport:
    """Download a public archive while enforcing the actual streamed byte count."""

    def download(
        self,
        url: str,
        destination: BinaryIO,
        *,
        timeout_seconds: float,
        user_agent: str,
        max_archive_bytes: int,
    ) -> ArchiveDownloadResult:
        """Stream an archive into an already-open private staging file."""
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/zip,application/octet-stream;q=0.9,*/*;q=0.1",
            },
        )
        byte_size = 0
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            final_url = response.geturl()
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                byte_size += len(block)
                if byte_size > max_archive_bytes:
                    raise ValueError("archive download exceeded max_archive_bytes")
                destination.write(block)
        return ArchiveDownloadResult(url, final_url, byte_size)


@dataclass(frozen=True, slots=True)
class ArchiveLoadSummary:
    """Diagnostics exposed after a pinned or mutable archive load."""

    loader_type: str
    archive_sha256: str
    archive_byte_size: int
    source_page_count: int
    cache_reused: bool
    offline: bool
    requested_url: str
    final_url: str | None


@dataclass(frozen=True, slots=True)
class _ArchiveStreamIdentity:
    device: int
    inode: int
    byte_size: int
    modified_ns: int
    changed_ns: int


class SafeZipHtmlLoader:
    """Validate, extract, and enumerate one already-identified ZIP archive."""

    def __init__(
        self,
        archive_path: Path,
        extraction_root: Path,
        *,
        archive_sha256: str,
        archive_root: str,
        source_base_url: str,
        include_path_prefixes: Sequence[str],
        source_kind: str,
        max_archive_bytes: int,
        max_members: int,
        max_member_bytes: int,
        max_extracted_bytes: int,
        fetched_at: str | None = None,
    ) -> None:
        self._archive_path = archive_path
        self._extraction_root = extraction_root
        self._archive_sha256 = archive_sha256
        self._archive_root = archive_root
        self._source_base_url = source_base_url
        self._prefixes = tuple(include_path_prefixes)
        self._source_kind = source_kind
        self._max_archive_bytes = max_archive_bytes
        self._max_members = max_members
        self._max_member_bytes = max_member_bytes
        self._max_extracted_bytes = max_extracted_bytes
        self._fetched_at = fetched_at
        self.cache_reused = False
        self.member_count = 0
        self.extracted_byte_size = 0
        self.archive_byte_size = 0
        self._member_digests: dict[str, str] = {}

    def load(self) -> tuple[SourceDocument, ...]:
        """Validate bytes/cache and emit selected HTML in deterministic order."""
        _recover_extraction_publication(self._extraction_root)
        _reject_unsafe_existing_extraction(self._extraction_root)
        with _verified_archive_stream(
            self._archive_path,
            max_bytes=self._max_archive_bytes,
            expected_sha256=self._archive_sha256,
        ) as (archive_stream, archive_size, archive_identity):
            self.archive_byte_size = archive_size
            reused_extraction = self._reuse_valid_extraction(
                archive_stream,
                archive_size=archive_size,
                archive_identity=archive_identity,
            )
            if reused_extraction:
                backup = self._extraction_root.with_name(
                    f".{self._extraction_root.name}.backup"
                )
                if backup.exists() or backup.is_symlink():
                    _retire_directory(backup)
            else:
                self._extract_safely(
                    archive_stream,
                    archive_size=archive_size,
                    archive_identity=archive_identity,
                )
            document_root = self._extraction_root.joinpath(
                *PurePosixPath(self._archive_root).parts
            )
            if not document_root.is_dir() or _is_link_like(document_root):
                raise ValueError(
                    f"configured archive_root is absent: {self._archive_root}"
                )
            files = _selected_html_files(document_root, self._prefixes)
            documents: list[SourceDocument] = []
            for path in files:
                logical_path = path.relative_to(document_root).as_posix()
                member_path = path.relative_to(self._extraction_root).as_posix()
                expected_digest = self._member_digests.get(member_path)
                if expected_digest is None:
                    raise ValueError("selected HTML is absent from extraction identity")
                encoded = _read_bounded_regular_file(
                    path,
                    max_bytes=self._max_member_bytes,
                    label="extracted HTML member",
                )
                observed_digest = hashlib.sha256(encoded).hexdigest()
                if observed_digest != expected_digest:
                    raise ValueError("extracted HTML member SHA-256 is inconsistent")
                content = encoded.decode("utf-8")
                source_url = build_source_url(self._source_base_url, logical_path)
                documents.append(
                    SourceDocument(
                        source_url=source_url,
                        canonical_url=source_url,
                        content=content,
                        content_sha256=observed_digest,
                        source_kind=self._source_kind,
                        logical_path=logical_path,
                        category=PurePosixPath(logical_path).parts[0],
                        fetched_at=self._fetched_at,
                    )
                )
            return tuple(documents)

    def _reuse_valid_extraction(
        self,
        archive_stream: BinaryIO,
        *,
        archive_size: int,
        archive_identity: _ArchiveStreamIdentity,
    ) -> bool:
        manifest_path = self._extraction_root / "extraction_manifest.json"
        if not manifest_path.is_file() or _is_link_like(self._extraction_root):
            return False
        try:
            manifest = _read_bounded_json_object(
                manifest_path,
                max_bytes=_MAX_EXTRACTION_MANIFEST_BYTES,
                label="extraction manifest",
            )
            if (
                set(manifest) != _EXTRACTION_MANIFEST_KEYS
                or manifest.get("schema_revision") != EXTRACTION_MANIFEST_REVISION
                or manifest.get("complete") is not True
                or manifest.get("archive_sha256") != self._archive_sha256
                or manifest.get("archive_byte_size") != archive_size
                or manifest.get("archive_format") != "zip"
                or manifest.get("archive_root") != self._archive_root
            ):
                return False
            records = manifest.get("members")
            if (
                not isinstance(records, list)
                or len(records) > self._max_members
                or manifest.get("member_count") != len(records)
            ):
                return False
            archive_stream.seek(0)
            _preflight_zip_member_count(archive_stream, self._max_members)
            archive_stream.seek(0)
            with zipfile.ZipFile(archive_stream) as archive:
                archive_members = _validated_members(
                    archive,
                    max_members=self._max_members,
                    max_member_bytes=self._max_member_bytes,
                    max_extracted_bytes=self._max_extracted_bytes,
                )
                expected_members = {
                    relative.as_posix(): (
                        info,
                        "directory" if is_directory else "file",
                        info.file_size,
                    )
                    for info, relative, is_directory in archive_members
                }
                total = 0
                member_digests: dict[str, str] = {}
                root = self._extraction_root.resolve(strict=True)
                recorded_paths: set[str] = set()
                for record in records:
                    if not isinstance(record, dict):
                        return False
                    relative = _safe_member_name(str(record.get("path", "")))
                    relative_name = relative.as_posix()
                    if relative_name in recorded_paths:
                        return False
                    recorded_paths.add(relative_name)
                    candidate = self._extraction_root.joinpath(*relative.parts)
                    if _is_link_like(candidate):
                        return False
                    resolved = candidate.resolve(strict=True)
                    if not resolved.is_relative_to(root):
                        return False
                    kind = record.get("kind")
                    expected = expected_members.get(relative_name)
                    if expected is None or kind != expected[1]:
                        return False
                    if kind == "directory":
                        if set(record) != {"path", "kind"} or not resolved.is_dir():
                            return False
                        continue
                    if (
                        kind != "file"
                        or set(record) != {"path", "kind", "size", "sha256"}
                        or not resolved.is_file()
                    ):
                        return False
                    size = resolved.stat().st_size
                    if (
                        size != expected[2]
                        or size != record.get("size")
                        or size > self._max_member_bytes
                    ):
                        return False
                    archive_digest, member_archive_size = _zip_member_sha256(
                        archive,
                        expected[0],
                        max_bytes=self._max_member_bytes,
                    )
                    cached_digest, cached_size = _bounded_file_sha256(
                        resolved,
                        self._max_member_bytes,
                    )
                    if (
                        member_archive_size != size
                        or cached_size != size
                        or archive_digest != cached_digest
                        or archive_digest != record.get("sha256")
                    ):
                        return False
                    member_digests[relative_name] = archive_digest
                    total += size
                    if total > self._max_extracted_bytes:
                        return False
            if recorded_paths != set(expected_members):
                return False
            if manifest.get("extracted_byte_size") != total:
                return False
            expected_tree_paths = _tree_paths_for_archive_members(recorded_paths)
            expected_tree_paths.add("extraction_manifest.json")
            if _existing_tree_paths(self._extraction_root) != expected_tree_paths:
                return False
            _verify_open_archive(
                archive_stream,
                identity=archive_identity,
                expected_sha256=self._archive_sha256,
                max_bytes=self._max_archive_bytes,
            )
            self.member_count = len(records)
            self.extracted_byte_size = total
            self._member_digests = member_digests
            self.cache_reused = True
            return True
        except (
            KeyError,
            OSError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
        ):
            return False

    def _extract_safely(
        self,
        archive_stream: BinaryIO,
        *,
        archive_size: int,
        archive_identity: _ArchiveStreamIdentity,
    ) -> None:
        destination_parent = self._extraction_root.parent
        _prepare_real_directory(destination_parent)
        staging = Path(
            tempfile.mkdtemp(
                dir=destination_parent,
                prefix=f".{self._extraction_root.name}.staging-",
            )
        )
        try:
            archive_stream.seek(0)
            _preflight_zip_member_count(archive_stream, self._max_members)
            archive_stream.seek(0)
            with zipfile.ZipFile(archive_stream) as archive:
                members = _validated_members(
                    archive,
                    max_members=self._max_members,
                    max_member_bytes=self._max_member_bytes,
                    max_extracted_bytes=self._max_extracted_bytes,
                )
                records: list[dict[str, Any]] = []
                extracted_total = 0
                for info, relative, is_directory in members:
                    target = staging.joinpath(*relative.parts)
                    _ensure_no_symlink_ancestors(staging, target.parent)
                    if is_directory:
                        target.mkdir(parents=True, exist_ok=True)
                        records.append(
                            {"path": relative.as_posix(), "kind": "directory"}
                        )
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    written = 0
                    with archive.open(info, "r") as source, target.open("xb") as output:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            written += len(block)
                            extracted_total += len(block)
                            if written > self._max_member_bytes:
                                raise ValueError("ZIP member exceeded max_member_bytes")
                            if extracted_total > self._max_extracted_bytes:
                                raise ValueError(
                                    "ZIP content exceeded max_extracted_bytes"
                                )
                            output.write(block)
                            digest.update(block)
                    if written != info.file_size:
                        raise ValueError("ZIP member size changed during extraction")
                    records.append(
                        {
                            "path": relative.as_posix(),
                            "kind": "file",
                            "size": written,
                            "sha256": digest.hexdigest(),
                        }
                    )
            _verify_open_archive(
                archive_stream,
                identity=archive_identity,
                expected_sha256=self._archive_sha256,
                max_bytes=self._max_archive_bytes,
            )
            archive_root_path = staging.joinpath(
                *PurePosixPath(self._archive_root).parts
            )
            if not archive_root_path.is_dir() or _is_link_like(archive_root_path):
                raise ValueError(
                    f"configured archive_root is absent: {self._archive_root}"
                )
            payload = {
                "schema_revision": EXTRACTION_MANIFEST_REVISION,
                "complete": True,
                "archive_sha256": self._archive_sha256,
                "archive_byte_size": archive_size,
                "archive_format": "zip",
                "archive_root": self._archive_root,
                "member_count": len(records),
                "extracted_byte_size": extracted_total,
                "members": records,
            }
            _write_json_atomic(payload, staging / "extraction_manifest.json")
            _publish_directory(staging, self._extraction_root)
            self.member_count = len(records)
            self.extracted_byte_size = extracted_total
            self._member_digests = {
                str(record["path"]): str(record["sha256"])
                for record in records
                if record["kind"] == "file"
            }
            self.cache_reused = False
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


class PinnedLocalArchiveHtmlLoader:
    """Load a config-relative, SHA-pinned ZIP without any network access."""

    def __init__(
        self,
        settings: PinnedLocalArchiveLoaderSettings,
        raw_root: Path,
        *,
        config_directory: Path,
        source_config_sha256: str,
        offline: bool = False,
        refresh: bool = False,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if refresh:
            raise ValueError(
                "--refresh is not supported for pinned-local-archive; use --rebuild"
            )
        self._settings = settings
        self._raw_root = raw_root
        self._archive_path = (config_directory / settings.archive_path).resolve()
        self._source_config_sha256 = source_config_sha256
        self._offline = offline
        self._now = now or (lambda: datetime.now(UTC))
        self.summary: ArchiveLoadSummary | None = None

    @property
    def source_snapshot_sha256(self) -> str | None:
        """Return the verified archive content identity after loading."""
        return None if self.summary is None else self.summary.archive_sha256

    def load(self) -> tuple[SourceDocument, ...]:
        """Verify the repository snapshot and publish portable source metadata."""
        if not self._archive_path.is_file():
            raise FileNotFoundError(
                f"pinned local archive is missing: {self._settings.archive_path}"
            )
        safe_loader = SafeZipHtmlLoader(
            self._archive_path,
            self._raw_root / "extracted" / self._settings.archive_sha256,
            archive_sha256=self._settings.archive_sha256,
            archive_root=self._settings.archive_root,
            source_base_url=self._settings.source_base_url,
            include_path_prefixes=self._settings.include_path_prefixes,
            source_kind="pinned-local-archive",
            max_archive_bytes=self._settings.max_archive_bytes,
            max_members=self._settings.max_members,
            max_member_bytes=self._settings.max_member_bytes,
            max_extracted_bytes=self._settings.max_extracted_bytes,
        )
        documents = safe_loader.load()
        observed_sha = self._settings.archive_sha256
        archive_size = safe_loader.archive_byte_size
        manifest = _archive_source_manifest(
            loader_type=self._settings.type,
            requested_url=self._settings.original_archive_url,
            final_url=None,
            expected_sha256=self._settings.archive_sha256,
            observed_sha256=observed_sha,
            sha256_origin="project-pinned-local-archive",
            archive_byte_size=archive_size,
            archive_root=self._settings.archive_root,
            source_base_url=self._settings.source_base_url,
            include_path_prefixes=self._settings.include_path_prefixes,
            source_config_sha256=self._source_config_sha256,
            documents=documents,
            acquired_at=None,
            cache_reused=safe_loader.cache_reused,
            offline=True,
            archive_cache_path=None,
        )
        _write_json_atomic(manifest, self._raw_root / "fetch_manifest.json")
        self.summary = ArchiveLoadSummary(
            loader_type=self._settings.type,
            archive_sha256=observed_sha,
            archive_byte_size=archive_size,
            source_page_count=len(documents),
            cache_reused=safe_loader.cache_reused,
            offline=self._offline,
            requested_url=self._settings.original_archive_url,
            final_url=None,
        )
        return documents

    def rollback_source(self) -> None:
        """Pinned source bytes never change, so rollback is intentionally empty."""

    def commit_source(self) -> None:
        """Pinned source bytes never change, so commit is intentionally empty."""


class SnapshotHttpArchiveHtmlLoader:
    """Download once, lock observed bytes, and refresh only when requested."""

    def __init__(
        self,
        settings: SnapshotHttpArchiveLoaderSettings,
        raw_root: Path,
        *,
        source_config_sha256: str,
        offline: bool = False,
        refresh: bool = False,
        transport: ArchiveTransport | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._raw_root = raw_root
        self._source_config_sha256 = source_config_sha256
        self._offline = offline
        self._refresh = refresh
        self._transport = transport or UrllibArchiveTransport()
        self._now = now or (lambda: datetime.now(UTC))
        self._rollback_records: list[tuple[Path, bytes | None]] = []
        self.summary: ArchiveLoadSummary | None = None

    @property
    def source_snapshot_sha256(self) -> str | None:
        """Return the source-locked archive content identity after loading."""
        return None if self.summary is None else self.summary.archive_sha256

    def load(self) -> tuple[SourceDocument, ...]:
        """Reuse a valid lock/cache or acquire one candidate archive."""
        if self._refresh:
            if self._offline:
                raise RuntimeError("--refresh cannot be combined with --offline")
        self._recover_pending_source_publication()
        if self._refresh:
            return self._download_and_lock()
        lock_path = self._raw_root / "source.lock.json"
        existing = _load_optional_json(lock_path)
        if existing is not None:
            return self._load_locked(existing)
        if self._offline:
            raise RuntimeError(
                "offline replay requires a complete valid source lock/cache"
            )
        return self._download_and_lock()

    def _load_locked(self, lock: dict[str, Any]) -> tuple[SourceDocument, ...]:
        validated_lock = validate_snapshot_http_archive_cache(
            settings=self._settings,
            raw_root=self._raw_root,
            source_config_sha256=self._source_config_sha256,
        )
        if validated_lock != lock:
            raise RuntimeError("source lock changed during validation")
        relative = _safe_cache_relative(str(validated_lock["cache_relative_path"]))
        archive_path = self._raw_root / relative
        observed = str(validated_lock["observed_sha256"])
        size = int(validated_lock["byte_size"])
        documents, safe_loader = self._documents_from_archive(
            archive_path,
            observed,
            fetched_at=str(validated_lock["fetched_at"]),
        )
        self.summary = ArchiveLoadSummary(
            loader_type=self._settings.type,
            archive_sha256=observed,
            archive_byte_size=size,
            source_page_count=len(documents),
            cache_reused=True,
            offline=self._offline,
            requested_url=str(validated_lock["requested_url"]),
            final_url=str(validated_lock["final_url"]),
        )
        manifest = _archive_source_manifest(
            loader_type=self._settings.type,
            requested_url=str(validated_lock["requested_url"]),
            final_url=str(validated_lock["final_url"]),
            expected_sha256=None,
            observed_sha256=observed,
            sha256_origin="observed-on-first-acquisition",
            archive_byte_size=size,
            archive_root=self._settings.archive_root,
            source_base_url=self._settings.source_base_url,
            include_path_prefixes=self._settings.include_path_prefixes,
            source_config_sha256=self._source_config_sha256,
            documents=documents,
            acquired_at=str(validated_lock["fetched_at"]),
            cache_reused=True,
            offline=self._offline,
            archive_cache_path=str(validated_lock["cache_relative_path"]),
        )
        _write_json_atomic(manifest, self._raw_root / "fetch_manifest.json")
        safe_loader.cache_reused = True
        return documents

    def _download_and_lock(self) -> tuple[SourceDocument, ...]:
        _prepare_real_directory(self._raw_root)
        archive_directory = self._raw_root / "archives"
        _reject_existing_symlink_components(archive_directory)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._raw_root,
            prefix=".archive-download-",
            suffix=".zip",
        )
        temporary = Path(temporary_name)
        archive_path: Path | None = None
        corrupt_archive: Path | None = None
        try:
            with os.fdopen(descriptor, "w+b") as temporary_stream:
                result = self._download_candidate(temporary_stream)
                temporary_stream.flush()
                os.fsync(temporary_stream.fileno())
                observed, size = _bounded_open_file_sha256(
                    temporary_stream,
                    self._settings.max_archive_bytes,
                )
                _require_open_file_path_identity(temporary_stream, temporary)
            if result.requested_url != self._settings.archive_url:
                raise ValueError("archive transport requested URL is inconsistent")
            _validate_final_https_url(result.final_url)
            if size != result.byte_size:
                raise ValueError(
                    "downloaded archive size did not match transport result"
                )
            archive_relative = f"archives/{observed}.zip"
            archive_path = self._raw_root / archive_relative
            _reject_existing_symlink_components(archive_path)
            if archive_path.exists():
                cached_sha, cached_size = _bounded_file_sha256(
                    archive_path,
                    self._settings.max_archive_bytes,
                )
                if cached_sha != observed or cached_size != size:
                    corrupt_archive = archive_path.with_name(
                        f".{archive_path.name}.corrupt-{uuid.uuid4().hex}"
                    )
                    os.rename(archive_path, corrupt_archive)
                    candidate_archive = temporary
                else:
                    temporary.unlink()
                    candidate_archive = archive_path
            else:
                candidate_archive = temporary
            fetched_at = self._now().isoformat()
            documents, safe_loader = self._documents_from_archive(
                candidate_archive,
                observed,
                fetched_at=fetched_at,
            )
            if candidate_archive == temporary:
                _prepare_real_directory(archive_path.parent)
                _reject_existing_symlink_components(archive_path)
                try:
                    _publish_verified_archive(
                        temporary,
                        archive_path,
                        expected_sha256=observed,
                        expected_size=size,
                        max_archive_bytes=self._settings.max_archive_bytes,
                    )
                except BaseException:
                    if (
                        corrupt_archive is not None
                        and corrupt_archive.exists()
                        and not archive_path.exists()
                    ):
                        os.rename(corrupt_archive, archive_path)
                    raise
            if corrupt_archive is not None and corrupt_archive.exists():
                corrupt_archive.unlink(missing_ok=True)
            lock = {
                "schema_revision": SOURCE_LOCK_REVISION,
                "complete": True,
                "loader_type": self._settings.type,
                "requested_url": self._settings.archive_url,
                "final_url": result.final_url,
                "observed_sha256": observed,
                "byte_size": size,
                "archive_format": self._settings.archive_format,
                "archive_root": self._settings.archive_root,
                "source_base_url": self._settings.source_base_url,
                "include_path_prefixes": list(self._settings.include_path_prefixes),
                "fetched_at": fetched_at,
                "source_config_sha256": self._source_config_sha256,
                "source_snapshot_sha256": observed,
                "cache_relative_path": archive_relative,
            }
            manifest = _archive_source_manifest(
                loader_type=self._settings.type,
                requested_url=self._settings.archive_url,
                final_url=result.final_url,
                expected_sha256=None,
                observed_sha256=observed,
                sha256_origin="observed-on-acquisition",
                archive_byte_size=size,
                archive_root=self._settings.archive_root,
                source_base_url=self._settings.source_base_url,
                include_path_prefixes=self._settings.include_path_prefixes,
                source_config_sha256=self._source_config_sha256,
                documents=documents,
                acquired_at=fetched_at,
                cache_reused=safe_loader.cache_reused,
                offline=False,
                archive_cache_path=archive_relative,
            )
            self._begin_source_publication(observed)
            self._record_and_replace(
                lock_path=self._raw_root / "source.lock.json", payload=lock
            )
            self._record_and_replace(
                lock_path=self._raw_root / "fetch_manifest.json",
                payload=manifest,
            )
            self.summary = ArchiveLoadSummary(
                loader_type=self._settings.type,
                archive_sha256=observed,
                archive_byte_size=size,
                source_page_count=len(documents),
                cache_reused=False,
                offline=False,
                requested_url=self._settings.archive_url,
                final_url=result.final_url,
            )
            return documents
        except BaseException:
            temporary.unlink(missing_ok=True)
            if (
                archive_path is not None
                and corrupt_archive is not None
                and corrupt_archive.exists()
                and not archive_path.exists()
            ):
                os.rename(corrupt_archive, archive_path)
            if self._rollback_records:
                self.rollback_source()
            raise

    def _download_candidate(self, destination: BinaryIO) -> ArchiveDownloadResult:
        for attempt in range(self._settings.max_retries + 1):
            destination.seek(0)
            destination.truncate(0)
            try:
                return self._transport.download(
                    self._settings.archive_url,
                    destination,
                    timeout_seconds=self._settings.timeout_seconds,
                    user_agent=self._settings.user_agent,
                    max_archive_bytes=self._settings.max_archive_bytes,
                )
            except Exception:
                if attempt >= self._settings.max_retries:
                    raise
        raise AssertionError("unreachable archive retry boundary")

    def _documents_from_archive(
        self,
        archive_path: Path,
        observed_sha: str,
        *,
        fetched_at: str,
    ) -> tuple[tuple[SourceDocument, ...], SafeZipHtmlLoader]:
        safe_loader = SafeZipHtmlLoader(
            archive_path,
            self._raw_root / "extracted" / observed_sha,
            archive_sha256=observed_sha,
            archive_root=self._settings.archive_root,
            source_base_url=self._settings.source_base_url,
            include_path_prefixes=self._settings.include_path_prefixes,
            source_kind="snapshot-http-archive",
            max_archive_bytes=self._settings.max_archive_bytes,
            max_members=self._settings.max_members,
            max_member_bytes=self._settings.max_member_bytes,
            max_extracted_bytes=self._settings.max_extracted_bytes,
            fetched_at=fetched_at,
        )
        return safe_loader.load(), safe_loader

    def _record_and_replace(self, *, lock_path: Path, payload: dict[str, Any]) -> None:
        if not any(path == lock_path for path, _previous in self._rollback_records):
            previous = lock_path.read_bytes() if lock_path.is_file() else None
            self._rollback_records.append((lock_path, previous))
        _write_json_atomic(payload, lock_path)

    def _begin_source_publication(self, candidate_sha256: str) -> None:
        """Durably preserve pre-refresh source metadata before replacing either file."""
        targets = (
            self._raw_root / "source.lock.json",
            self._raw_root / "fetch_manifest.json",
        )
        records: list[tuple[Path, bytes | None]] = []
        for path in targets:
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                previous = None
            else:
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise ValueError(
                        "source publication metadata must be a regular file"
                    )
                previous = _read_bounded_regular_file(
                    path,
                    max_bytes=_MAX_SOURCE_ROLLBACK_BACKUP_BYTES,
                    label="source publication metadata",
                )
            records.append((path, previous))
        self._rollback_records = records
        if not self._refresh or not any(
            previous is not None for _path, previous in records
        ):
            return
        journal = self._source_rollback_root
        if journal.exists() or journal.is_symlink():
            raise RuntimeError("unfinished source refresh rollback journal exists")
        staging = Path(
            tempfile.mkdtemp(
                dir=self._raw_root,
                prefix=".source-refresh-rollback.staging-",
            )
        )
        try:
            journal_records: list[dict[str, Any]] = []
            for path, previous in records:
                backup_name = f"{path.name}.previous"
                if previous is not None:
                    _write_bytes_atomic(previous, staging / backup_name)
                journal_records.append(
                    {
                        "target": path.name,
                        "existed": previous is not None,
                        "backup": backup_name if previous is not None else None,
                        "sha256": (
                            hashlib.sha256(previous).hexdigest()
                            if previous is not None
                            else None
                        ),
                    }
                )
            _write_json_atomic(
                {
                    "schema_revision": SOURCE_ROLLBACK_REVISION,
                    "complete": True,
                    "candidate_source_snapshot_sha256": candidate_sha256,
                    "source_config_sha256": self._source_config_sha256,
                    "records": journal_records,
                },
                staging / "rollback.json",
            )
            os.rename(staging, journal)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @property
    def _source_rollback_root(self) -> Path:
        return self._raw_root / ".source-refresh-rollback"

    def _recover_pending_source_publication(self) -> None:
        recover_snapshot_http_archive_refresh(
            self._raw_root,
            settings=self._settings,
            source_config_sha256=self._source_config_sha256,
        )
        self._rollback_records.clear()

    def rollback_source(self) -> None:
        """Restore source lock/manifest state after a downstream refresh failure."""
        journal = self._source_rollback_root
        if journal.exists() or journal.is_symlink():
            _payload, records = _read_source_rollback_journal(journal, self._raw_root)
            self._rollback_records = records
        errors: list[BaseException] = []
        for path, previous in reversed(self._rollback_records):
            try:
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    _write_bytes_atomic(previous, path)
            except BaseException as error:
                errors.append(error)
        if errors:
            failure = RuntimeError(
                f"source rollback failed for {len(errors)} publication record(s)"
            )
            for error in errors:
                failure.add_note(f"{type(error).__name__}: {error}")
            raise failure from errors[0]
        if journal.exists() or journal.is_symlink():
            _retire_directory(journal)
        self._rollback_records.clear()

    def commit_source(self) -> None:
        """Forget rollback bytes after all downstream artifacts are complete."""
        journal = self._source_rollback_root
        if journal.exists() or journal.is_symlink():
            _retire_directory(journal)
        self._rollback_records.clear()


def _archive_source_manifest(
    *,
    loader_type: str,
    requested_url: str,
    final_url: str | None,
    expected_sha256: str | None,
    observed_sha256: str,
    sha256_origin: str,
    archive_byte_size: int,
    archive_root: str,
    source_base_url: str,
    include_path_prefixes: Sequence[str],
    source_config_sha256: str,
    documents: tuple[SourceDocument, ...],
    acquired_at: str | None,
    cache_reused: bool,
    offline: bool,
    archive_cache_path: str | None,
) -> dict[str, Any]:
    return {
        "schema_revision": ARCHIVE_SOURCE_MANIFEST_REVISION,
        "complete": True,
        "loader_type": loader_type,
        "requested_archive_url": requested_url,
        "final_archive_url": final_url,
        "expected_archive_sha256": expected_sha256,
        "observed_archive_sha256": observed_sha256,
        "archive_sha256_origin": sha256_origin,
        "archive_byte_size": archive_byte_size,
        "archive_format": "zip",
        "archive_root": archive_root,
        "source_base_url": source_base_url,
        "include_path_prefixes": list(include_path_prefixes),
        "source_page_count": len(documents),
        "source_config_fingerprint_revision": "source-config-v1",
        "source_config_sha256": source_config_sha256,
        "source_snapshot_sha256": observed_sha256,
        "document_snapshot_sha256": source_snapshot_sha256(documents),
        "acquired_at": acquired_at,
        "cache_reused": cache_reused,
        "offline": offline,
        "archive_cache_path": archive_cache_path,
        "pages": [
            {
                "source_url": document.source_url,
                "canonical_url": document.canonical_url,
                "logical_path": document.logical_path,
                "category": document.category,
                "content_sha256": document.content_sha256,
            }
            for document in documents
        ],
    }


def _validated_members(
    archive: zipfile.ZipFile,
    *,
    max_members: int,
    max_member_bytes: int,
    max_extracted_bytes: int,
) -> tuple[tuple[zipfile.ZipInfo, PurePosixPath, bool], ...]:
    infos = archive.infolist()
    if len(infos) > max_members:
        raise ValueError("ZIP member count exceeded max_members")
    seen: set[str] = set()
    casefold_seen: set[str] = set()
    path_kinds: dict[str, str] = {}
    total = 0
    validated: list[tuple[zipfile.ZipInfo, PurePosixPath, bool]] = []
    for info in infos:
        original = getattr(info, "orig_filename", info.filename)
        if "\x00" in original:
            raise ValueError("ZIP member contains NUL")
        relative = _safe_member_name(original)
        normalized = relative.as_posix()
        if normalized == "extraction_manifest.json":
            raise ValueError("ZIP member path is reserved for extraction metadata")
        key = normalized.rstrip("/")
        folded = _portable_path_identity(key)
        if folded == _portable_path_identity("extraction_manifest.json"):
            raise ValueError("ZIP member path is reserved for extraction metadata")
        if key in seen or folded in casefold_seen:
            raise ValueError(f"duplicate normalized ZIP output path: {key}")
        seen.add(key)
        casefold_seen.add(folded)
        is_directory = info.is_dir() or original.endswith("/")
        _validate_zip_entry_type(info, is_directory=is_directory)
        size = info.file_size
        if size < 0 or size > max_member_bytes:
            raise ValueError("ZIP member exceeded max_member_bytes")
        if not is_directory:
            total += size
            if total > max_extracted_bytes:
                raise ValueError("ZIP content exceeded max_extracted_bytes")
        path_kinds[key] = "directory" if is_directory else "file"
        validated.append((info, PurePosixPath(key), is_directory))
    tree_kinds: dict[str, str] = {}
    folded_tree_paths: dict[str, str] = {}
    for path, kind in path_kinds.items():
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts) + 1):
            tree_path = PurePosixPath(*parts[:index]).as_posix()
            tree_kind = kind if index == len(parts) else "directory"
            previous_kind = tree_kinds.get(tree_path)
            if previous_kind is not None and previous_kind != tree_kind:
                raise ValueError("ZIP file/directory output collision")
            tree_kinds[tree_path] = tree_kind
            folded = _portable_path_identity(tree_path)
            previous_path = folded_tree_paths.get(folded)
            if previous_path is not None and previous_path != tree_path:
                raise ValueError(f"duplicate normalized ZIP output path: {tree_path}")
            folded_tree_paths[folded] = tree_path
    return tuple(validated)


def _tree_paths_for_archive_members(member_paths: set[str]) -> set[str]:
    """Return explicit members plus directories created for implicit parents."""
    paths = set(member_paths)
    for member_path in member_paths:
        parts = PurePosixPath(member_path).parts
        paths.update(
            PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts))
        )
    return paths


def _portable_path_identity(value: str) -> str:
    """Return a stable Unicode/case-insensitive filesystem identity."""
    normalized = unicodedata.normalize("NFC", value)
    return unicodedata.normalize("NFC", normalized.casefold())


def _existing_tree_paths(root: Path) -> set[str]:
    """Enumerate an already symlink-free extraction tree without following links."""
    paths: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = Path(entry.path).relative_to(root).as_posix()
                paths.add(relative)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
    return paths


def _safe_member_name(value: str) -> PurePosixPath:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("unsafe ZIP member path")
    if (
        value.startswith("/")
        or value.startswith("//")
        or _WINDOWS_DRIVE_PATTERN.match(value)
    ):
        raise ValueError("unsafe absolute ZIP member path")
    stripped = value[:-1] if value.endswith("/") else value
    path = PurePosixPath(stripped)
    if path == PurePosixPath(".") or ".." in path.parts or "." in path.parts:
        raise ValueError("unsafe traversing ZIP member path")
    if path.as_posix() != stripped:
        raise ValueError("ZIP member path is not normalized POSIX")
    for component in path.parts:
        if (
            any(character in _WINDOWS_FORBIDDEN_CHARS for character in component)
            or any(ord(character) < 32 for character in component)
            or component.endswith((" ", "."))
            or len(component.encode("utf-16-le")) // 2 > 255
        ):
            raise ValueError("unsafe Windows ZIP member path")
        basename = component.split(".", maxsplit=1)[0].casefold()
        if basename in _WINDOWS_RESERVED_NAMES:
            raise ValueError("unsafe Windows device ZIP member path")
    return path


def _validate_zip_entry_type(info: zipfile.ZipInfo, *, is_directory: bool) -> None:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type == stat.S_IFLNK:
        raise ValueError("ZIP symlink entries are not allowed")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError("ZIP special/device entries are not allowed")
    if is_directory and file_type == stat.S_IFREG:
        raise ValueError("ZIP directory has regular-file mode")
    if not is_directory and file_type == stat.S_IFDIR:
        raise ValueError("ZIP file has directory mode")


def _selected_html_files(
    document_root: Path,
    prefixes: Sequence[str],
) -> tuple[Path, ...]:
    resolved_root = document_root.resolve(strict=True)
    selected: dict[str, Path] = {}
    for prefix in prefixes:
        prefix_path = PurePosixPath(prefix.rstrip("/"))
        candidate_root = document_root.joinpath(*prefix_path.parts)
        if not candidate_root.is_dir() or _is_link_like(candidate_root):
            continue
        for candidate in candidate_root.rglob("*.html"):
            if _is_link_like(candidate) or not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(resolved_root):
                raise ValueError("extracted HTML escaped archive_root")
            logical = resolved.relative_to(resolved_root).as_posix()
            selected[logical] = resolved
    return tuple(selected[key] for key in sorted(selected))


@contextmanager
def _verified_archive_stream(
    path: Path,
    *,
    max_bytes: int,
    expected_sha256: str,
) -> Iterator[tuple[BinaryIO, int, _ArchiveStreamIdentity]]:
    """Keep one verified no-follow archive inode open through all ZIP reads."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("archive cache path must be a regular file")
    if before.st_size > max_bytes:
        raise ValueError("archive exceeded max_archive_bytes")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise ValueError("archive changed while it was opened")
        observed_sha256, byte_size = _hash_open_archive(stream, max_bytes=max_bytes)
        verified = os.fstat(stream.fileno())
        identity = _ArchiveStreamIdentity(
            device=verified.st_dev,
            inode=verified.st_ino,
            byte_size=verified.st_size,
            modified_ns=verified.st_mtime_ns,
            changed_ns=verified.st_ctime_ns,
        )
        if (
            opened.st_size != byte_size
            or opened.st_mtime_ns != verified.st_mtime_ns
            or opened.st_ctime_ns != verified.st_ctime_ns
            or observed_sha256 != expected_sha256
        ):
            raise ValueError(
                "archive SHA-256 mismatch or archive changed while it was verified"
            )
        stream.seek(0)
        try:
            yield stream, byte_size, identity
        except BaseException:
            raise
        else:
            _verify_open_archive(
                stream,
                identity=identity,
                expected_sha256=expected_sha256,
                max_bytes=max_bytes,
            )


def _hash_open_archive(stream: BinaryIO, *, max_bytes: int) -> tuple[str, int]:
    """Hash one already-open regular file with an explicit byte bound."""
    stream.seek(0)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        size += len(block)
        if size > max_bytes:
            raise ValueError("archive exceeded max_archive_bytes")
        digest.update(block)
    return digest.hexdigest(), size


def _verify_open_archive(
    stream: BinaryIO,
    *,
    identity: _ArchiveStreamIdentity,
    expected_sha256: str,
    max_bytes: int,
) -> None:
    """Recheck the held archive inode and bytes after ZIP processing."""
    observed_sha256, byte_size = _hash_open_archive(stream, max_bytes=max_bytes)
    current = os.fstat(stream.fileno())
    if (
        current.st_dev != identity.device
        or current.st_ino != identity.inode
        or current.st_size != identity.byte_size
        or current.st_mtime_ns != identity.modified_ns
        or current.st_ctime_ns != identity.changed_ns
        or byte_size != identity.byte_size
        or observed_sha256 != expected_sha256
    ):
        raise ValueError("archive changed while ZIP content was processed")
    stream.seek(0)


def _preflight_zip_member_count(stream: BinaryIO, max_members: int) -> None:
    """Bound central-directory allocation before ``ZipFile`` parses all entries."""
    original_position = stream.tell()
    try:
        stream.seek(0, os.SEEK_END)
        archive_size = stream.tell()
        tail_size = min(archive_size, 65_557)
        stream.seek(archive_size - tail_size)
        tail = stream.read(tail_size)
        eocd_offset = -1
        search_end = len(tail)
        while True:
            candidate = tail.rfind(b"PK\x05\x06", 0, search_end)
            if candidate < 0:
                break
            if candidate + 22 <= len(tail):
                comment_size = struct.unpack_from("<H", tail, candidate + 20)[0]
                if candidate + 22 + comment_size == len(tail):
                    eocd_offset = archive_size - tail_size + candidate
                    break
            search_end = candidate
        if eocd_offset < 0:
            raise zipfile.BadZipFile("ZIP end-of-central-directory record is absent")
        eocd = tail[eocd_offset - (archive_size - tail_size) :]
        disk_number, central_disk = struct.unpack_from("<HH", eocd, 4)
        disk_member_count = struct.unpack_from("<H", eocd, 8)[0]
        member_count = struct.unpack_from("<H", eocd, 10)[0]
        central_size = struct.unpack_from("<L", eocd, 12)[0]
        central_offset = struct.unpack_from("<L", eocd, 16)[0]
        if disk_number != 0 or central_disk != 0:
            raise zipfile.BadZipFile("multi-disk ZIP archives are not supported")
        if member_count == 0xFFFF or central_size == 0xFFFFFFFF:
            locator_offset = eocd_offset - 20
            if locator_offset < 0:
                raise zipfile.BadZipFile("ZIP64 locator is absent")
            stream.seek(locator_offset)
            locator = stream.read(20)
            if len(locator) != 20 or locator[:4] != b"PK\x06\x07":
                raise zipfile.BadZipFile("ZIP64 locator is invalid")
            zip64_offset = struct.unpack_from("<Q", locator, 8)[0]
            stream.seek(zip64_offset)
            zip64 = stream.read(56)
            if len(zip64) < 56 or zip64[:4] != b"PK\x06\x06":
                raise zipfile.BadZipFile("ZIP64 end record is invalid")
            member_count = struct.unpack_from("<Q", zip64, 32)[0]
            central_size = struct.unpack_from("<Q", zip64, 40)[0]
            central_offset = struct.unpack_from("<Q", zip64, 48)[0]
        elif disk_member_count != member_count:
            raise zipfile.BadZipFile("ZIP central-directory counts are inconsistent")
        if member_count > max_members:
            raise ValueError("ZIP member count exceeded max_members")
        max_central_bytes = max(1024 * 1024, max_members * 8192)
        if central_size > max_central_bytes:
            raise ValueError("ZIP central directory exceeded its safe allocation bound")
        if central_offset + central_size > eocd_offset:
            raise zipfile.BadZipFile("ZIP central directory bounds are invalid")
        stream.seek(central_offset)
        remaining = central_size
        observed_members = 0
        while remaining:
            if remaining < 46:
                raise zipfile.BadZipFile("ZIP central-directory record is truncated")
            header = stream.read(46)
            if len(header) != 46 or header[:4] != b"PK\x01\x02":
                raise zipfile.BadZipFile("ZIP central-directory record is invalid")
            name_size, extra_size, comment_size = struct.unpack_from("<HHH", header, 28)
            variable_size = name_size + extra_size + comment_size
            record_size = 46 + variable_size
            if record_size > remaining:
                raise zipfile.BadZipFile(
                    "ZIP central-directory record exceeds its bounds"
                )
            stream.seek(variable_size, os.SEEK_CUR)
            remaining -= record_size
            observed_members += 1
            if observed_members > max_members:
                raise ValueError("ZIP member count exceeded max_members")
        if observed_members != member_count:
            raise zipfile.BadZipFile("ZIP central-directory count is inconsistent")
    finally:
        stream.seek(original_position)


def _bounded_open_file_sha256(stream: BinaryIO, max_bytes: int) -> tuple[str, int]:
    """Hash one held regular-file descriptor without trusting its path."""
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("archive staging must be a regular file")
    digest = hashlib.sha256()
    size = 0
    stream.seek(0)
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        size += len(block)
        if size > max_bytes:
            raise ValueError("archive exceeded max_archive_bytes")
        digest.update(block)
    after = os.fstat(stream.fileno())
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        raise ValueError("archive changed while its SHA-256 was computed")
    return digest.hexdigest(), size


def _require_open_file_path_identity(stream: BinaryIO, path: Path) -> None:
    """Require a temporary pathname to still name its held regular-file inode."""
    opened = os.fstat(stream.fileno())
    current = path.lstat()
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or current.st_dev != opened.st_dev
        or current.st_ino != opened.st_ino
    ):
        raise ValueError("archive staging path changed during download")


def _bounded_file_sha256(path: Path, max_bytes: int) -> tuple[str, int]:
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("archive cache path must be a regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("archive cache path must be a regular file")
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > max_bytes:
                raise ValueError("archive exceeded max_archive_bytes")
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (
        opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != size
        or after.st_mtime_ns != opened.st_mtime_ns
        or after.st_ctime_ns != opened.st_ctime_ns
    ):
        raise ValueError("archive changed while its SHA-256 was computed")
    return digest.hexdigest(), size


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read one no-follow regular file with an explicit memory bound."""
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError(f"{label} must be a bounded regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a regular file")
        payload = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    if (
        len(payload) > max_bytes
        or opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != len(payload)
        or after.st_mtime_ns != opened.st_mtime_ns
        or after.st_ctime_ns != opened.st_ctime_ns
    ):
        raise ValueError(f"{label} changed while it was read")
    return payload


def _zip_member_sha256(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int,
) -> tuple[str, int]:
    """Hash one streamed ZIP member while enforcing its actual output size."""
    digest = hashlib.sha256()
    size = 0
    with archive.open(info, "r") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > max_bytes:
                raise ValueError("ZIP member exceeded max_member_bytes")
            digest.update(block)
    if size != info.file_size:
        raise ValueError("ZIP member size changed during validation")
    return digest.hexdigest(), size


def _ensure_no_symlink_ancestors(root: Path, path: Path) -> None:
    if not path.is_relative_to(root):
        raise ValueError("ZIP extraction target escaped its staging root")
    current = path
    while current != root:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(mode) or _is_link_like(current):
                raise ValueError("ZIP extraction destination contains a symlink")
        parent = current.parent
        if parent == current:
            raise ValueError("ZIP extraction target escaped its staging root")
        current = parent


def _reject_existing_symlink_components(path: Path) -> None:
    """Reject an existing symlink at or above a cache publication path."""
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode) or _is_link_like(candidate):
            raise ValueError("archive cache path contains a symlink")


def _is_link_like(path: Path) -> bool:
    """Return whether a path is a symlink or native Windows junction."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _prepare_real_directory(path: Path) -> None:
    """Create a cache directory without following pre-existing symlink components."""
    _reject_existing_symlink_components(path)
    path.mkdir(parents=True, exist_ok=True)
    _reject_existing_symlink_components(path)
    mode = path.lstat().st_mode
    if not stat.S_ISDIR(mode):
        raise ValueError("archive cache path must be a directory")


def _publish_directory(staging: Path, destination: Path) -> None:
    _reject_unsafe_existing_extraction(destination)
    if not destination.exists():
        os.rename(staging, destination)
        return
    backup = destination.parent / f".{destination.name}.backup"
    if backup.exists():
        raise RuntimeError(f"archive extraction backup already exists: {backup}")
    os.rename(destination, backup)
    try:
        os.rename(staging, destination)
    except BaseException:
        os.rename(backup, destination)
        raise
    _retire_directory(backup)


def _recover_extraction_publication(destination: Path) -> None:
    """Restore an old extraction left between the two directory renames."""
    backup = destination.with_name(f".{destination.name}.backup")
    if not backup.exists() and not backup.is_symlink():
        return
    _reject_unsafe_existing_extraction(backup)
    if destination.exists() or destination.is_symlink():
        return
    os.rename(backup, destination)


def _publish_verified_archive(
    staging: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    max_archive_bytes: int,
) -> None:
    """Publish one verified archive and verify the exact destination bytes again."""
    observed, size = _bounded_file_sha256(staging, max_archive_bytes)
    if observed != expected_sha256 or size != expected_size:
        raise RuntimeError("verified archive staging changed before publication")
    os.replace(staging, destination)
    try:
        observed, size = _bounded_file_sha256(destination, max_archive_bytes)
        if observed != expected_sha256 or size != expected_size:
            raise RuntimeError("published archive identity is inconsistent")
    except BaseException:
        if destination.exists() or destination.is_symlink():
            quarantine = destination.with_name(
                f".{destination.name}.failed-publication-{uuid.uuid4().hex}"
            )
            os.rename(destination, quarantine)
        raise


def _reject_unsafe_existing_extraction(root: Path) -> None:
    absolute_root = root.absolute()
    for ancestor in absolute_root.parents:
        try:
            ancestor_mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(ancestor_mode) or _is_link_like(ancestor):
            raise ValueError("ZIP extraction destination ancestor is a symlink")
    try:
        mode = absolute_root.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or _is_link_like(absolute_root):
        raise ValueError("ZIP extraction destination contains a symlink")
    if not stat.S_ISDIR(mode):
        raise ValueError("ZIP extraction destination must be a directory")
    pending = [absolute_root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink() or _is_link_like(Path(entry.path)):
                    raise ValueError("ZIP extraction destination contains a symlink")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))


def _safe_cache_relative(value: str) -> Path:
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or _WINDOWS_DRIVE_PATTERN.match(value)
    ):
        raise ValueError("unsafe source lock cache path")
    pure_path = PurePosixPath(value)
    if (
        pure_path.is_absolute()
        or pure_path == PurePosixPath(".")
        or ".." in pure_path.parts
        or "." in pure_path.parts
        or pure_path.as_posix() != value
    ):
        raise ValueError("unsafe source lock cache path")
    return Path(*pure_path.parts)


def _validate_final_https_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as error:
        raise ValueError("archive redirect final URL must be absolute HTTPS") from error
    if parsed.scheme != "https" or not hostname:
        raise ValueError("archive redirect final URL must be absolute HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("archive redirect final URL authentication is not allowed")
    if parsed.query or parsed.fragment:
        raise ValueError("archive redirect final URL query/fragment is not allowed")


def _validate_source_lock(
    lock: dict[str, Any],
    *,
    settings: SnapshotHttpArchiveLoaderSettings,
    source_config_sha256: str,
) -> None:
    required = {
        "schema_revision",
        "complete",
        "loader_type",
        "requested_url",
        "final_url",
        "observed_sha256",
        "byte_size",
        "archive_format",
        "archive_root",
        "source_base_url",
        "include_path_prefixes",
        "fetched_at",
        "source_config_sha256",
        "source_snapshot_sha256",
        "cache_relative_path",
    }
    if set(lock) != required:
        raise RuntimeError("source lock schema is invalid")
    if (
        lock.get("schema_revision") != SOURCE_LOCK_REVISION
        or lock.get("complete") is not True
        or lock.get("loader_type") != settings.type
    ):
        raise RuntimeError("source lock is incomplete or incompatible")
    if lock.get("source_config_sha256") != source_config_sha256:
        raise RuntimeError(
            "source config does not match the existing source lock; "
            "use a different data-root or explicit --refresh"
        )
    if (
        lock.get("requested_url") != settings.archive_url
        or lock.get("archive_format") != settings.archive_format
        or lock.get("archive_root") != settings.archive_root
        or lock.get("source_base_url") != settings.source_base_url
        or lock.get("include_path_prefixes") != list(settings.include_path_prefixes)
    ):
        raise RuntimeError("source lock settings identity is inconsistent")
    observed = lock.get("observed_sha256")
    if not isinstance(observed, str) or not re.fullmatch(r"[0-9a-f]{64}", observed):
        raise RuntimeError("source lock observed SHA-256 is invalid")
    if lock.get("source_snapshot_sha256") != observed:
        raise RuntimeError("source lock snapshot identity is inconsistent")
    if (
        not isinstance(lock.get("byte_size"), int)
        or isinstance(lock["byte_size"], bool)
        or lock["byte_size"] < 1
    ):
        raise RuntimeError("source lock byte size is invalid")
    if not isinstance(lock.get("fetched_at"), str) or not lock["fetched_at"].strip():
        raise RuntimeError("source lock fetched timestamp is invalid")
    try:
        _validate_final_https_url(str(lock.get("requested_url")))
        _validate_final_https_url(str(lock.get("final_url")))
    except ValueError as error:
        raise RuntimeError("source lock URL identity is invalid") from error
    cache_relative = _safe_cache_relative(str(lock.get("cache_relative_path")))
    if cache_relative.as_posix() != f"archives/{observed}.zip":
        raise RuntimeError("source lock cache identity is inconsistent")


def validate_snapshot_http_archive_cache(
    settings: SnapshotHttpArchiveLoaderSettings,
    raw_root: Path,
    *,
    source_config_sha256: str,
) -> dict[str, Any]:
    """Strictly validate a source lock and its local archive without network I/O."""
    lock_path = raw_root / "source.lock.json"
    try:
        payload = _read_bounded_regular_file(
            lock_path,
            max_bytes=_MAX_SOURCE_ROLLBACK_JSON_BYTES,
            label="source lock",
        )
        lock = json.loads(payload.decode("utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError("source lock is missing") from error
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"existing source lock is invalid: {error}") from error
    if not isinstance(lock, dict):
        raise RuntimeError("existing source lock is invalid: JSON object required")
    _validate_source_lock(
        lock,
        settings=settings,
        source_config_sha256=source_config_sha256,
    )
    archive_path = raw_root / _safe_cache_relative(str(lock["cache_relative_path"]))
    try:
        _reject_existing_symlink_components(archive_path)
        observed, size = _bounded_file_sha256(
            archive_path,
            settings.max_archive_bytes,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "source lock cache is missing or invalid; "
            "use --refresh to acquire a new snapshot"
        ) from error
    if observed != lock["observed_sha256"] or size != lock["byte_size"]:
        raise RuntimeError(
            "source lock cache is missing or invalid; use --refresh to acquire a new snapshot"
        )
    return lock


def _read_source_rollback_journal(
    journal: Path,
    raw_root: Path,
) -> tuple[dict[str, Any], list[tuple[Path, bytes | None]]]:
    """Read and authenticate the structure and backup bytes of a refresh journal."""
    _reject_unsafe_existing_extraction(journal)
    try:
        encoded = _read_bounded_regular_file(
            journal / "rollback.json",
            max_bytes=_MAX_SOURCE_ROLLBACK_JSON_BYTES,
            label="source refresh rollback journal",
        )
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("source refresh rollback journal is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("source refresh rollback journal must be a JSON object")
    if set(payload) != {
        "schema_revision",
        "complete",
        "candidate_source_snapshot_sha256",
        "source_config_sha256",
        "records",
    }:
        raise RuntimeError("source refresh rollback journal schema is invalid")
    if (
        payload.get("schema_revision") != SOURCE_ROLLBACK_REVISION
        or payload.get("complete") is not True
        or not isinstance(payload.get("candidate_source_snapshot_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(payload["candidate_source_snapshot_sha256"]),
        )
        or not isinstance(payload.get("source_config_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(payload["source_config_sha256"]))
    ):
        raise RuntimeError("source refresh rollback journal is incomplete")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != 2:
        raise RuntimeError("source refresh rollback records are invalid")
    expected_targets = {"source.lock.json", "fetch_manifest.json"}
    seen_targets: set[str] = set()
    validated_records: list[tuple[str, str | None, str | None]] = []
    expected_files = {"rollback.json"}
    for record in raw_records:
        if not isinstance(record, dict) or set(record) != {
            "target",
            "existed",
            "backup",
            "sha256",
        }:
            raise RuntimeError("source refresh rollback record schema is invalid")
        target = record.get("target")
        if not isinstance(target, str) or target not in expected_targets:
            raise RuntimeError("source refresh rollback target is invalid")
        if target in seen_targets or not isinstance(record.get("existed"), bool):
            raise RuntimeError("source refresh rollback target is duplicated")
        seen_targets.add(target)
        if record["existed"]:
            backup_name = record.get("backup")
            expected_backup = f"{target}.previous"
            expected_sha = record.get("sha256")
            if (
                backup_name != expected_backup
                or not isinstance(expected_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            ):
                raise RuntimeError("source refresh rollback backup identity is invalid")
            expected_files.add(expected_backup)
            validated_records.append((target, expected_backup, expected_sha))
        else:
            if record.get("backup") is not None or record.get("sha256") is not None:
                raise RuntimeError("source refresh rollback missing marker is invalid")
            validated_records.append((target, None, None))
    if seen_targets != expected_targets:
        raise RuntimeError("source refresh rollback targets are incomplete")
    actual_files: set[str] = set()
    with os.scandir(journal) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                raise RuntimeError(
                    "source refresh rollback journal contains a special entry"
                )
            actual_files.add(entry.name)
    if actual_files != expected_files:
        raise RuntimeError("source refresh rollback journal file set is invalid")
    records: list[tuple[Path, bytes | None]] = []
    for target, backup_name, expected_sha in validated_records:
        if backup_name is None:
            previous = None
        else:
            try:
                previous = _read_bounded_regular_file(
                    journal / backup_name,
                    max_bytes=_MAX_SOURCE_ROLLBACK_BACKUP_BYTES,
                    label="source refresh rollback backup",
                )
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    "source refresh rollback backup is invalid"
                ) from error
            if hashlib.sha256(previous).hexdigest() != expected_sha:
                raise RuntimeError("source refresh rollback backup SHA-256 is invalid")
        records.append((raw_root / target, previous))
    return payload, records


def _dataset_committing_source_candidate(
    raw_root: Path,
    *,
    candidate_sha256: str,
    source_config_sha256: str,
) -> DatasetManifest | None:
    """Return the strict v2 dataset manifest that names the source candidate."""
    if raw_root.name != "raw" or raw_root.parent.name != "data":
        return None
    try:
        encoded_dataset = _read_bounded_regular_file(
            raw_root.parent.parent / "dataset_manifest.json",
            max_bytes=_MAX_SOURCE_ROLLBACK_JSON_BYTES,
            label="dataset manifest",
        )
        dataset_value = json.loads(encoded_dataset.decode("utf-8"))
        if not isinstance(dataset_value, dict):
            return None
        dataset = DatasetManifest(**dataset_value)
    except (
        OSError,
        RuntimeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    if (
        dataset.schema_revision == DATASET_MANIFEST_REVISION
        and dataset.source_snapshot_sha256 == candidate_sha256
        and dataset.source_config_sha256 == source_config_sha256
    ):
        return dataset
    return None


def _snapshot_source_manifest_commits_candidate(
    settings: SnapshotHttpArchiveLoaderSettings,
    raw_root: Path,
    *,
    dataset: DatasetManifest,
    lock: dict[str, Any],
) -> bool:
    """Anchor a refresh candidate to augmented provenance and verified ZIP pages."""
    try:
        encoded = _read_bounded_regular_file(
            raw_root / "fetch_manifest.json",
            max_bytes=_MAX_SOURCE_ROLLBACK_BACKUP_BYTES,
            label="archive source manifest",
        )
        source = json.loads(encoded.decode("utf-8"))
        if (
            not isinstance(source, dict)
            or set(source) != _COMMITTED_ARCHIVE_SOURCE_KEYS
        ):
            return False
        archive_sha256 = str(lock["observed_sha256"])
        archive_relative = str(lock["cache_relative_path"])
        archive_path = raw_root / _safe_cache_relative(archive_relative)
        safe_loader = SafeZipHtmlLoader(
            archive_path,
            raw_root / "extracted" / archive_sha256,
            archive_sha256=archive_sha256,
            archive_root=settings.archive_root,
            source_base_url=settings.source_base_url,
            include_path_prefixes=settings.include_path_prefixes,
            source_kind=settings.type,
            max_archive_bytes=settings.max_archive_bytes,
            max_members=settings.max_members,
            max_member_bytes=settings.max_member_bytes,
            max_extracted_bytes=settings.max_extracted_bytes,
            fetched_at=str(lock["fetched_at"]),
        )
        documents = safe_loader.load()
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        return False
    pages = [
        {
            "source_url": document.source_url,
            "canonical_url": document.canonical_url,
            "logical_path": document.logical_path,
            "category": document.category,
            "content_sha256": document.content_sha256,
        }
        for document in documents
    ]
    raw_pages = source.get("pages")
    if (
        not isinstance(raw_pages, list)
        or any(
            not isinstance(page, dict) or set(page) != _ARCHIVE_PAGE_KEYS
            for page in raw_pages
        )
        or raw_pages != pages
        or not isinstance(source.get("parser_settings"), dict)
        or not isinstance(source.get("cache_reused"), bool)
        or source.get("offline") is not False
    ):
        return False
    expected = {
        "schema_revision": ARCHIVE_SOURCE_MANIFEST_REVISION,
        "complete": True,
        "loader_type": settings.type,
        "requested_archive_url": lock["requested_url"],
        "final_archive_url": lock["final_url"],
        "expected_archive_sha256": None,
        "observed_archive_sha256": archive_sha256,
        "archive_sha256_origin": "observed-on-acquisition",
        "archive_byte_size": lock["byte_size"],
        "archive_format": settings.archive_format,
        "archive_root": settings.archive_root,
        "source_base_url": settings.source_base_url,
        "include_path_prefixes": list(settings.include_path_prefixes),
        "source_page_count": dataset.source_page_count,
        "source_config_fingerprint_revision": "source-config-v1",
        "source_config_sha256": dataset.source_config_sha256,
        "source_snapshot_sha256": archive_sha256,
        "document_snapshot_sha256": source_snapshot_sha256(documents),
        "acquired_at": lock["fetched_at"],
        "archive_cache_path": archive_relative,
        "parser_type": dataset.parser_type,
        "site_config_file_sha256": dataset.site_config_sha256,
        "processing_config_fingerprint_revision": "processing-config-v1",
        "processing_config_sha256": dataset.processing_config_sha256,
        "configured_source": source_config_payload(
            settings,
            category=dataset.dataset_slug,
        )["source"],
        "pages": pages,
    }
    return all(source.get(key) == value for key, value in expected.items())


def snapshot_http_archive_refresh_candidate_is_committed(
    settings: SnapshotHttpArchiveLoaderSettings,
    raw_root: Path,
    *,
    source_config_sha256: str,
) -> bool:
    """Validate a pending refresh candidate before retiring its rollback journal."""
    journal = raw_root / ".source-refresh-rollback"
    if not journal.exists() and not journal.is_symlink():
        return True
    payload, _records = _read_source_rollback_journal(journal, raw_root)
    candidate_sha256 = str(payload["candidate_source_snapshot_sha256"])
    candidate_source_config = str(payload["source_config_sha256"])
    dataset = _dataset_committing_source_candidate(
        raw_root,
        candidate_sha256=candidate_sha256,
        source_config_sha256=candidate_source_config,
    )
    if dataset is None:
        return False
    if candidate_source_config != source_config_sha256:
        raise RuntimeError(
            "pending source refresh config does not match the active config"
        )
    lock = validate_snapshot_http_archive_cache(
        settings,
        raw_root,
        source_config_sha256=source_config_sha256,
    )
    if (
        lock.get("observed_sha256") != candidate_sha256
        or lock.get("source_snapshot_sha256") != candidate_sha256
    ):
        raise RuntimeError("pending source refresh cache identity is inconsistent")
    return _snapshot_source_manifest_commits_candidate(
        settings,
        raw_root,
        dataset=dataset,
        lock=lock,
    )


def recover_snapshot_http_archive_refresh(
    raw_root: Path,
    *,
    settings: SnapshotHttpArchiveLoaderSettings | None = None,
    source_config_sha256: str | None = None,
    force_rollback: bool = False,
) -> bool:
    """Recover source metadata replaced before derived publication committed."""
    journal = raw_root / ".source-refresh-rollback"
    if not journal.exists() and not journal.is_symlink():
        return True
    payload, records = _read_source_rollback_journal(journal, raw_root)
    committed = False
    if not force_rollback:
        if settings is None or source_config_sha256 is None:
            if (
                _dataset_committing_source_candidate(
                    raw_root,
                    candidate_sha256=str(payload["candidate_source_snapshot_sha256"]),
                    source_config_sha256=str(payload["source_config_sha256"]),
                )
                is not None
            ):
                raise RuntimeError(
                    "source refresh candidate validation requires active settings"
                )
        else:
            committed = snapshot_http_archive_refresh_candidate_is_committed(
                settings,
                raw_root,
                source_config_sha256=source_config_sha256,
            )
    if committed:
        _retire_directory(journal)
        return True
    errors: list[BaseException] = []
    for path, previous in reversed(records):
        try:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                _write_bytes_atomic(previous, path)
        except BaseException as error:
            errors.append(error)
    if errors:
        failure = RuntimeError(
            f"source rollback failed for {len(errors)} publication record(s)"
        )
        for error in errors:
            failure.add_note(f"{type(error).__name__}: {error}")
        raise failure from errors[0]
    _retire_directory(journal)
    return False


def _retire_directory(path: Path) -> None:
    """Atomically deactivate a completed journal before best-effort cleanup."""
    _reject_unsafe_existing_extraction(path)
    retired = path.with_name(f".{path.name}.retired-{uuid.uuid4().hex}")
    os.rename(path, retired)
    try:
        shutil.rmtree(retired)
    except BaseException:
        # The transaction committed at rename; an orphan is safer than rollback.
        pass


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise RuntimeError("existing source lock is not a regular file")
    try:
        encoded = _read_bounded_regular_file(
            path,
            max_bytes=_MAX_SOURCE_ROLLBACK_JSON_BYTES,
            label="source lock",
        )
        value = json.loads(encoded.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        return value
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"existing source lock is invalid: {error}") from error


def _read_bounded_json_object(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> dict[str, Any]:
    encoded = _read_bounded_regular_file(
        path,
        max_bytes=max_bytes,
        label=label,
    )
    value = json.loads(encoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    _write_bytes_atomic(
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        path,
    )


def _write_bytes_atomic(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
