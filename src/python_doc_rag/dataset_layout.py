"""Portable dataset artifact manifests with a legacy Python compatibility layer."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_MANIFEST_REVISION = "dataset-artifact-layout-v2"
LEGACY_DATASET_MANIFEST_REVISION = "dataset-artifact-layout-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = frozenset(
    {
        "schema_revision",
        "dataset_name",
        "dataset_slug",
        "loader_type",
        "parser_type",
        "site_config_sha256",
        "source_manifest_path",
        "processed_chunks_path",
        "corpus_manifest_path",
        "dense_index_path",
        "dense_metadata_path",
        "dense_manifest_path",
        "profile_artifact_root",
        "created_at",
        "source_page_count",
        "section_count",
        "chunk_count",
        "source_config_sha256",
        "processing_config_sha256",
        "source_snapshot_sha256",
    }
)
_LEGACY_MANIFEST_KEYS = _MANIFEST_KEYS.difference(
    {"source_config_sha256", "processing_config_sha256", "source_snapshot_sha256"}
)
_PENDING_PUBLICATION_PATHS = (
    Path(".prepare-publication.json"),
    Path(".prepare-rebuild-recovery.json"),
    Path("data/raw/.source-refresh-rollback"),
    Path("data/.raw.backup"),
)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Portable identity and relative artifact paths for one prepared dataset."""

    schema_revision: str
    dataset_name: str
    dataset_slug: str
    loader_type: str
    parser_type: str
    site_config_sha256: str
    source_manifest_path: str
    processed_chunks_path: str
    corpus_manifest_path: str
    dense_index_path: str
    dense_metadata_path: str
    dense_manifest_path: str
    profile_artifact_root: str
    created_at: str
    source_page_count: int
    section_count: int
    chunk_count: int
    source_config_sha256: str | None = None
    processing_config_sha256: str | None = None
    source_snapshot_sha256: str | None = None

    def __post_init__(self) -> None:
        """Reject unsafe paths, incomplete identities, and invalid counts."""
        if self.schema_revision not in {
            DATASET_MANIFEST_REVISION,
            LEGACY_DATASET_MANIFEST_REVISION,
        }:
            raise ValueError(f"unsupported dataset manifest: {self.schema_revision}")
        for label, value in (
            ("dataset_name", self.dataset_name),
            ("dataset_slug", self.dataset_slug),
            ("loader_type", self.loader_type),
            ("parser_type", self.parser_type),
            ("created_at", self.created_at),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must not be empty")
        if not _SHA256_PATTERN.fullmatch(self.site_config_sha256):
            raise ValueError("site_config_sha256 must be lowercase SHA-256")
        fingerprint_values = (
            self.source_config_sha256,
            self.processing_config_sha256,
            self.source_snapshot_sha256,
        )
        if self.schema_revision == DATASET_MANIFEST_REVISION:
            if not all(
                isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)
                for value in fingerprint_values
            ):
                raise ValueError("v2 dataset fingerprints must be lowercase SHA-256")
        elif any(value is not None for value in fingerprint_values):
            raise ValueError("legacy dataset manifest must not contain v2 fingerprints")
        for label in (
            "source_manifest_path",
            "processed_chunks_path",
            "corpus_manifest_path",
            "dense_index_path",
            "dense_metadata_path",
            "dense_manifest_path",
            "profile_artifact_root",
        ):
            safe_relative_path(getattr(self, label), label)
        for label in ("source_page_count", "section_count", "chunk_count"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON object persisted at the data root."""
        result = asdict(self)
        if self.schema_revision == LEGACY_DATASET_MANIFEST_REVISION:
            for key in (
                "source_config_sha256",
                "processing_config_sha256",
                "source_snapshot_sha256",
            ):
                result.pop(key)
        return result


@dataclass(frozen=True, slots=True)
class ResolvedDatasetArtifacts:
    """Absolute artifact paths resolved within one trusted data root."""

    data_root: Path
    manifest_path: Path | None
    processed_jsonl: Path
    corpus_manifest_path: Path | None
    index_path: Path
    metadata_path: Path
    index_manifest_path: Path
    source_manifest_path: Path | None
    profile_artifact_root: Path | None
    dataset_manifest: DatasetManifest | None

    @property
    def legacy_python_layout(self) -> bool:
        """Return whether fixed historical Python paths are in use."""
        return self.dataset_manifest is None


def generic_dataset_manifest(
    *,
    dataset_name: str,
    dataset_slug: str,
    loader_type: str,
    parser_type: str,
    site_config_sha256: str,
    created_at: str,
    source_page_count: int,
    section_count: int,
    chunk_count: int,
    source_config_sha256: str | None = None,
    processing_config_sha256: str | None = None,
    source_snapshot_sha256: str | None = None,
) -> DatasetManifest:
    """Create the canonical generic artifact layout without absolute paths."""
    fingerprints = (
        source_config_sha256,
        processing_config_sha256,
        source_snapshot_sha256,
    )
    supplied_fingerprints = tuple(value is not None for value in fingerprints)
    if any(supplied_fingerprints) and not all(supplied_fingerprints):
        raise ValueError(
            "source_config_sha256, processing_config_sha256, and "
            "source_snapshot_sha256 must be provided together"
        )
    schema_revision = (
        DATASET_MANIFEST_REVISION
        if all(supplied_fingerprints)
        else LEGACY_DATASET_MANIFEST_REVISION
    )
    return DatasetManifest(
        schema_revision=schema_revision,
        dataset_name=dataset_name,
        dataset_slug=dataset_slug,
        loader_type=loader_type,
        parser_type=parser_type,
        site_config_sha256=site_config_sha256,
        source_manifest_path="data/raw/fetch_manifest.json",
        processed_chunks_path="data/processed/chunks.jsonl",
        corpus_manifest_path="data/processed/corpus_manifest.json",
        dense_index_path="indexes/dense.faiss",
        dense_metadata_path="indexes/metadata.jsonl",
        dense_manifest_path="indexes/index_manifest.json",
        profile_artifact_root="profiles",
        created_at=created_at,
        source_page_count=source_page_count,
        section_count=section_count,
        chunk_count=chunk_count,
        source_config_sha256=source_config_sha256,
        processing_config_sha256=processing_config_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
    )


def resolve_dataset_artifacts(
    data_root: Path,
    *,
    allow_pending_publication: bool = False,
) -> ResolvedDatasetArtifacts:
    """Resolve a generic manifest when present, otherwise legacy Python paths."""
    root = data_root.expanduser().resolve()
    pending = [
        relative.as_posix()
        for relative in _PENDING_PUBLICATION_PATHS
        if (root / relative).exists() or (root / relative).is_symlink()
    ]
    if pending and not allow_pending_publication:
        raise RuntimeError(
            "dataset publication is incomplete; run prepare to recover before use: "
            + ", ".join(pending)
        )
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return ResolvedDatasetArtifacts(
            data_root=root,
            manifest_path=None,
            processed_jsonl=root / "data/processed/python_3_13_ja_chunks.jsonl",
            corpus_manifest_path=None,
            index_path=root / "indexes/python_3_13_ja.faiss",
            metadata_path=root / "indexes/python_3_13_ja_metadata.jsonl",
            index_manifest_path=root / "indexes/python_3_13_ja_index_manifest.json",
            source_manifest_path=None,
            profile_artifact_root=None,
            dataset_manifest=None,
        )
    manifest = load_dataset_manifest(manifest_path)
    return ResolvedDatasetArtifacts(
        data_root=root,
        manifest_path=manifest_path,
        processed_jsonl=resolve_manifest_path(root, manifest.processed_chunks_path),
        corpus_manifest_path=resolve_manifest_path(root, manifest.corpus_manifest_path),
        index_path=resolve_manifest_path(root, manifest.dense_index_path),
        metadata_path=resolve_manifest_path(root, manifest.dense_metadata_path),
        index_manifest_path=resolve_manifest_path(root, manifest.dense_manifest_path),
        source_manifest_path=resolve_manifest_path(root, manifest.source_manifest_path),
        profile_artifact_root=resolve_manifest_path(root, manifest.profile_artifact_root),
        dataset_manifest=manifest,
    )


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Read a strict dataset manifest without accepting unknown fields."""
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid dataset manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("dataset manifest must be a JSON object")
    revision = value.get("schema_revision")
    expected_keys = (
        _LEGACY_MANIFEST_KEYS
        if revision == LEGACY_DATASET_MANIFEST_REVISION
        else _MANIFEST_KEYS
    )
    unknown = sorted(set(value).difference(expected_keys))
    missing = sorted(expected_keys.difference(value))
    if unknown or missing:
        raise ValueError(
            f"dataset manifest schema mismatch; unknown={unknown}, missing={missing}"
        )
    try:
        if revision == LEGACY_DATASET_MANIFEST_REVISION:
            value = {
                **value,
                "source_config_sha256": None,
                "processing_config_sha256": None,
                "source_snapshot_sha256": None,
            }
        return DatasetManifest(**value)
    except TypeError as error:
        raise ValueError(f"invalid dataset manifest values: {error}") from error


def write_dataset_manifest_atomic(manifest: DatasetManifest, path: Path) -> None:
    """Persist one complete manifest through a same-filesystem temporary file."""
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def safe_relative_path(value: str, label: str) -> Path:
    """Require a non-empty relative path with no parent traversal."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path == Path(".") or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} is not a safe relative path: {value}")
    return path


def resolve_manifest_path(data_root: Path, value: str) -> Path:
    """Resolve one manifest path and reject symlink escape from data-root."""
    root = data_root.expanduser().resolve()
    candidate = (root / safe_relative_path(value, "manifest path")).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"dataset manifest path escapes data-root: {value}")
    return candidate
