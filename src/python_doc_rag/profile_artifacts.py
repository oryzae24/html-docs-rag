"""Resolve and validate runtime-profile artifacts without loading ML models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from python_doc_rag.profiles import RuntimeProfile


@dataclass(frozen=True, slots=True)
class ProfileArtifactPaths:
    """Safe absolute paths derived from one runtime profile and data root."""

    data_root: Path
    required_paths: tuple[Path, ...]
    embedding_index_path: Path | None
    embedding_metadata_path: Path | None
    embedding_manifest_path: Path | None
    symbol_index_path: Path | None
    identity_manifest_path: Path | None


PROFILE_ARTIFACT_MANIFEST_REVISION = "dataset-profile-artifacts-v1"
_GENERIC_PROFILE_ROOT = Path("profiles/recommended-v2")


@dataclass(frozen=True, slots=True)
class DatasetProfileArtifactManifest:
    """Dataset-local content identity separated from algorithm profile settings."""

    schema_revision: str
    profile_revision: str
    dataset_name: str
    dataset_slug: str
    record_count: int
    input_metadata_sha256: str
    embedding_index_path: str
    embedding_index_sha256: str
    embedding_metadata_path: str
    embedding_metadata_sha256: str
    embedding_manifest_path: str
    embedding_manifest_sha256: str
    symbol_index_path: str
    symbol_index_sha256: str
    model_name: str
    model_revision: str
    embedding_dimension: int
    normalized_embeddings: bool
    query_prefix: str
    document_prefix: str
    trust_remote_code: bool
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_revision != PROFILE_ARTIFACT_MANIFEST_REVISION:
            raise ValueError("unsupported dataset profile artifact manifest")
        if self.record_count < 1 or self.embedding_dimension < 1:
            raise ValueError("profile artifact counts and dimension must be positive")
        for label in (
            "input_metadata_sha256",
            "embedding_index_sha256",
            "embedding_metadata_sha256",
            "embedding_manifest_sha256",
            "symbol_index_sha256",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{label} must be a complete SHA-256")
        for label in (
            "embedding_index_path",
            "embedding_metadata_path",
            "embedding_manifest_path",
            "symbol_index_path",
        ):
            safe_profile_relative_path(getattr(self, label), label)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProfileArtifactValidation:
    """Model-free validation details for profile-specific local artifacts."""

    profile_name: str
    chunk_count: int | None
    index_count: int | None
    embedding_dimension: int | None
    model_name: str | None
    model_revision: str | None
    index_sha256: str | None
    metadata_sha256: str | None
    manifest_sha256: str | None
    symbol_sha256: str | None
    errors: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether all requested profile invariants were satisfied."""
        return not self.errors


def safe_profile_relative_path(value: str | None, label: str) -> Path:
    """Reject missing, absolute, parent-traversing, or empty profile paths."""
    if value is None or not value.strip():
        raise ValueError(f"profileに{label} pathがありません。")
    path = Path(value)
    if path == Path(".") or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"profileの{label} pathが安全な相対pathではありません: {value}"
        )
    return path


def profile_artifact_paths(
    profile: RuntimeProfile,
    data_root: Path,
    *,
    allow_pending_publication: bool = False,
) -> ProfileArtifactPaths:
    """Resolve every profile artifact below the data root without escaping it."""
    root = data_root.expanduser().resolve()
    dataset_paths = _generic_dataset_paths(
        profile,
        root,
        allow_pending_publication=allow_pending_publication,
    )
    if dataset_paths is not None:
        return dataset_paths
    required_paths = tuple(
        _below_data_root(root, value, "required artifact")
        for value in profile.required_artifacts
    )
    if profile.embedding_model_key is None:
        return ProfileArtifactPaths(
            data_root=root,
            required_paths=required_paths,
            embedding_index_path=None,
            embedding_metadata_path=None,
            embedding_manifest_path=None,
            symbol_index_path=None,
            identity_manifest_path=None,
        )
    return ProfileArtifactPaths(
        data_root=root,
        required_paths=required_paths,
        embedding_index_path=_below_data_root(
            root,
            profile.embedding_index_path,
            "embedding index",
        ),
        embedding_metadata_path=_below_data_root(
            root,
            profile.embedding_metadata_path,
            "embedding metadata",
        ),
        embedding_manifest_path=_below_data_root(
            root,
            profile.embedding_manifest_path,
            "embedding manifest",
        ),
        symbol_index_path=_below_data_root(
            root,
            profile.symbol_index_path,
            "symbol index",
        ),
        identity_manifest_path=None,
    )


def validate_profile_artifacts(
    profile: RuntimeProfile,
    data_root: Path,
    *,
    expected_chunk_count: int | None = None,
    baseline_metadata_path: Path | None = None,
    allow_pending_publication: bool = False,
) -> ProfileArtifactValidation:
    """Validate profile paths, manifests, counts, hashes, and pinned model spec."""
    errors: list[str] = []
    try:
        paths = profile_artifact_paths(
            profile,
            data_root,
            allow_pending_publication=allow_pending_publication,
        )
    except ValueError as error:
        return _validation(profile, errors=(str(error),))
    for path in paths.required_paths:
        if not path.is_file():
            errors.append(f"profile artifactが見つかりません: {path}")
    if errors or profile.embedding_model_key is None:
        return _validation(profile, errors=tuple(errors))

    index_path = _present(paths.embedding_index_path)
    metadata_path = _present(paths.embedding_metadata_path)
    manifest_path = _present(paths.embedding_manifest_path)
    symbol_path = _present(paths.symbol_index_path)
    identity = _load_identity_manifest(paths.identity_manifest_path, errors)
    try:
        manifest = _read_json_object(manifest_path)
    except ValueError as error:
        return _validation(
            profile,
            errors=(f"profile manifestを読み取れません: {error}",),
        )

    from python_doc_rag.embedding_models import retrieval_embedding_spec

    spec = retrieval_embedding_spec(profile.embedding_model_key)
    expected_manifest = {
        "model_name": spec.model_name,
        "model_revision": spec.revision,
        "embedding_dimension": spec.embedding_dimension,
        "normalized_embeddings": spec.normalize_embeddings,
        "query_prefix": spec.query_prefix,
        "document_prefix": spec.document_prefix,
        "trust_remote_code": spec.trust_remote_code,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            errors.append(
                f"profile manifestの{key}が不一致です: "
                f"{manifest.get(key)!r} != {expected!r}"
            )

    manifest_count = _positive_integer(manifest.get("chunk_count"))
    if manifest_count is None:
        errors.append("profile manifestのchunk_countが正の整数ではありません。")
    metadata_count = _safe_metadata_count(metadata_path, errors)
    symbol_count = _safe_symbol_count(metadata_path, symbol_path, errors)
    index_count, index_dimension = _safe_index_summary(index_path, errors)
    _compare_counts(
        "profile manifest", manifest_count, "metadata", metadata_count, errors
    )
    _compare_counts(
        "profile manifest", manifest_count, "symbol index", symbol_count, errors
    )
    _compare_counts(
        "profile manifest", manifest_count, "FAISS index", index_count, errors
    )
    if expected_chunk_count is not None:
        _compare_counts(
            "expected chunk count",
            expected_chunk_count,
            "profile manifest",
            manifest_count,
            errors,
        )
    if index_dimension is not None and index_dimension != spec.embedding_dimension:
        errors.append(
            "profile FAISS index次元が不一致です: "
            f"{index_dimension} != {spec.embedding_dimension}"
        )

    actual_index_sha = _compare_file_hash(
        index_path,
        manifest.get("index_sha256"),
        "profile FAISS",
        errors,
    )
    actual_metadata_sha = _compare_file_hash(
        metadata_path,
        manifest.get("metadata_sha256"),
        "profile metadata",
        errors,
    )
    actual_symbol_sha = _compare_file_hash(
        symbol_path,
        identity.symbol_index_sha256
        if identity is not None
        else profile.symbol_index_sha256,
        "symbol index",
        errors,
    )
    if identity is not None:
        _validate_identity_manifest(
            identity,
            profile,
            spec,
            paths,
            actual_index_sha,
            actual_metadata_sha,
            actual_symbol_sha,
            manifest_count,
            errors,
        )
    if baseline_metadata_path is not None:
        _compare_baseline_metadata(
            baseline_metadata_path,
            metadata_count,
            actual_metadata_sha,
            manifest,
            errors,
        )
    return ProfileArtifactValidation(
        profile_name=profile.name,
        chunk_count=manifest_count,
        index_count=index_count,
        embedding_dimension=index_dimension,
        model_name=_optional_string(manifest.get("model_name")),
        model_revision=_optional_string(manifest.get("model_revision")),
        index_sha256=actual_index_sha,
        metadata_sha256=actual_metadata_sha,
        manifest_sha256=sha256_file(manifest_path),
        symbol_sha256=actual_symbol_sha,
        errors=tuple(errors),
    )


def profile_artifact_errors(
    profile: RuntimeProfile,
    data_root: Path,
) -> tuple[str, ...]:
    """Return CLI-compatible profile artifact errors."""
    return validate_profile_artifacts(profile, data_root).errors


def sha256_file(path: Path) -> str:
    """Return the complete SHA-256 digest of one local file."""
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_dataset_profile_artifact_manifest(
    *,
    profile: RuntimeProfile,
    dataset_name: str,
    dataset_slug: str,
    record_count: int,
    input_metadata_path: Path,
    paths: ProfileArtifactPaths,
) -> DatasetProfileArtifactManifest:
    """Create dataset-local identity from completed, already-validated files."""
    from python_doc_rag.embedding_models import retrieval_embedding_spec

    if profile.embedding_model_key is None:
        raise ValueError("profile does not use a retrieval embedding")
    spec = retrieval_embedding_spec(profile.embedding_model_key)
    index_path = _present(paths.embedding_index_path)
    metadata_path = _present(paths.embedding_metadata_path)
    embedding_manifest_path = _present(paths.embedding_manifest_path)
    symbol_path = _present(paths.symbol_index_path)
    return DatasetProfileArtifactManifest(
        schema_revision=PROFILE_ARTIFACT_MANIFEST_REVISION,
        profile_revision=profile.revision,
        dataset_name=dataset_name,
        dataset_slug=dataset_slug,
        record_count=record_count,
        input_metadata_sha256=sha256_file(input_metadata_path),
        embedding_index_path=index_path.relative_to(paths.data_root).as_posix(),
        embedding_index_sha256=sha256_file(index_path),
        embedding_metadata_path=metadata_path.relative_to(paths.data_root).as_posix(),
        embedding_metadata_sha256=sha256_file(metadata_path),
        embedding_manifest_path=embedding_manifest_path.relative_to(
            paths.data_root
        ).as_posix(),
        embedding_manifest_sha256=sha256_file(embedding_manifest_path),
        symbol_index_path=symbol_path.relative_to(paths.data_root).as_posix(),
        symbol_index_sha256=sha256_file(symbol_path),
        model_name=spec.model_name,
        model_revision=spec.revision,
        embedding_dimension=spec.embedding_dimension,
        normalized_embeddings=spec.normalize_embeddings,
        query_prefix=spec.query_prefix,
        document_prefix=spec.document_prefix,
        trust_remote_code=spec.trust_remote_code,
        created_at=datetime.now(UTC).isoformat(),
    )


def write_dataset_profile_artifact_manifest_atomic(
    manifest: DatasetProfileArtifactManifest,
    path: Path,
) -> None:
    """Write dataset-local profile identity atomically."""
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
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
        os.replace(name, destination)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _generic_dataset_paths(
    profile: RuntimeProfile,
    root: Path,
    *,
    allow_pending_publication: bool,
) -> ProfileArtifactPaths | None:
    """Resolve dataset-neutral paths when a dataset manifest or identity exists."""
    from python_doc_rag.dataset_layout import resolve_dataset_artifacts

    resolved = resolve_dataset_artifacts(
        root,
        allow_pending_publication=allow_pending_publication,
    )
    identity_path = root / _GENERIC_PROFILE_ROOT / "artifact_manifest.json"
    uses_generic_layout = resolved.dataset_manifest is not None
    if profile.embedding_model_key is None:
        if not uses_generic_layout:
            return None
        required = (
            resolved.index_path,
            resolved.metadata_path,
            resolved.index_manifest_path,
        )
        return ProfileArtifactPaths(
            data_root=root,
            required_paths=required,
            embedding_index_path=None,
            embedding_metadata_path=None,
            embedding_manifest_path=None,
            symbol_index_path=None,
            identity_manifest_path=None,
        )

    identity = _load_identity_manifest(identity_path, None)
    if identity is not None:
        index_path = _below_data_root(
            root, identity.embedding_index_path, "embedding index"
        )
        metadata_path = _below_data_root(
            root, identity.embedding_metadata_path, "embedding metadata"
        )
        manifest_path = _below_data_root(
            root, identity.embedding_manifest_path, "embedding manifest"
        )
        symbol_path = _below_data_root(root, identity.symbol_index_path, "symbol index")
    elif uses_generic_layout:
        profile_root = _GENERIC_PROFILE_ROOT
        index_path = _below_data_root(
            root, (profile_root / "bge-m3/index.faiss").as_posix(), "embedding index"
        )
        metadata_path = _below_data_root(
            root,
            (profile_root / "bge-m3/metadata.jsonl").as_posix(),
            "embedding metadata",
        )
        manifest_path = _below_data_root(
            root,
            (profile_root / "bge-m3/manifest.json").as_posix(),
            "embedding manifest",
        )
        symbol_path = _below_data_root(
            root,
            (profile_root / "symbol_fields.jsonl").as_posix(),
            "symbol index",
        )
    else:
        return None

    return ProfileArtifactPaths(
        data_root=root,
        required_paths=(
            index_path,
            metadata_path,
            manifest_path,
            symbol_path,
            identity_path,
        ),
        embedding_index_path=index_path,
        embedding_metadata_path=metadata_path,
        embedding_manifest_path=manifest_path,
        symbol_index_path=symbol_path,
        identity_manifest_path=identity_path,
    )


def _load_identity_manifest(
    path: Path | None,
    errors: list[str] | None,
) -> DatasetProfileArtifactManifest | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = _read_json_object(path)
        expected_keys = {field.name for field in fields(DatasetProfileArtifactManifest)}
        unknown = sorted(set(payload).difference(expected_keys))
        missing = sorted(expected_keys.difference(payload))
        if unknown or missing:
            raise ValueError(
                "dataset profile artifact manifest schema mismatch; "
                f"unknown={unknown}, missing={missing}"
            )
        return DatasetProfileArtifactManifest(**payload)
    except (TypeError, ValueError) as error:
        if errors is not None:
            errors.append(f"dataset profile artifact manifestが不正です: {error}")
            return None
        raise


def _validate_identity_manifest(
    identity: DatasetProfileArtifactManifest,
    profile: RuntimeProfile,
    spec: Any,
    paths: ProfileArtifactPaths,
    actual_index_sha: str | None,
    actual_metadata_sha: str | None,
    actual_symbol_sha: str | None,
    manifest_count: int | None,
    errors: list[str],
) -> None:
    expected_values = {
        "schema_revision": PROFILE_ARTIFACT_MANIFEST_REVISION,
        "profile_revision": profile.revision,
        "model_name": spec.model_name,
        "model_revision": spec.revision,
        "embedding_dimension": spec.embedding_dimension,
        "normalized_embeddings": spec.normalize_embeddings,
        "query_prefix": spec.query_prefix,
        "document_prefix": spec.document_prefix,
        "trust_remote_code": spec.trust_remote_code,
    }
    for label, expected in expected_values.items():
        actual = getattr(identity, label)
        if actual != expected:
            errors.append(
                f"dataset profile artifact manifestの{label}が不一致です: "
                f"{actual!r} != {expected!r}"
            )
    path_values = {
        "embedding_index_path": paths.embedding_index_path,
        "embedding_metadata_path": paths.embedding_metadata_path,
        "embedding_manifest_path": paths.embedding_manifest_path,
        "symbol_index_path": paths.symbol_index_path,
    }
    for label, path in path_values.items():
        if path is None:
            continue
        actual = path.relative_to(paths.data_root).as_posix()
        expected = getattr(identity, label)
        if actual != expected:
            errors.append(
                f"dataset profile artifact manifestの{label}が解決pathと"
                f"一致しません: {expected} != {actual}"
            )
    hashes = {
        "embedding_index_sha256": actual_index_sha,
        "embedding_metadata_sha256": actual_metadata_sha,
        "input_metadata_sha256": actual_metadata_sha,
        "symbol_index_sha256": actual_symbol_sha,
    }
    for label, actual in hashes.items():
        expected = getattr(identity, label)
        if actual is not None and actual != expected:
            errors.append(
                f"dataset profile artifact manifestの{label}が不一致です: "
                f"{expected} != {actual}"
            )
    embedding_manifest = _present(paths.embedding_manifest_path)
    actual_manifest_sha = (
        sha256_file(embedding_manifest) if embedding_manifest.is_file() else None
    )
    if (
        actual_manifest_sha is not None
        and identity.embedding_manifest_sha256 != actual_manifest_sha
    ):
        errors.append(
            "dataset profile artifact manifestのembedding_manifest_sha256が不一致です。"
        )
    _compare_counts(
        "dataset profile artifact manifest",
        identity.record_count,
        "embedding manifest",
        manifest_count,
        errors,
    )
    try:
        from python_doc_rag.dataset_layout import resolve_dataset_artifacts

        dataset = resolve_dataset_artifacts(paths.data_root).dataset_manifest
    except ValueError as error:
        errors.append(f"dataset manifestを検証できません: {error}")
        return
    if dataset is not None:
        if identity.dataset_name != dataset.dataset_name:
            errors.append(
                "profile artifactのdataset_nameがdataset manifestと不一致です。"
            )
        if identity.dataset_slug != dataset.dataset_slug:
            errors.append(
                "profile artifactのdataset_slugがdataset manifestと不一致です。"
            )
        _compare_counts(
            "dataset manifest",
            dataset.chunk_count,
            "profile artifact manifest",
            identity.record_count,
            errors,
        )


def _below_data_root(root: Path, value: str | None, label: str) -> Path:
    relative = safe_profile_relative_path(value, label)
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"profileの{label} pathがdata-root外を参照します: {value}")
    return candidate


def _present(path: Path | None) -> Path:
    if path is None:
        raise AssertionError("embedding profile path was not resolved")
    return path


def _validation(
    profile: RuntimeProfile,
    *,
    errors: tuple[str, ...],
) -> ProfileArtifactValidation:
    return ProfileArtifactValidation(
        profile_name=profile.name,
        chunk_count=None,
        index_count=None,
        embedding_dimension=None,
        model_name=None,
        model_revision=None,
        index_sha256=None,
        metadata_sha256=None,
        manifest_sha256=None,
        symbol_sha256=None,
        errors=errors,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"{path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: JSONが不正です（{error.msg}）") from error
    if not isinstance(data, dict):
        raise ValueError(f"JSON objectではありません: {path}")
    return data


def _positive_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _safe_metadata_count(path: Path, errors: list[str]) -> int | None:
    try:
        from python_doc_rag.vector_store import load_chunks_jsonl

        count = len(load_chunks_jsonl(path))
    except Exception as error:
        errors.append(f"profile metadataを読み取れません: {_message(error)}")
        return None
    if count < 1:
        errors.append(f"profile metadataにチャンクがありません: {path}")
    return count


def _safe_symbol_count(
    metadata_path: Path,
    symbol_path: Path,
    errors: list[str],
) -> int | None:
    try:
        from python_doc_rag.technical_retrieval import load_symbol_sidecar
        from python_doc_rag.vector_store import load_chunks_jsonl

        chunks = load_chunks_jsonl(metadata_path)
        return len(load_symbol_sidecar(chunks, symbol_path))
    except Exception as error:
        errors.append(f"symbol indexを読み取れません: {_message(error)}")
        return None


def _safe_index_summary(
    path: Path,
    errors: list[str],
) -> tuple[int | None, int | None]:
    try:
        from python_doc_rag.vector_store import _import_faiss

        faiss = _import_faiss()
        index = faiss.read_index(str(path))
        count = int(index.ntotal)
        dimension = int(index.d)
    except Exception as error:
        errors.append(
            "FAISS indexを読み取れません。faiss-cpuの導入とファイル内容を"
            f"確認してください: {_message(error)}"
        )
        return None, None
    if count < 1:
        errors.append(f"FAISS indexにベクトルがありません: {path}")
    if dimension < 1:
        errors.append(f"FAISS indexの次元が不正です: {dimension}")
    return count, dimension


def _compare_file_hash(
    path: Path,
    expected: object,
    label: str,
    errors: list[str],
) -> str | None:
    if not path.is_file():
        return None
    actual = sha256_file(path)
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(f"{label}の期待SHA-256が不正です。")
    elif actual != expected:
        errors.append(f"{label}のSHA-256が不一致です: {actual} != {expected}")
    return actual


def _compare_baseline_metadata(
    baseline_path: Path,
    metadata_count: int | None,
    metadata_sha256: str | None,
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    if not baseline_path.is_file():
        errors.append(f"baseline metadataが見つかりません: {baseline_path}")
        return
    baseline_sha = sha256_file(baseline_path)
    try:
        with baseline_path.open("r", encoding="utf-8") as stream:
            baseline_count = sum(1 for line in stream if line.strip())
    except OSError as error:
        errors.append(f"baseline metadataを読み取れません: {_message(error)}")
        return
    _compare_counts(
        "baseline metadata", baseline_count, "profile metadata", metadata_count, errors
    )
    if metadata_sha256 is not None and metadata_sha256 != baseline_sha:
        errors.append(
            "baseline metadataとprofile metadataのSHA-256が一致しません: "
            f"{baseline_sha} != {metadata_sha256}"
        )
    if manifest.get("input_jsonl_sha256") != baseline_sha:
        errors.append(
            "profile manifestのinput_jsonl_sha256がbaseline metadataと一致しません。"
        )


def _compare_counts(
    first_label: str,
    first: int | None,
    second_label: str,
    second: int | None,
    errors: list[str],
) -> None:
    if first is not None and second is not None and first != second:
        errors.append(
            f"{first_label}と{second_label}の件数が一致しません: {first} != {second}"
        )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _message(error: BaseException) -> str:
    return str(error).strip() or error.__class__.__name__
