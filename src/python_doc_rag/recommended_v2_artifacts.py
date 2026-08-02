"""Prepare the complete Git-ignored artifact set required by recommended-v2."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from python_doc_rag.embedding_models import retrieval_embedding_spec
from python_doc_rag.profile_artifacts import (
    DatasetProfileArtifactManifest,
    ProfileArtifactPaths,
    ProfileArtifactValidation,
    create_dataset_profile_artifact_manifest,
    profile_artifact_paths,
    safe_profile_relative_path,
    sha256_file,
    validate_profile_artifacts,
    write_dataset_profile_artifact_manifest_atomic,
)
from python_doc_rag.profiles import RuntimeProfile, runtime_profile
from python_doc_rag.technical_retrieval import write_symbol_sidecar_atomic
from python_doc_rag.vector_store import build_vector_index, load_chunks_jsonl

RECOMMENDED_V2_EXPECTED_CHUNK_COUNT = 8_677
BASELINE_METADATA_RELATIVE_PATH = Path("indexes/python_3_13_ja_metadata.jsonl")
_PROTECTED_BASELINE_RELATIVE_PATHS = (
    Path("data/processed/python_3_13_ja_chunks.jsonl"),
    Path("indexes/python_3_13_ja.faiss"),
    BASELINE_METADATA_RELATIVE_PATH,
    Path("indexes/python_3_13_ja_index_manifest.json"),
)


class ArtifactPreparationError(RuntimeError):
    """Represent a safe, actionable artifact preparation refusal."""


@dataclass(frozen=True, slots=True)
class RecommendedV2PreparationResult:
    """Paths and validation details for created or reused artifacts."""

    reused_existing: bool
    paths: ProfileArtifactPaths
    validation: ProfileArtifactValidation


def adopt_legacy_recommended_v2_artifacts(
    data_root: Path,
    *,
    dataset_name: str = "Python 3.13 Japanese documentation",
    dataset_slug: str = "python-3-13-ja",
) -> RecommendedV2PreparationResult:
    """Add dataset-local identity to valid legacy artifacts without rebuilding."""
    root = data_root.expanduser().resolve()
    profile = runtime_profile("recommended-v2")
    identity_path = root / "profiles/recommended-v2/artifact_manifest.json"
    if identity_path.exists():
        validation = validate_profile_artifacts(
            profile,
            root,
            expected_chunk_count=RECOMMENDED_V2_EXPECTED_CHUNK_COUNT,
            baseline_metadata_path=root / BASELINE_METADATA_RELATIVE_PATH,
        )
        _require_valid(validation, "既存legacy adoption")
        return RecommendedV2PreparationResult(
            True,
            profile_artifact_paths(profile, root),
            validation,
        )
    legacy_paths = profile_artifact_paths(profile, root)
    baseline = root / BASELINE_METADATA_RELATIVE_PATH
    validation = validate_profile_artifacts(
        profile,
        root,
        expected_chunk_count=RECOMMENDED_V2_EXPECTED_CHUNK_COUNT,
        baseline_metadata_path=baseline,
    )
    _require_valid(validation, "legacy recommended-v2 artifact")
    identity = create_dataset_profile_artifact_manifest(
        profile=profile,
        dataset_name=dataset_name,
        dataset_slug=dataset_slug,
        record_count=RECOMMENDED_V2_EXPECTED_CHUNK_COUNT,
        input_metadata_path=baseline,
        paths=legacy_paths,
    )
    write_dataset_profile_artifact_manifest_atomic(identity, identity_path)
    final_paths = profile_artifact_paths(profile, root)
    final_validation = validate_profile_artifacts(
        profile,
        root,
        expected_chunk_count=RECOMMENDED_V2_EXPECTED_CHUNK_COUNT,
        baseline_metadata_path=baseline,
    )
    _require_valid(final_validation, "adopt済みlegacy artifact")
    return RecommendedV2PreparationResult(False, final_paths, final_validation)


def prepare_dataset_recommended_v2_artifacts(
    data_root: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    validate_only: bool = False,
) -> RecommendedV2PreparationResult:
    """Prepare recommended-v2 artifacts for a generic dataset manifest."""
    from python_doc_rag.dataset_layout import resolve_dataset_artifacts

    root = data_root.expanduser().resolve()
    resolved = resolve_dataset_artifacts(root)
    dataset = resolved.dataset_manifest
    if dataset is None:
        raise ArtifactPreparationError(
            "generic recommended-v2 preparation requires dataset_manifest.json"
        )
    profile = runtime_profile("recommended-v2")
    paths = profile_artifact_paths(profile, root)
    target_root = root / "profiles/recommended-v2"
    if all(path.is_file() for path in paths.required_paths):
        validation = validate_profile_artifacts(
            profile,
            root,
            expected_chunk_count=dataset.chunk_count,
            baseline_metadata_path=resolved.metadata_path,
        )
        _require_valid(validation, "既存dataset profile artifact")
        return RecommendedV2PreparationResult(True, paths, validation)
    if target_root.exists():
        raise ArtifactPreparationError(
            "dataset profile artifactがpartialまたは不整合な状態です。"
            f"暗黙には上書きしません: {target_root}"
        )
    if validate_only:
        raise ArtifactPreparationError(
            "--validate-onlyを指定しましたがdataset profile artifactがありません。"
        )
    chunks = load_chunks_jsonl(resolved.metadata_path)
    if len(chunks) != dataset.chunk_count or not chunks:
        raise ArtifactPreparationError(
            "dataset metadataとdataset manifestのchunk件数が一致しません: "
            f"{len(chunks)} != {dataset.chunk_count}"
        )
    spec = retrieval_embedding_spec(_required_model_key(profile))
    if spec.trust_remote_code or not spec.normalize_embeddings:
        raise ArtifactPreparationError("recommended-v2 embedding spec is unsafe")

    target_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            dir=target_root.parent,
            prefix=".recommended-v2.prepare-",
        )
    )
    staged_index = staging_root / "bge-m3/index.faiss"
    staged_metadata = staging_root / "bge-m3/metadata.jsonl"
    staged_manifest = staging_root / "bge-m3/manifest.json"
    staged_symbol = staging_root / "symbol_fields.jsonl"
    staged_identity = staging_root / "artifact_manifest.json"
    published = False
    try:
        symbol_summary = write_symbol_sidecar_atomic(chunks, staged_symbol)
        build_result = build_vector_index(
            resolved.metadata_path,
            staged_index,
            staged_metadata,
            staged_manifest,
            model_name=spec.model_name,
            model_revision=spec.revision,
            batch_size=batch_size,
            device=device,
            document_prefix=spec.document_prefix,
            query_prefix=spec.query_prefix,
            trust_remote_code=spec.trust_remote_code,
        )
        if build_result.chunk_count != dataset.chunk_count:
            raise ArtifactPreparationError("built profile index count is inconsistent")
        if build_result.embedding_dimension != spec.embedding_dimension:
            raise ArtifactPreparationError(
                "built profile index dimension is inconsistent"
            )
        _record_reproducible_manifest(
            staged_manifest,
            spec=spec,
            baseline_metadata=resolved.metadata_path,
            input_jsonl_label=resolved.metadata_path.relative_to(root).as_posix(),
        )
        staged_paths = ProfileArtifactPaths(
            data_root=root,
            required_paths=(
                staged_index,
                staged_metadata,
                staged_manifest,
                staged_symbol,
                staged_identity,
            ),
            embedding_index_path=staged_index,
            embedding_metadata_path=staged_metadata,
            embedding_manifest_path=staged_manifest,
            symbol_index_path=staged_symbol,
            identity_manifest_path=staged_identity,
        )
        identity = create_dataset_profile_artifact_manifest(
            profile=profile,
            dataset_name=dataset.dataset_name,
            dataset_slug=dataset.dataset_slug,
            record_count=dataset.chunk_count,
            input_metadata_path=resolved.metadata_path,
            paths=staged_paths,
        )
        identity = _identity_with_published_paths(identity)
        if identity.symbol_index_sha256 != symbol_summary.sha256:
            raise ArtifactPreparationError("symbol sidecar SHA-256 is inconsistent")
        write_dataset_profile_artifact_manifest_atomic(identity, staged_identity)
        if target_root.exists():
            raise ArtifactPreparationError(
                f"profile artifact公開先が処理中に作成されました: {target_root}"
            )
        os.rename(staging_root, target_root)
        published = True
        validation = validate_profile_artifacts(
            profile,
            root,
            expected_chunk_count=dataset.chunk_count,
            baseline_metadata_path=resolved.metadata_path,
        )
        _require_valid(validation, "公開後dataset profile artifact")
    except BaseException:
        if published and target_root.exists() and not staging_root.exists():
            os.rename(target_root, staging_root)
            published = False
        raise
    finally:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)
    return RecommendedV2PreparationResult(
        False,
        profile_artifact_paths(profile, root),
        validation,
    )


def prepare_recommended_v2_artifacts(
    data_root: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    validate_only: bool = False,
    profile: RuntimeProfile | None = None,
) -> RecommendedV2PreparationResult:
    """Build all recommended-v2 artifacts once, or validate an existing set."""
    selected_profile = profile or runtime_profile("recommended-v2")
    if selected_profile.revision != "recommended-v2":
        raise ArtifactPreparationError(
            "artifact準備はrecommended-v2 profileだけを対象とします。"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be at least 1")

    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise ArtifactPreparationError(f"data-rootが存在しません: {root}")
    paths = profile_artifact_paths(selected_profile, root)
    phase_relative = _phase_relative_path(selected_profile)
    target_phase = (root / phase_relative).resolve()
    _validate_output_scope(selected_profile, root, phase_relative, paths)
    baseline_metadata = (root / BASELINE_METADATA_RELATIVE_PATH).resolve()
    if not baseline_metadata.is_file():
        raise ArtifactPreparationError(
            f"baseline metadataが見つかりません: {baseline_metadata}"
        )

    existing_files = tuple(path.is_file() for path in paths.required_paths)
    if all(existing_files):
        validation = validate_profile_artifacts(
            selected_profile,
            root,
            expected_chunk_count=RECOMMENDED_V2_EXPECTED_CHUNK_COUNT,
            baseline_metadata_path=baseline_metadata,
        )
        _require_valid(validation, "既存artifact")
        return RecommendedV2PreparationResult(True, paths, validation)
    if any(path.exists() for path in paths.required_paths) or target_phase.exists():
        present = [str(path) for path in paths.required_paths if path.exists()]
        details = "\n".join(f"- {path}" for path in present) or f"- {target_phase}"
        raise ArtifactPreparationError(
            "recommended-v2 artifactがpartialまたは不整合な状態です。"
            "暗黙には上書きしません。既存内容を確認してください。\n"
            f"{details}"
        )
    if validate_only:
        raise ArtifactPreparationError(
            "--validate-onlyを指定しましたがrecommended-v2 artifactがありません。"
        )

    chunks = load_chunks_jsonl(baseline_metadata)
    if len(chunks) != RECOMMENDED_V2_EXPECTED_CHUNK_COUNT:
        raise ArtifactPreparationError(
            "baseline metadataのchunk件数が不一致です: "
            f"{len(chunks)} != {RECOMMENDED_V2_EXPECTED_CHUNK_COUNT}"
        )
    spec = retrieval_embedding_spec(_required_model_key(selected_profile))
    if spec.trust_remote_code or not spec.normalize_embeddings:
        raise ArtifactPreparationError(
            "recommended-v2 embedding specの安全設定が不正です。"
        )

    target_phase.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            dir=target_phase.parent,
            prefix=f".{target_phase.name}.prepare-",
        )
    )
    final_validation: ProfileArtifactValidation | None = None
    try:
        staging_root = staging_directory / "data-root"
        staged_paths = profile_artifact_paths(selected_profile, staging_root)
        staged_phase = staging_root / phase_relative
        symbol_path = _required_path(staged_paths.symbol_index_path)
        write_symbol_sidecar_atomic(chunks, symbol_path)
        build_result = build_vector_index(
            baseline_metadata,
            _required_path(staged_paths.embedding_index_path),
            _required_path(staged_paths.embedding_metadata_path),
            _required_path(staged_paths.embedding_manifest_path),
            model_name=spec.model_name,
            model_revision=spec.revision,
            batch_size=batch_size,
            device=device,
            document_prefix=spec.document_prefix,
            query_prefix=spec.query_prefix,
            trust_remote_code=spec.trust_remote_code,
        )
        if build_result.embedding_dimension != spec.embedding_dimension:
            raise ArtifactPreparationError(
                "built indexの次元がpinned BGE-M3 specと一致しません: "
                f"{build_result.embedding_dimension} != {spec.embedding_dimension}"
            )
        _record_reproducible_manifest(
            _required_path(staged_paths.embedding_manifest_path),
            spec=spec,
            baseline_metadata=baseline_metadata,
            input_jsonl_label=BASELINE_METADATA_RELATIVE_PATH.as_posix(),
        )
        staged_validation = validate_profile_artifacts(
            selected_profile,
            staging_root,
            expected_chunk_count=RECOMMENDED_V2_EXPECTED_CHUNK_COUNT,
            baseline_metadata_path=baseline_metadata,
        )
        _require_valid(staged_validation, "staging artifact")
        if target_phase.exists():
            raise ArtifactPreparationError(
                f"公開先が処理中に作成されたため停止しました: {target_phase}"
            )
        os.rename(staged_phase, target_phase)
        try:
            final_validation = validate_profile_artifacts(
                selected_profile,
                root,
                expected_chunk_count=RECOMMENDED_V2_EXPECTED_CHUNK_COUNT,
                baseline_metadata_path=baseline_metadata,
            )
            _require_valid(final_validation, "公開後artifact")
        except BaseException:
            if target_phase.exists() and not staged_phase.exists():
                staged_phase.parent.mkdir(parents=True, exist_ok=True)
                os.rename(target_phase, staged_phase)
            raise
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)

    final_paths = profile_artifact_paths(selected_profile, root)
    if final_validation is None:
        raise AssertionError("published artifacts were not validated")
    return RecommendedV2PreparationResult(False, final_paths, final_validation)


def _phase_relative_path(profile: RuntimeProfile) -> Path:
    symbol_relative = safe_profile_relative_path(
        profile.symbol_index_path,
        "symbol index",
    )
    phase_relative = symbol_relative.parent
    if phase_relative == Path("."):
        raise ArtifactPreparationError(
            "symbol indexは専用のartifact directory内に配置する必要があります。"
        )
    return phase_relative


def _validate_output_scope(
    profile: RuntimeProfile,
    root: Path,
    phase_relative: Path,
    paths: ProfileArtifactPaths,
) -> None:
    protected = {
        (root / relative).resolve() for relative in _PROTECTED_BASELINE_RELATIVE_PATHS
    }
    for path in paths.required_paths:
        if path in protected:
            raise ArtifactPreparationError(
                f"baseline artifactを出力先にはできません: {path}"
            )
    for value in profile.required_artifacts:
        relative = safe_profile_relative_path(value, "required artifact")
        if not relative.is_relative_to(phase_relative):
            raise ArtifactPreparationError(
                f"recommended-v2 required artifactが単一のatomic公開範囲外です: {value}"
            )


def _record_reproducible_manifest(
    manifest_path: Path,
    *,
    spec: Any,
    baseline_metadata: Path,
    input_jsonl_label: str,
) -> None:
    manifest = _read_json_object(manifest_path)
    manifest.update(
        {
            "experiment_revision": "recommended-v2-artifact-setup-v1",
            "model_key": spec.key,
            "model_name": spec.model_name,
            "model_revision": spec.revision,
            "license": spec.license,
            "model_card_url": spec.model_card_url,
            "language_evidence": spec.language_evidence,
            "embedding_dimension": spec.embedding_dimension,
            "max_sequence_length": spec.max_sequence_length,
            "pooling": spec.pooling,
            "normalized_embeddings": spec.normalize_embeddings,
            "query_prefix": spec.query_prefix,
            "document_prefix": spec.document_prefix,
            "trust_remote_code": spec.trust_remote_code,
            "input_jsonl": input_jsonl_label,
            "input_jsonl_sha256": sha256_file(baseline_metadata),
            "openai_api_used": False,
            "contains_secrets": False,
        }
    )
    _write_json_atomic(manifest, manifest_path)


def _identity_with_published_paths(
    identity: DatasetProfileArtifactManifest,
) -> DatasetProfileArtifactManifest:
    return replace(
        identity,
        embedding_index_path="profiles/recommended-v2/bge-m3/index.faiss",
        embedding_metadata_path="profiles/recommended-v2/bge-m3/metadata.jsonl",
        embedding_manifest_path="profiles/recommended-v2/bge-m3/manifest.json",
        symbol_index_path="profiles/recommended-v2/symbol_fields.jsonl",
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ArtifactPreparationError(f"manifestがJSON objectではありません: {path}")
    return data


def _write_json_atomic(data: dict[str, Any], path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _require_valid(
    validation: ProfileArtifactValidation,
    label: str,
) -> None:
    if validation.succeeded:
        return
    details = "\n".join(f"- {error}" for error in validation.errors)
    raise ArtifactPreparationError(f"{label}の検証に失敗しました。\n{details}")


def _required_model_key(profile: RuntimeProfile) -> str:
    if profile.embedding_model_key is None:
        raise ArtifactPreparationError("profileにembedding model keyがありません。")
    return profile.embedding_model_key


def _required_path(path: Path | None) -> Path:
    if path is None:
        raise ArtifactPreparationError("profileに必要なartifact pathがありません。")
    return path
