"""One atomic preparation pipeline for independently configured ingestion."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from python_doc_rag.dataset_layout import (
    DATASET_MANIFEST_REVISION,
    DatasetManifest,
    generic_dataset_manifest,
    load_dataset_manifest,
    write_dataset_manifest_atomic,
)
from python_doc_rag.ingestion.orchestration import ingest_documents
from python_doc_rag.ingestion.protocols import DocumentLoader, IngestionResult
from python_doc_rag.ingestion.registry import (
    LoaderRuntimeOptions,
    build_document_loader,
    build_document_parser,
    validate_loader_parser_combination,
)
from python_doc_rag.ingestion.serialization import write_chunks_jsonl_atomic
from python_doc_rag.loaders.zip_archive import (
    SafeZipHtmlLoader,
    recover_snapshot_http_archive_refresh,
    snapshot_http_archive_refresh_candidate_is_committed,
    validate_snapshot_http_archive_cache,
)
from python_doc_rag.loading import (
    bounded_http_refresh_candidate_is_committed,
    recover_bounded_http_refresh,
    validate_bounded_http_cache,
    validate_bounded_http_replayable_cache,
)
from python_doc_rag.profile_artifacts import (
    profile_artifact_paths,
    sha256_file,
    validate_profile_artifacts,
)
from python_doc_rag.profiles import runtime_profile
from python_doc_rag.recommended_v2_artifacts import (
    ArtifactPreparationError,
    RecommendedV2PreparationResult,
    prepare_dataset_recommended_v2_artifacts,
)
from python_doc_rag.site_config import (
    BoundedHttpLoaderSettings,
    LocalCompatibilityLoaderSettings,
    PinnedLocalArchiveLoaderSettings,
    SiteConfig,
    SnapshotHttpArchiveLoaderSettings,
    load_site_config,
)
from python_doc_rag.source_identity import (
    PROCESSING_CONFIG_FINGERPRINT_REVISION,
    SOURCE_CONFIG_FINGERPRINT_REVISION,
    processing_config_sha256,
    source_config_payload,
    source_config_sha256,
    source_snapshot_sha256,
)
from python_doc_rag.vector_store import VectorIndexBuildResult, build_vector_index

PreparationTarget = Literal["corpus", "index", "profile", "all"]
_PUBLICATION_JOURNAL_REVISION = "prepare-publication-v1"
_PUBLICATION_JOURNAL_NAME = ".prepare-publication.json"
_REBUILD_RECOVERY_REVISION = "prepare-rebuild-recovery-v1"
_REBUILD_RECOVERY_NAME = ".prepare-rebuild-recovery.json"
_MANAGED_DIRECTORIES = (
    Path("data"),
    Path("data/raw"),
    Path("data/raw/archives"),
    Path("data/raw/extracted"),
    Path("data/raw/pages"),
    Path("data/processed"),
    Path("data/.raw.staging"),
    Path("data/.raw.staging/pages"),
    Path("data/.raw.backup"),
    Path("data/.raw.failed-refresh"),
    Path("data/raw/.source-refresh-rollback"),
    Path("indexes"),
    Path("profiles"),
    Path("profiles/recommended-v2"),
    Path(".prepare-staging"),
    Path(".prepare-staging/data"),
    Path(".prepare-staging/data/processed"),
    Path(".prepare-staging/indexes"),
    Path(".prepare-staging/profiles"),
    Path(".prepare-staging/profiles/recommended-v2"),
)
_MANAGED_FILES = (
    Path("dataset_manifest.json"),
    Path(_PUBLICATION_JOURNAL_NAME),
    Path(_REBUILD_RECOVERY_NAME),
    Path("data/raw/source.lock.json"),
    Path("data/raw/fetch_manifest.json"),
    Path("data/processed/chunks.jsonl"),
    Path("data/processed/corpus_manifest.json"),
    Path("indexes/dense.faiss"),
    Path("indexes/metadata.jsonl"),
    Path("indexes/index_manifest.json"),
    Path("data/.raw.staging/fetch_manifest.json"),
    Path(".prepare-staging/prepare_state.json"),
    Path(".prepare-staging/dataset_manifest.json"),
    Path(".prepare-staging/data/processed/chunks.jsonl"),
    Path(".prepare-staging/data/processed/corpus_manifest.json"),
    Path(".prepare-staging/indexes/dense.faiss"),
    Path(".prepare-staging/indexes/metadata.jsonl"),
    Path(".prepare-staging/indexes/index_manifest.json"),
)
_LEGACY_FIXED_ARTIFACTS = (
    Path("data/processed/python_3_13_ja_chunks.jsonl"),
    Path("indexes/python_3_13_ja.faiss"),
    Path("indexes/python_3_13_ja_metadata.jsonl"),
    Path("indexes/python_3_13_ja_index_manifest.json"),
)
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


@dataclass(frozen=True, slots=True)
class DatasetPreparationResult:
    """Summary of prepared or safely reused dataset artifacts."""

    data_root: Path
    dataset_manifest: DatasetManifest
    source_page_count: int
    parsed_page_count: int
    failed_page_count: int
    section_count: int
    chunk_count: int
    excluded_section_count: int
    excluded_node_count: int
    code_block_count: int
    table_count: int
    fallback_page_count: int
    corpus_sha256: str
    dense_index_result: VectorIndexBuildResult | None
    profile_result: RecommendedV2PreparationResult | None
    reused_dataset: bool
    source_cache_reused: bool
    source_snapshot_sha256: str


def prepare_dataset(
    site_config_path: Path,
    data_root: Path,
    *,
    source_root: Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    rebuild: bool = False,
    resume: bool = False,
    until: PreparationTarget = "all",
    device: str = "cuda",
) -> DatasetPreparationResult:
    """Load, parse, chunk, index, and prepare one configured dataset."""
    if until not in {"corpus", "index", "profile", "all"}:
        raise ValueError(f"unsupported preparation target: {until}")
    if refresh and rebuild:
        raise ValueError("--refresh and --rebuild are mutually exclusive")
    if refresh and offline:
        raise ValueError("--offline and --refresh cannot be used together")
    config_path = site_config_path.expanduser().resolve(strict=True)
    config = load_site_config(config_path)
    validate_loader_parser_combination(config.loader, config.parser)
    _validate_runtime_source_options(config, source_root=source_root, refresh=refresh)
    root = data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _validate_managed_data_root(root)
    source_config = source_config_sha256(
        config.loader,
        category=config.dataset.slug,
    )
    processing_config = processing_config_sha256(config)
    rebuild_recovery = _pending_rebuild_recovery(root)
    pending_publication = _pending_derived_publication(root)
    if (
        rebuild_recovery is not None
        and rebuild_recovery["complete"] is True
        and pending_publication is not None
        and pending_publication["phase"] == "committed"
    ):
        candidate_valid, _candidate_error = (
            _validate_committed_publication_candidate(
                config,
                config_path=config_path,
                root=root,
                publication=pending_publication,
                source_config=source_config,
                processing_config=processing_config,
                source_root=source_root,
            )
        )
        if candidate_valid:
            _complete_validated_committed_publication(
                config,
                root=root,
                publication=pending_publication,
                source_config=source_config,
            )
    if rebuild_recovery is not None:
        if _rebuild_recovery_generation_is_committed(
            config,
            config_path=config_path,
            root=root,
            marker=rebuild_recovery,
            source_config=source_config,
            processing_config=processing_config,
            source_root=source_root,
        ):
            _finalize_rebuild_recovery(root)
            rebuild_recovery = None
        elif not rebuild and not refresh:
            raise RuntimeError(
                "a recoverable derived rebuild is pending; rerun prepare with "
                "--rebuild, or --refresh --resume for a refreshable source"
            )
        else:
            rebuild_recovery["until"] = until
            _write_json_atomic(
                rebuild_recovery,
                root / _REBUILD_RECOVERY_NAME,
            )
            _complete_rebuild_recovery_quarantine(root, rebuild_recovery)
    pending_publication = _pending_derived_publication(root)
    if pending_publication is not None and pending_publication["phase"] == "committed":
        candidate_valid, candidate_error = _validate_committed_publication_candidate(
            config,
            config_path=config_path,
            root=root,
            publication=pending_publication,
            source_config=source_config,
            processing_config=processing_config,
            source_root=source_root,
        )
        if candidate_valid:
            _complete_validated_committed_publication(
                config,
                root=root,
                publication=pending_publication,
                source_config=source_config,
            )
        else:
            if _loader_source_rollback_available(config, root=root):
                source_was_rolled_back = not _recover_loader_publication(
                    config,
                    root=root,
                    source_config=source_config,
                    force_rollback=True,
                )
                if not source_was_rolled_back:
                    raise RuntimeError("source publication rollback was not completed")
                _mark_derived_publication_for_rollback(root, pending_publication)
                _recover_derived_publication(root)
            elif _current_source_matches_backed_up_dataset(
                root,
                pending_publication,
            ):
                _mark_derived_publication_for_rollback(root, pending_publication)
                _recover_derived_publication(root)
            elif refresh or (
                rebuild
                and _loader_current_source_is_replayable(
                    config,
                    root=root,
                    source_config=source_config,
                )
            ):
                _begin_rebuild_recovery_quarantine(root, pending_publication)
                pending_publication = None
            else:
                error = RuntimeError(
                    "committed source has no rollback backup and the derived "
                    "generation is invalid; recoverable derived backups were retained"
                )
                if candidate_error is not None:
                    raise error from candidate_error
                raise error
    else:
        _recover_derived_publication(root)
        _recover_loader_publication(
            config,
            root=root,
            source_config=source_config,
            force_rollback=pending_publication is not None,
        )
    existing_manifest_path = root / "dataset_manifest.json"
    if not existing_manifest_path.is_file():
        _reject_manifestless_legacy_artifacts(root)
    existing = (
        load_dataset_manifest(existing_manifest_path)
        if existing_manifest_path.is_file()
        else None
    )
    staging = root / ".prepare-staging"
    state = {
        "schema_revision": "prepare-staging-v1",
        "source_config_sha256": source_config,
        "processing_config_sha256": processing_config,
        "until": until,
    }
    if staging.exists():
        if not resume:
            raise RuntimeError(
                f"partial preparation staging exists; use --resume: {staging}"
            )
        if _read_json_object(staging / "prepare_state.json") != state:
            raise RuntimeError("preparation staging does not match the requested settings")
    if existing is not None:
        _validate_existing_source_identity(
            existing,
            config,
            source_config=source_config,
            refresh=refresh,
        )
        if existing.schema_revision != DATASET_MANIFEST_REVISION and (
            refresh
            or rebuild
            or staging.exists()
            or not _requested_artifacts_complete(config, root, existing, until=until)
        ):
            raise RuntimeError(
                "legacy dataset cannot be upgraded in place; use a different data-root"
            )
        if not staging.exists() and not refresh and not rebuild and _requested_artifacts_complete(
            config,
            root,
            existing,
            until=until,
        ):
            _validate_processing_identity(
                existing,
                config,
                processing_config=processing_config,
            )
            _validate_committed_generation(
                config,
                config_path=config_path,
                root=root,
                manifest=existing,
                source_config=source_config,
                processing_config=processing_config,
                source_root=source_root,
                until=until,
            )
            return _reuse_prepared_dataset(
                config,
                root,
                existing,
                until=until,
                device=device,
            )
        if not refresh and not rebuild:
            _validate_processing_identity(
                existing,
                config,
                processing_config=processing_config,
            )

    if not staging.exists():
        staging.mkdir(parents=True)
        _write_json_atomic(state, staging / "prepare_state.json")

    runtime = LoaderRuntimeOptions(
        data_root=root,
        config_directory=config_path.parent,
        dataset_slug=config.dataset.slug,
        source_config_sha256=source_config,
        source_root=source_root,
        offline=offline or rebuild,
        refresh=refresh,
        resume=resume,
    )
    loader = build_document_loader(config.loader, runtime)
    parser = build_document_parser(config.parser)
    source_manifest_path = root / "data/raw/fetch_manifest.json"
    previous_source_manifest = (
        source_manifest_path.read_bytes() if source_manifest_path.is_file() else None
    )
    publication: dict[str, Any] | None = None
    try:
        documents = tuple(loader.load())
        snapshot_identity = _loader_snapshot_identity(loader, documents)
        if (
            existing is not None
            and existing.schema_revision == DATASET_MANIFEST_REVISION
            and existing.source_snapshot_sha256 != snapshot_identity
            and not refresh
        ):
            raise RuntimeError(
                "local source content does not match the prepared dataset; "
                "use a different data-root or refresh-capable source"
            )
        _augment_source_manifest(
            source_manifest_path,
            config=config,
            documents=documents,
            source_config=source_config,
            processing_config=processing_config,
            source_snapshot=snapshot_identity,
        )
        ingestion = ingest_documents(
            documents,
            parser,
            chunking_config=config.chunking,
        )
        if not ingestion.chunks:
            raise RuntimeError("configured ingestion produced no searchable chunks")
        staged_chunks = staging / "data/processed/chunks.jsonl"
        staged_corpus_manifest = staging / "data/processed/corpus_manifest.json"
        write_chunks_jsonl_atomic(ingestion.chunks, staged_chunks)
        _write_json_atomic(
            _corpus_manifest(
                config,
                documents,
                ingestion,
                staged_chunks,
                source_config=source_config,
                processing_config=processing_config,
                source_snapshot=snapshot_identity,
            ),
            staged_corpus_manifest,
        )

        dense_result = _prepare_dense_index(
            config,
            staging,
            staged_chunks,
            until=until,
            device=device,
        )
        manifest = generic_dataset_manifest(
            dataset_name=config.dataset.name,
            dataset_slug=config.dataset.slug,
            loader_type=config.loader.type,
            parser_type=config.parser.type,
            site_config_sha256=config.config_sha256,
            source_config_sha256=source_config,
            processing_config_sha256=processing_config,
            source_snapshot_sha256=snapshot_identity,
            created_at=datetime.now(UTC).isoformat(),
            source_page_count=len(documents),
            section_count=len(ingestion.sections),
            chunk_count=len(ingestion.chunks),
        )
        write_dataset_manifest_atomic(manifest, staging / "dataset_manifest.json")
        profile_result = _prepare_profile(
            config,
            staging,
            until=until,
            device=device,
        )
        publication = _publish_derived_transaction(
            staging,
            root,
            include_index=dense_result is not None,
            include_profile=profile_result is not None,
            remove_unbuilt=refresh or rebuild,
        )
    except BaseException:
        rollback_errors = _rollback_preparation_transaction(
            root=root,
            loader=loader,
            publication=publication,
            source_manifest_path=source_manifest_path,
            previous_source_manifest=previous_source_manifest,
            restore_source_manifest=existing is not None,
        )
        if rollback_errors:
            raise RuntimeError(
                "preparation failed and source rollback was incomplete"
            ) from rollback_errors[0]
        raise
    try:
        _commit_loader_source(loader)
    except BaseException:
        rollback_errors = _rollback_preparation_transaction(
            root=root,
            loader=loader,
            publication=publication,
            source_manifest_path=source_manifest_path,
            previous_source_manifest=previous_source_manifest,
            restore_source_manifest=existing is not None,
        )
        if rollback_errors:
            raise RuntimeError(
                "preparation failed and source rollback was incomplete"
            ) from rollback_errors[0]
        raise
    _finalize_committed_publication(root, publication)
    shutil.rmtree(staging, ignore_errors=True)

    published_dense = (
        None
        if dense_result is None
        else VectorIndexBuildResult(
            index_path=root / manifest.dense_index_path,
            metadata_path=root / manifest.dense_metadata_path,
            manifest_path=root / manifest.dense_manifest_path,
            chunk_count=dense_result.chunk_count,
            embedding_dimension=dense_result.embedding_dimension,
            elapsed_seconds=dense_result.elapsed_seconds,
        )
    )
    published_profile: RecommendedV2PreparationResult | None = None
    if profile_result is not None:
        validated_profile = prepare_dataset_recommended_v2_artifacts(
            root,
            device=device,
            batch_size=min(config.index.embedding_batch_size, 32),
            validate_only=True,
        )
        published_profile = RecommendedV2PreparationResult(
            reused_existing=profile_result.reused_existing,
            paths=validated_profile.paths,
            validation=validated_profile.validation,
        )
    result = _result(
        root,
        manifest,
        ingestion=ingestion,
        dense_result=published_dense,
        profile_result=published_profile,
        source_cache_reused=_loader_cache_reused(loader),
    )
    _finalize_rebuild_recovery(root)
    return result


def _validate_runtime_source_options(
    config: SiteConfig,
    *,
    source_root: Path | None,
    refresh: bool,
) -> None:
    if isinstance(config.loader, LocalCompatibilityLoaderSettings):
        if source_root is None:
            raise ValueError("--source-root is required for local-html-tree")
        if refresh:
            raise ValueError(
                "--refresh is not supported for local-html-tree; use --rebuild"
            )
        return
    if source_root is not None:
        raise ValueError(f"--source-root is not accepted for {config.loader.type}")
    if refresh and isinstance(config.loader, PinnedLocalArchiveLoaderSettings):
        raise ValueError(
            "--refresh is not supported for pinned-local-archive; use --rebuild"
        )


def _pending_rebuild_recovery(root: Path) -> dict[str, Any] | None:
    """Return one validated explicit-rebuild recovery marker when present."""
    marker_path = root / _REBUILD_RECOVERY_NAME
    if not marker_path.exists() and not marker_path.is_symlink():
        return None
    payload = _read_json_object(marker_path)
    _validate_rebuild_recovery_marker(root, payload)
    return payload


def _validate_rebuild_recovery_marker(root: Path, payload: dict[str, Any]) -> None:
    if set(payload) != {"schema_revision", "complete", "backup_root", "until"}:
        raise RuntimeError("derived rebuild recovery marker schema is invalid")
    if payload.get("schema_revision") != _REBUILD_RECOVERY_REVISION:
        raise RuntimeError("derived rebuild recovery marker revision is invalid")
    if not isinstance(payload.get("complete"), bool):
        raise RuntimeError("derived rebuild recovery marker state is invalid")
    if payload.get("until") not in {"corpus", "index", "profile", "all"}:
        raise RuntimeError("derived rebuild recovery target is invalid")
    backup_root = payload.get("backup_root")
    if (
        not isinstance(backup_root, str)
        or not backup_root.startswith(".prepare-publication-backup-")
        or PurePosixPath(backup_root).parent != PurePosixPath(".")
    ):
        raise RuntimeError("derived rebuild recovery backup path is invalid")
    backup_path = _safe_publication_relative(
        root,
        backup_root,
        "derived rebuild recovery backup",
    )
    if _is_link_like(backup_path) or not backup_path.is_dir():
        raise RuntimeError("derived rebuild recovery backup is unavailable")


def _publication_validation_target(
    publication: dict[str, Any],
) -> PreparationTarget:
    """Recover the highest artifact target represented by a publication journal."""
    destinations = {
        entry["destination"]
        for entry in publication["entries"]
        if entry["source"] is not None
    }
    if "profiles/recommended-v2" in destinations:
        return "profile"
    if "indexes" in destinations:
        return "index"
    return "corpus"


def _validate_committed_publication_layout(
    root: Path,
    publication: dict[str, Any],
) -> None:
    """Require intentionally removed destinations to remain absent after commit."""
    for entry in publication["entries"]:
        if entry["source"] is not None:
            continue
        destination = root / str(entry["destination"])
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(
                f"committed publication retained an unbuilt artifact: {destination}"
            )


def _rebuild_recovery_generation_is_committed(
    config: SiteConfig,
    *,
    config_path: Path,
    root: Path,
    marker: dict[str, Any],
    source_config: str,
    processing_config: str,
    source_root: Path | None,
) -> bool:
    """Recognize a rebuild that committed before its final marker unlink."""
    if (
        (root / _PUBLICATION_JOURNAL_NAME).exists()
        or (root / _PUBLICATION_JOURNAL_NAME).is_symlink()
        or (root / ".prepare-staging").exists()
        or (root / ".prepare-staging").is_symlink()
        or _loader_source_rollback_available(config, root=root)
    ):
        return False
    try:
        manifest = load_dataset_manifest(root / "dataset_manifest.json")
        if manifest.schema_revision != DATASET_MANIFEST_REVISION:
            raise RuntimeError("rebuild recovery requires a current dataset manifest")
        _validate_committed_generation(
            config,
            config_path=config_path,
            root=root,
            manifest=manifest,
            source_config=source_config,
            processing_config=processing_config,
            source_root=source_root,
            until=marker["until"],
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _begin_rebuild_recovery_quarantine(
    root: Path,
    publication: dict[str, Any],
) -> None:
    """Preserve an unrecoverable old generation before an explicit rebuild."""
    _validate_publication_journal(root, publication)
    if publication["phase"] != "committed":
        raise RuntimeError("only a committed failed generation can be quarantined")
    marker_path = root / _REBUILD_RECOVERY_NAME
    if marker_path.exists() or marker_path.is_symlink():
        raise RuntimeError("derived rebuild recovery marker already exists")
    marker = {
        "schema_revision": _REBUILD_RECOVERY_REVISION,
        "complete": False,
        "backup_root": publication["backup_root"],
        "until": _publication_validation_target(publication),
    }
    _write_json_atomic(marker, marker_path)
    _complete_rebuild_recovery_quarantine(root, marker)


def _complete_rebuild_recovery_quarantine(
    root: Path,
    marker: dict[str, Any],
) -> None:
    """Idempotently move the failed journal/staging into its retained backup."""
    _validate_rebuild_recovery_marker(root, marker)
    backup_root = root / str(marker["backup_root"])
    journal_path = root / _PUBLICATION_JOURNAL_NAME
    retained_journal = backup_root / "recovery-journal.json"
    staging = root / ".prepare-staging"
    retained_staging = backup_root / "recovery-staging"
    publication: dict[str, Any]
    if marker["complete"] is True:
        if not retained_journal.is_file() or _is_link_like(retained_journal):
            raise RuntimeError("derived rebuild recovery journal is unavailable")
        publication = _read_json_object(retained_journal)
        _validate_publication_journal(root, publication)
        if (
            publication["phase"] != "committed"
            or publication["backup_root"] != marker["backup_root"]
        ):
            raise RuntimeError(
                "derived rebuild recovery journal identity is inconsistent"
            )
        return
    if journal_path.exists() or journal_path.is_symlink():
        if retained_journal.exists() or retained_journal.is_symlink():
            raise RuntimeError("derived rebuild recovery has conflicting journals")
        publication = _read_json_object(journal_path)
    elif retained_journal.is_file() and not _is_link_like(retained_journal):
        publication = _read_json_object(retained_journal)
    else:
        raise RuntimeError("derived rebuild recovery journal is unavailable")
    _validate_publication_journal(root, publication)
    if (
        publication["phase"] != "committed"
        or publication["backup_root"] != marker["backup_root"]
    ):
        raise RuntimeError("derived rebuild recovery journal identity is inconsistent")
    if staging.exists() or staging.is_symlink():
        if retained_staging.exists() or retained_staging.is_symlink():
            raise RuntimeError("derived rebuild recovery has conflicting staging")
        os.rename(staging, retained_staging)
    if journal_path.exists() or journal_path.is_symlink():
        os.rename(journal_path, retained_journal)
    marker["complete"] = True
    _write_json_atomic(marker, root / _REBUILD_RECOVERY_NAME)


def _finalize_rebuild_recovery(root: Path) -> None:
    """Expose a successfully rebuilt generation while retaining its old bundle."""
    marker_path = root / _REBUILD_RECOVERY_NAME
    if marker_path.exists() or marker_path.is_symlink():
        marker = _read_json_object(marker_path)
        _validate_rebuild_recovery_marker(root, marker)
        if marker["complete"] is not True:
            raise RuntimeError("derived rebuild recovery quarantine is incomplete")
        marker_path.unlink()


def _pending_derived_publication(root: Path) -> dict[str, Any] | None:
    """Return one validated pending publication journal without mutating it."""
    journal_path = root / _PUBLICATION_JOURNAL_NAME
    if not journal_path.exists() and not journal_path.is_symlink():
        return None
    payload = _read_json_object(journal_path)
    _validate_publication_journal(root, payload)
    return payload


def _validate_committed_publication_candidate(
    config: SiteConfig,
    *,
    config_path: Path,
    root: Path,
    publication: dict[str, Any],
    source_config: str,
    processing_config: str,
    source_root: Path | None,
) -> tuple[bool, Exception | None]:
    """Strictly validate a committed source and derived publication candidate."""
    try:
        source_committed = _loader_publication_candidate_is_committed(
            config,
            root=root,
            source_config=source_config,
        )
        if not source_committed:
            return False, None
        committed_manifest = load_dataset_manifest(root / "dataset_manifest.json")
        if committed_manifest.schema_revision != DATASET_MANIFEST_REVISION:
            raise RuntimeError(
                "committed publication requires a current dataset manifest"
            )
        _validate_committed_generation(
            config,
            config_path=config_path,
            root=root,
            manifest=committed_manifest,
            source_config=source_config,
            processing_config=processing_config,
            source_root=source_root,
            until=_publication_validation_target(publication),
        )
        _validate_committed_publication_layout(root, publication)
    except Exception as error:
        return False, error
    return True, None


def _complete_validated_committed_publication(
    config: SiteConfig,
    *,
    root: Path,
    publication: dict[str, Any],
    source_config: str,
) -> None:
    """Finalize a validated generation, or roll derived data back with its source."""
    source_kept = _recover_loader_publication(
        config,
        root=root,
        source_config=source_config,
    )
    if source_kept:
        _recover_derived_publication(root)
        return
    _mark_derived_publication_for_rollback(root, publication)
    _recover_derived_publication(root)


def _recover_loader_publication(
    config: SiteConfig,
    *,
    root: Path,
    source_config: str,
    force_rollback: bool = False,
) -> bool:
    """Recover a loader source swap and report whether its candidate committed."""
    if isinstance(config.loader, SnapshotHttpArchiveLoaderSettings):
        return recover_snapshot_http_archive_refresh(
            root / "data/raw",
            settings=config.loader,
            source_config_sha256=source_config,
            force_rollback=force_rollback,
        )
    if isinstance(config.loader, BoundedHttpLoaderSettings):
        return recover_bounded_http_refresh(
            root / "data/raw",
            force_rollback=force_rollback,
        )
    return True


def _loader_publication_candidate_is_committed(
    config: SiteConfig,
    *,
    root: Path,
    source_config: str,
) -> bool:
    """Validate a source candidate before a committed derived journal is finalized."""
    if isinstance(config.loader, SnapshotHttpArchiveLoaderSettings):
        return snapshot_http_archive_refresh_candidate_is_committed(
            config.loader,
            root / "data/raw",
            source_config_sha256=source_config,
        )
    if isinstance(config.loader, BoundedHttpLoaderSettings):
        backup = root / "data/.raw.backup"
        return not backup.exists() or bounded_http_refresh_candidate_is_committed(
            root / "data/raw"
        )
    return True


def _loader_source_rollback_available(config: SiteConfig, *, root: Path) -> bool:
    """Return whether the active loader still has its canonical old-source backup."""
    if isinstance(config.loader, SnapshotHttpArchiveLoaderSettings):
        path = root / "data/raw/.source-refresh-rollback"
        return path.exists() or path.is_symlink()
    if isinstance(config.loader, BoundedHttpLoaderSettings):
        path = root / "data/.raw.backup"
        return path.exists() or path.is_symlink()
    return False


def _loader_current_source_is_replayable(
    config: SiteConfig,
    *,
    root: Path,
    source_config: str,
) -> bool:
    """Validate source bytes before quarantining the only old derived generation."""
    if isinstance(config.loader, SnapshotHttpArchiveLoaderSettings):
        try:
            validate_snapshot_http_archive_cache(
                config.loader,
                root / "data/raw",
                source_config_sha256=source_config,
            )
        except (OSError, RuntimeError, ValueError):
            return False
        return True
    if isinstance(config.loader, BoundedHttpLoaderSettings):
        try:
            validate_bounded_http_replayable_cache(
                root / "data/raw",
                expected_source_config_sha256=source_config,
            )
        except (OSError, RuntimeError, ValueError):
            return False
        return True
    return True


def _mark_derived_publication_for_rollback(
    root: Path,
    payload: dict[str, Any],
) -> None:
    """Durably downgrade a failed committed candidate before restoring backups."""
    payload["phase"] = "publishing"
    _write_json_atomic(payload, root / _PUBLICATION_JOURNAL_NAME)


def _current_source_matches_backed_up_dataset(
    root: Path,
    publication: dict[str, Any],
) -> bool:
    """Identify a source rollback that completed before derived rollback failed."""
    dataset_backup: Path | None = None
    for entry in publication["entries"]:
        if entry["destination"] == "dataset_manifest.json" and entry["had_destination"]:
            candidate = root / str(entry["backup"])
            if candidate.is_file() and not _is_link_like(candidate):
                dataset_backup = candidate
            break
    if dataset_backup is None:
        return False
    try:
        previous = load_dataset_manifest(dataset_backup)
        source = _read_json_object(root / "data/raw/fetch_manifest.json")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    return (
        previous.schema_revision == DATASET_MANIFEST_REVISION
        and source.get("source_config_sha256") == previous.source_config_sha256
        and source.get("source_snapshot_sha256") == previous.source_snapshot_sha256
    )


def _validate_managed_data_root(root: Path) -> None:
    """Reject symlinked or mistyped canonical preparation destinations."""
    for relative in _MANAGED_DIRECTORIES:
        path = root / relative
        if _is_link_like(path):
            raise RuntimeError(
                f"managed data-root directory must not be a symlink/junction: {relative}"
            )
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"managed data-root directory has the wrong type: {relative}")
    for relative in _MANAGED_FILES:
        path = root / relative
        if _is_link_like(path):
            raise RuntimeError(
                f"managed data-root file must not be a symlink/junction: {relative}"
            )
        if path.exists() and not path.is_file():
            raise RuntimeError(f"managed data-root file has the wrong type: {relative}")


def _is_link_like(path: Path) -> bool:
    """Return whether a path is a symlink or native Windows junction."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _reject_manifestless_legacy_artifacts(root: Path) -> None:
    """Preserve fixed-layout legacy datasets that lack an identity manifest."""
    found = [
        path
        for path in _LEGACY_FIXED_ARTIFACTS
        if (root / path).exists() or _is_link_like(root / path)
    ]
    if found:
        names = ", ".join(path.as_posix() for path in found)
        raise RuntimeError(
            "legacy dataset artifacts exist without dataset_manifest.json; "
            f"use a different data-root: {names}"
        )


def _validate_existing_source_identity(
    existing: DatasetManifest,
    config: SiteConfig,
    *,
    source_config: str,
    refresh: bool,
) -> None:
    if existing.schema_revision == DATASET_MANIFEST_REVISION:
        if existing.source_config_sha256 != source_config and not refresh:
            hint = (
                "explicit --refresh"
                if isinstance(
                    config.loader,
                    (BoundedHttpLoaderSettings, SnapshotHttpArchiveLoaderSettings),
                )
                else "a different data-root"
            )
            raise RuntimeError(
                "existing dataset was built from a different source config; "
                f"use {hint}"
            )
        return
    if existing.site_config_sha256 != config.config_sha256 and not refresh:
        raise RuntimeError(
            "existing legacy dataset was built from a different site config; "
            "use a different data-root or explicit --refresh"
        )


def _validate_processing_identity(
    existing: DatasetManifest,
    config: SiteConfig,
    *,
    processing_config: str,
) -> None:
    if existing.schema_revision == DATASET_MANIFEST_REVISION:
        if existing.processing_config_sha256 != processing_config:
            raise RuntimeError(
                "processing settings differ from the prepared dataset; use --rebuild"
            )
    elif existing.site_config_sha256 != config.config_sha256:
        raise RuntimeError(
            "existing legacy dataset was built from a different site config"
        )


def _requested_artifacts_complete(
    config: SiteConfig,
    root: Path,
    manifest: DatasetManifest,
    *,
    until: PreparationTarget,
) -> bool:
    required = [
        root / manifest.source_manifest_path,
        root / manifest.processed_chunks_path,
        root / manifest.corpus_manifest_path,
    ]
    if until in {"index", "profile", "all"}:
        required.extend(
            (
                root / manifest.dense_index_path,
                root / manifest.dense_metadata_path,
                root / manifest.dense_manifest_path,
            )
        )
    if until in {"profile", "all"} and config.profile.prepare == "recommended-v2":
        required.extend(
            profile_artifact_paths(runtime_profile("recommended-v2"), root).required_paths
        )
    return all(path.is_file() for path in required)


def _validate_committed_generation(
    config: SiteConfig,
    *,
    config_path: Path,
    root: Path,
    manifest: DatasetManifest,
    source_config: str,
    processing_config: str,
    source_root: Path | None,
    until: PreparationTarget,
) -> None:
    """Reject a complete-looking dataset whose committed artifacts do not agree."""
    if manifest.schema_revision != DATASET_MANIFEST_REVISION:
        return
    if manifest.source_config_sha256 != source_config:
        raise RuntimeError("committed dataset source fingerprint is inconsistent")
    if manifest.processing_config_sha256 != processing_config:
        raise RuntimeError("committed dataset processing fingerprint is inconsistent")
    source = _read_json_object(root / manifest.source_manifest_path)
    _require_manifest_values(
        source,
        "source manifest",
        {
            "complete": True,
            "loader_type": manifest.loader_type,
            "parser_type": manifest.parser_type,
            "parser_settings": _json_compatible(asdict(config.parser)),
            "source_config_fingerprint_revision": SOURCE_CONFIG_FINGERPRINT_REVISION,
            "source_config_sha256": source_config,
            "processing_config_fingerprint_revision": (
                PROCESSING_CONFIG_FINGERPRINT_REVISION
            ),
            "processing_config_sha256": processing_config,
            "source_snapshot_sha256": manifest.source_snapshot_sha256,
            "source_page_count": manifest.source_page_count,
            "configured_source": source_config_payload(
                config.loader,
                category=config.dataset.slug,
            )["source"],
        },
    )
    _validate_committed_source_bytes(
        config,
        config_path=config_path,
        root=root,
        source=source,
        source_config=source_config,
        source_root=source_root,
        expected_snapshot=manifest.source_snapshot_sha256 or "",
    )

    chunks_path = root / manifest.processed_chunks_path
    chunks_sha256 = sha256_file(chunks_path)
    corpus = _read_json_object(root / manifest.corpus_manifest_path)
    _require_manifest_values(
        corpus,
        "corpus manifest",
        {
            "loader_type": manifest.loader_type,
            "parser_type": manifest.parser_type,
            "source_config_sha256": source_config,
            "processing_config_sha256": processing_config,
            "source_snapshot_sha256": manifest.source_snapshot_sha256,
            "source_page_count": manifest.source_page_count,
            "section_count": manifest.section_count,
            "chunk_count": manifest.chunk_count,
            "chunks_sha256": chunks_sha256,
        },
    )
    if _count_nonempty_lines(chunks_path) != manifest.chunk_count:
        raise RuntimeError("committed processed corpus count is inconsistent")

    if until not in {"index", "profile", "all"}:
        return
    index_path = root / manifest.dense_index_path
    metadata_path = root / manifest.dense_metadata_path
    index_manifest = _read_json_object(root / manifest.dense_manifest_path)
    _require_manifest_values(
        index_manifest,
        "dense index manifest",
        {
            "chunk_count": manifest.chunk_count,
            "input_jsonl_sha256": chunks_sha256,
            "index_sha256": sha256_file(index_path),
            "metadata_sha256": sha256_file(metadata_path),
        },
    )
    if _count_nonempty_lines(metadata_path) != manifest.chunk_count:
        raise RuntimeError("committed dense metadata count is inconsistent")
    if until in {"profile", "all"} and config.profile.prepare == "recommended-v2":
        validation = validate_profile_artifacts(
            runtime_profile("recommended-v2"),
            root,
            expected_chunk_count=manifest.chunk_count,
            baseline_metadata_path=metadata_path,
            allow_pending_publication=True,
        )
        if not validation.succeeded:
            raise RuntimeError(
                "committed recommended-v2 profile is inconsistent: "
                + "; ".join(validation.errors)
            )


def _validate_committed_source_bytes(
    config: SiteConfig,
    *,
    config_path: Path,
    root: Path,
    source: dict[str, Any],
    source_config: str,
    source_root: Path | None,
    expected_snapshot: str,
) -> None:
    """Validate the loader-specific local bytes without performing network I/O."""
    if isinstance(config.loader, LocalCompatibilityLoaderSettings):
        runtime = LoaderRuntimeOptions(
            data_root=root,
            config_directory=config_path.parent,
            dataset_slug=config.dataset.slug,
            source_config_sha256=source_config,
            source_root=source_root,
            offline=True,
        )
        loader = build_document_loader(config.loader, runtime)
        documents = tuple(loader.load())
        if _loader_snapshot_identity(loader, documents) != expected_snapshot:
            raise RuntimeError(
                "local source content does not match the prepared dataset; "
                "use a different data-root"
            )
        return
    if isinstance(config.loader, PinnedLocalArchiveLoaderSettings):
        archive = (config_path.parent / config.loader.archive_path).resolve()
        if not archive.is_file():
            raise RuntimeError("committed pinned source archive is missing")
        if sha256_file(archive) != config.loader.archive_sha256:
            raise RuntimeError("committed pinned source archive SHA-256 is inconsistent")
        _validate_committed_archive_source(
            config,
            root=root,
            source=source,
            archive_path=archive,
            archive_sha256=config.loader.archive_sha256,
            archive_cache_path=None,
            requested_url=config.loader.original_archive_url,
            final_url=None,
            expected_archive_sha256=config.loader.archive_sha256,
            acquired_at=None,
        )
        return
    if isinstance(config.loader, SnapshotHttpArchiveLoaderSettings):
        lock = validate_snapshot_http_archive_cache(
            config.loader,
            root / "data/raw",
            source_config_sha256=source_config,
        )
        if (
            lock.get("source_snapshot_sha256") != expected_snapshot
            or lock.get("observed_sha256") != expected_snapshot
        ):
            raise RuntimeError("committed source lock snapshot is inconsistent")
        if lock.get("cache_relative_path") != source.get("archive_cache_path"):
            raise RuntimeError("source lock and source manifest cache paths differ")
        archive_path = root / "data/raw" / str(lock["cache_relative_path"])
        _validate_committed_archive_source(
            config,
            root=root,
            source=source,
            archive_path=archive_path,
            archive_sha256=expected_snapshot,
            archive_cache_path=str(lock["cache_relative_path"]),
            requested_url=str(lock["requested_url"]),
            final_url=str(lock["final_url"]),
            expected_archive_sha256=None,
            acquired_at=str(lock["fetched_at"]),
        )
        return
    if isinstance(config.loader, BoundedHttpLoaderSettings):
        validate_bounded_http_cache(
            root / "data/raw",
            expected_source_config_sha256=source_config,
            expected_processing_config_sha256=str(
                source["processing_config_sha256"]
            ),
            expected_source_snapshot_sha256=expected_snapshot,
            expected_page_count=int(source["source_page_count"]),
        )


def _validate_committed_archive_source(
    config: SiteConfig,
    *,
    root: Path,
    source: dict[str, Any],
    archive_path: Path,
    archive_sha256: str,
    archive_cache_path: str | None,
    requested_url: str,
    final_url: str | None,
    expected_archive_sha256: str | None,
    acquired_at: str | None,
) -> None:
    """Anchor committed archive provenance records to verified archive documents."""
    if set(source) != _COMMITTED_ARCHIVE_SOURCE_KEYS:
        raise RuntimeError("committed archive source manifest schema is invalid")
    settings = config.loader
    if not isinstance(
        settings,
        (PinnedLocalArchiveLoaderSettings, SnapshotHttpArchiveLoaderSettings),
    ):
        raise TypeError("archive source validation requires archive loader settings")
    loader = SafeZipHtmlLoader(
        archive_path,
        root / "data/raw/extracted" / archive_sha256,
        archive_sha256=archive_sha256,
        archive_root=settings.archive_root,
        source_base_url=settings.source_base_url,
        include_path_prefixes=settings.include_path_prefixes,
        source_kind=settings.type,
        max_archive_bytes=settings.max_archive_bytes,
        max_members=settings.max_members,
        max_member_bytes=settings.max_member_bytes,
        max_extracted_bytes=settings.max_extracted_bytes,
        fetched_at=acquired_at,
    )
    documents = loader.load()
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
    expected_values = {
        "schema_revision": "zip-html-source-v1",
        "complete": True,
        "loader_type": settings.type,
        "requested_archive_url": requested_url,
        "final_archive_url": final_url,
        "expected_archive_sha256": expected_archive_sha256,
        "observed_archive_sha256": archive_sha256,
        "archive_byte_size": archive_path.stat().st_size,
        "archive_format": settings.archive_format,
        "archive_root": settings.archive_root,
        "source_base_url": settings.source_base_url,
        "include_path_prefixes": list(settings.include_path_prefixes),
        "source_page_count": len(documents),
        "source_snapshot_sha256": archive_sha256,
        "document_snapshot_sha256": source_snapshot_sha256(documents),
        "acquired_at": acquired_at,
        "archive_cache_path": archive_cache_path,
        "pages": pages,
    }
    _require_manifest_values(source, "archive source manifest", expected_values)
    allowed_origins = (
        {"project-pinned-local-archive"}
        if isinstance(settings, PinnedLocalArchiveLoaderSettings)
        else {"observed-on-acquisition", "observed-on-first-acquisition"}
    )
    if source.get("archive_sha256_origin") not in allowed_origins:
        raise RuntimeError("committed archive source SHA-256 origin is inconsistent")
    if not isinstance(source.get("cache_reused"), bool) or not isinstance(
        source.get("offline"), bool
    ):
        raise RuntimeError("committed archive source replay flags are invalid")


def _require_manifest_values(
    payload: dict[str, Any],
    label: str,
    expected: dict[str, Any],
) -> None:
    mismatched = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatched:
        raise RuntimeError(f"committed {label} is inconsistent: {', '.join(mismatched)}")


def _json_compatible(value: Any) -> Any:
    """Return the value after the same tuple-to-list normalization as JSON."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _prepare_dense_index(
    config: SiteConfig,
    staging: Path,
    staged_chunks: Path,
    *,
    until: PreparationTarget,
    device: str,
) -> VectorIndexBuildResult | None:
    if until not in {"index", "profile", "all"}:
        return None
    staged_index = staging / "indexes/dense.faiss"
    staged_metadata = staging / "indexes/metadata.jsonl"
    staged_manifest = staging / "indexes/index_manifest.json"
    if _complete_staged_index(
        staged_index,
        staged_metadata,
        staged_manifest,
        staged_chunks,
    ):
        manifest_data = _read_json_object(staged_manifest)
        return VectorIndexBuildResult(
            index_path=staged_index,
            metadata_path=staged_metadata,
            manifest_path=staged_manifest,
            chunk_count=int(manifest_data["chunk_count"]),
            embedding_dimension=int(manifest_data["embedding_dimension"]),
            elapsed_seconds=float(manifest_data.get("elapsed_seconds", 0.0)),
        )
    if any(path.exists() for path in (staged_index, staged_metadata, staged_manifest)):
        _remove_path(staging / "indexes")
    result = build_vector_index(
        staged_chunks,
        staged_index,
        staged_metadata,
        staged_manifest,
        model_name=config.index.embedding_model,
        batch_size=config.index.embedding_batch_size,
        device=device,
        trust_remote_code=False,
    )
    _make_index_manifest_portable(
        staged_manifest,
        input_jsonl="data/processed/chunks.jsonl",
    )
    return result


def _prepare_profile(
    config: SiteConfig,
    staging: Path,
    *,
    until: PreparationTarget,
    device: str,
) -> RecommendedV2PreparationResult | None:
    if until not in {"profile", "all"} or config.profile.prepare != "recommended-v2":
        return None
    batch_size = min(config.index.embedding_batch_size, 32)
    target = staging / "profiles/recommended-v2"
    try:
        return prepare_dataset_recommended_v2_artifacts(
            staging,
            device=device,
            batch_size=batch_size,
        )
    except ArtifactPreparationError:
        if not target.exists() and not target.is_symlink():
            raise
        _remove_path(target)
        return prepare_dataset_recommended_v2_artifacts(
            staging,
            device=device,
            batch_size=batch_size,
        )


def _reuse_prepared_dataset(
    config: SiteConfig,
    root: Path,
    manifest: DatasetManifest,
    *,
    until: PreparationTarget,
    device: str,
) -> DatasetPreparationResult:
    chunks_path = root / manifest.processed_chunks_path
    corpus_manifest = _read_json_object(root / manifest.corpus_manifest_path)
    if sha256_file(chunks_path) != corpus_manifest.get("chunks_sha256"):
        raise RuntimeError("existing processed corpus SHA-256 is inconsistent")
    dense_result: VectorIndexBuildResult | None = None
    if until in {"index", "profile", "all"}:
        index_manifest = _read_json_object(root / manifest.dense_manifest_path)
        if int(index_manifest.get("chunk_count", -1)) != manifest.chunk_count:
            raise RuntimeError("existing dense index count is inconsistent")
        if sha256_file(root / manifest.dense_index_path) != index_manifest.get(
            "index_sha256"
        ):
            raise RuntimeError("existing dense index SHA-256 is inconsistent")
        dense_result = VectorIndexBuildResult(
            index_path=root / manifest.dense_index_path,
            metadata_path=root / manifest.dense_metadata_path,
            manifest_path=root / manifest.dense_manifest_path,
            chunk_count=manifest.chunk_count,
            embedding_dimension=int(index_manifest["embedding_dimension"]),
            elapsed_seconds=float(index_manifest.get("elapsed_seconds", 0.0)),
        )
    profile_result: RecommendedV2PreparationResult | None = None
    if until in {"profile", "all"} and config.profile.prepare == "recommended-v2":
        profile_result = prepare_dataset_recommended_v2_artifacts(
            root,
            device=device,
            batch_size=min(config.index.embedding_batch_size, 32),
            validate_only=True,
        )
    return DatasetPreparationResult(
        data_root=root,
        dataset_manifest=manifest,
        source_page_count=manifest.source_page_count,
        parsed_page_count=int(corpus_manifest.get("parsed_page_count", 0)),
        failed_page_count=int(corpus_manifest.get("failed_page_count", 0)),
        section_count=manifest.section_count,
        chunk_count=manifest.chunk_count,
        excluded_section_count=int(corpus_manifest.get("excluded_section_count", 0)),
        excluded_node_count=int(corpus_manifest.get("excluded_node_count", 0)),
        code_block_count=int(corpus_manifest.get("code_block_count", 0)),
        table_count=int(corpus_manifest.get("table_count", 0)),
        fallback_page_count=int(corpus_manifest.get("fallback_page_count", 0)),
        corpus_sha256=str(corpus_manifest["chunks_sha256"]),
        dense_index_result=dense_result,
        profile_result=profile_result,
        reused_dataset=True,
        source_cache_reused=True,
        source_snapshot_sha256=(
            manifest.source_snapshot_sha256
            or str(corpus_manifest.get("source_snapshot_sha256", ""))
        ),
    )


def _result(
    root: Path,
    manifest: DatasetManifest,
    *,
    ingestion: IngestionResult,
    dense_result: VectorIndexBuildResult | None,
    profile_result: RecommendedV2PreparationResult | None,
    source_cache_reused: bool,
) -> DatasetPreparationResult:
    chunks_path = root / manifest.processed_chunks_path
    return DatasetPreparationResult(
        data_root=root,
        dataset_manifest=manifest,
        source_page_count=manifest.source_page_count,
        parsed_page_count=len(ingestion.documents),
        failed_page_count=len(ingestion.failures),
        section_count=manifest.section_count,
        chunk_count=manifest.chunk_count,
        excluded_section_count=ingestion.excluded_section_count,
        excluded_node_count=ingestion.excluded_node_count,
        code_block_count=ingestion.code_block_count,
        table_count=ingestion.table_count,
        fallback_page_count=ingestion.fallback_page_count,
        corpus_sha256=sha256_file(chunks_path),
        dense_index_result=dense_result,
        profile_result=profile_result,
        reused_dataset=False,
        source_cache_reused=source_cache_reused,
        source_snapshot_sha256=manifest.source_snapshot_sha256 or "",
    )


def _loader_snapshot_identity(
    loader: DocumentLoader,
    documents: tuple[Any, ...],
) -> str:
    archive_identity = getattr(loader, "source_snapshot_sha256", None)
    if isinstance(archive_identity, str) and archive_identity:
        return archive_identity
    return source_snapshot_sha256(documents)


def _loader_cache_reused(loader: DocumentLoader) -> bool:
    summary = getattr(loader, "summary", None)
    return bool(getattr(summary, "cache_reused", False))


def _commit_loader_source(loader: DocumentLoader) -> None:
    operation = getattr(loader, "commit_source", None)
    if callable(operation):
        operation()


def _rollback_loader_source(loader: DocumentLoader) -> None:
    operation = getattr(loader, "rollback_source", None)
    if callable(operation):
        operation()


def _rollback_preparation_transaction(
    *,
    root: Path,
    loader: DocumentLoader,
    publication: dict[str, Any] | None,
    source_manifest_path: Path,
    previous_source_manifest: bytes | None,
    restore_source_manifest: bool,
) -> list[BaseException]:
    """Attempt every pre-commit rollback step and retain recoverable backups."""
    errors: list[BaseException] = []
    if publication is not None:
        derived_rollback_ready = True
        if publication.get("phase") == "committed":
            try:
                _mark_derived_publication_for_rollback(root, publication)
            except BaseException as error:
                errors.append(error)
                derived_rollback_ready = False
        if derived_rollback_ready:
            try:
                _rollback_derived_publication(root, publication)
            except BaseException as error:
                errors.append(error)
    try:
        _rollback_loader_source(loader)
    except BaseException as error:
        errors.append(error)
    if restore_source_manifest:
        try:
            _restore_source_manifest(
                source_manifest_path,
                previous_source_manifest,
            )
        except BaseException as error:
            errors.append(error)
    return errors


def _restore_source_manifest(path: Path, previous: bytes | None) -> None:
    """Restore the exact committed source manifest after downstream failure."""
    if previous is None:
        path.unlink(missing_ok=True)
        return
    _write_bytes_atomic(previous, path)


def _augment_source_manifest(
    path: Path,
    *,
    config: SiteConfig,
    documents: tuple[Any, ...],
    source_config: str,
    processing_config: str,
    source_snapshot: str,
) -> None:
    if path.is_file():
        payload = _read_json_object(path)
    else:
        payload = {
            "schema_revision": "local-html-tree-source-v2",
            "complete": True,
            "loader_type": config.loader.type,
            "source_page_count": len(documents),
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
    payload.update(
        {
            "complete": True,
            "loader_type": config.loader.type,
            "parser_type": config.parser.type,
            "parser_settings": asdict(config.parser),
            "source_page_count": len(documents),
            "site_config_file_sha256": config.config_sha256,
            "source_config_fingerprint_revision": SOURCE_CONFIG_FINGERPRINT_REVISION,
            "source_config_sha256": source_config,
            "processing_config_fingerprint_revision": (
                PROCESSING_CONFIG_FINGERPRINT_REVISION
            ),
            "processing_config_sha256": processing_config,
            "source_snapshot_sha256": source_snapshot,
            "configured_source": source_config_payload(
                config.loader,
                category=config.dataset.slug,
            )["source"],
        }
    )
    _write_json_atomic(payload, path)


def _corpus_manifest(
    config: SiteConfig,
    documents: tuple[Any, ...],
    ingestion: IngestionResult,
    chunks_path: Path,
    *,
    source_config: str,
    processing_config: str,
    source_snapshot: str,
) -> dict[str, Any]:
    return {
        "schema_revision": "configurable-corpus-v2",
        "dataset_name": config.dataset.name,
        "dataset_slug": config.dataset.slug,
        "loader_type": config.loader.type,
        "parser_type": config.parser.type,
        "parser_settings": asdict(config.parser),
        "site_config_file_sha256": config.config_sha256,
        "source_config_sha256": source_config,
        "processing_config_sha256": processing_config,
        "source_snapshot_sha256": source_snapshot,
        "source_page_count": len(documents),
        "parsed_page_count": len(ingestion.documents),
        "failed_page_count": len(ingestion.failures),
        "section_count": len(ingestion.sections),
        "chunk_count": len(ingestion.chunks),
        "excluded_section_count": ingestion.excluded_section_count,
        "excluded_node_count": ingestion.excluded_node_count,
        "code_block_count": ingestion.code_block_count,
        "table_count": ingestion.table_count,
        "fallback_page_count": ingestion.fallback_page_count,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "chunks_sha256": sha256_file(chunks_path),
        "source_content_sha256": [document.content_sha256 for document in documents],
        "failures": [asdict(failure) for failure in ingestion.failures],
        "created_at": datetime.now(UTC).isoformat(),
        "openai_api_used": False,
        "contains_secrets": False,
    }


def _complete_staged_index(
    index_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    input_path: Path,
) -> bool:
    if not all(path.is_file() for path in (index_path, metadata_path, manifest_path)):
        return False
    try:
        manifest = _read_json_object(manifest_path)
        return (
            manifest.get("input_jsonl_sha256") == sha256_file(input_path)
            and manifest.get("index_sha256") == sha256_file(index_path)
            and manifest.get("metadata_sha256") == sha256_file(metadata_path)
        )
    except (OSError, ValueError):
        return False


def _make_index_manifest_portable(path: Path, *, input_jsonl: str) -> None:
    manifest = _read_json_object(path)
    manifest["input_jsonl"] = input_jsonl
    _write_json_atomic(manifest, path)


def _publish_derived_transaction(
    staging: Path,
    root: Path,
    *,
    include_index: bool,
    include_profile: bool,
    remove_unbuilt: bool,
) -> dict[str, Any]:
    _validate_managed_data_root(root)
    items: list[tuple[Path | None, Path]] = [
        (staging / "data/processed", root / "data/processed"),
    ]
    if include_index:
        items.append((staging / "indexes", root / "indexes"))
    elif remove_unbuilt:
        items.append((None, root / "indexes"))
    profile_destination = root / "profiles/recommended-v2"
    if include_profile:
        items.append(
            (
                staging / "profiles/recommended-v2",
                profile_destination,
            )
        )
    elif remove_unbuilt:
        items.append((None, profile_destination))
    items.append((staging / "dataset_manifest.json", root / "dataset_manifest.json"))
    journal_path = root / _PUBLICATION_JOURNAL_NAME
    if journal_path.exists() or journal_path.is_symlink():
        raise RuntimeError("unresolved derived publication journal already exists")
    try:
        staging_relative = staging.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError("preparation staging must remain within the data-root") from error
    backup_root = Path(
        tempfile.mkdtemp(dir=root, prefix=".prepare-publication-backup-")
    )
    entries = [
        {
            "source": (
                None
                if source is None
                else source.relative_to(root).as_posix()
            ),
            "destination": destination.relative_to(root).as_posix(),
            "backup": (backup_root / str(index)).relative_to(root).as_posix(),
            "had_destination": destination.exists() or destination.is_symlink(),
        }
        for index, (source, destination) in enumerate(items)
    ]
    payload: dict[str, Any] = {
        "schema_revision": _PUBLICATION_JOURNAL_REVISION,
        "phase": "publishing",
        "staging": staging_relative,
        "backup_root": backup_root.relative_to(root).as_posix(),
        "entries": entries,
    }
    try:
        _write_json_atomic(payload, journal_path)
    except BaseException:
        _remove_path(backup_root)
        raise
    try:
        for entry, (source, destination) in zip(entries, items, strict=True):
            _validate_managed_data_root(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination_exists = destination.exists() or destination.is_symlink()
            if destination_exists != entry["had_destination"]:
                raise RuntimeError(
                    f"publication destination changed concurrently: {destination}"
                )
            if destination_exists:
                os.rename(destination, root / str(entry["backup"]))
            if source is not None:
                os.rename(source, destination)
        payload["phase"] = "committed"
        _write_json_atomic(payload, journal_path)
    except BaseException as publication_error:
        if payload.get("phase") == "committed":
            try:
                _mark_derived_publication_for_rollback(root, payload)
            except BaseException as mark_error:
                failure = RuntimeError(
                    "derived artifact publication failed after its commit marker; "
                    f"recoverable backups retained at {backup_root}"
                )
                failure.add_note(
                    f"original publication error: "
                    f"{type(publication_error).__name__}: {publication_error}"
                )
                raise failure from mark_error
        try:
            _rollback_derived_publication(root, payload)
        except BaseException as rollback_error:
            raise RuntimeError(
                "derived artifact publication failed and rollback was incomplete; "
                f"recoverable backups retained at {backup_root}"
            ) from rollback_error
        raise
    return payload


def _recover_derived_publication(root: Path) -> None:
    """Recover or finalize one publication interrupted by process termination."""
    journal_path = root / _PUBLICATION_JOURNAL_NAME
    if not journal_path.is_file():
        return
    payload = _read_json_object(journal_path)
    _validate_publication_journal(root, payload)
    if payload["phase"] == "committed":
        _finalize_committed_publication(root, payload)
        return
    try:
        _rollback_derived_publication(root, payload)
    except BaseException as error:
        raise RuntimeError(
            "interrupted derived publication could not be rolled back; "
            f"recoverable backups remain at {root / str(payload['backup_root'])}"
        ) from error


def _rollback_derived_publication(root: Path, payload: dict[str, Any]) -> None:
    _validate_publication_journal(root, payload)
    if payload["phase"] != "publishing":
        raise RuntimeError("committed publication must be marked for rollback first")
    errors: list[BaseException] = []
    for entry in reversed(payload["entries"]):
        try:
            _rollback_publication_entry(root, entry)
        except BaseException as error:
            errors.append(error)
    if errors:
        raise RuntimeError("one or more publication entries could not be restored") from errors[0]
    backup_root = root / str(payload["backup_root"])
    _remove_path(backup_root)
    (root / _PUBLICATION_JOURNAL_NAME).unlink(missing_ok=True)


def _rollback_publication_entry(root: Path, entry: dict[str, Any]) -> None:
    source_value = entry["source"]
    source = None if source_value is None else root / str(source_value)
    destination = root / str(entry["destination"])
    backup = root / str(entry["backup"])
    had_destination = bool(entry["had_destination"])
    backup_exists = backup.exists() or backup.is_symlink()
    source_exists = source is not None and (source.exists() or source.is_symlink())
    destination_exists = destination.exists() or destination.is_symlink()
    if backup_exists:
        if source is not None and not source_exists:
            if destination_exists:
                source.parent.mkdir(parents=True, exist_ok=True)
                os.rename(destination, source)
                destination_exists = False
        elif destination_exists:
            raise RuntimeError("publication destination conflicts with its retained backup")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(backup, destination)
        return
    if had_destination:
        if not destination_exists:
            raise RuntimeError("committed artifact backup and destination are both missing")
        if source is not None and not source_exists:
            raise RuntimeError("committed artifact backup is missing after publication")
        return
    if source is not None and not source_exists:
        if not destination_exists:
            return
        source.parent.mkdir(parents=True, exist_ok=True)
        os.rename(destination, source)
    elif source is None and destination_exists:
        raise RuntimeError("unexpected artifact appeared at an unbuilt destination")


def _validate_publication_journal(root: Path, payload: dict[str, Any]) -> None:
    if payload.get("schema_revision") != _PUBLICATION_JOURNAL_REVISION:
        raise RuntimeError("derived publication journal revision is invalid")
    if payload.get("phase") not in {"publishing", "committed"}:
        raise RuntimeError("derived publication journal phase is invalid")
    backup_root = payload.get("backup_root")
    if (
        not isinstance(backup_root, str)
        or not backup_root.startswith(".prepare-publication-backup-")
        or PurePosixPath(backup_root).parent != PurePosixPath(".")
    ):
        raise RuntimeError("derived publication backup path is invalid")
    _safe_publication_relative(root, backup_root, "publication backup root")
    staging = payload.get("staging")
    if staging != ".prepare-staging":
        raise RuntimeError("derived publication staging path is invalid")
    _safe_publication_relative(root, staging, "publication staging")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("derived publication journal entries are invalid")
    allowed_sources = {
        "data/processed": ".prepare-staging/data/processed",
        "indexes": ".prepare-staging/indexes",
        "profiles/recommended-v2": ".prepare-staging/profiles/recommended-v2",
        "dataset_manifest.json": ".prepare-staging/dataset_manifest.json",
    }
    seen_destinations: set[str] = set()
    seen_backups: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "destination",
            "backup",
            "had_destination",
        }:
            raise RuntimeError("derived publication journal entry schema is invalid")
        destination = entry["destination"]
        if destination not in allowed_sources or destination in seen_destinations:
            raise RuntimeError("derived publication destination path is invalid")
        _safe_publication_relative(root, destination, "publication destination")
        seen_destinations.add(destination)
        source = entry["source"]
        expected_source = allowed_sources[destination]
        if source is None:
            if destination not in {"indexes", "profiles/recommended-v2"}:
                raise RuntimeError("derived publication source path is invalid")
        elif source != expected_source:
            raise RuntimeError("derived publication source path is invalid")
        else:
            _safe_publication_relative(root, source, "publication source")
        backup = entry["backup"]
        if (
            not isinstance(backup, str)
            or PurePosixPath(backup).parent.as_posix() != backup_root
            or not PurePosixPath(backup).name.isdecimal()
            or backup in seen_backups
        ):
            raise RuntimeError("derived publication entry backup path is invalid")
        _safe_publication_relative(root, backup, "publication entry backup")
        seen_backups.add(backup)
        if not isinstance(entry["had_destination"], bool):
            raise RuntimeError("derived publication destination state is invalid")
    if not {"data/processed", "dataset_manifest.json"}.issubset(seen_destinations):
        raise RuntimeError("derived publication journal is missing required entries")
    _validate_managed_data_root(root)


def _safe_publication_relative(root: Path, value: str, label: str) -> Path:
    if not value or "\\" in value:
        raise RuntimeError(f"{label} is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{label} is unsafe")
    if relative.as_posix() != value:
        raise RuntimeError(f"{label} is not normalized")
    candidate = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    if not candidate.resolve(strict=False).is_relative_to(resolved_root):
        raise RuntimeError(f"{label} escaped the data-root")
    current = root
    for part in relative.parts:
        current = current / part
        if _is_link_like(current):
            raise RuntimeError(f"{label} has a symlink component")
    return candidate


def _finalize_committed_publication(root: Path, payload: dict[str, Any]) -> None:
    """Best-effort cleanup after the dataset manifest committed the generation."""
    try:
        _remove_path(root / str(payload["backup_root"]))
        _remove_path(root / str(payload["staging"]))
        (root / _PUBLICATION_JOURNAL_NAME).unlink(missing_ok=True)
    except OSError:
        return


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif _is_link_like(path):
        raise RuntimeError(f"refusing to remove a junction: {path}")
    elif path.exists():
        shutil.rmtree(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


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
