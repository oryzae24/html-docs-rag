"""Build full-section parents and search-only child FAISS artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from python_doc_rag.config import DEFAULT_EMBEDDING_MODEL, ChunkingConfig
from python_doc_rag.corpus import enumerate_html_files, write_chunks_jsonl_atomic
from python_doc_rag.html_parser import parse_python_doc_html_result
from python_doc_rag.models import DocumentSection
from python_doc_rag.section_parent import (
    SECTION_ID_ALGORITHM,
    SECTION_PARENT_REVISION,
    SectionStore,
    create_section_children,
    write_sections_jsonl_atomic,
)
from python_doc_rag.vector_store import build_vector_index

DEFAULT_CHILD_CONFIGS = ((300, 75), (400, 100))


def parse_args() -> argparse.Namespace:
    """Parse local input and Git-ignored artifact destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--child-config",
        action="append",
        type=_child_config,
        dest="child_configs",
        help="Repeat at most twice; format SIZE:OVERLAP (default: 300:75, 400:100)",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    """Parse every target HTML file, validate mappings, and build indexes."""
    args = parse_args()
    configs = args.child_configs or list(DEFAULT_CHILD_CONFIGS)
    if len(configs) > 2:
        raise ValueError("at most two child configurations may be built")
    if len(set(configs)) != len(configs):
        raise ValueError("child configurations must be unique")

    document_root = args.document_root.expanduser().resolve(strict=True)
    artifact_root = args.artifact_root.expanduser()
    artifact_root.mkdir(parents=True, exist_ok=True)
    sections_path = artifact_root / "sections.jsonl"
    section_manifest_path = artifact_root / "section_manifest.json"
    started_at = datetime.now(UTC)
    initial_rss = _maximum_rss_bytes()

    sections, parse_summary = _parse_all_sections(document_root)
    written = write_sections_jsonl_atomic(sections, sections_path)
    if written != len(sections):
        raise RuntimeError("written section count did not match parser result count")
    section_store = SectionStore.from_jsonl(sections_path)
    if section_store.section_count != len(sections):
        raise RuntimeError("section store count did not match parser result count")

    section_manifest = {
        "revision": SECTION_PARENT_REVISION,
        "document_root": str(document_root),
        "section_path": str(sections_path),
        "section_sha256": _sha256(sections_path),
        "section_count": len(sections),
        "section_id_algorithm": SECTION_ID_ALGORITHM,
        "section_text_hash_algorithm": "sha256(utf-8)",
        "duplicate_section_count": 0,
        **parse_summary,
        "started_at": started_at.isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
    }
    _write_json_atomic(section_manifest, section_manifest_path)

    built_configs: list[dict[str, Any]] = []
    for size, overlap in configs:
        built_configs.append(
            _build_child_artifacts(
                sections,
                section_store=section_store,
                section_manifest_path=section_manifest_path,
                section_manifest=section_manifest,
                artifact_root=artifact_root,
                config=ChunkingConfig(size, overlap),
                embedding_model=args.embedding_model,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    finished_at = datetime.now(UTC)
    complete_manifest = dict(section_manifest)
    complete_manifest.update(
        {
            "child_configurations": built_configs,
            "finished_at": finished_at.isoformat(),
            "total_build_time_seconds": (finished_at - started_at).total_seconds(),
            "peak_cpu_rss_bytes": _maximum_rss_bytes(),
            "cpu_rss_increase_bytes": max(0, _maximum_rss_bytes() - initial_rss),
            "peak_gpu_vram_allocated_bytes": _peak_gpu_vram(),
        }
    )
    _write_json_atomic(complete_manifest, section_manifest_path)
    print(json.dumps(complete_manifest, ensure_ascii=False, indent=2))
    return 0


def _parse_all_sections(
    document_root: Path,
) -> tuple[list[DocumentSection], dict[str, Any]]:
    enumeration = enumerate_html_files(document_root)
    if enumeration.missing_categories:
        raise RuntimeError(
            "raw documentation is missing target categories: "
            + ", ".join(enumeration.missing_categories)
        )
    if enumeration.skipped_unsafe_paths:
        raise RuntimeError("raw documentation contains unsafe target paths")
    sections: list[DocumentSection] = []
    failures: list[dict[str, str]] = []
    excluded_count = 0
    successful_count = 0
    for html_path in enumeration.files:
        source_path = html_path.relative_to(document_root).as_posix()
        category = PurePosixPath(source_path).parts[0]
        try:
            html = html_path.read_text(encoding="utf-8")
            result = parse_python_doc_html_result(
                html,
                source_path=source_path,
                category=category,
            )
        except Exception as error:  # noqa: BLE001 - audit every input before failing
            failures.append(
                {
                    "source_path": source_path,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        successful_count += 1
        excluded_count += result.excluded_section_count
        sections.extend(result.sections)
    if failures:
        raise RuntimeError(
            f"section parsing failed for {len(failures)} files: "
            + json.dumps(failures, ensure_ascii=False)
        )
    return sections, {
        "available_html_count": enumeration.available_html_count,
        "detected_html_count": len(enumeration.files),
        "successful_html_count": successful_count,
        "failed_html_count": 0,
        "extracted_section_count": len(sections),
        "excluded_empty_or_short_section_count": excluded_count,
        "missing_categories": [],
        "skipped_unsafe_paths": [],
        "failures": [],
    }


def _build_child_artifacts(
    sections: list[DocumentSection],
    *,
    section_store: SectionStore,
    section_manifest_path: Path,
    section_manifest: dict[str, Any],
    artifact_root: Path,
    config: ChunkingConfig,
    embedding_model: str,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    config_root = artifact_root / (
        f"child-{config.chunk_size}-overlap-{config.chunk_overlap}"
    )
    config_root.mkdir(parents=True, exist_ok=True)
    child_jsonl = config_root / "child_chunks.jsonl"
    index_path = config_root / "child.faiss"
    metadata_path = config_root / "child_metadata.jsonl"
    index_manifest_path = config_root / "child_index_manifest.json"
    mapping_manifest_path = config_root / "parent_mapping_manifest.json"
    started_at = datetime.now(UTC)
    children, summary = create_section_children(sections, config)
    written = write_chunks_jsonl_atomic(children, child_jsonl)
    if written != summary.child_count:
        raise RuntimeError("written child count did not match generated child count")
    resolved_count = section_store.validate_children(children)
    unresolved_count = summary.child_count - resolved_count
    if unresolved_count:
        raise RuntimeError(f"unresolved child mappings: {unresolved_count}")

    result = build_vector_index(
        child_jsonl,
        index_path,
        metadata_path,
        index_manifest_path,
        model_name=embedding_model,
        batch_size=batch_size,
        device=device,
    )
    finished_at = datetime.now(UTC)
    hashes = {
        "child_jsonl_sha256": _sha256(child_jsonl),
        "child_index_sha256": _sha256(index_path),
        "child_metadata_sha256": _sha256(metadata_path),
    }
    index_manifest = _read_json(index_manifest_path)
    index_manifest.update(
        {
            "revision": SECTION_PARENT_REVISION,
            "section_manifest_path": str(section_manifest_path),
            "section_sha256": section_manifest["section_sha256"],
            "section_count": summary.section_count,
            "child_count": summary.child_count,
            "child_size": config.chunk_size,
            "child_overlap": config.chunk_overlap,
            "average_children_per_section": summary.average_children_per_section,
            "maximum_children_per_section": summary.maximum_children_per_section,
            "duplicate_section_count": summary.duplicate_section_count,
            "unresolved_child_count": unresolved_count,
            **hashes,
            "embedding_model": embedding_model,
            "embedding_dimension": result.embedding_dimension,
            "normalized_embeddings": True,
            "faiss_index_type": "IndexFlatIP",
            "index_build_time_seconds": result.elapsed_seconds,
            "total_config_build_time_seconds": (
                finished_at - started_at
            ).total_seconds(),
            "created_at": finished_at.isoformat(),
            "child_index_size_bytes": index_path.stat().st_size,
            "child_metadata_size_bytes": metadata_path.stat().st_size,
        }
    )
    _write_json_atomic(index_manifest, index_manifest_path)
    mapping_manifest = {
        "revision": SECTION_PARENT_REVISION,
        "section_path": section_manifest["section_path"],
        "section_sha256": section_manifest["section_sha256"],
        "section_count": summary.section_count,
        "child_path": str(child_jsonl),
        "child_count": summary.child_count,
        "child_size": config.chunk_size,
        "child_overlap": config.chunk_overlap,
        "section_id_algorithm": SECTION_ID_ALGORITHM,
        "section_text_hash_algorithm": "sha256(utf-8)",
        "duplicate_section_count": summary.duplicate_section_count,
        "unresolved_child_count": unresolved_count,
        **hashes,
        "created_at": finished_at.isoformat(),
    }
    _write_json_atomic(mapping_manifest, mapping_manifest_path)
    return {
        "child_size": config.chunk_size,
        "child_overlap": config.chunk_overlap,
        "artifact_root": str(config_root),
        "index_manifest_path": str(index_manifest_path),
        "index_manifest_sha256": _sha256(index_manifest_path),
        "mapping_manifest_path": str(mapping_manifest_path),
        "mapping_manifest_sha256": _sha256(mapping_manifest_path),
        **hashes,
        "child_count": summary.child_count,
    }


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write_json_atomic(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _maximum_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024)


def _peak_gpu_vram() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _child_config(value: str) -> tuple[int, int]:
    try:
        size_text, overlap_text = value.split(":", maxsplit=1)
        size = int(size_text)
        overlap = int(overlap_text)
        ChunkingConfig(size, overlap)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "child config must be SIZE:OVERLAP with positive size and valid overlap"
        ) from error
    return size, overlap


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
