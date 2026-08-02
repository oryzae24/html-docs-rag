"""Build child-search artifacts for existing-chunk parent retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from python_doc_rag.config import DEFAULT_EMBEDDING_MODEL, ChunkingConfig
from python_doc_rag.corpus import write_chunks_jsonl_atomic
from python_doc_rag.parent_retrieval import (
    PARENT_ID_ALGORITHM,
    PARENT_RETRIEVAL_REVISION,
    ParentStore,
    create_child_chunks,
)
from python_doc_rag.vector_store import (
    build_vector_index,
    load_chunks_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse portable input, output, and child chunk settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-metadata-path", type=Path)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--child-size", type=_positive_int, required=True)
    parser.add_argument("--child-overlap", type=_non_negative_int, required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    """Build deterministic children, validate mappings, and build FAISS."""
    args = parse_args()
    data_root = args.data_root.expanduser()
    parent_path = args.parent_metadata_path or (
        data_root / "indexes/python_3_13_ja_metadata.jsonl"
    )
    artifact_root = args.artifact_root.expanduser()
    artifact_root.mkdir(parents=True, exist_ok=True)
    child_jsonl = artifact_root / "child_chunks.jsonl"
    index_path = artifact_root / "child.faiss"
    metadata_path = artifact_root / "child_metadata.jsonl"
    index_manifest_path = artifact_root / "child_index_manifest.json"
    mapping_manifest_path = artifact_root / "parent_mapping_manifest.json"
    config = ChunkingConfig(args.child_size, args.child_overlap)

    started_at = datetime.now(UTC)
    initial_rss = _maximum_rss_bytes()
    parents = load_chunks_jsonl(parent_path)
    store = ParentStore(parents)
    children, summary = create_child_chunks(parents, config)
    written = write_chunks_jsonl_atomic(children, child_jsonl)
    if written != summary.child_count:
        raise RuntimeError("written child count did not match generated child count")
    unresolved = summary.child_count - store.validate_children(children)
    if unresolved:
        raise RuntimeError(f"unresolved child mappings: {unresolved}")

    build_result = build_vector_index(
        child_jsonl,
        index_path,
        metadata_path,
        index_manifest_path,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
    )
    finished_at = datetime.now(UTC)
    child_jsonl_sha256 = _sha256(child_jsonl)
    index_sha256 = _sha256(index_path)
    metadata_sha256 = _sha256(metadata_path)
    parent_sha256 = _sha256(parent_path)
    index_manifest = _read_json(index_manifest_path)
    index_manifest.update(
        {
            "revision": PARENT_RETRIEVAL_REVISION,
            "parent_metadata_path": str(parent_path),
            "parent_metadata_sha256": parent_sha256,
            "parent_count": summary.parent_count,
            "child_count": summary.child_count,
            "child_size": config.chunk_size,
            "child_overlap": config.chunk_overlap,
            "average_children_per_parent": summary.average_children_per_parent,
            "maximum_children_per_parent": summary.maximum_children_per_parent,
            "parent_id_algorithm": PARENT_ID_ALGORITHM,
            "parent_id_collision_count": summary.parent_id_collision_count,
            "unresolved_child_count": unresolved,
            "child_jsonl_sha256": child_jsonl_sha256,
            "child_index_sha256": index_sha256,
            "child_metadata_sha256": metadata_sha256,
            "embedding_model": args.embedding_model,
            "embedding_dimension": build_result.embedding_dimension,
            "normalized_embeddings": True,
            "faiss_index_type": "IndexFlatIP",
            "build_time_seconds": build_result.elapsed_seconds,
            "total_build_time_seconds": (finished_at - started_at).total_seconds(),
            "python_version": platform.python_version(),
            "execution_device": args.device,
            "created_at": finished_at.isoformat(),
            "peak_cpu_rss_bytes": _maximum_rss_bytes(),
            "cpu_rss_increase_bytes": max(0, _maximum_rss_bytes() - initial_rss),
            "peak_gpu_vram_allocated_bytes": _peak_gpu_vram(),
            "child_index_size_bytes": index_path.stat().st_size,
            "child_metadata_size_bytes": metadata_path.stat().st_size,
        }
    )
    _write_json_atomic(index_manifest, index_manifest_path)
    mapping_manifest = {
        "revision": PARENT_RETRIEVAL_REVISION,
        "parent_metadata_path": str(parent_path),
        "parent_metadata_sha256": parent_sha256,
        "parent_count": summary.parent_count,
        "child_count": summary.child_count,
        "child_size": config.chunk_size,
        "child_overlap": config.chunk_overlap,
        "average_children_per_parent": summary.average_children_per_parent,
        "maximum_children_per_parent": summary.maximum_children_per_parent,
        "parent_id_algorithm": PARENT_ID_ALGORITHM,
        "parent_id_collision_count": summary.parent_id_collision_count,
        "unresolved_child_count": unresolved,
        "parent_text_hash_algorithm": "sha256(utf-8)",
        "child_jsonl_path": str(child_jsonl),
        "child_jsonl_sha256": child_jsonl_sha256,
        "child_metadata_path": str(metadata_path),
        "child_metadata_sha256": metadata_sha256,
        "created_at": finished_at.isoformat(),
    }
    _write_json_atomic(mapping_manifest, mapping_manifest_path)
    print(json.dumps(index_manifest, ensure_ascii=False, indent=2))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
