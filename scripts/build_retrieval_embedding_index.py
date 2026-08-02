"""Build one pinned retrieval-specialized embedding index outside baseline paths."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from python_doc_rag.embedding_models import retrieval_embedding_spec
from python_doc_rag.vector_store import build_vector_index


def parse_args() -> argparse.Namespace:
    """Parse an explicit input and experiment-only output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model-key",
        choices=("bge-m3", "multilingual-e5-base"),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    return parser.parse_args()


def main() -> int:
    """Build and validate one model-specific normalized FAISS index."""
    args = parse_args()
    spec = retrieval_embedding_spec(args.model_key)
    output_root = args.output_root.expanduser() / spec.key
    index_path = output_root / "index.faiss"
    metadata_path = output_root / "metadata.jsonl"
    manifest_path = output_root / "manifest.json"
    baseline_paths = {
        args.input_jsonl.expanduser().resolve(),
        args.input_jsonl.expanduser().parent.parent / "indexes/python_3_13_ja.faiss",
    }
    if any(
        path.expanduser().resolve() in baseline_paths
        for path in (index_path, metadata_path, manifest_path)
    ):
        raise ValueError("experiment outputs must not replace baseline artifacts")
    result = build_vector_index(
        args.input_jsonl,
        index_path,
        metadata_path,
        manifest_path,
        model_name=spec.model_name,
        model_revision=spec.revision,
        batch_size=args.batch_size,
        device=args.device,
        document_prefix=spec.document_prefix,
        query_prefix=spec.query_prefix,
        trust_remote_code=spec.trust_remote_code,
    )
    if result.embedding_dimension != spec.embedding_dimension:
        raise RuntimeError(
            "built index dimension differs from pinned model specification"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment_revision": "retrieval-embedding-tournament-v1",
            "model_key": spec.key,
            "license": spec.license,
            "model_card_url": spec.model_card_url,
            "language_evidence": spec.language_evidence,
            "max_sequence_length": spec.max_sequence_length,
            "pooling": spec.pooling,
            "openai_api_used": False,
            "contains_secrets": False,
        }
    )
    _write_json_atomic(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
