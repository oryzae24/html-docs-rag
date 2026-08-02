"""Evaluate baseline or existing-chunk parent retrieval without generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from python_doc_rag.evaluation import (
    evaluate_retrieval,
    load_evaluation_questions,
    retrieval_evaluation_to_dict,
)
from python_doc_rag.parent_retrieval import (
    PARENT_RETRIEVAL_REVISION,
    ParentDocumentRetriever,
    ParentStore,
)
from python_doc_rag.retrieval import (
    BM25Retriever,
    CodeAwareNgramTokenizer,
    ReciprocalRankFusionRetriever,
    VectorIndexRetriever,
)
from python_doc_rag.vector_store import load_chunks_jsonl, load_vector_index

_HYBRID_NGRAM_SIZES = (2,)
_HYBRID_RRF_K = 10
_HYBRID_CANDIDATE_K = 30


class _TracingParentSearcher:
    """Capture one parent diagnostic trace per evaluation query."""

    def __init__(self, retriever: ParentDocumentRetriever) -> None:
        self._retriever = retriever
        self.traces: list[Any] = []

    def search(self, query: str, *, top_k: int = 5) -> Any:
        results = self._retriever.search(query, top_k=top_k)
        self.traces.append(self._retriever.last_trace)
        return results


def parse_args() -> argparse.Namespace:
    """Parse fixed retrieval settings and portable artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--context-mode", choices=("chunk", "parent"), required=True)
    parser.add_argument("--retriever", choices=("dense", "hybrid"), required=True)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--index-manifest-path", type=Path)
    parser.add_argument("--parent-metadata-path", type=Path)
    parser.add_argument("--child-index-path", type=Path)
    parser.add_argument("--child-metadata-path", type=Path)
    parser.add_argument("--child-manifest-path", type=Path)
    parser.add_argument("--child-candidate-k", type=_positive_int, default=60)
    parser.add_argument("--embedding-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=_top_k, default=10)
    return parser.parse_args()


def main() -> int:
    """Build one search runtime, evaluate every question, and save atomically."""
    args = parse_args()
    data_root = args.data_root.expanduser()
    baseline_index = args.index_path or data_root / "indexes/python_3_13_ja.faiss"
    baseline_metadata = args.metadata_path or (
        data_root / "indexes/python_3_13_ja_metadata.jsonl"
    )
    baseline_manifest = args.index_manifest_path or (
        data_root / "indexes/python_3_13_ja_index_manifest.json"
    )
    parent_metadata = args.parent_metadata_path or baseline_metadata
    selected_root = data_root / "indexes/parent_document"
    child_index = args.child_index_path or selected_root / "child.faiss"
    child_metadata = args.child_metadata_path or selected_root / "child_metadata.jsonl"
    child_manifest = args.child_manifest_path or (
        selected_root / "child_index_manifest.json"
    )

    index_path = child_index if args.context_mode == "parent" else baseline_index
    metadata_path = (
        child_metadata if args.context_mode == "parent" else baseline_metadata
    )
    manifest_path = (
        child_manifest if args.context_mode == "parent" else baseline_manifest
    )
    manifest = _read_json(manifest_path)
    configured_model = manifest.get("model_name")
    if not isinstance(configured_model, str) or not configured_model.strip():
        configured_model = manifest.get("embedding_model")
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ValueError(f"manifest has no embedding model: {manifest_path}")
    embedding_model_name = args.embedding_model or configured_model

    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(embedding_model_name, device=args.device)
    vector_index = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=embedding_model,
    )
    chunks = load_chunks_jsonl(metadata_path)
    child_searcher: Any = vector_index
    if args.retriever == "hybrid":
        bm25 = BM25Retriever(
            chunks,
            tokenizer=CodeAwareNgramTokenizer(_HYBRID_NGRAM_SIZES),
        )
        child_searcher = ReciprocalRankFusionRetriever(
            [VectorIndexRetriever(vector_index), bm25],
            rrf_k=_HYBRID_RRF_K,
            candidate_k=_HYBRID_CANDIDATE_K,
        )

    tracing: _TracingParentSearcher | None = None
    searcher: Any = child_searcher
    parent_store: ParentStore | None = None
    if args.context_mode == "parent":
        _validate_parent_manifest(manifest, parent_metadata)
        parent_store = ParentStore.from_jsonl(parent_metadata)
        parent_store.validate_children(chunks)
        parent_retriever = ParentDocumentRetriever(
            child_searcher,
            parent_store,
            child_candidate_k=args.child_candidate_k,
        )
        tracing = _TracingParentSearcher(parent_retriever)
        searcher = tracing

    questions = load_evaluation_questions(args.questions)
    report = evaluate_retrieval(searcher, questions, top_k=args.top_k)
    payload = retrieval_evaluation_to_dict(report)
    payload["settings"] = {
        "context_mode": args.context_mode,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "questions_path": str(args.questions.expanduser()),
        "questions_sha256": _sha256(args.questions),
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "embedding_model": embedding_model_name,
        "search_fields": ["page_title", "section_title", "text"],
        "parent_retrieval_revision": (
            PARENT_RETRIEVAL_REVISION if args.context_mode == "parent" else None
        ),
        "child_candidate_k": (
            args.child_candidate_k if args.context_mode == "parent" else None
        ),
        "japanese_ngram_sizes": (
            list(_HYBRID_NGRAM_SIZES) if args.retriever == "hybrid" else None
        ),
        "rrf_k": _HYBRID_RRF_K if args.retriever == "hybrid" else None,
        "hybrid_candidate_k": (
            _HYBRID_CANDIDATE_K if args.retriever == "hybrid" else None
        ),
        "retriever_weights": [1.0, 1.0] if args.retriever == "hybrid" else None,
    }
    payload["environment"] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "sentence_transformers_version": importlib.metadata.version(
            "sentence-transformers"
        ),
        "faiss_version": importlib.metadata.version("faiss-cpu"),
        "device": str(getattr(embedding_model, "device", args.device)),
        "indexed_chunk_count": len(chunks),
        "parent_count": parent_store.parent_count if parent_store else None,
    }
    if tracing is not None:
        _attach_parent_diagnostics(payload, tracing.traces)
    _write_json_atomic(payload, args.output_path.expanduser())
    summary = payload["summary"]
    print(
        f"{args.context_mode}/{args.retriever}: "
        f"Hit@5={summary['hit_at_5']:.4f} "
        f"Hit@10={summary['hit_at_10']:.4f} "
        f"MRR@10={summary['mrr_at_10']:.4f}"
    )
    print(f"Saved: {args.output_path}")
    return 0


def _validate_parent_manifest(manifest: dict[str, Any], parent_path: Path) -> None:
    if manifest.get("revision") != PARENT_RETRIEVAL_REVISION:
        raise ValueError("child manifest has an unsupported parent revision")
    if manifest.get("parent_metadata_sha256") != _sha256(parent_path):
        raise ValueError("parent metadata SHA-256 does not match child manifest")


def _attach_parent_diagnostics(
    payload: dict[str, Any],
    traces: list[Any],
) -> None:
    questions = payload["questions"]
    if len(questions) != len(traces):
        raise RuntimeError("parent diagnostic count does not match questions")
    for question, trace in zip(questions, traces, strict=True):
        question["parent_diagnostics"] = {
            "child_candidate_count": trace.child_candidate_count,
            "unique_parent_candidate_count": trace.unique_parent_candidate_count,
            "child_to_parent_compression_ratio": trace.compression_ratio,
            "maximum_children_for_one_parent": (trace.maximum_children_for_one_parent),
            "matches": [
                {
                    "parent_rank": match.parent_rank,
                    "matched_child_rank": match.matched_child_rank,
                    "matched_child_text": match.matched_child_text,
                    "matched_child_index": match.matched_child_index,
                    "parent_id": match.parent_id,
                    "child_hits_for_parent": match.child_hits_for_parent,
                }
                for match in trace.matches
            ],
        }
    payload["summary"]["average_unique_parent_candidates"] = sum(
        trace.unique_parent_candidate_count for trace in traces
    ) / len(traces)
    payload["summary"]["average_child_to_parent_compression_ratio"] = sum(
        trace.compression_ratio for trace in traces
    ) / len(traces)
    payload["summary"]["maximum_children_for_one_parent"] = max(
        (trace.maximum_children_for_one_parent for trace in traces),
        default=0,
    )


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
    with path.expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _top_k(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 10:
        raise argparse.ArgumentTypeError("top-k must be at least 10")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
