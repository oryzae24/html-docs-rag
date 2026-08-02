"""Evaluate Dense, BM25, or RRF Hybrid retrieval without a generation model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

from python_doc_rag.evaluation import (
    evaluate_retrieval,
    load_evaluation_questions,
    save_retrieval_evaluation,
)
from python_doc_rag.retrieval import (
    BM25Retriever,
    CodeAwareNgramTokenizer,
    ReciprocalRankFusionRetriever,
    VectorIndexRetriever,
)
from python_doc_rag.vector_store import load_chunks_jsonl, load_vector_index


def parse_args() -> argparse.Namespace:
    """Parse portable evaluation paths and settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--index-manifest-path", type=Path)
    parser.add_argument("--embedding-model")
    parser.add_argument("--device")
    parser.add_argument("--top-k", type=_top_k, default=10)
    parser.add_argument(
        "--retriever",
        choices=("dense", "bm25", "hybrid"),
        default="dense",
    )
    parser.add_argument("--ngram-sizes", type=_ngram_sizes, default=(2, 3))
    parser.add_argument("--rrf-k", type=_positive_int, default=60)
    parser.add_argument("--candidate-k", type=_positive_int, default=20)
    return parser.parse_args()


def main() -> int:
    """Build the selected retriever once, evaluate all questions, and save JSON."""
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    data_root = _resolve_data_root(args.data_root)
    index_path = args.index_path or data_root / "indexes/python_3_13_ja.faiss"
    metadata_path = (
        args.metadata_path
        or data_root / "indexes/python_3_13_ja_metadata.jsonl"
    )
    manifest_path = (
        args.index_manifest_path
        or data_root / "indexes/python_3_13_ja_index_manifest.json"
    )
    questions_path = args.questions or repository_root / "evaluation/questions.jsonl"
    output_path = args.output_path or _default_output_path(data_root, args.retriever)

    questions = load_evaluation_questions(questions_path)
    chunks = load_chunks_jsonl(metadata_path)
    tokenizer = CodeAwareNgramTokenizer(args.ngram_sizes)
    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "metadata_path": str(metadata_path.expanduser()),
        "index_chunk_count": len(chunks),
    }
    settings: dict[str, Any] = {
        "retriever": args.retriever,
        "questions_path": str(questions_path.expanduser()),
        "output_path": str(output_path.expanduser()),
        "top_k": args.top_k,
        "search_fields": ["page_title", "section_title", "text"],
        "field_combination": "labeled fields joined by newlines",
    }

    searcher: Any
    if args.retriever == "bm25":
        searcher = BM25Retriever(chunks, tokenizer=tokenizer)
        settings.update(
            {
                "japanese_ngram_sizes": list(args.ngram_sizes),
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
            }
        )
    else:
        manifest = _read_json_object(manifest_path)
        configured_model = manifest.get("model_name")
        if not isinstance(configured_model, str) or not configured_model.strip():
            raise ValueError(f"manifest has no valid model_name: {manifest_path}")
        embedding_model_name = args.embedding_model or configured_model

        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            embedding_model_name,
            device=args.device,
        )
        vector_index = load_vector_index(
            index_path,
            metadata_path,
            embedding_model=embedding_model,
        )
        environment.update(
            {
                "sentence_transformers_version": importlib.metadata.version(
                    "sentence-transformers"
                ),
                "faiss_version": importlib.metadata.version("faiss-cpu"),
                "embedding_model": embedding_model_name,
                "device": str(getattr(embedding_model, "device", "unknown")),
                "index_path": str(index_path.expanduser()),
                "manifest_path": str(manifest_path.expanduser()),
                "embedding_dimension": vector_index.embedding_dimension,
            }
        )
        if args.retriever == "dense":
            searcher = vector_index
        else:
            bm25 = BM25Retriever(chunks, tokenizer=tokenizer)
            searcher = ReciprocalRankFusionRetriever(
                [VectorIndexRetriever(vector_index), bm25],
                rrf_k=args.rrf_k,
                candidate_k=args.candidate_k,
            )
            settings.update(
                {
                    "japanese_ngram_sizes": list(args.ngram_sizes),
                    "bm25_k1": 1.5,
                    "bm25_b": 0.75,
                    "rrf_k": args.rrf_k,
                    "candidate_k": args.candidate_k,
                    "retriever_weights": [1.0, 1.0],
                }
            )

    report = evaluate_retrieval(searcher, questions, top_k=args.top_k)
    save_retrieval_evaluation(
        report,
        output_path,
        environment=environment,
        settings=settings,
    )
    _print_summary(report, output_path)
    return 0


def _resolve_data_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.expanduser()
    configured = os.environ.get("PYTHON_DOC_RAG_DATA_ROOT")
    if configured and configured.strip():
        return Path(configured).expanduser()
    raise ValueError("set PYTHON_DOC_RAG_DATA_ROOT or pass --data-root")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _top_k(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-k must be an integer") from error
    if parsed < 10:
        raise argparse.ArgumentTypeError("top-k must be at least 10")
    return parsed



def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _ngram_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ngram-sizes must be comma-separated integers"
        ) from error
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError(
            "ngram-sizes must contain positive integers"
        )
    return sizes


def _default_output_path(data_root: Path, retriever: str) -> Path:
    filenames = {
        "dense": "dense_baseline.json",
        "bm25": "bm25_baseline.json",
        "hybrid": "hybrid_rrf.json",
    }
    try:
        filename = filenames[retriever]
    except KeyError as error:
        raise ValueError(f"unsupported retriever: {retriever}") from error
    return data_root / "evaluation" / filename

def _print_summary(report: Any, output_path: Path) -> None:
    print(f"Questions: {report.question_count}")
    print(f"Hit@1: {report.top_1_hit_rate:.4f}")
    print(f"Hit@3: {report.top_3_hit_rate:.4f}")
    print(f"Hit@5: {report.top_5_hit_rate:.4f}")
    print(f"Hit@10: {report.top_10_hit_rate:.4f}")
    print(f"MRR@10: {report.mrr_at_10:.4f}")
    print(f"Average retrieval: {report.average_retrieval_seconds:.4f}s")
    print(f"Median retrieval: {report.median_retrieval_seconds:.4f}s")
    for name, metrics in report.query_type_metrics.items():
        print(
            f"{name}: Hit@5={metrics.hit_at_5:.4f}, "
            f"Hit@10={metrics.hit_at_10:.4f}, MRR@10={metrics.mrr_at_10:.4f}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
