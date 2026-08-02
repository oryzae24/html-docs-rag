"""Evaluate bounded candidate-recall strategies before local reranking."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
from pathlib import Path
from time import perf_counter
from typing import Any

from python_doc_rag.candidate_evaluation import (
    evaluate_candidate_recall,
    save_candidate_evaluation_atomic,
)
from python_doc_rag.embedding_models import retrieval_embedding_spec
from python_doc_rag.evaluation import (
    evaluate_retrieval,
    load_evaluation_questions,
    retrieval_evaluation_to_dict,
)
from python_doc_rag.reranking import (
    CrossEncoderPairScorer,
    RerankingRetriever,
    model_spec_for_key,
)
from python_doc_rag.retrieval import (
    BM25Retriever,
    CodeAwareNgramTokenizer,
    ReciprocalRankFusionRetriever,
    VectorIndexRetriever,
)
from python_doc_rag.technical_retrieval import (
    FieldBM25Retriever,
    SymbolRetriever,
    WeightedRankFusionRetriever,
    load_symbol_sidecar,
    write_symbol_sidecar_atomic,
)
from python_doc_rag.vector_store import load_chunks_jsonl, load_vector_index

_FIELD_CONFIGS: dict[str, dict[str, float]] = {
    "equal": {
        "identifiers": 1.0,
        "section_title": 1.0,
        "page_title": 1.0,
        "body_dense": 1.0,
        "body_lexical": 1.0,
    },
    "identifier-priority": {
        "identifiers": 3.0,
        "section_title": 1.0,
        "page_title": 1.0,
        "body_dense": 1.0,
        "body_lexical": 1.0,
    },
    "title-priority": {
        "identifiers": 1.0,
        "section_title": 2.0,
        "page_title": 2.0,
        "body_dense": 1.0,
        "body_lexical": 1.0,
    },
    "identifier-title-priority": {
        "identifiers": 3.0,
        "section_title": 2.0,
        "page_title": 2.0,
        "body_dense": 1.0,
        "body_lexical": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse one frozen candidate configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--symbol-sidecar", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("recommended-v1", "field", "embedding", "field-embedding"),
        required=True,
    )
    parser.add_argument("--field-config", choices=tuple(_FIELD_CONFIGS))
    parser.add_argument(
        "--embedding-key", choices=("bge-m3", "multilingual-e5-base")
    )
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--hard-case-id", action="append", default=[])
    parser.add_argument("--source-code-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-k", type=_candidate_k, default=30)
    parser.add_argument("--reranker-batch-size", type=_positive_int, default=16)
    return parser.parse_args()


def main() -> int:
    """Build reusable searchers, measure candidates, then rerank the same mode."""
    args = parse_args()
    _validate_args(args)
    import torch
    from sentence_transformers import SentenceTransformer

    data_root = args.data_root.expanduser()
    metadata_path = data_root / "indexes/python_3_13_ja_metadata.jsonl"
    baseline_index_path = data_root / "indexes/python_3_13_ja.faiss"
    baseline_manifest_path = data_root / "indexes/python_3_13_ja_index_manifest.json"
    chunks = load_chunks_jsonl(metadata_path)
    if not args.symbol_sidecar.exists():
        symbol_summary = write_symbol_sidecar_atomic(chunks, args.symbol_sidecar)
    else:
        symbol_summary = None
    records = load_symbol_sidecar(chunks, args.symbol_sidecar)

    embedding_started = perf_counter()
    embedding_spec = None
    if args.mode in {"embedding", "field-embedding"}:
        embedding_spec = retrieval_embedding_spec(args.embedding_key)
        embedding_model_name = embedding_spec.model_name
        index_root = args.embedding_root.expanduser() / embedding_spec.key
        index_path = index_root / "index.faiss"
        model = SentenceTransformer(
            embedding_spec.model_name,
            revision=embedding_spec.revision,
            device=args.device,
            trust_remote_code=False,
        )
        body_searcher = load_vector_index(
            index_path,
            index_root / "metadata.jsonl",
            embedding_model=model,
            query_prefix=embedding_spec.query_prefix,
        )
        selected_manifest_path = index_root / "manifest.json"
    else:
        baseline_manifest = _read_json(baseline_manifest_path)
        embedding_model_name = str(baseline_manifest["model_name"])
        model = SentenceTransformer(
            embedding_model_name, device=args.device
        )
        body_searcher = load_vector_index(
            baseline_index_path,
            metadata_path,
            embedding_model=model,
        )
        index_path = baseline_index_path
        selected_manifest_path = baseline_manifest_path
    embedding_load_seconds = perf_counter() - embedding_started

    lexical_all = BM25Retriever(
        chunks, tokenizer=CodeAwareNgramTokenizer((2,))
    )
    if args.mode == "recommended-v1":
        candidate_searcher: Any = ReciprocalRankFusionRetriever(
            [VectorIndexRetriever(body_searcher), lexical_all],
            rrf_k=10,
            candidate_k=30,
        )
        field_weights = None
    elif args.mode == "embedding":
        candidate_searcher = body_searcher
        field_weights = None
    else:
        field_weights = _FIELD_CONFIGS[args.field_config]
        field_retrievers = {
            "identifiers": SymbolRetriever(chunks, records),
            "section_title": FieldBM25Retriever(
                chunks, field="section_title"
            ),
            "page_title": FieldBM25Retriever(chunks, field="page_title"),
            "body_dense": VectorIndexRetriever(body_searcher),
            "body_lexical": FieldBM25Retriever(chunks, field="body"),
        }
        candidate_searcher = WeightedRankFusionRetriever(
            [
                (name, field_retrievers[name], weight)
                for name, weight in field_weights.items()
            ],
            rrf_k=10,
            candidate_k=30,
        )

    questions = load_evaluation_questions(args.questions)
    candidate_payload = evaluate_candidate_recall(
        candidate_searcher,
        questions,
        candidate_k=args.candidate_k,
        hard_case_ids=args.hard_case_id,
    )
    reranker_spec = model_spec_for_key("mmarco-minilm")
    reranker_load_started = perf_counter()
    scorer = CrossEncoderPairScorer.from_pretrained(
        reranker_spec,
        device=args.device,
        max_length=512,
    )
    reranker_load_seconds = perf_counter() - reranker_load_started
    reranker = RerankingRetriever(
        candidate_searcher,
        scorer,
        candidate_k=args.candidate_k,
        batch_size=args.reranker_batch_size,
    )
    reranked = retrieval_evaluation_to_dict(
        evaluate_retrieval(reranker, questions, top_k=10)
    )
    candidate_payload.update(
        {
            "reranked": reranked,
            "settings": {
                "mode": args.mode,
                "field_config": args.field_config,
                "field_weights": field_weights,
                "candidate_k": args.candidate_k,
                "field_rrf_k": 10,
                "recommended_v1_rrf_k": 10,
                "recommended_v1_hybrid_candidate_k": 30,
                "japanese_ngram_sizes": [2],
                "questions_path": str(args.questions.expanduser()),
                "questions_sha256": _sha256(args.questions),
                "source_code_commit": args.source_code_commit,
                "input_metadata_sha256": _sha256(metadata_path),
                "input_index_sha256": _sha256(index_path),
                "input_manifest_sha256": _sha256(selected_manifest_path),
                "symbol_sidecar_sha256": _sha256(args.symbol_sidecar),
                "embedding_model": embedding_model_name,
                "embedding_spec": (
                    {
                        "key": embedding_spec.key,
                        "model_name": embedding_spec.model_name,
                        "revision": embedding_spec.revision,
                        "license": embedding_spec.license,
                        "model_card_url": embedding_spec.model_card_url,
                        "dimension": embedding_spec.embedding_dimension,
                        "max_sequence_length": embedding_spec.max_sequence_length,
                        "pooling": embedding_spec.pooling,
                        "query_prefix": embedding_spec.query_prefix,
                        "document_prefix": embedding_spec.document_prefix,
                        "normalized": embedding_spec.normalize_embeddings,
                        "trust_remote_code": embedding_spec.trust_remote_code,
                    }
                    if embedding_spec is not None
                    else None
                ),
                "reranker_model": reranker_spec.model_name,
                "reranker_revision": reranker_spec.revision,
                "openai_api_used": False,
                "contains_secrets": False,
            },
            "environment": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "sentence_transformers_version": importlib.metadata.version(
                    "sentence-transformers"
                ),
                "transformers_version": importlib.metadata.version("transformers"),
                "faiss_version": importlib.metadata.version("faiss-cpu"),
                "device": args.device,
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "embedding_load_seconds": embedding_load_seconds,
                "reranker_load_seconds": reranker_load_seconds,
                "peak_cpu_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
                "gpu_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else None
                ),
                "gpu_peak_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved())
                    if torch.cuda.is_available()
                    else None
                ),
                "index_size_bytes": index_path.stat().st_size,
                "symbol_sidecar_size_bytes": args.symbol_sidecar.stat().st_size,
                "symbol_sidecar_built_in_run": symbol_summary is not None,
            },
        }
    )
    save_candidate_evaluation_atomic(candidate_payload, args.output_path)
    summary = candidate_payload["summary"]
    reranked_summary = reranked["summary"]
    print(
        f"{args.mode}/{args.field_config or args.embedding_key or 'baseline'}: "
        f"Recall@30={summary['recall_at_30']:.4f} "
        f"Hit@5={reranked_summary['hit_at_5']:.4f} "
        f"MRR@10={reranked_summary['mrr_at_10']:.4f}"
    )
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode in {"field", "field-embedding"} and args.field_config is None:
        raise ValueError("field modes require --field-config")
    if args.mode in {"embedding", "field-embedding"}:
        if args.embedding_key is None or args.embedding_root is None:
            raise ValueError("embedding modes require key and root")
    elif args.embedding_key is not None or args.embedding_root is not None:
        raise ValueError("embedding options are only valid for embedding modes")
    if len(args.source_code_commit) != 40:
        raise ValueError("source-code-commit must be a full Git hash")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _candidate_k(value: str) -> int:
    parsed = _positive_int(value)
    if parsed != 30:
        raise argparse.ArgumentTypeError("candidate-k is frozen to 30")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
