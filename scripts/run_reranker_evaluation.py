"""Evaluate two bounded local-reranker candidates on Development questions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from python_doc_rag.evaluation import (
    evaluate_retrieval,
    load_evaluation_questions,
    retrieval_evaluation_to_dict,
)
from python_doc_rag.reranking import (
    RERANKING_REVISION,
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
from python_doc_rag.vector_store import load_chunks_jsonl, load_vector_index

_HYBRID_NGRAM_SIZES = (2,)
_HYBRID_RRF_K = 10
_HYBRID_CANDIDATE_K = 30


class _TracingReranker:
    """Collect one internal-only reranking trace per question."""

    def __init__(self, reranker: RerankingRetriever) -> None:
        self._reranker = reranker
        self.traces: list[Any] = []

    def search(self, query: str, *, top_k: int = 5) -> Any:
        """Rerank once and retain diagnostics outside returned chunks."""
        results = self._reranker.search(query, top_k=top_k)
        self.traces.append(self._reranker.last_trace)
        return results


def parse_args() -> argparse.Namespace:
    """Parse one-model bounded Development sweep settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-label",
        choices=("development", "holdout"),
        default="development",
    )
    parser.add_argument("--model-key", choices=("bge-m3", "mmarco-minilm"), required=True)
    parser.add_argument("--candidate-source", nargs="+", choices=("dense", "hybrid"), default=("dense", "hybrid"))
    parser.add_argument("--candidate-k", nargs="+", type=_candidate_k, default=(20, 30))
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--top-k", type=_top_k, default=10)
    return parser.parse_args()


def main() -> int:
    """Load each model once and evaluate only the bounded parameter grid."""
    args = parse_args()
    data_root = args.data_root.expanduser()
    questions_path = args.questions.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = data_root / "indexes/python_3_13_ja.faiss"
    metadata_path = data_root / "indexes/python_3_13_ja_metadata.jsonl"
    manifest_path = data_root / "indexes/python_3_13_ja_index_manifest.json"
    manifest = _read_json(manifest_path)
    embedding_model_name = str(manifest["model_name"])

    import torch
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(embedding_model_name, device=args.device)
    vector_index = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=embedding_model,
    )
    chunks = load_chunks_jsonl(metadata_path)
    dense_retriever = VectorIndexRetriever(vector_index)
    searchers: dict[str, Any] = {"dense": vector_index}
    if "hybrid" in args.candidate_source:
        bm25 = BM25Retriever(
            chunks,
            tokenizer=CodeAwareNgramTokenizer(_HYBRID_NGRAM_SIZES),
        )
        searchers["hybrid"] = ReciprocalRankFusionRetriever(
            [dense_retriever, bm25],
            rrf_k=_HYBRID_RRF_K,
            candidate_k=_HYBRID_CANDIDATE_K,
        )

    spec = model_spec_for_key(args.model_key)
    model_load_started = perf_counter()
    scorer = CrossEncoderPairScorer.from_pretrained(
        spec,
        device=args.device,
        max_length=args.max_length,
    )
    model_load_seconds = perf_counter() - model_load_started
    questions = load_evaluation_questions(questions_path)

    for source in dict.fromkeys(args.candidate_source):
        for candidate_k in dict.fromkeys(args.candidate_k):
            reranker = RerankingRetriever(
                searchers[source],
                scorer,
                candidate_k=candidate_k,
                batch_size=args.batch_size,
            )
            tracing = _TracingReranker(reranker)
            report = evaluate_retrieval(tracing, questions, top_k=args.top_k)
            payload = retrieval_evaluation_to_dict(report)
            _attach_traces(payload, tracing.traces)
            payload["settings"] = {
                "revision": RERANKING_REVISION,
                "model_key": spec.key,
                "model_name": spec.model_name,
                "model_revision": spec.revision,
                "model_license": spec.license,
                "model_card_url": spec.model_card_url,
                "language_evidence": spec.language_evidence,
                "trust_remote_code": False,
                "candidate_source": source,
                "candidate_k": candidate_k,
                "top_k": args.top_k,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "questions_path": str(questions_path),
                "questions_sha256": _sha256(questions_path),
                "index_sha256": _sha256(index_path),
                "metadata_sha256": _sha256(metadata_path),
                "manifest_sha256": _sha256(manifest_path),
                "embedding_model": embedding_model_name,
                "reranker_document_fields": [
                    "page_title",
                    "section_title",
                    "text",
                ],
            }
            payload["environment"] = {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "sentence_transformers_version": importlib.metadata.version(
                    "sentence-transformers"
                ),
                "transformers_version": importlib.metadata.version("transformers"),
                "faiss_version": importlib.metadata.version("faiss-cpu"),
                "device": args.device,
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if args.device.startswith("cuda") and torch.cuda.is_available()
                    else None
                ),
                "model_load_seconds": model_load_seconds,
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
            }
            output_path = output_dir / (
                f"{spec.key}_{source}_k{candidate_k}_{args.dataset_label}.json"
            )
            _write_json_atomic(payload, output_path)
            summary = payload["summary"]
            print(
                f"{spec.key}/{source}/k={candidate_k}: "
                f"Hit@5={summary['hit_at_5']:.4f} "
                f"MRR@10={summary['mrr_at_10']:.4f}",
                flush=True,
            )
    return 0


def _attach_traces(payload: dict[str, Any], traces: list[Any]) -> None:
    questions = payload["questions"]
    if len(questions) != len(traces):
        raise RuntimeError("reranking trace count does not match questions")
    for question, trace in zip(questions, traces, strict=True):
        question["reranking"] = {
            "candidate_count": trace.candidate_count,
            "batch_size": trace.batch_size,
            "scoring_seconds": trace.scoring_seconds,
            "matches": [
                {
                    "rank": match.rank,
                    "original_rank": match.original_rank,
                    "rerank_score": match.rerank_score,
                    "original_score": match.original_score,
                    "source_url": match.source_url,
                    "chunk_index": match.chunk_index,
                    "start_index": match.start_index,
                }
                for match in trace.matches
            ],
        }
    payload["summary"]["average_reranking_seconds"] = sum(
        trace.scoring_seconds for trace in traces
    ) / len(traces)


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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _candidate_k(value: str) -> int:
    parsed = _positive_int(value)
    if parsed not in {20, 30}:
        raise argparse.ArgumentTypeError("candidate-k must be 20 or 30")
    return parsed


def _top_k(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 10:
        raise argparse.ArgumentTypeError("top-k must be at least 10")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
