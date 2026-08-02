"""Evaluate the bounded full-section parent retrieval grid."""

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
from python_doc_rag.section_parent import (
    SECTION_PARENT_REVISION,
    SectionParentRetriever,
    SectionStore,
)
from python_doc_rag.vector_store import load_chunks_jsonl, load_vector_index

_HYBRID_NGRAM_SIZES = (2,)
_HYBRID_RRF_K = 10
_HYBRID_CANDIDATE_K = 30
_DEFAULT_CONFIGS = ((300, 75), (400, 100))


class _TracingSectionSearcher:
    """Capture one section-parent diagnostic trace per evaluation query."""

    def __init__(
        self,
        searcher: Any,
        section_retriever: SectionParentRetriever,
        reranker: RerankingRetriever | None,
    ) -> None:
        self._searcher = searcher
        self._section_retriever = section_retriever
        self._reranker = reranker
        self.section_traces: list[Any] = []
        self.reranking_traces: list[Any] = []

    def search(self, query: str, *, top_k: int = 5) -> Any:
        """Retrieve sections and retain internal diagnostics."""
        results = self._searcher.search(query, top_k=top_k)
        self.section_traces.append(self._section_retriever.last_trace)
        if self._reranker is not None:
            self.reranking_traces.append(self._reranker.last_trace)
        return results


def parse_args() -> argparse.Namespace:
    """Parse the two-setting Development grid or one frozen setting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-label",
        choices=("development", "holdout"),
        default="development",
    )
    parser.add_argument(
        "--child-config",
        nargs="+",
        type=_child_config,
        default=_DEFAULT_CONFIGS,
    )
    parser.add_argument(
        "--retriever",
        nargs="+",
        choices=("dense", "hybrid"),
        default=("dense", "hybrid"),
    )
    parser.add_argument("--child-candidate-k", type=_candidate_k, default=30)
    parser.add_argument(
        "--reranker-model-key",
        choices=("mmarco-minilm",),
    )
    parser.add_argument("--reranker-candidate-k", type=_candidate_k, default=30)
    parser.add_argument("--reranker-batch-size", type=_positive_int, default=16)
    parser.add_argument("--reranker-max-length", type=_positive_int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=_top_k, default=10)
    return parser.parse_args()


def main() -> int:
    """Load sections and embedding once, then evaluate the bounded grid."""
    args = parse_args()
    configs = list(dict.fromkeys(args.child_config))
    if len(configs) > 2:
        raise ValueError("at most two section child configurations may be evaluated")
    retrievers = list(dict.fromkeys(args.retriever))
    artifact_root = args.artifact_root.expanduser()
    questions_path = args.questions.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    section_path = artifact_root / "sections.jsonl"
    section_manifest_path = artifact_root / "section_manifest.json"
    section_manifest = _read_json(section_manifest_path)
    _validate_section_manifest(section_manifest, section_path)
    section_store = SectionStore.from_jsonl(section_path)
    questions = load_evaluation_questions(questions_path)

    import torch
    from sentence_transformers import SentenceTransformer

    embedding_model_name = _configured_model(artifact_root, configs[0])
    embedding_model = SentenceTransformer(embedding_model_name, device=args.device)
    reranker_spec = None
    reranker_scorer = None
    if args.reranker_model_key is not None:
        reranker_spec = model_spec_for_key(args.reranker_model_key)
        reranker_scorer = CrossEncoderPairScorer.from_pretrained(
            reranker_spec,
            device=args.device,
            max_length=args.reranker_max_length,
        )
    for size, overlap in configs:
        config_root = artifact_root / f"child-{size}-overlap-{overlap}"
        index_path = config_root / "child.faiss"
        metadata_path = config_root / "child_metadata.jsonl"
        manifest_path = config_root / "child_index_manifest.json"
        manifest = _read_json(manifest_path)
        _validate_child_manifest(
            manifest,
            section_path=section_path,
            metadata_path=metadata_path,
            index_path=index_path,
            child_size=size,
            child_overlap=overlap,
        )
        if _configured_model(artifact_root, (size, overlap)) != embedding_model_name:
            raise ValueError("section child configurations use different embedding models")
        vector_index = load_vector_index(
            index_path,
            metadata_path,
            embedding_model=embedding_model,
        )
        children = load_chunks_jsonl(metadata_path)
        if section_store.validate_children(children) != len(children):
            raise RuntimeError("not all section children resolved")
        searchers: dict[str, Any] = {"dense": vector_index}
        if "hybrid" in retrievers:
            searchers["hybrid"] = ReciprocalRankFusionRetriever(
                [
                    VectorIndexRetriever(vector_index),
                    BM25Retriever(
                        children,
                        tokenizer=CodeAwareNgramTokenizer(_HYBRID_NGRAM_SIZES),
                    ),
                ],
                rrf_k=_HYBRID_RRF_K,
                candidate_k=_HYBRID_CANDIDATE_K,
            )
        for retriever_name in retrievers:
            section_retriever = SectionParentRetriever(
                searchers[retriever_name],
                section_store,
                child_candidate_k=args.child_candidate_k,
            )
            reranker = None
            searcher: Any = section_retriever
            if reranker_scorer is not None:
                reranker = RerankingRetriever(
                    section_retriever,
                    reranker_scorer,
                    candidate_k=args.reranker_candidate_k,
                    batch_size=args.reranker_batch_size,
                )
                searcher = reranker
            tracing = _TracingSectionSearcher(
                searcher,
                section_retriever,
                reranker,
            )
            report = evaluate_retrieval(tracing, questions, top_k=args.top_k)
            payload = retrieval_evaluation_to_dict(report)
            _attach_section_diagnostics(payload, tracing.section_traces)
            if tracing.reranking_traces:
                _attach_reranking_diagnostics(payload, tracing.reranking_traces)
            payload["settings"] = {
                "revision": SECTION_PARENT_REVISION,
                "retriever": retriever_name,
                "child_size": size,
                "child_overlap": overlap,
                "child_candidate_k": args.child_candidate_k,
                "top_k": args.top_k,
                "questions_path": str(questions_path),
                "questions_sha256": _sha256(questions_path),
                "section_path": str(section_path),
                "section_sha256": _sha256(section_path),
                "section_manifest_path": str(section_manifest_path),
                "section_manifest_sha256": _sha256(section_manifest_path),
                "index_path": str(index_path),
                "index_sha256": _sha256(index_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "index_manifest_path": str(manifest_path),
                "index_manifest_sha256": _sha256(manifest_path),
                "embedding_model": embedding_model_name,
                "search_fields": ["page_title", "section_title", "text"],
                "reranking_revision": (
                    RERANKING_REVISION if reranker_spec is not None else None
                ),
                "reranker_model_key": (
                    reranker_spec.key if reranker_spec is not None else None
                ),
                "reranker_model": (
                    reranker_spec.model_name if reranker_spec is not None else None
                ),
                "reranker_model_revision": (
                    reranker_spec.revision if reranker_spec is not None else None
                ),
                "reranker_model_license": (
                    reranker_spec.license if reranker_spec is not None else None
                ),
                "reranker_trust_remote_code": (
                    False if reranker_spec is not None else None
                ),
                "reranker_candidate_k": (
                    args.reranker_candidate_k if reranker_spec is not None else None
                ),
                "reranker_batch_size": (
                    args.reranker_batch_size if reranker_spec is not None else None
                ),
                "reranker_max_length": (
                    args.reranker_max_length if reranker_spec is not None else None
                ),
                "japanese_ngram_sizes": (
                    list(_HYBRID_NGRAM_SIZES)
                    if retriever_name == "hybrid"
                    else None
                ),
                "rrf_k": _HYBRID_RRF_K if retriever_name == "hybrid" else None,
                "hybrid_candidate_k": (
                    _HYBRID_CANDIDATE_K if retriever_name == "hybrid" else None
                ),
            }
            payload["environment"] = {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "sentence_transformers_version": importlib.metadata.version(
                    "sentence-transformers"
                ),
                "faiss_version": importlib.metadata.version("faiss-cpu"),
                "device": args.device,
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if args.device.startswith("cuda") and torch.cuda.is_available()
                    else None
                ),
                "section_count": section_store.section_count,
                "indexed_child_count": len(children),
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
            reranker_suffix = (
                f"_{reranker_spec.key}" if reranker_spec is not None else ""
            )
            output_path = output_dir / (
                f"section_{size}_{overlap}_{retriever_name}{reranker_suffix}_"
                f"{args.dataset_label}.json"
            )
            _write_json_atomic(payload, output_path)
            summary = payload["summary"]
            print(
                f"{size}/{overlap}/{retriever_name}: "
                f"Hit@5={summary['hit_at_5']:.4f} "
                f"MRR@10={summary['mrr_at_10']:.4f}",
                flush=True,
            )
    return 0


def _configured_model(artifact_root: Path, config: tuple[int, int]) -> str:
    size, overlap = config
    manifest = _read_json(
        artifact_root
        / f"child-{size}-overlap-{overlap}"
        / "child_index_manifest.json"
    )
    model = manifest.get("model_name")
    if not isinstance(model, str) or not model:
        model = manifest.get("embedding_model")
    if not isinstance(model, str) or not model:
        raise ValueError("section child manifest has no embedding model")
    return model


def _validate_section_manifest(manifest: dict[str, Any], section_path: Path) -> None:
    if manifest.get("revision") != SECTION_PARENT_REVISION:
        raise ValueError("section manifest has unsupported revision")
    if manifest.get("section_sha256") != _sha256(section_path):
        raise ValueError("section SHA-256 does not match section manifest")
    if manifest.get("section_count") != _line_count(section_path):
        raise ValueError("section count does not match section manifest")
    if manifest.get("extracted_section_count") != manifest.get("section_count"):
        raise ValueError("parser result count does not match section manifest")


def _validate_child_manifest(
    manifest: dict[str, Any],
    *,
    section_path: Path,
    metadata_path: Path,
    index_path: Path,
    child_size: int,
    child_overlap: int,
) -> None:
    expected = {
        "revision": SECTION_PARENT_REVISION,
        "section_sha256": _sha256(section_path),
        "child_metadata_sha256": _sha256(metadata_path),
        "child_index_sha256": _sha256(index_path),
        "child_size": child_size,
        "child_overlap": child_overlap,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{key} does not match section child manifest")
    if manifest.get("child_count") != _line_count(metadata_path):
        raise ValueError("child count does not match section child manifest")


def _attach_section_diagnostics(
    payload: dict[str, Any],
    traces: list[Any],
) -> None:
    questions = payload["questions"]
    if len(questions) != len(traces):
        raise RuntimeError("section trace count does not match questions")
    for question, trace in zip(questions, traces, strict=True):
        question["section_parent_diagnostics"] = {
            "child_candidate_count": trace.child_candidate_count,
            "unique_section_candidate_count": trace.unique_parent_candidate_count,
            "child_to_section_compression_ratio": trace.compression_ratio,
            "maximum_children_for_one_section": (
                trace.maximum_children_for_one_parent
            ),
            "matches": [
                {
                    "section_rank": match.parent_rank,
                    "matched_child_rank": match.matched_child_rank,
                    "section_id": match.section_id,
                    "matched_child_start": match.matched_child_start,
                    "matched_child_end": match.matched_child_end,
                    "child_hits_for_section": match.child_hits_for_section,
                }
                for match in trace.matches
            ],
        }
    payload["summary"]["average_unique_section_candidates"] = sum(
        trace.unique_parent_candidate_count for trace in traces
    ) / len(traces)
    payload["summary"]["average_child_to_section_compression_ratio"] = sum(
        trace.compression_ratio for trace in traces
    ) / len(traces)
    payload["summary"]["maximum_children_for_one_section"] = max(
        (trace.maximum_children_for_one_parent for trace in traces),
        default=0,
    )


def _attach_reranking_diagnostics(
    payload: dict[str, Any],
    traces: list[Any],
) -> None:
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


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _child_config(value: str) -> tuple[int, int]:
    try:
        size_text, overlap_text = value.split(":", maxsplit=1)
        size = int(size_text)
        overlap = int(overlap_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("child config must be SIZE:OVERLAP") from error
    if (size, overlap) not in {(300, 75), (400, 100)}:
        raise argparse.ArgumentTypeError("child config must be 300:75 or 400:100")
    return size, overlap


def _candidate_k(value: str) -> int:
    parsed = _positive_int(value)
    if parsed not in {30, 60}:
        raise argparse.ArgumentTypeError("candidate-k must be 30 or 60")
    return parsed


def _top_k(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 10:
        raise argparse.ArgumentTypeError("top-k must be at least 10")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
