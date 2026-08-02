"""Run Dense or Hybrid Qwen RAG quality and answerability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
from pathlib import Path
from time import perf_counter
from typing import Any

from python_doc_rag.answer_contract import (
    ANSWER_MODE_LEGACY,
    ANSWER_MODES,
    contract_revision_for_mode,
    generation_contract_for_mode,
)
from python_doc_rag.config import DEFAULT_GENERATION_MODEL
from python_doc_rag.generation import (
    ChatTemplatePromptSerializer,
    GenerationConfig,
)
from python_doc_rag.parent_retrieval import (
    PARENT_RETRIEVAL_REVISION,
    ParentDocumentRetriever,
    ParentStore,
)
from python_doc_rag.pipeline import RagPipeline
from python_doc_rag.rag_evaluation import (
    DERIVED_RESULT_REVISION,
    JUDGE_PROMPT_REVISION,
    JUDGE_SCHEMA_REVISION,
    OpenAIResponsesJudge,
    RecordingRetriever,
    apply_judge_to_records,
    evaluate_rag_questions,
    load_answerability_questions,
    load_rag_quality_questions,
    load_rag_records,
    save_rag_evaluation,
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
    SectionContextResolver,
    SectionParentRetriever,
    SectionStore,
)
from python_doc_rag.transformers_generation import (
    TransformersAnswerGenerator,
    resolve_device,
)
from python_doc_rag.vector_store import load_chunks_jsonl, load_vector_index

_HYBRID_NGRAM_SIZES = (2,)
_HYBRID_RRF_K = 10
_HYBRID_CANDIDATE_K = 30


def parse_args() -> argparse.Namespace:
    """Parse explicit evaluation and model settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument(
        "--dataset-type",
        choices=("auto", "rag-quality", "answerability"),
        default="auto",
    )
    parser.add_argument("--input-results", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--index-manifest-path", type=Path)
    parser.add_argument(
        "--context-mode",
        choices=("chunk", "parent", "section-parent"),
        default="chunk",
    )
    parser.add_argument("--parent-metadata-path", type=Path)
    parser.add_argument("--child-index-path", type=Path)
    parser.add_argument("--child-metadata-path", type=Path)
    parser.add_argument("--child-manifest-path", type=Path)
    parser.add_argument("--section-path", type=Path)
    parser.add_argument("--section-manifest-path", type=Path)
    parser.add_argument("--child-candidate-k", type=_positive_int, default=60)
    parser.add_argument(
        "--reranker-model-key",
        choices=("bge-m3", "mmarco-minilm"),
    )
    parser.add_argument("--reranker-candidate-k", type=_positive_int, default=30)
    parser.add_argument("--reranker-batch-size", type=_positive_int, default=16)
    parser.add_argument("--reranker-max-length", type=_positive_int, default=512)
    parser.add_argument("--embedding-model")
    parser.add_argument("--model-name", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--retriever", choices=("dense", "hybrid"), default="dense")
    parser.add_argument(
        "--answer-mode",
        choices=ANSWER_MODES,
        default=ANSWER_MODE_LEGACY,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--max-input-tokens", type=_positive_int, default=8192)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=512)
    parser.add_argument("--judge", choices=("none", "openai"), default="none")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--judge-max-retries", type=_non_negative_int, default=2)
    return parser.parse_args()


def main() -> int:
    """Build runtime once or apply a judge to previously saved Qwen output."""
    args = parse_args()
    if (
        args.input_results is not None
        and args.input_results.expanduser().resolve()
        == args.output_path.expanduser().resolve()
    ):
        raise ValueError("--output-path must differ from --input-results")
    judge = _build_judge(args)
    settings = _settings(args, judge_model=getattr(judge, "model_name", None))

    if args.input_results is not None:
        if judge is None:
            raise ValueError("--input-results requires --judge openai")
        records = load_rag_records(args.input_results)
        progress = _progress_writer(args.output_path, settings, {})
        judged = apply_judge_to_records(records, judge, on_case=progress)
        save_rag_evaluation(judged, args.output_path, settings=settings)
        _print_summary(args.output_path)
        return 0

    repository_root = Path(__file__).resolve().parents[1]
    questions_path = args.questions or (
        repository_root / "evaluation/rag_quality_questions.jsonl"
    )
    dataset_type = _resolve_dataset_type(questions_path, args.dataset_type)
    questions = (
        load_rag_quality_questions(questions_path)
        if dataset_type == "rag-quality"
        else load_answerability_questions(questions_path)
    )
    data_root = _resolve_data_root(args.data_root)
    pipeline, trace, environment = build_runtime(args, data_root)
    settings.update(
        {
            "dataset_type": dataset_type,
            "questions_path": str(questions_path.expanduser()),
            "question_count": len(questions),
        }
    )
    progress = _progress_writer(args.output_path, settings, environment)
    records = evaluate_rag_questions(
        pipeline,
        trace,
        questions,
        judge=judge,
        on_case=progress,
    )
    environment.update(_resource_metrics())
    save_rag_evaluation(
        records,
        args.output_path,
        settings=settings,
        environment=environment,
    )
    _print_summary(args.output_path)
    return 0


def build_runtime(
    args: argparse.Namespace,
    data_root: Path,
) -> tuple[RagPipeline, RecordingRetriever, dict[str, Any]]:
    """Load one embedding model, index, retriever, and Qwen generator."""
    import torch
    from sentence_transformers import SentenceTransformer

    runtime_started = perf_counter()
    baseline_index_path = args.index_path or data_root / "indexes/python_3_13_ja.faiss"
    baseline_metadata_path = (
        args.metadata_path or data_root / "indexes/python_3_13_ja_metadata.jsonl"
    )
    baseline_manifest_path = (
        args.index_manifest_path
        or data_root / "indexes/python_3_13_ja_index_manifest.json"
    )
    context_mode = getattr(args, "context_mode", "chunk")
    parent_metadata_path = (
        getattr(args, "parent_metadata_path", None) or baseline_metadata_path
    )
    parent_root = data_root / "indexes/parent_document"
    section_root = data_root / "indexes/section_parent"
    if context_mode in {"parent", "section-parent"}:
        selected_root = parent_root if context_mode == "parent" else section_root
        index_path = (
            getattr(args, "child_index_path", None) or selected_root / "child.faiss"
        )
        metadata_path = (
            getattr(args, "child_metadata_path", None)
            or selected_root / "child_metadata.jsonl"
        )
        manifest_path = (
            getattr(args, "child_manifest_path", None)
            or selected_root / "child_index_manifest.json"
        )
    else:
        index_path = baseline_index_path
        metadata_path = baseline_metadata_path
        manifest_path = baseline_manifest_path
    manifest = _read_json_object(manifest_path)
    configured_model = manifest.get("model_name") or manifest.get("embedding_model")
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ValueError(f"manifest has no valid model_name: {manifest_path}")
    embedding_model_name = args.embedding_model or configured_model
    actual_device = resolve_device(args.device, torch_module=torch)
    embedding_model = SentenceTransformer(
        embedding_model_name,
        device=actual_device,
    )
    index_load_started = perf_counter()
    vector_index = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=embedding_model,
    )
    index_load_seconds = perf_counter() - index_load_started
    dense_retriever = VectorIndexRetriever(vector_index)
    child_searcher: Any = vector_index
    candidate_searcher: Any = vector_index
    retriever: Any = dense_retriever
    if args.retriever == "hybrid":
        chunks = load_chunks_jsonl(metadata_path)
        bm25 = BM25Retriever(
            chunks,
            tokenizer=CodeAwareNgramTokenizer(_HYBRID_NGRAM_SIZES),
        )
        child_searcher = ReciprocalRankFusionRetriever(
            [dense_retriever, bm25],
            rrf_k=_HYBRID_RRF_K,
            candidate_k=_HYBRID_CANDIDATE_K,
        )
        candidate_searcher = child_searcher
        retriever = child_searcher
    parent_store: ParentStore | None = None
    section_store: SectionStore | None = None
    context_selector: SectionContextResolver | None = None
    parent_store_seconds = 0.0
    if context_mode == "parent":
        if manifest.get("revision") != PARENT_RETRIEVAL_REVISION:
            raise ValueError("child manifest has an unsupported parent revision")
        parent_store_started = perf_counter()
        parent_store = ParentStore.from_jsonl(parent_metadata_path)
        parent_store.validate_children(load_chunks_jsonl(metadata_path))
        parent_store_seconds = perf_counter() - parent_store_started
        parent_retriever = ParentDocumentRetriever(
            child_searcher,
            parent_store,
            child_candidate_k=getattr(args, "child_candidate_k", 60),
        )
        candidate_searcher = parent_retriever
        retriever = parent_retriever
    elif context_mode == "section-parent":
        if manifest.get("revision") != SECTION_PARENT_REVISION:
            raise ValueError("child manifest has an unsupported section revision")
        section_path = getattr(args, "section_path", None) or (
            section_root / "sections.jsonl"
        )
        section_manifest_path = getattr(args, "section_manifest_path", None) or (
            section_root / "section_manifest.json"
        )
        section_manifest = _read_json_object(section_manifest_path)
        section_sha256 = _sha256(section_path)
        if section_manifest.get("revision") != SECTION_PARENT_REVISION:
            raise ValueError("section manifest has an unsupported revision")
        if section_manifest.get("section_sha256") != section_sha256:
            raise ValueError("section artifact SHA-256 does not match its manifest")
        if manifest.get("section_sha256") != section_sha256:
            raise ValueError("section artifact SHA-256 does not match child manifest")
        parent_store_started = perf_counter()
        section_store = SectionStore.from_jsonl(section_path)
        section_store.validate_children(load_chunks_jsonl(metadata_path))
        parent_store_seconds = perf_counter() - parent_store_started
        section_retriever = SectionParentRetriever(
            child_searcher,
            section_store,
            child_candidate_k=getattr(args, "child_candidate_k", 30),
        )
        candidate_searcher = section_retriever
        retriever = section_retriever
        context_selector = SectionContextResolver()
    reranker_spec = None
    reranker_load_seconds = 0.0
    reranker_model_key = getattr(args, "reranker_model_key", None)
    if reranker_model_key is not None:
        reranker_spec = model_spec_for_key(reranker_model_key)
        reranker_load_started = perf_counter()
        scorer = CrossEncoderPairScorer.from_pretrained(
            reranker_spec,
            device=actual_device,
            max_length=getattr(args, "reranker_max_length", 512),
        )
        reranker_load_seconds = perf_counter() - reranker_load_started
        retriever = RerankingRetriever(
            candidate_searcher,
            scorer,
            candidate_k=getattr(args, "reranker_candidate_k", 30),
            batch_size=getattr(args, "reranker_batch_size", 16),
        )
    trace = RecordingRetriever(retriever)
    qwen_load_started = perf_counter()
    generator = TransformersAnswerGenerator.from_pretrained(
        args.model_name,
        revision=args.revision,
        device=actual_device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        trust_remote_code=False,
    )
    qwen_load_seconds = perf_counter() - qwen_load_started
    pipeline = RagPipeline(
        retriever=trace,
        generator=generator,
        tokenizer=generator.prompt_tokenizer,
        prompt_serializer=ChatTemplatePromptSerializer(generator.tokenizer),
        config=GenerationConfig(
            retrieval_limit=args.top_k,
            max_prompt_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        ),
        generation_contract=generation_contract_for_mode(
            getattr(args, "answer_mode", ANSWER_MODE_LEGACY)
        ),
        context_selector=context_selector,
    )
    environment = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "sentence_transformers_version": importlib.metadata.version(
            "sentence-transformers"
        ),
        "faiss_version": importlib.metadata.version("faiss-cpu"),
        "device": actual_device,
        "gpu_name": (
            torch.cuda.get_device_name(0) if actual_device.startswith("cuda") else None
        ),
        "dtype": generator.dtype_name,
        "generation_model": generator.model_name,
        "generation_model_revision": generator.model_revision,
        "embedding_model": embedding_model_name,
        "index_path": str(index_path.expanduser()),
        "metadata_path": str(metadata_path.expanduser()),
        "index_manifest_path": str(manifest_path.expanduser()),
        "context_mode": context_mode,
        "parent_retrieval_revision": (
            PARENT_RETRIEVAL_REVISION if context_mode == "parent" else None
        ),
        "parent_metadata_path": (
            str(parent_metadata_path.expanduser()) if context_mode == "parent" else None
        ),
        "parent_count": parent_store.parent_count if parent_store else None,
        "section_parent_revision": (
            SECTION_PARENT_REVISION if context_mode == "section-parent" else None
        ),
        "section_path": (
            str(section_path.expanduser()) if context_mode == "section-parent" else None
        ),
        "section_count": section_store.section_count if section_store else None,
        "indexed_chunk_count": vector_index.chunk_count,
        "index_load_seconds": index_load_seconds,
        "parent_store_build_seconds": parent_store_seconds,
        "reranking_revision": (
            RERANKING_REVISION if reranker_spec is not None else None
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
        "reranker_trust_remote_code": (False if reranker_spec is not None else None),
        "reranker_load_seconds": reranker_load_seconds,
        "qwen_load_seconds": qwen_load_seconds,
        "runtime_build_seconds": perf_counter() - runtime_started,
    }
    return pipeline, trace, environment


def _build_judge(args: argparse.Namespace) -> OpenAIResponsesJudge | None:
    if args.judge == "none":
        return None
    if not args.judge_model:
        raise ValueError("--judge-model is required with --judge openai")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is required with --judge openai")
    return OpenAIResponsesJudge(
        model_name=args.judge_model,
        timeout_seconds=args.judge_timeout,
        max_retries=args.judge_max_retries,
    )


def _settings(
    args: argparse.Namespace,
    *,
    judge_model: str | None,
) -> dict[str, Any]:
    answer_mode = getattr(args, "answer_mode", ANSWER_MODE_LEGACY)
    settings: dict[str, Any] = {
        "retriever": args.retriever,
        "answer_mode": answer_mode,
        "contract_revision": contract_revision_for_mode(answer_mode),
        "top_k": args.top_k,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "judge": args.judge,
        "judge_model": judge_model,
        "judge_prompt_revision": JUDGE_PROMPT_REVISION,
        "judge_schema_revision": JUDGE_SCHEMA_REVISION,
        "derived_result_revision": DERIVED_RESULT_REVISION,
        "output_path": str(args.output_path.expanduser()),
        "generation_history_shared_between_questions": False,
        "context_mode": getattr(args, "context_mode", "chunk"),
        "child_candidate_k": (
            getattr(args, "child_candidate_k", 60)
            if getattr(args, "context_mode", "chunk") in {"parent", "section-parent"}
            else None
        ),
        "section_parent_revision": (
            SECTION_PARENT_REVISION
            if getattr(args, "context_mode", "chunk") == "section-parent"
            else None
        ),
        "reranking_revision": (
            RERANKING_REVISION
            if getattr(args, "reranker_model_key", None) is not None
            else None
        ),
        "reranker_model_key": getattr(args, "reranker_model_key", None),
        "reranker_candidate_k": (
            getattr(args, "reranker_candidate_k", 30)
            if getattr(args, "reranker_model_key", None) is not None
            else None
        ),
        "reranker_batch_size": (
            getattr(args, "reranker_batch_size", 16)
            if getattr(args, "reranker_model_key", None) is not None
            else None
        ),
        "reranker_max_length": (
            getattr(args, "reranker_max_length", 512)
            if getattr(args, "reranker_model_key", None) is not None
            else None
        ),
    }
    if args.retriever == "hybrid":
        settings.update(
            {
                "japanese_ngram_sizes": list(_HYBRID_NGRAM_SIZES),
                "rrf_k": _HYBRID_RRF_K,
                "candidate_k": _HYBRID_CANDIDATE_K,
                "retriever_weights": [1.0, 1.0],
            }
        )
    return settings


def _resource_metrics() -> dict[str, Any]:
    """Return process and CUDA peak metrics without external services."""
    metrics: dict[str, Any] = {
        "peak_cpu_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
    }
    try:
        import torch
    except ImportError:
        return metrics
    if torch.cuda.is_available():
        metrics.update(
            {
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
    return metrics


def _progress_writer(
    path: Path,
    settings: dict[str, Any],
    environment: dict[str, Any],
) -> Any:
    def write(records: Any) -> None:
        save_rag_evaluation(
            records,
            path,
            settings=settings,
            environment=environment,
        )
        latest = records[-1]
        status = "ok" if latest.generation_succeeded else "failed"
        print(f"[{len(records)}] {latest.id}: {status}", flush=True)

    return write


def _resolve_dataset_type(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    with path.expanduser().open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    break
                return "rag-quality" if "required_facts" in row else "answerability"
    raise ValueError(f"cannot detect dataset type from empty file: {path}")


def _resolve_data_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.expanduser()
    configured = os.environ.get("PYTHON_DOC_RAG_DATA_ROOT")
    if configured and configured.strip():
        return Path(configured).expanduser()
    raise ValueError("set PYTHON_DOC_RAG_DATA_ROOT or pass --data-root")


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _print_summary(path: Path) -> None:
    data = _read_json_object(path)
    summary = data["summary"]
    print(f"Questions: {summary['question_count']}")
    print(f"Generation success: {summary['generation_success_rate']:.4f}")
    print(
        f"First-attempt contract: {summary['first_attempt_contract_success_rate']:.4f}"
    )
    print(f"Retry used: {summary['retry_used_rate']:.4f}")
    print(f"Contract failed: {summary['contract_failed_rate']:.4f}")
    print(f"Citation format: {summary['citation_format_success_rate']:.4f}")
    print(f"Correct source Top-5: {summary['expected_source_top_5_rate']:.4f}")
    print(f"Correct source cited: {summary['expected_source_cited_rate']:.4f}")
    print(f"False answer: {summary['false_answer_rate']:.4f}")
    print(f"False abstention: {summary['false_abstain_rate']:.4f}")
    if (
        summary["judge_completed_count"]
        or summary["raw_judge_error_count"]
        or summary["derived_semantic_error_count"]
    ):
        print(f"Judge completed: {summary['judge_completed_count']}")
        print(f"Answer relevance: {summary['average_answer_relevance']:.4f}")
        print(f"Faithfulness: {summary['average_faithfulness']:.4f}")
        print(f"Citation support: {summary['average_citation_support']:.4f}")
        print(f"Completeness: {summary['average_completeness']:.4f}")
        print(f"Unsupported claims: {summary['unsupported_claim_count']}")
        print(f"Missing required facts: {summary['missing_required_fact_count']}")
        print(f"Grounded: {summary['grounded_count']} ({summary['grounded_rate']:.4f})")
        print(f"Coverage labels: {summary['coverage_label_counts']}")
        print(f"Answerability labels: {summary['derived_answerability_label_counts']}")
        print(f"Raw judge errors: {summary['raw_judge_error_count']}")
        print(f"Derived semantic errors: {summary['derived_semantic_error_count']}")
        print(f"Judge total tokens: {summary['judge_total_tokens']}")
    print(f"Saved: {path}")


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
