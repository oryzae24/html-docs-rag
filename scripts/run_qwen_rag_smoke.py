"""Run a three-question Qwen RAG smoke test and save structured metrics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from python_doc_rag import (
    ChatTemplatePromptSerializer,
    GenerationConfig,
    RagPipeline,
    TransformersAnswerGenerator,
    VectorIndexRetriever,
    load_vector_index,
)
from python_doc_rag.config import DEFAULT_GENERATION_MODEL
from python_doc_rag.evaluation import load_evaluation_questions
from python_doc_rag.retrieval import Retriever
from python_doc_rag.transformers_generation import resolve_device

_URL_PATTERN = re.compile(r"(?i)(?:https?://)\S+")
_MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]\r\n]+\]\([^\)\r\n]+\)")
_CITATION_PATTERN = re.compile(r"\[S([1-9]\d*)\]")
_TRUSTED_SOURCE_PREFIX = "https://docs.python.org/ja/3.13/"


class TimingRetriever:
    """Record retrieval wall time while preserving the Retriever contract."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self.history: list[float] = []

    def retrieve(self, question: str, *, limit: int) -> Any:
        started_at = time.perf_counter()
        chunks = self._retriever.retrieve(question, limit=limit)
        self.history.append(time.perf_counter() - started_at)
        return chunks


def parse_args() -> argparse.Namespace:
    """Parse portable paths and generation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--index-manifest-path", type=Path)
    parser.add_argument("--questions-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--embedding-model")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    """Load models once, answer three questions, and persist measurements."""
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    data_root = _resolve_data_root(args.data_root)
    index_path = args.index_path or data_root / "indexes/python_3_13_ja.faiss"
    metadata_path = (
        args.metadata_path or data_root / "indexes/python_3_13_ja_metadata.jsonl"
    )
    manifest_path = (
        args.index_manifest_path
        or data_root / "indexes/python_3_13_ja_index_manifest.json"
    )
    questions_path = (
        args.questions_path or repository_root / "evaluation/qwen_smoke_questions.jsonl"
    )
    output_path = args.output_path or data_root / "experiments/qwen_rag_smoke.json"

    import torch
    import transformers

    actual_device = resolve_device(args.device, torch_module=torch)
    manifest = _read_json(manifest_path)
    embedding_model_name = args.embedding_model or str(manifest["model_name"])
    questions = load_evaluation_questions(questions_path)
    if len(questions) != 3:
        raise ValueError(
            f"expected exactly 3 evaluation questions, got {len(questions)}"
        )

    model_cached_before = _has_cached_snapshot(args.model_name)
    embedding_started_at = time.perf_counter()
    embedding_model = SentenceTransformer(
        embedding_model_name,
        device=actual_device,
    )
    embedding_load_seconds = time.perf_counter() - embedding_started_at
    vector_index = load_vector_index(
        index_path,
        metadata_path,
        embedding_model=embedding_model,
    )

    generator = TransformersAnswerGenerator.from_pretrained(
        args.model_name,
        revision=args.revision,
        device=actual_device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        trust_remote_code=False,
    )
    model_downloaded = not model_cached_before and _has_cached_snapshot(args.model_name)
    serializer = ChatTemplatePromptSerializer(generator.tokenizer)
    timing_retriever = TimingRetriever(VectorIndexRetriever(vector_index))
    pipeline = RagPipeline(
        retriever=timing_retriever,
        generator=generator,
        tokenizer=generator.prompt_tokenizer,
        prompt_serializer=serializer,
        config=GenerationConfig(
            retrieval_limit=args.top_k,
            max_prompt_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        ),
    )

    if actual_device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    for item in questions:
        history_offset = len(generator.generation_history)
        if actual_device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        answer_started_at = time.perf_counter()
        answer = pipeline.answer(item.question)
        answer_seconds = time.perf_counter() - answer_started_at
        attempts = generator.generation_history[history_offset:]
        citation_numbers = tuple(
            int(number) for number in _CITATION_PATTERN.findall(answer.answer_text)
        )
        selected_urls = {chunk.source_url for chunk in answer.retrieved_chunks}
        source_urls = [source.url for source in answer.sources]
        url_count = len(_URL_PATTERN.findall(answer.answer_text))
        markdown_link_count = len(_MARKDOWN_LINK_PATTERN.findall(answer.answer_text))
        citation_valid = bool(citation_numbers) and all(
            1 <= number <= len(answer.retrieved_chunks) for number in citation_numbers
        )
        sources_from_selected_chunks = all(url in selected_urls for url in source_urls)
        trusted_source_urls = all(
            url.startswith(_TRUSTED_SOURCE_PREFIX) for url in source_urls
        )
        question_result = {
            "question": item.question,
            "expected_url_keywords": list(item.expected_url_keywords),
            "retrieval_seconds": timing_retriever.history[-1],
            "answer_seconds": answer_seconds,
            "generation_seconds": sum(metric.elapsed_seconds for metric in attempts),
            "input_tokens": [metric.input_tokens for metric in attempts],
            "generated_tokens": [metric.generated_tokens for metric in attempts],
            "generation_attempts": answer.generation_attempts,
            "answer_text": answer.answer_text,
            "citation_numbers": list(citation_numbers),
            "source_urls": source_urls,
            "answer_url_count": url_count,
            "markdown_link_count": markdown_link_count,
            "citation_validation_success": citation_valid,
            "sources_from_selected_chunks": sources_from_selected_chunks,
            "trusted_source_urls": trusted_source_urls,
            "selected_chunk_count": len(answer.retrieved_chunks),
            "max_memory_allocated_bytes": _cuda_metric(
                torch,
                actual_device,
                "max_memory_allocated",
            ),
            "max_memory_reserved_bytes": _cuda_metric(
                torch,
                actual_device,
                "max_memory_reserved",
            ),
        }
        results.append(question_result)
        _print_question_result(question_result)

    payload = {
        "environment": _environment_payload(
            torch=torch,
            transformers=transformers,
            generator=generator,
            actual_device=actual_device,
            model_downloaded=model_downloaded,
            embedding_model_name=embedding_model_name,
            embedding_load_seconds=embedding_load_seconds,
        ),
        "settings": {
            "top_k": args.top_k,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "trust_remote_code": False,
            "greedy": True,
            "question_count": len(questions),
        },
        "questions": results,
        "summary": {
            "completed_questions": len(results),
            "answer_url_count": sum(item["answer_url_count"] for item in results),
            "citation_violation_count": sum(
                not item["citation_validation_success"] for item in results
            ),
            "retry_count": sum(
                max(0, item["generation_attempts"] - 1) for item in results
            ),
            "all_sources_trusted": all(item["trusted_source_urls"] for item in results),
            "peak_memory_allocated_bytes": max(
                item["max_memory_allocated_bytes"] for item in results
            ),
            "peak_memory_reserved_bytes": max(
                item["max_memory_reserved_bytes"] for item in results
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved metrics: {output_path}")


def _resolve_data_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument.expanduser()
    configured = os.environ.get("PYTHON_DOC_RAG_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    raise ValueError("set PYTHON_DOC_RAG_DATA_ROOT or pass --data-root")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _has_cached_snapshot(model_name: str) -> bool:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    model_directory = hf_home / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots = model_directory / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _cuda_metric(torch: Any, device: str, name: str) -> int:
    if not device.startswith("cuda"):
        return 0
    return int(getattr(torch.cuda, name)())


def _environment_payload(
    *,
    torch: Any,
    transformers: Any,
    generator: TransformersAnswerGenerator,
    actual_device: str,
    model_downloaded: bool,
    embedding_model_name: str,
    embedding_load_seconds: float,
) -> dict[str, Any]:
    gpu_name: str | None = None
    gpu_vram_bytes = 0
    if actual_device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(actual_device)
        gpu_name = str(properties.name)
        gpu_vram_bytes = int(properties.total_memory)
    return {
        "gpu_name": gpu_name,
        "gpu_vram_bytes": gpu_vram_bytes,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": importlib.metadata.version(
            "sentence-transformers"
        ),
        "model_name": generator.model_name,
        "model_revision": generator.model_revision,
        "actual_dtype": generator.dtype_name,
        "device": generator.device,
        "model_downloaded": model_downloaded,
        "model_load_seconds": generator.model_load_seconds,
        "embedding_model_name": embedding_model_name,
        "embedding_load_seconds": embedding_load_seconds,
    }


def _print_question_result(result: dict[str, Any]) -> None:
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer_text']}")
    print(f"Citations: {result['citation_numbers']}")
    for url in result["source_urls"]:
        print(f"Source: {url}")


if __name__ == "__main__":
    main()
