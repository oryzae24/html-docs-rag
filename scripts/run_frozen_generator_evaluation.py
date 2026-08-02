"""Compare pinned local generators over one immutable retrieved-context artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import resource
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from python_doc_rag.answer_contract import AnswerOrAbstainGenerationContract
from python_doc_rag.frozen_contexts import (
    FrozenContextRetriever,
    load_frozen_contexts,
)
from python_doc_rag.generation import (
    ChatTemplatePromptSerializer,
    GenerationConfig,
    PromptSerializer,
    TokenizerProtocol,
    count_prompt_tokens,
)
from python_doc_rag.generator_models import generator_model_spec
from python_doc_rag.models import SearchChunk
from python_doc_rag.pipeline import RagPipeline
from python_doc_rag.rag_evaluation import (
    RecordingRetriever,
    evaluate_rag_questions,
    load_answerability_questions,
    load_rag_quality_questions,
    save_rag_evaluation,
)
from python_doc_rag.transformers_generation import TransformersAnswerGenerator


def parse_args() -> argparse.Namespace:
    """Parse one model and one frozen dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("answerability", "rag-quality", "hard-cases"),
        required=True,
    )
    parser.add_argument(
        "--model-key",
        choices=(
            "baseline-qwen3-4b",
            "qwen3-8b",
            "qwen2.5-7b-instruct",
        ),
        required=True,
    )
    parser.add_argument("--source-code-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-input-tokens", type=_positive_int, default=8192)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=512)
    return parser.parse_args()


class _ExactFrozenSelector:
    """Keep all five saved chunks or fail before implicit truncation."""

    def __call__(
        self,
        question: str,
        candidates: Sequence[SearchChunk],
        *,
        tokenizer: TokenizerProtocol,
        max_prompt_tokens: int,
        prompt_serializer: PromptSerializer,
        initial_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
        retry_prompt_builder: Callable[[str, Sequence[SearchChunk]], str],
    ) -> tuple[SearchChunk, ...]:
        selected = tuple(candidates)
        if len(selected) != 5:
            raise RuntimeError("generator comparison requires exactly five contexts")
        prompts = (
            prompt_serializer.serialize(initial_prompt_builder(question, selected)),
            prompt_serializer.serialize(retry_prompt_builder(question, selected)),
        )
        if any(
            count_prompt_tokens(tokenizer, prompt) > max_prompt_tokens
            for prompt in prompts
        ):
            raise RuntimeError("a frozen context exceeds this model's prompt budget")
        return selected


def main() -> int:
    """Load one model once, evaluate all cases, and retain per-case failures."""
    args = parse_args()
    import torch

    spec = generator_model_spec(args.model_key)
    records = load_frozen_contexts(args.contexts, dataset=args.dataset)
    retriever = FrozenContextRetriever(records)
    trace = RecordingRetriever(retriever)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        generator = TransformersAnswerGenerator.from_pretrained(
            spec.model_name,
            revision=spec.revision,
            device=args.device,
            dtype=spec.dtype,
            max_input_tokens=args.max_input_tokens,
            trust_remote_code=spec.trust_remote_code,
            generation_kwargs=spec.generation_options(),
            sampling_seed=spec.sampling_seed,
        )
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            f"{spec.model_name} did not fit unquantized in GPU memory; "
            "the tournament will not switch to quantization"
        ) from error
    serializer = ChatTemplatePromptSerializer(
        generator.tokenizer,
        template_kwargs=spec.template_options(),
    )
    pipeline = RagPipeline(
        retriever=trace,
        generator=generator,
        tokenizer=generator.prompt_tokenizer,
        prompt_serializer=serializer,
        config=GenerationConfig(
            retrieval_limit=5,
            max_prompt_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        ),
        generation_contract=AnswerOrAbstainGenerationContract(),
        context_selector=_ExactFrozenSelector(),
    )
    questions = (
        load_answerability_questions(args.questions)
        if args.dataset == "answerability"
        else load_rag_quality_questions(args.questions)
    )
    settings = {
        "revision": "local-generator-tournament-v1",
        "dataset": args.dataset,
        "model_key": spec.key,
        "model_name": spec.model_name,
        "model_revision": spec.revision,
        "model_license": spec.license,
        "model_card_url": spec.model_card_url,
        "language_evidence": spec.language_evidence,
        "parameter_billions": spec.parameter_billions,
        "dtype": spec.dtype,
        "trust_remote_code": spec.trust_remote_code,
        "template_kwargs": spec.template_options(),
        "generation_kwargs": spec.generation_options(),
        "sampling_seed": spec.sampling_seed,
        "thinking_visible_to_user": False,
        "contexts_path": str(args.contexts.expanduser()),
        "contexts_sha256": _sha256(args.contexts),
        "questions_path": str(args.questions.expanduser()),
        "questions_sha256": _sha256(args.questions),
        "source_code_commit": args.source_code_commit,
        "selected_context_tuple_frozen": True,
        "retriever_reexecuted": False,
        "answer_mode": "answer-or-abstain",
        "contract_revision": "answer-or-abstain-v1",
        "top_k": 5,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "openai_api_used": False,
        "contains_secrets": False,
    }
    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "device": generator.device,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "model_load_seconds": generator.model_load_seconds,
    }

    def save_progress(evaluated: Sequence[Any]) -> None:
        save_rag_evaluation(
            evaluated,
            args.output_path,
            settings=settings,
            environment=environment,
        )
        print(f"[{len(evaluated)}/{len(questions)}] {evaluated[-1].id}", flush=True)

    evaluated = evaluate_rag_questions(
        pipeline,
        trace,
        questions,
        on_case=save_progress,
    )
    history = generator.generation_history
    environment.update(
        {
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
            "generation_call_count": len(history),
            "generation_input_tokens": sum(item.input_tokens for item in history),
            "generation_output_tokens": sum(item.generated_tokens for item in history),
            "frozen_retriever_call_count": retriever.calls,
        }
    )
    save_rag_evaluation(
        evaluated,
        args.output_path,
        settings=settings,
        environment=environment,
    )
    print(f"Saved: {args.output_path}")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
