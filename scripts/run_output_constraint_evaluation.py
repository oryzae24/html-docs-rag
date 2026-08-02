"""Evaluate exact-choice two-stage answerability on frozen local contexts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import resource
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from python_doc_rag.constrained_generation import exact_choice_settings
from python_doc_rag.frozen_contexts import (
    FrozenContextRetriever,
    load_frozen_contexts,
)
from python_doc_rag.generation import (
    ChatTemplatePromptSerializer,
    GenerationConfig,
)
from python_doc_rag.generator_models import generator_model_spec
from python_doc_rag.models import SearchChunk
from python_doc_rag.rag_evaluation import (
    RecordingRetriever,
    evaluate_rag_questions,
    load_answerability_questions,
    load_rag_quality_questions,
    save_rag_evaluation,
)
from python_doc_rag.transformers_generation import TransformersAnswerGenerator
from python_doc_rag.two_stage_answerability import (
    TWO_STAGE_CONTRACT_REVISION,
    TwoStageAnswerabilityPipeline,
    select_two_stage_contexts_within_budget,
)


def parse_args() -> argparse.Namespace:
    """Parse one frozen dataset and one external result path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("answerability", "rag-quality", "hard-cases"),
        required=True,
    )
    parser.add_argument("--source-code-commit", required=True)
    parser.add_argument(
        "--model-key",
        choices=("baseline-qwen3-4b", "qwen3-8b"),
        default="qwen3-8b",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-input-tokens", type=_positive_int, default=8192)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=512)
    return parser.parse_args()


class _ExactFrozenTwoStageSelector:
    """Require the complete saved five-chunk tuple to fit every stage."""

    def __call__(
        self,
        question: str,
        candidates: Sequence[SearchChunk],
        **kwargs: Any,
    ) -> tuple[SearchChunk, ...]:
        selected = select_two_stage_contexts_within_budget(
            question,
            candidates,
            **kwargs,
        )
        if len(candidates) != 5 or selected != tuple(candidates):
            raise RuntimeError(
                "output-constraint comparison requires all five frozen contexts"
            )
        return selected


def main() -> int:
    """Load Qwen3-8B once and evaluate exact choice plus cited prose."""
    args = parse_args()
    import torch

    spec = generator_model_spec(args.model_key)
    records = load_frozen_contexts(args.contexts, dataset=args.dataset)
    retriever = FrozenContextRetriever(records)
    trace = RecordingRetriever(retriever)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
    serializer = ChatTemplatePromptSerializer(
        generator.tokenizer,
        template_kwargs=spec.template_options(),
    )
    pipeline = TwoStageAnswerabilityPipeline(
        retriever=trace,
        generator=generator,
        tokenizer=generator.prompt_tokenizer,
        prompt_serializer=serializer,
        config=GenerationConfig(
            retrieval_limit=5,
            max_prompt_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        ),
        context_selector=_ExactFrozenTwoStageSelector(),
    )
    questions = (
        load_answerability_questions(args.questions)
        if args.dataset == "answerability"
        else load_rag_quality_questions(args.questions)
    )
    status_trace: list[dict[str, Any]] = []
    previous_generation_calls = 0
    settings: dict[str, Any] = {
        "revision": "output-constraint-tournament-v1",
        "dataset": args.dataset,
        "model_name": spec.model_name,
        "model_revision": spec.revision,
        "model_license": spec.license,
        "dtype": spec.dtype,
        "trust_remote_code": spec.trust_remote_code,
        "template_kwargs": spec.template_options(),
        "stage_1": dict(exact_choice_settings(("answer", "abstain"))),
        "stage_2": "legacy-cited-answer-v1-with-one-retry",
        "contract_revision": TWO_STAGE_CONTRACT_REVISION,
        "answer_text_in_json": False,
        "contexts_sha256": _sha256(args.contexts),
        "questions_sha256": _sha256(args.questions),
        "source_code_commit": args.source_code_commit,
        "selected_context_tuple_frozen": True,
        "top_k": 5,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "schema_constrained_candidate": "skipped-no-lightweight-strict-dependency",
        "openai_api_used": False,
        "contains_secrets": False,
        "status_trace": status_trace,
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
        nonlocal previous_generation_calls
        generation_call_count = len(generator.generation_history)
        status_trace.append(
            {
                "id": evaluated[-1].id,
                "status_choice": pipeline.last_status_choice,
                "generation_calls": generation_call_count - previous_generation_calls,
                "prompt_token_counts": list(pipeline.last_prompt_token_counts),
            }
        )
        previous_generation_calls = generation_call_count
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
