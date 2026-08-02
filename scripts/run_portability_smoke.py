"""Run the configured-dataset portability smoke with one reusable local service."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import resource
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from python_doc_rag.cli import build_profile_service
from python_doc_rag.portability_evaluation import (
    evaluate_portability_questions,
    load_portability_questions,
    save_portability_evaluation_atomic,
    summarize_portability,
)
from python_doc_rag.profile_artifacts import sha256_file


def parse_args() -> argparse.Namespace:
    """Parse explicit local smoke paths without accepting remote services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--profile", default="recommended-v2")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--allowed-source-prefix", action="append", required=True)
    return parser.parse_args()


def main() -> int:
    """Build the service once, continue per case, and save progress atomically."""
    args = parse_args()
    questions = load_portability_questions(args.questions)
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_started = perf_counter()
    service = build_profile_service(
        args.data_root,
        profile_name=args.profile,
        device=args.device,
    )
    service_load_seconds = perf_counter() - load_started
    settings: dict[str, Any] = {
        "revision": "uv-docs-portability-smoke-v1",
        "profile": args.profile,
        "question_count": len(questions),
        "questions_sha256": sha256_file(args.questions),
        "allowed_source_prefixes": args.allowed_source_prefix,
        "service_reused": True,
        "openai_api_used": False,
        "contains_secrets": False,
    }
    environment: dict[str, Any] = {
        "executed_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": _driver_version(torch),
        "transformers": _version("transformers"),
        "sentence_transformers": _version("sentence-transformers"),
        "service_load_seconds": service_load_seconds,
    }

    def save_progress(records: Any) -> None:
        save_portability_evaluation_atomic(
            records,
            args.output_path,
            settings=settings,
            environment=environment,
        )

    records = evaluate_portability_questions(
        service,
        questions,
        allowed_source_prefixes=args.allowed_source_prefix,
        on_case=save_progress,
    )
    environment.update(
        {
            "peak_cpu_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024
            / 1024,
            "gpu_peak_allocated_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
            "gpu_peak_reserved_gb": (
                torch.cuda.max_memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
        }
    )
    save_portability_evaluation_atomic(
        records,
        args.output_path,
        settings=settings,
        environment=environment,
    )
    summary = summarize_portability(records)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _driver_version(torch: Any) -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return str(torch._C._cuda_getDriverVersion())
    except AttributeError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
