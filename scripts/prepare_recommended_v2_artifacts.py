"""Build or validate every Git-ignored artifact required by recommended-v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from python_doc_rag.cli import CliError, resolve_data_root
from python_doc_rag.recommended_v2_artifacts import (
    ArtifactPreparationError,
    prepare_recommended_v2_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse data-root, device, batching, and validation-only options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "data/、indexes/、experiments/を含むroot。未指定時は"
            "PYTHON_DOC_RAG_DATA_ROOTを使用"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
        help="BGE-M3 index構築device（既定: cuda）",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=32,
        help="embedding batch size（既定: 32）",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="既存artifactだけを検証し、不足時は構築しない",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Prepare recommended-v2 artifacts and print reproducibility details."""
    args = parse_args(argv)
    try:
        data_root = resolve_data_root(args.data_root).resolve()
        result = prepare_recommended_v2_artifacts(
            data_root,
            device=args.device,
            batch_size=args.batch_size,
            validate_only=args.validate_only,
        )
    except (ArtifactPreparationError, CliError, OSError, ValueError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1

    validation = result.validation
    payload = {
        "status": "reused" if result.reused_existing else "created",
        "profile": validation.profile_name,
        "chunk_count": validation.chunk_count,
        "embedding_dimension": validation.embedding_dimension,
        "model_name": validation.model_name,
        "model_revision": validation.model_revision,
        "artifacts": {
            "index": _artifact_payload(
                result.paths.embedding_index_path,
                data_root,
                validation.index_sha256,
            ),
            "metadata": _artifact_payload(
                result.paths.embedding_metadata_path,
                data_root,
                validation.metadata_sha256,
            ),
            "manifest": _artifact_payload(
                result.paths.embedding_manifest_path,
                data_root,
                validation.manifest_sha256,
            ),
            "symbol": _artifact_payload(
                result.paths.symbol_index_path,
                data_root,
                validation.symbol_sha256,
            ),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _artifact_payload(
    path: Path | None,
    data_root: Path,
    sha256: str | None,
) -> dict[str, str | None]:
    if path is None:
        raise AssertionError("recommended-v2 artifact path is missing")
    return {
        "path": path.relative_to(data_root).as_posix(),
        "sha256": sha256,
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
