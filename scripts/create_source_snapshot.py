"""Create the deterministic Python documentation source snapshot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from python_doc_rag.source_snapshot import create_deterministic_source_snapshot


def parse_args() -> argparse.Namespace:
    """Parse explicit source and destination paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--archive-root",
        default="python-3.13-docs-html",
        help="Single fixed top-level directory stored in the ZIP",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the snapshot and print its measured identity as JSON."""
    args = parse_args()
    result = create_deterministic_source_snapshot(
        args.source_root,
        args.output_path,
        archive_root=args.archive_root,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
