"""Portable serialization helpers shared by ingestion pipelines."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from python_doc_rag.models import SearchChunk


def write_chunks_jsonl_atomic(
    chunks: Iterable[SearchChunk],
    output_path: Path,
) -> int:
    """Stream chunks to UTF-8 JSONL and atomically replace the destination."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                json.dump(chunk.to_dict(), stream, ensure_ascii=False)
                stream.write("\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


__all__ = ["write_chunks_jsonl_atomic"]
