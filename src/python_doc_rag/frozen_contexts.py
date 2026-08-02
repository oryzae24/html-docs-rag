"""Persist and replay immutable retrieved contexts for generator comparison."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python_doc_rag.generation import build_prompt_contexts
from python_doc_rag.models import SearchChunk

FROZEN_CONTEXT_REVISION = "frozen-generator-context-v1"


@dataclass(frozen=True, slots=True)
class FrozenContextRecord:
    """One exact question and its selected citation-ready chunk tuple."""

    id: str
    question: str
    dataset: str
    chunks: tuple[SearchChunk, ...]
    rerank_scores: tuple[float, ...]
    original_ranks: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return full external diagnostics and prompt-visible sanitized text."""
        prompt_contexts = build_prompt_contexts(self.chunks)
        return {
            "id": self.id,
            "question": self.question,
            "dataset": self.dataset,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "chunk_tuple_sha256": chunk_tuple_sha256(self.chunks),
            "rerank_scores": list(self.rerank_scores),
            "original_ranks": list(self.original_ranks),
            "prompt_visible_contexts": [
                {
                    "label": context.label,
                    "page_title": context.page_title,
                    "section_title": context.section_title,
                    "text": context.text,
                }
                for context in prompt_contexts
            ],
        }


class FrozenContextRetriever:
    """Replay one preselected tuple without invoking a retrieval model."""

    def __init__(self, records: Sequence[FrozenContextRecord]) -> None:
        if not records:
            raise ValueError("frozen context retriever requires records")
        by_question: dict[str, FrozenContextRecord] = {}
        for record in records:
            if record.question in by_question:
                raise ValueError(f"duplicate frozen question: {record.question}")
            by_question[record.question] = record
        self._by_question = by_question
        self.calls = 0

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        """Return exactly the saved prefix and reject unknown questions."""
        if limit < 1:
            raise ValueError("limit must be positive")
        try:
            record = self._by_question[question]
        except KeyError as error:
            raise ValueError("question is absent from frozen contexts") from error
        if limit < len(record.chunks):
            raise ValueError("retrieval limit would truncate the frozen tuple")
        self.calls += 1
        return record.chunks


def save_frozen_contexts_atomic(
    records: Sequence[FrozenContextRecord],
    path: Path,
    *,
    settings: Mapping[str, Any],
) -> None:
    """Validate and atomically save a complete frozen-context artifact."""
    if not records:
        raise ValueError("at least one frozen context is required")
    payload = {
        "revision": FROZEN_CONTEXT_REVISION,
        "settings": dict(settings),
        "record_count": len(records),
        "records": [record.to_dict() for record in records],
    }
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_frozen_contexts(
    path: Path,
    *,
    dataset: str | None = None,
) -> tuple[FrozenContextRecord, ...]:
    """Load and fail closed on identity, hash, or sanitized-context drift."""
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("revision") != FROZEN_CONTEXT_REVISION:
        raise ValueError("unsupported frozen context artifact")
    rows = data.get("records")
    if not isinstance(rows, list) or data.get("record_count") != len(rows):
        raise ValueError("frozen context record count mismatch")
    records = tuple(_record_from_mapping(row) for row in rows)
    if dataset is not None:
        records = tuple(record for record in records if record.dataset == dataset)
        if not records:
            raise ValueError(f"no frozen contexts for dataset: {dataset}")
    FrozenContextRetriever(records)
    return records


def chunk_tuple_sha256(chunks: Sequence[SearchChunk]) -> str:
    """Hash the exact selected tuple, including citation metadata and order."""
    canonical = json.dumps(
        [chunk.to_dict() for chunk in chunks],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _record_from_mapping(value: object) -> FrozenContextRecord:
    if not isinstance(value, dict):
        raise TypeError("frozen context record must be an object")
    chunks_value = value.get("chunks")
    if not isinstance(chunks_value, list) or not chunks_value:
        raise ValueError("frozen chunks must be a non-empty list")
    chunks = tuple(SearchChunk.from_dict(item) for item in chunks_value)
    if value.get("chunk_tuple_sha256") != chunk_tuple_sha256(chunks):
        raise ValueError("frozen chunk tuple SHA-256 mismatch")
    scores = value.get("rerank_scores")
    ranks = value.get("original_ranks")
    if (
        not isinstance(scores, list)
        or not isinstance(ranks, list)
        or len(scores) != len(chunks)
        or len(ranks) != len(chunks)
    ):
        raise ValueError("frozen reranking diagnostics must align with chunks")
    prompt_contexts = build_prompt_contexts(chunks)
    expected_visible = [
        {
            "label": context.label,
            "page_title": context.page_title,
            "section_title": context.section_title,
            "text": context.text,
        }
        for context in prompt_contexts
    ]
    if value.get("prompt_visible_contexts") != expected_visible:
        raise ValueError("frozen prompt-visible context mismatch")
    identifiers = {name: value.get(name) for name in ("id", "question", "dataset")}
    if not all(isinstance(item, str) and item for item in identifiers.values()):
        raise TypeError("frozen id, question, and dataset must be strings")
    return FrozenContextRecord(
        id=str(identifiers["id"]),
        question=str(identifiers["question"]),
        dataset=str(identifiers["dataset"]),
        chunks=chunks,
        rerank_scores=tuple(float(score) for score in scores),
        original_ranks=tuple(int(rank) for rank in ranks),
    )
