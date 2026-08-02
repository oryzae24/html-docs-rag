"""Build a deterministic, citation-ready JSONL corpus from Python docs HTML."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from python_doc_rag.chunking import chunk_sections
from python_doc_rag.config import ChunkingConfig
from python_doc_rag.html_parser import parse_python_doc_html_result
from python_doc_rag.ingestion.serialization import write_chunks_jsonl_atomic
from python_doc_rag.models import SearchChunk
from python_doc_rag.sites.python_docs.constants import (
    PYTHON_DOC_BASE_URL,
    PYTHON_DOC_VERSION,
    TARGET_CATEGORIES,
)


@dataclass(frozen=True, slots=True)
class HtmlFileEnumeration:
    """Selected HTML files and diagnostics from archive enumeration."""

    document_root: Path
    files: tuple[Path, ...]
    available_html_count: int
    missing_categories: tuple[str, ...]
    skipped_unsafe_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorpusBuildFailure:
    """A source file that could not be decoded or parsed."""

    source_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class CorpusBuildResult:
    """Summary and output locations for a completed corpus build."""

    output_jsonl: Path
    manifest_path: Path
    detected_html_count: int
    successful_html_count: int
    failed_html_count: int
    extracted_section_count: int
    generated_chunk_count: int
    excluded_section_count: int
    missing_categories: tuple[str, ...]
    failures: tuple[CorpusBuildFailure, ...]


@dataclass(slots=True)
class _BuildCounters:
    successful_html_count: int = 0
    extracted_section_count: int = 0
    generated_chunk_count: int = 0
    excluded_section_count: int = 0


def enumerate_html_files(
    document_root: Path,
    *,
    categories: Sequence[str] = TARGET_CATEGORIES,
    test_mode: bool = False,
    max_files_per_category: int = 3,
) -> HtmlFileEnumeration:
    """Enumerate target-category HTML files without following symbolic links."""
    if test_mode and max_files_per_category < 1:
        raise ValueError("max_files_per_category must be greater than zero")

    root = document_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"document_root is not a directory: {document_root}")

    selected_files: list[Path] = []
    available_html_count = 0
    missing_categories: list[str] = []
    skipped_unsafe_paths: list[str] = []

    for category in categories:
        _validate_category(category)
        category_root = root / category
        if not category_root.is_dir() or category_root.is_symlink():
            missing_categories.append(category)
            continue

        category_files: list[tuple[str, Path]] = []
        for candidate in category_root.rglob("*.html"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                resolved_candidate = candidate.resolve(strict=True)
                relative_path = resolved_candidate.relative_to(root).as_posix()
            except (OSError, ValueError):
                skipped_unsafe_paths.append(candidate.relative_to(root).as_posix())
                continue
            category_files.append((relative_path, resolved_candidate))

        category_files.sort(key=lambda item: item[0])
        available_html_count += len(category_files)
        if test_mode:
            category_files = category_files[:max_files_per_category]
        selected_files.extend(path for _, path in category_files)

    selected_files.sort(key=lambda path: path.relative_to(root).as_posix())
    return HtmlFileEnumeration(
        document_root=root,
        files=tuple(selected_files),
        available_html_count=available_html_count,
        missing_categories=tuple(missing_categories),
        skipped_unsafe_paths=tuple(sorted(skipped_unsafe_paths)),
    )


def build_corpus(
    document_root: Path,
    output_jsonl: Path,
    manifest_path: Path,
    *,
    test_mode: bool = False,
    max_files_per_category: int = 3,
    archive_sha256: str | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> CorpusBuildResult:
    """Build JSONL and a manifest from a local Python documentation archive."""
    settings = chunking_config or ChunkingConfig()
    if output_jsonl.expanduser().resolve() == manifest_path.expanduser().resolve():
        raise ValueError("output_jsonl and manifest_path must be different paths")

    started_at = datetime.now(UTC)
    enumeration = enumerate_html_files(
        document_root,
        test_mode=test_mode,
        max_files_per_category=max_files_per_category,
    )
    counters = _BuildCounters()
    failures: list[CorpusBuildFailure] = []

    chunks = _iter_corpus_chunks(enumeration, settings, counters, failures)
    written_count = write_chunks_jsonl_atomic(chunks, output_jsonl)
    if written_count != counters.generated_chunk_count:
        raise RuntimeError("written JSONL count did not match generated chunk count")

    finished_at = datetime.now(UTC)
    result = CorpusBuildResult(
        output_jsonl=output_jsonl,
        manifest_path=manifest_path,
        detected_html_count=len(enumeration.files),
        successful_html_count=counters.successful_html_count,
        failed_html_count=len(failures),
        extracted_section_count=counters.extracted_section_count,
        generated_chunk_count=counters.generated_chunk_count,
        excluded_section_count=counters.excluded_section_count,
        missing_categories=enumeration.missing_categories,
        failures=tuple(failures),
    )
    manifest = _build_manifest(
        result,
        enumeration=enumeration,
        settings=settings,
        started_at=started_at,
        finished_at=finished_at,
        test_mode=test_mode,
        max_files_per_category=max_files_per_category,
        archive_sha256=archive_sha256,
    )
    _write_json_atomic(manifest, manifest_path)
    return result


def _iter_corpus_chunks(
    enumeration: HtmlFileEnumeration,
    settings: ChunkingConfig,
    counters: _BuildCounters,
    failures: list[CorpusBuildFailure],
) -> Iterator[SearchChunk]:
    """Process one file at a time so the complete corpus is never retained."""
    for html_path in enumeration.files:
        source_path = html_path.relative_to(enumeration.document_root).as_posix()
        category = PurePosixPath(source_path).parts[0]
        try:
            html = html_path.read_text(encoding="utf-8")
            parse_result = parse_python_doc_html_result(
                html,
                source_path=source_path,
                category=category,
            )
            chunks = chunk_sections(parse_result.sections, settings)
        except Exception as error:  # noqa: BLE001 - one bad source must not stop a build
            failures.append(
                CorpusBuildFailure(
                    source_path=source_path,
                    reason=_relative_failure_reason(error, html_path, source_path),
                )
            )
            continue

        counters.successful_html_count += 1
        counters.extracted_section_count += len(parse_result.sections)
        counters.excluded_section_count += parse_result.excluded_section_count
        counters.generated_chunk_count += len(chunks)
        yield from chunks


def _build_manifest(
    result: CorpusBuildResult,
    *,
    enumeration: HtmlFileEnumeration,
    settings: ChunkingConfig,
    started_at: datetime,
    finished_at: datetime,
    test_mode: bool,
    max_files_per_category: int,
    archive_sha256: str | None,
) -> dict[str, Any]:
    """Create a JSON-serializable corpus build manifest."""
    return {
        "python_documentation_version": PYTHON_DOC_VERSION,
        "base_url": PYTHON_DOC_BASE_URL,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "target_categories": list(TARGET_CATEGORIES),
        "test_mode": test_mode,
        "max_files_per_category": max_files_per_category,
        "available_html_count": enumeration.available_html_count,
        "detected_html_count": result.detected_html_count,
        "successful_html_count": result.successful_html_count,
        "failed_html_count": result.failed_html_count,
        "extracted_section_count": result.extracted_section_count,
        "generated_chunk_count": result.generated_chunk_count,
        "excluded_empty_or_short_section_count": result.excluded_section_count,
        "archive_sha256": archive_sha256,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "missing_categories": list(result.missing_categories),
        "skipped_unsafe_paths": list(enumeration.skipped_unsafe_paths),
        "failures": [asdict(failure) for failure in result.failures],
    }


def _write_json_atomic(data: dict[str, Any], output_path: Path) -> None:
    """Write one JSON object through a sibling temporary file."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_sibling(output_path)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _temporary_sibling(output_path: Path) -> Path:
    """Reserve a unique temporary path on the destination filesystem."""
    descriptor, name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _validate_category(category: str) -> None:
    """Reject category names that could escape the documentation root."""
    path = PurePosixPath(category)
    if (
        not category
        or path.is_absolute()
        or len(path.parts) != 1
        or category in {".", ".."}
    ):
        raise ValueError(f"invalid documentation category: {category!r}")


def _relative_failure_reason(
    error: Exception,
    html_path: Path,
    source_path: str,
) -> str:
    """Format an error without leaking the local absolute archive path."""
    message = str(error).replace(str(html_path), source_path)
    return f"{type(error).__name__}: {message}"
