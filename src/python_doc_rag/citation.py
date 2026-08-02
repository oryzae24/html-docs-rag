"""Validate answer citations against retrieval-controlled metadata."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from python_doc_rag.models import CitationSource, CitedAnswer, SearchChunk

_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_CANONICAL_CITATION_PATTERN = re.compile(r"\[S([1-9]\d*)\]")
_CITATION_LIKE_PATTERN = re.compile(r"\[[sS][^\]\r\n]*\]")
_URL_PATTERN = re.compile(r"(?i)(?:https?://|ftp://|www\.)\S+")
_MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]\r\n]+\]\([^\)\r\n]+\)")


class CitationContractError(ValueError):
    """Raised when generated text violates the fail-closed citation contract."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("invalid generated answer: " + ", ".join(self.reasons))


@dataclass(frozen=True, slots=True)
class CitationValidation:
    """Validated citation numbers and their deduplicated source entries."""

    valid_numbers: tuple[int, ...]
    invalid_numbers: tuple[int, ...]
    sources: tuple[CitationSource, ...]


def finalize_cited_answer(
    answer: str,
    retrieved_chunks: tuple[SearchChunk, ...],
    *,
    generation_attempts: int,
) -> CitedAnswer:
    """Validate generated text and attach only retrieval-controlled sources."""
    reasons: list[str] = []
    canonical_numbers = tuple(
        int(match) for match in _CANONICAL_CITATION_PATTERN.findall(answer)
    )
    citation_like = tuple(_CITATION_LIKE_PATTERN.findall(answer))
    canonical_markers = tuple(_CANONICAL_CITATION_PATTERN.findall(answer))

    if not answer.strip():
        reasons.append("empty_answer")
    if _URL_PATTERN.search(answer):
        reasons.append("url_detected")
    if _MARKDOWN_LINK_PATTERN.search(answer):
        reasons.append("markdown_link_detected")
    if len(citation_like) != len(canonical_markers):
        reasons.append("malformed_citation")
    if any(number > len(retrieved_chunks) for number in canonical_numbers):
        reasons.append("citation_out_of_range")
    if not canonical_numbers:
        reasons.append("citation_required")

    if reasons:
        raise CitationContractError(_unique_strings_in_order(reasons))

    validation = validate_citations(answer, retrieved_chunks)
    if validation.invalid_numbers:
        raise CitationContractError(("citation_out_of_range",))
    return CitedAnswer(
        answer_text=answer.strip(),
        sources=validation.sources,
        retrieved_chunks=retrieved_chunks,
        generation_attempts=generation_attempts,
    )


def validate_citations(
    answer: str,
    search_results: Sequence[SearchChunk],
) -> CitationValidation:
    """Resolve [S1] references by search-result position and detect invalid ones."""
    numbers = _unique_in_order(int(match) for match in _CITATION_PATTERN.findall(answer))
    valid_numbers = tuple(number for number in numbers if 1 <= number <= len(search_results))
    invalid_numbers = tuple(number for number in numbers if number not in valid_numbers)

    sources: list[CitationSource] = []
    seen_urls: set[str] = set()
    for number in valid_numbers:
        result = search_results[number - 1]
        if result.source_url in seen_urls:
            continue
        seen_urls.add(result.source_url)
        sources.append(
            CitationSource(
                label=f"S{number}",
                page_title=result.page_title,
                section_title=result.section_title,
                url=result.source_url,
            )
        )

    return CitationValidation(valid_numbers, invalid_numbers, tuple(sources))


def remove_invalid_citations(answer: str, invalid_numbers: Sequence[int]) -> str:
    """Remove citation markers that cannot be resolved to a search result."""
    invalid = set(invalid_numbers)
    return _CITATION_PATTERN.sub(
        lambda match: "" if int(match.group(1)) in invalid else match.group(0),
        answer,
    )


def format_sources(sources: Sequence[CitationSource]) -> list[str]:
    """Format validated metadata as a human-readable source list."""
    return [
        f"[{source.label}] {source.page_title} - {source.section_title}: {source.url}"
        for source in sources
    ]


def _unique_in_order(numbers: Iterable[int]) -> tuple[int, ...]:
    ordered: list[int] = []
    seen: set[int] = set()
    for number in numbers:
        if number not in seen:
            seen.add(number)
            ordered.append(number)
    return tuple(ordered)


def _unique_strings_in_order(values: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)
