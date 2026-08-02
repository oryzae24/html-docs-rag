import pytest

from python_doc_rag.citation import (
    CitationContractError,
    finalize_cited_answer,
    format_sources,
    remove_invalid_citations,
    validate_citations,
)
from python_doc_rag.models import SearchChunk


def make_result(section: str, url: str) -> SearchChunk:
    return SearchChunk(
        text="検索された本文",
        page_title="Python 3.13 ドキュメント",
        section_title=section,
        source_url=url,
        category="tutorial",
        chunk_index=0,
        start_index=0,
    )


def test_validate_citations_detects_invalid_and_deduplicates_urls() -> None:
    first_url = "https://docs.python.org/ja/3.13/tutorial/controlflow.html#if"
    results = [
        make_result("if 文", first_url),
        make_result("if 文の続き", first_url),
        make_result(
            "for 文",
            "https://docs.python.org/ja/3.13/tutorial/controlflow.html#for",
        ),
    ]
    answer = "条件分岐にはifを使います[S1][S2]。反復にはforです[S3][S9]。"

    validation = validate_citations(answer, results)

    assert validation.valid_numbers == (1, 2, 3)
    assert validation.invalid_numbers == (9,)
    assert [source.label for source in validation.sources] == ["S1", "S3"]
    assert [source.url for source in validation.sources] == [
        first_url,
        results[2].source_url,
    ]
    assert format_sources(validation.sources)[0].startswith(
        "[S1] Python 3.13 ドキュメント - if 文:"
    )


def test_invalid_citation_is_removed_without_using_answer_url() -> None:
    trusted_url = "https://docs.python.org/ja/3.13/tutorial/controlflow.html#if"
    answer = "回答[S1] 偽URL https://example.com [S4]"
    validation = validate_citations(answer, [make_result("if 文", trusted_url)])

    cleaned = remove_invalid_citations(answer, validation.invalid_numbers)

    assert cleaned == "回答[S1] 偽URL https://example.com "
    assert validation.sources[0].url == trusted_url


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("回答[S0]", "malformed_citation"),
        ("回答[S01]", "malformed_citation"),
        ("回答[S2]", "citation_out_of_range"),
        ("回答[s1]", "malformed_citation"),
        ("引用のない回答", "citation_required"),
    ],
)
def test_finalizer_rejects_invalid_or_missing_citations(
    answer: str,
    reason: str,
) -> None:
    chunk = make_result("if 文", "https://trusted.invalid/if")
    with pytest.raises(CitationContractError) as error:
        finalize_cited_answer(
            answer,
            (chunk,),
            generation_attempts=1,
        )

    assert reason in error.value.reasons


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("回答https://untrusted.invalid [S1]", "url_detected"),
        ("[資料](relative/path)に基づく回答[S1]", "markdown_link_detected"),
    ],
)
def test_finalizer_rejects_urls_and_markdown_links(
    answer: str,
    reason: str,
) -> None:
    with pytest.raises(CitationContractError) as error:
        finalize_cited_answer(
            answer,
            (make_result("if 文", "https://trusted.invalid/if"),),
            generation_attempts=1,
        )

    assert reason in error.value.reasons


def test_finalizer_builds_sources_only_from_chunk_metadata() -> None:
    trusted_url = "https://docs.python.org/ja/3.13/tutorial/controlflow.html#if"
    chunk = make_result("if 文", trusted_url)
    contexts = (chunk,)

    result = finalize_cited_answer(
        "if文を使います[S1]",
        contexts,
        generation_attempts=1,
    )

    assert result.answer_text == "if文を使います[S1]"
    assert result.sources[0].label == "S1"
    assert result.sources[0].url == trusted_url
    assert result.retrieved_chunks is contexts
