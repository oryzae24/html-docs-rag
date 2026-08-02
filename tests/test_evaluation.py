import json
from pathlib import Path

import pytest

from python_doc_rag.evaluation import (
    EvaluationQuestion,
    compare_retrieval_reports,
    evaluate_retrieval,
    load_evaluation_questions,
    save_retrieval_evaluation,
)
from python_doc_rag.models import SearchChunk, SearchResult


def _result(rank: int, url: str) -> SearchResult:
    chunk = SearchChunk(
        text="本文",
        page_title="ページ",
        section_title="節",
        source_url=url,
        category="tutorial",
        chunk_index=rank - 1,
        start_index=0,
    )
    return SearchResult(rank, 1.0 / rank, chunk, "ページ", "節", url, "tutorial")


class EvaluationSearcher:
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        del top_k
        if query == "top1":
            urls = ["https://docs.python.org/ja/3.13/tutorial/modules.html"]
        else:
            urls = [
                "https://docs.python.org/ja/3.13/reference/index.html",
                "https://docs.python.org/ja/3.13/tutorial/controlflow.html",
                "https://docs.python.org/ja/3.13/library/unittest.html",
            ]
        return [_result(rank, url) for rank, url in enumerate(urls, start=1)]


def test_load_evaluation_questions(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    rows = [
        {"question": "質問1", "expected_url_keywords": ["tutorial/modules"]},
        {"question": "質問2", "expected_url_keywords": ["library/unittest"]},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    questions = load_evaluation_questions(path)

    assert questions == [
        EvaluationQuestion("質問1", ("tutorial/modules",)),
        EvaluationQuestion("質問2", ("library/unittest",)),
    ]


def test_evaluate_retrieval_calculates_hit_rates_and_details() -> None:
    questions = [
        EvaluationQuestion("top1", ("tutorial/modules",)),
        EvaluationQuestion("top3", ("library/unittest",)),
    ]

    report = evaluate_retrieval(EvaluationSearcher(), questions)

    assert report.top_1_hit_rate == pytest.approx(0.5)
    assert report.top_3_hit_rate == pytest.approx(1.0)
    assert report.top_5_hit_rate == pytest.approx(1.0)
    assert report.questions[0].top_1_hit is True
    assert report.questions[1].top_1_hit is False
    assert report.questions[1].top_3_hit is True
    assert report.questions[1].results[2].source_url.endswith("library/unittest.html")


def test_evaluate_retrieval_accepts_any_expected_url_keyword() -> None:
    questions = [
        EvaluationQuestion("top1", ("library/__main__", "tutorial/modules")),
    ]

    report = evaluate_retrieval(EvaluationSearcher(), questions)

    assert report.questions[0].top_1_hit is True


def test_load_evaluation_questions_accepts_metadata(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "io-isatty-001",
                "question": "isatty()とは？",
                "query_type": "exact_identifier",
                "topic": "io",
                "expected_url_keywords": ["library/io.html"],
                "ignored_note": "unknown fields remain compatible",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert load_evaluation_questions(path) == [
        EvaluationQuestion(
            "isatty()とは？",
            ("library/io.html",),
            "io-isatty-001",
            "exact_identifier",
            "io",
        )
    ]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {
                    "id": "duplicate",
                    "question": "質問1",
                    "query_type": "conceptual",
                    "topic": "topic",
                    "expected_url_keywords": ["first"],
                },
                {
                    "id": "duplicate",
                    "question": "質問2",
                    "query_type": "operational",
                    "topic": "topic",
                    "expected_url_keywords": ["second"],
                },
            ],
            "duplicate evaluation question id",
        ),
        (
            [
                {
                    "id": "invalid-type",
                    "question": "質問",
                    "query_type": "hybrid",
                    "topic": "topic",
                    "expected_url_keywords": ["url"],
                }
            ],
            "unsupported query_type",
        ),
        (
            [{"question": "質問", "expected_url_keywords": []}],
            "expected_url_keywords",
        ),
    ],
)
def test_load_evaluation_questions_rejects_invalid_rows(
    rows: list[dict[str, object]],
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_evaluation_questions(path)


def test_evaluate_retrieval_reports_top_10_mrr_and_groups() -> None:
    class RankedSearcher:
        def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
            relevant_rank = {"first": 1, "eighth": 8}.get(query)
            return [
                _result(
                    rank,
                    "https://docs.python.org/ja/3.13/"
                    + ("second-answer" if rank == relevant_rank else f"noise-{rank}"),
                )
                for rank in range(1, top_k + 1)
            ]

    questions = [
        EvaluationQuestion("first", ("second-answer",), "q1", "exact_identifier", "io"),
        EvaluationQuestion(
            "eighth",
            ("unused-answer", "second-answer"),
            "q2",
            "conceptual",
            "io",
        ),
        EvaluationQuestion("missing", ("absent",), "q3", "operational", "subprocess"),
    ]

    report = evaluate_retrieval(RankedSearcher(), questions, top_k=10)

    assert report.top_1_hit_rate == pytest.approx(1 / 3)
    assert report.top_3_hit_rate == pytest.approx(1 / 3)
    assert report.top_5_hit_rate == pytest.approx(1 / 3)
    assert report.top_10_hit_rate == pytest.approx(2 / 3)
    assert report.mrr_at_10 == pytest.approx((1 + 1 / 8) / 3)
    assert [item.first_relevant_rank for item in report.questions] == [1, 8, None]
    assert report.query_type_metrics["conceptual"].hit_at_10 == 1.0
    assert report.topic_metrics["io"].hit_at_5 == 0.5
    assert report.unmatched_top_10 == ("q3",)


def test_empty_evaluation_is_rejected_and_json_is_utf8(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_retrieval(EvaluationSearcher(), [])

    question = EvaluationQuestion(
        "日本語の質問", ("tutorial/modules",), "q1", "conceptual", "entrypoint"
    )
    report = evaluate_retrieval(EvaluationSearcher(), [question])
    output_path = tmp_path / "result.json"

    save_retrieval_evaluation(report, output_path)

    assert "日本語の質問" in output_path.read_text(encoding="utf-8")
    assert b"\\u65e5" not in output_path.read_bytes()


def test_compare_baseline_and_parent_rank_movements() -> None:
    class ParentSearcher:
        def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
            ranks = {"improved": 1, "worsened": 8, "same": 3}
            relevant = ranks[query]
            return [
                _result(
                    rank,
                    "https://docs.python.org/"
                    + ("answer" if rank == relevant else f"noise-{rank}"),
                )
                for rank in range(1, top_k + 1)
            ]

    class BaselineSearcher:
        def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
            ranks = {"improved": 8, "worsened": 1, "same": 3}
            relevant = ranks[query]
            return [
                _result(
                    rank,
                    "https://docs.python.org/"
                    + ("answer" if rank == relevant else f"noise-{rank}"),
                )
                for rank in range(1, top_k + 1)
            ]

    questions = [
        EvaluationQuestion(name, ("answer",), name, "conceptual", "topic")
        for name in ("improved", "worsened", "same")
    ]
    baseline = evaluate_retrieval(BaselineSearcher(), questions, top_k=10)
    parent = evaluate_retrieval(ParentSearcher(), questions, top_k=10)

    comparison = compare_retrieval_reports(baseline, parent)

    assert comparison.improved_count == 1
    assert comparison.tied_count == 1
    assert comparison.worsened_count == 1
    assert comparison.entered_top_5 == ("improved",)
    assert comparison.left_top_5 == ("worsened",)
    assert comparison.entered_top_10 == ()
    assert comparison.left_top_10 == ()


def test_compare_retrieval_requires_matching_stable_ids() -> None:
    first = evaluate_retrieval(
        EvaluationSearcher(),
        [EvaluationQuestion("top1", ("tutorial/modules",), "first")],
    )
    second = evaluate_retrieval(
        EvaluationSearcher(),
        [EvaluationQuestion("top1", ("tutorial/modules",), "second")],
    )
    with pytest.raises(ValueError, match="IDs must match"):
        compare_retrieval_reports(first, second)
