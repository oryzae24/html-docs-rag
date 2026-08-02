import json
from pathlib import Path

import pytest

from python_doc_rag.candidate_evaluation import (
    evaluate_candidate_recall,
    save_candidate_evaluation_atomic,
)
from python_doc_rag.evaluation import EvaluationQuestion
from python_doc_rag.models import SearchChunk, SearchResult


def _result(rank: int, *, relevant: bool = False, section: int = 0) -> SearchResult:
    suffix = "answer" if relevant else f"noise-{rank}"
    chunk = SearchChunk(
        text="本文",
        page_title=f"ページ{rank // 2}",
        section_title=f"節{section}",
        source_url=f"https://docs.python.org/ja/3.13/library/{suffix}.html#s{section}",
        category="library",
        chunk_index=rank,
        start_index=rank * 10,
    )
    return SearchResult(
        rank,
        1.0 / rank,
        chunk,
        chunk.page_title,
        chunk.section_title,
        chunk.source_url,
        chunk.category,
    )


class _Searcher:
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        relevant_rank = {"ten": 10, "twenty": 20, "thirty": 30}.get(query)
        return [
            _result(rank, relevant=rank == relevant_rank, section=rank % 3)
            for rank in range(1, top_k + 1)
        ]


def test_candidate_recall_reports_10_20_30_groups_and_hard_cases() -> None:
    questions = [
        EvaluationQuestion("ten", ("answer",), "q10", "exact_identifier", "io"),
        EvaluationQuestion("twenty", ("answer",), "q20", "conceptual", "io"),
        EvaluationQuestion("thirty", ("answer",), "q30", "operational", "io"),
        EvaluationQuestion("missing", ("answer",), "miss", "conceptual", "io"),
    ]

    report = evaluate_candidate_recall(
        _Searcher(), questions, candidate_k=30, hard_case_ids=("q30",)
    )

    assert report["summary"]["recall_at_10"] == pytest.approx(0.25)
    assert report["summary"]["recall_at_20"] == pytest.approx(0.5)
    assert report["summary"]["recall_at_30"] == pytest.approx(0.75)
    assert report["summary"]["expected_url_candidate_rate"] == pytest.approx(0.75)
    assert report["query_types"]["exact_identifier"]["recall_at_30"] == 1.0
    assert report["hard_cases"]["q30"]["first_relevant_rank"] == 30
    assert report["questions"][3]["first_relevant_rank"] is None
    assert report["questions"][0]["unique_url_count"] == 30
    assert "text" not in report["questions"][0]["results"][0]


def test_candidate_evaluation_rejects_tuning_below_required_depth() -> None:
    question = EvaluationQuestion("ten", ("answer",))

    with pytest.raises(ValueError, match="at least 30"):
        evaluate_candidate_recall(_Searcher(), [question], candidate_k=20)


def test_candidate_evaluation_save_is_utf8_and_replaces_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.json"
    payload = {"question": "日本語", "openai_api_used": False}

    save_candidate_evaluation_atomic(payload, path)
    save_candidate_evaluation_atomic(payload | {"revision": "v1"}, path)

    assert json.loads(path.read_text(encoding="utf-8"))["revision"] == "v1"
    assert not list(tmp_path.glob("*.tmp"))
