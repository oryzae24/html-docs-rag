from collections.abc import Sequence

import pytest

from python_doc_rag.models import SearchChunk, SearchResult
from python_doc_rag.reranking import RerankingRetriever


def chunk(name: str, *, text: str | None = None) -> SearchChunk:
    return SearchChunk(
        text=text or f"本文{name}",
        page_title=f"ページ{name}",
        section_title=f"節{name}",
        source_url=f"https://trusted.invalid/{name}",
        category="library",
        chunk_index=1,
        start_index=10,
        extra_metadata={"source_path": f"library/{name}.html"},
    )


def result(rank: int, item: SearchChunk) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=1.0 / rank,
        chunk=item,
        page_title=item.page_title,
        section_title=item.section_title,
        source_url=item.source_url,
        category=item.category,
    )


class Searcher:
    def __init__(self, items: Sequence[SearchResult]) -> None:
        self.items = list(items)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.items[:top_k]


class Scorer:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = tuple(scores)
        self.calls: list[tuple[tuple[tuple[str, str], ...], int]] = []

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> Sequence[float]:
        self.calls.append((tuple(pairs), batch_size))
        return self.scores


def test_reranker_batches_labeled_content_and_requests_bounded_candidates() -> None:
    items = [chunk("a"), chunk("b")]
    searcher = Searcher([result(1, items[0]), result(2, items[1])])
    scorer = Scorer([0.2, 0.9])
    reranker = RerankingRetriever(
        searcher,
        scorer,
        candidate_k=30,
        batch_size=8,
    )

    ranked = reranker.search("質問", top_k=2)

    assert [item.chunk for item in ranked] == [items[1], items[0]]
    assert searcher.calls == [("質問", 30)]
    pairs, batch_size = scorer.calls[0]
    assert batch_size == 8
    assert pairs[0][0] == "質問"
    assert pairs[0][1] == "ページタイトル: ページa\nセクションタイトル: 節a\n本文: 本文a"


def test_reranker_ties_preserve_original_rank_and_metadata() -> None:
    first = chunk("first")
    second = chunk("second")
    reranker = RerankingRetriever(
        Searcher([result(1, first), result(2, second)]),
        Scorer([0.5, 0.5]),
        candidate_k=20,
        batch_size=4,
    )

    ranked = reranker.search("質問", top_k=2)

    assert [item.chunk for item in ranked] == [first, second]
    assert ranked[0].chunk is first
    assert ranked[0].source_url == first.source_url
    assert ranked[0].chunk.extra_metadata == first.extra_metadata
    assert reranker.last_trace.matches[0].original_rank == 1
    assert reranker.last_trace.matches[0].original_score == 1.0


def test_reranker_retrieve_returns_only_original_chunks() -> None:
    first = chunk("first")
    second = chunk("second")
    reranker = RerankingRetriever(
        Searcher([result(1, first), result(2, second)]),
        Scorer([0.1, 0.8]),
        candidate_k=20,
        batch_size=4,
    )

    assert reranker.retrieve("質問", limit=1) == (second,)


@pytest.mark.parametrize("scores", [[0.1], [0.1, float("nan")]])
def test_reranker_rejects_invalid_scores(scores: list[float]) -> None:
    items = [chunk("a"), chunk("b")]
    reranker = RerankingRetriever(
        Searcher([result(1, items[0]), result(2, items[1])]),
        Scorer(scores),
        candidate_k=20,
        batch_size=4,
    )

    with pytest.raises(RuntimeError, match="reranker returned"):
        reranker.search("質問", top_k=2)


@pytest.mark.parametrize(
    ("candidate_k", "batch_size"),
    [(0, 1), (1, 0), (True, 1)],
)
def test_reranker_rejects_invalid_configuration(
    candidate_k: int,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        RerankingRetriever(
            Searcher([]),
            Scorer([]),
            candidate_k=candidate_k,
            batch_size=batch_size,
        )
