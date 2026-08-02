import pytest

from python_doc_rag.models import SearchChunk, SearchResult
from python_doc_rag.retrieval import (
    BM25Retriever,
    CodeAwareNgramTokenizer,
    ReciprocalRankFusionRetriever,
    VectorIndexRetriever,
)


def make_chunk(name: str) -> SearchChunk:
    return SearchChunk(
        text=f"本文{name}",
        page_title=f"ページ{name}",
        section_title=f"節{name}",
        source_url=f"https://trusted.invalid/{name}",
        category="tutorial",
        chunk_index=0,
        start_index=0,
    )


def make_result(rank: int, chunk: SearchChunk) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=1.0 / rank,
        chunk=chunk,
        page_title=chunk.page_title,
        section_title=chunk.section_title,
        source_url=chunk.source_url,
        category=chunk.category,
    )


class FakeVectorIndex:
    """Expose only the search surface used by VectorIndexRetriever."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_vector_index_retriever_preserves_search_rank() -> None:
    first = make_chunk("first")
    second = make_chunk("second")
    index = FakeVectorIndex([make_result(1, first), make_result(2, second)])
    retriever = VectorIndexRetriever(index)  # type: ignore[arg-type]

    chunks = retriever.retrieve("質問", limit=2)

    assert chunks == (first, second)
    assert index.calls == [("質問", 2)]


@pytest.mark.parametrize(
    "identifier",
    [
        "isatty",
        "isatty()",
        "__name__",
        "__main__",
        "shlex.split",
        "subprocess.run",
        "subprocess.Popen",
        "sys.argv",
        "argv[0]",
        "setUp",
        "tearDown",
        "KeyboardInterrupt",
        "EOFError",
        "os.pipe",
        "Path.exists",
    ],
)
def test_code_aware_tokenizer_preserves_identifiers(identifier: str) -> None:
    tokenizer = CodeAwareNgramTokenizer((2,))

    assert identifier in tokenizer.tokenize(identifier)


def test_code_aware_tokenizer_generates_japanese_bigrams() -> None:
    tokens = CodeAwareNgramTokenizer((2,)).tokenize("標準出力")

    assert tokens == ("標準", "準出", "出力")


def test_code_aware_tokenizer_generates_japanese_trigrams() -> None:
    tokens = CodeAwareNgramTokenizer((3,)).tokenize("標準出力")

    assert tokens == ("標準出", "準出力")


def test_japanese_ngrams_do_not_cross_whitespace_or_punctuation() -> None:
    tokens = CodeAwareNgramTokenizer((2,)).tokenize("標準 出力、履歴。保存")

    assert {"標準", "出力", "履歴", "保存"}.issubset(tokens)
    assert "準出" not in tokens
    assert "力履" not in tokens
    assert "歴保" not in tokens


def test_bm25_ranks_identifier_match_first() -> None:
    irrelevant = make_chunk("irrelevant")
    relevant = SearchChunk(
        text="IOBase.isatty() はストリームが端末なら True を返します。",
        page_title="io",
        section_title="I/O 基底クラス",
        source_url="https://trusted.invalid/io",
        category="library",
        chunk_index=3,
        start_index=120,
        extra_metadata={"anchor": "isatty"},
    )
    retriever = BM25Retriever([irrelevant, relevant])

    results = retriever.search("isatty()とは何ですか？", top_k=2)

    assert results[0].chunk == relevant
    assert len(results) == 1
    assert results[0].score > 0.0
    assert results[0].source_url == relevant.source_url
    assert results[0].chunk.extra_metadata == {"anchor": "isatty"}


def test_bm25_rejects_empty_corpus_query_and_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="corpus"):
        BM25Retriever([])

    retriever = BM25Retriever([make_chunk("first")])
    with pytest.raises(ValueError, match="empty"):
        retriever.search("  ")
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("質問", top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("質問", top_k=True)  # type: ignore[arg-type]


def test_bm25_ties_follow_corpus_order() -> None:
    first = SearchChunk(
        text="共通語",
        page_title="同じページ",
        section_title="同じ節",
        source_url="https://trusted.invalid/first",
        category="tutorial",
        chunk_index=0,
        start_index=0,
    )
    second = SearchChunk(
        text="共通語",
        page_title="同じページ",
        section_title="同じ節",
        source_url="https://trusted.invalid/second",
        category="tutorial",
        chunk_index=0,
        start_index=0,
    )
    retriever = BM25Retriever([first, second])

    chunks = retriever.retrieve("共通語", limit=10)

    assert chunks == (first, second)


def test_bm25_unknown_tokens_return_no_arbitrary_documents() -> None:
    retriever = BM25Retriever([make_chunk("first"), make_chunk("second")])

    assert retriever.retrieve("zzzz_completely_unknown", limit=2) == ()


class FakeRetriever:
    """Return a fixed ranking and record the requested candidate count."""

    def __init__(self, chunks: list[SearchChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, *, limit: int) -> tuple[SearchChunk, ...]:
        self.calls.append((question, limit))
        return tuple(self.chunks[:limit])


def _copy_chunk(chunk: SearchChunk) -> SearchChunk:
    return SearchChunk.from_dict(chunk.to_dict())


def test_rrf_merges_same_stable_chunk_and_calculates_score() -> None:
    shared = make_chunk("shared")
    dense_only = make_chunk("dense-only")
    sparse_only = make_chunk("sparse-only")
    dense = FakeRetriever([shared, dense_only])
    sparse = FakeRetriever([_copy_chunk(shared), sparse_only])
    hybrid = ReciprocalRankFusionRetriever(
        [dense, sparse],
        rrf_k=10,
        candidate_k=2,
    )

    results = hybrid.search("質問", top_k=3)

    assert [result.chunk for result in results] == [
        shared,
        dense_only,
        sparse_only,
    ]
    assert results[0].score == pytest.approx(2 / 11)
    assert results[1].score == pytest.approx(1 / 12)
    assert results[0].page_title == shared.page_title
    assert dense.calls == [("質問", 2)]
    assert sparse.calls == [("質問", 2)]


def test_rrf_does_not_double_count_duplicate_within_one_retriever() -> None:
    shared = make_chunk("shared")
    hybrid = ReciprocalRankFusionRetriever(
        [FakeRetriever([shared, _copy_chunk(shared)]), FakeRetriever([])],
        rrf_k=10,
        candidate_k=2,
    )

    results = hybrid.search("質問", top_k=2)

    assert len(results) == 1
    assert results[0].score == pytest.approx(1 / 11)


def test_rrf_key_keeps_same_text_at_different_urls_separate() -> None:
    first = make_chunk("first")
    data = first.to_dict()
    data["source_url"] = "https://trusted.invalid/second"
    second = SearchChunk.from_dict(data)
    hybrid = ReciprocalRankFusionRetriever(
        [FakeRetriever([first, second]), FakeRetriever([])],
        candidate_k=2,
    )

    chunks = hybrid.retrieve("質問", limit=2)

    assert chunks == (first, second)


def test_rrf_key_keeps_chunks_at_different_offsets_separate() -> None:
    first = make_chunk("first")
    data = first.to_dict()
    data["chunk_index"] = 1
    data["start_index"] = 100
    second = SearchChunk.from_dict(data)
    hybrid = ReciprocalRankFusionRetriever(
        [FakeRetriever([first, second]), FakeRetriever([])],
        candidate_k=2,
    )

    chunks = hybrid.retrieve("質問", limit=2)

    assert chunks == (first, second)


def test_rrf_retains_chunks_found_by_only_one_retriever() -> None:
    dense_only = make_chunk("dense-only")
    sparse_only = make_chunk("sparse-only")
    hybrid = ReciprocalRankFusionRetriever(
        [FakeRetriever([dense_only]), FakeRetriever([sparse_only])],
        candidate_k=1,
    )

    chunks = hybrid.retrieve("質問", limit=2)

    assert chunks == (dense_only, sparse_only)


@pytest.mark.parametrize(
    ("rrf_k", "candidate_k", "message"),
    [
        (0, 10, "rrf_k"),
        (10, 0, "candidate_k"),
        (True, 10, "rrf_k"),
        (10, True, "candidate_k"),
    ],
)
def test_rrf_validates_k_and_candidate_k(
    rrf_k: int,
    candidate_k: int,
    message: str,
) -> None:
    retrievers = [FakeRetriever([]), FakeRetriever([])]

    with pytest.raises(ValueError, match=message):
        ReciprocalRankFusionRetriever(
            retrievers,
            rrf_k=rrf_k,
            candidate_k=candidate_k,
        )


def test_rrf_rejects_one_retriever_and_invalid_search_limit() -> None:
    retriever = FakeRetriever([make_chunk("first")])
    with pytest.raises(ValueError, match="at least two"):
        ReciprocalRankFusionRetriever([retriever])

    hybrid = ReciprocalRankFusionRetriever([retriever, FakeRetriever([])])
    with pytest.raises(ValueError, match="top_k"):
        hybrid.search("質問", top_k=0)
    with pytest.raises(ValueError, match="empty"):
        hybrid.retrieve(" ", limit=1)
