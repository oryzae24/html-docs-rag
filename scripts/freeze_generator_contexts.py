"""Freeze the Phase A winner's Top-5 chunks for generator-only comparison."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from python_doc_rag.embedding_models import retrieval_embedding_spec
from python_doc_rag.frozen_contexts import (
    FrozenContextRecord,
    save_frozen_contexts_atomic,
)
from python_doc_rag.rag_evaluation import (
    load_answerability_questions,
    load_rag_quality_questions,
)
from python_doc_rag.reranking import (
    CrossEncoderPairScorer,
    RerankingRetriever,
    model_spec_for_key,
)
from python_doc_rag.retrieval import (
    BM25Retriever,
    CodeAwareNgramTokenizer,
    ReciprocalRankFusionRetriever,
    VectorIndexRetriever,
)
from python_doc_rag.technical_retrieval import (
    FieldBM25Retriever,
    SymbolRetriever,
    WeightedRankFusionRetriever,
    load_symbol_sidecar,
)
from python_doc_rag.vector_store import load_chunks_jsonl, load_vector_index


def parse_args() -> argparse.Namespace:
    """Parse explicit protected inputs and one external output artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--symbol-sidecar", type=Path, required=True)
    parser.add_argument("--answerability-questions", type=Path, required=True)
    parser.add_argument("--rag-questions", type=Path, required=True)
    parser.add_argument("--hard-case-questions", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--source-code-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--retrieval-mode",
        choices=("phase-a-winner", "recommended-v1"),
        default="phase-a-winner",
    )
    return parser.parse_args()


def main() -> int:
    """Retrieve each question once and save exact selected tuples."""
    args = parse_args()
    from sentence_transformers import SentenceTransformer

    data_root = args.data_root.expanduser()
    metadata_path = data_root / "indexes/python_3_13_ja_metadata.jsonl"
    chunks = load_chunks_jsonl(metadata_path)
    records = load_symbol_sidecar(chunks, args.symbol_sidecar)
    if args.retrieval_mode == "phase-a-winner":
        embedding_spec = retrieval_embedding_spec("bge-m3")
        index_root = args.embedding_root.expanduser() / embedding_spec.key
        embedding_model = SentenceTransformer(
            embedding_spec.model_name,
            revision=embedding_spec.revision,
            device=args.device,
            trust_remote_code=False,
        )
        vector_index = load_vector_index(
            index_root / "index.faiss",
            index_root / "metadata.jsonl",
            embedding_model=embedding_model,
            query_prefix=embedding_spec.query_prefix,
        )
        candidate_searcher: Any = WeightedRankFusionRetriever(
            [
                ("identifiers", SymbolRetriever(chunks, records), 1.0),
                (
                    "section_title",
                    FieldBM25Retriever(chunks, field="section_title"),
                    1.0,
                ),
                (
                    "page_title",
                    FieldBM25Retriever(chunks, field="page_title"),
                    1.0,
                ),
                ("body_dense", VectorIndexRetriever(vector_index), 1.0),
                ("body_lexical", FieldBM25Retriever(chunks, field="body"), 1.0),
            ],
            rrf_k=10,
            candidate_k=30,
        )
        embedding_revision: str | None = embedding_spec.revision
        index_path = index_root / "index.faiss"
        field_config: str | None = "equal"
    else:
        import json

        baseline_manifest = json.loads(
            (data_root / "indexes/python_3_13_ja_index_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        embedding_model_name = str(baseline_manifest["model_name"])
        embedding_model = SentenceTransformer(
            embedding_model_name,
            device=args.device,
            trust_remote_code=False,
        )
        index_path = data_root / "indexes/python_3_13_ja.faiss"
        vector_index = load_vector_index(
            index_path,
            metadata_path,
            embedding_model=embedding_model,
        )
        candidate_searcher = ReciprocalRankFusionRetriever(
            [
                VectorIndexRetriever(vector_index),
                BM25Retriever(
                    chunks,
                    tokenizer=CodeAwareNgramTokenizer((2,)),
                ),
            ],
            rrf_k=10,
            candidate_k=30,
        )
        embedding_spec = None
        embedding_revision = None
        field_config = None
    reranker_spec = model_spec_for_key("mmarco-minilm")
    scorer = CrossEncoderPairScorer.from_pretrained(
        reranker_spec,
        device=args.device,
        max_length=512,
    )
    reranker = RerankingRetriever(
        candidate_searcher,
        scorer,
        candidate_k=30,
        batch_size=16,
    )
    records_to_save: list[FrozenContextRecord] = []
    datasets: list[tuple[str, list[Any]]] = [
        (
            "answerability",
            load_answerability_questions(args.answerability_questions),
        ),
        ("rag-quality", load_rag_quality_questions(args.rag_questions)),
    ]
    if args.hard_case_questions is not None:
        datasets.append(
            (
                "hard-cases",
                load_rag_quality_questions(args.hard_case_questions),
            )
        )
    for dataset, questions in datasets:
        for question in questions:
            results = reranker.search(question.question, top_k=5)
            trace = reranker.last_trace
            if len(results) != 5 or len(trace.matches) != 5:
                raise RuntimeError("frozen context requires exactly five candidates")
            records_to_save.append(
                FrozenContextRecord(
                    id=question.id,
                    question=question.question,
                    dataset=dataset,
                    chunks=tuple(result.chunk for result in results),
                    rerank_scores=tuple(match.rerank_score for match in trace.matches),
                    original_ranks=tuple(match.original_rank for match in trace.matches),
                )
            )
    save_frozen_contexts_atomic(
        records_to_save,
        args.output_path,
        settings={
            "source_code_commit": args.source_code_commit,
            "retrieval_mode": args.retrieval_mode,
            "embedding_model": (
                embedding_spec.model_name
                if embedding_spec is not None
                else embedding_model_name
            ),
            "embedding_revision": embedding_revision,
            "symbol_sidecar_sha256": _sha256(args.symbol_sidecar),
            "embedding_index_sha256": _sha256(index_path),
            "answerability_questions_sha256": _sha256(
                args.answerability_questions
            ),
            "rag_questions_sha256": _sha256(args.rag_questions),
            "hard_case_questions_sha256": (
                _sha256(args.hard_case_questions)
                if args.hard_case_questions is not None
                else None
            ),
            "field_config": field_config,
            "field_rrf_k": 10,
            "field_candidate_k": 30,
            "reranker_model": reranker_spec.model_name,
            "reranker_revision": reranker_spec.revision,
            "reranker_candidate_k": 30,
            "top_k": 5,
            "openai_api_used": False,
            "contains_secrets": False,
        },
    )
    print(f"Frozen contexts: {len(records_to_save)} -> {args.output_path}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
