import hashlib
import json
from pathlib import Path

from python_doc_rag.preparation import prepare_dataset

PROTECTED_CHUNK_SHA256 = (
    "1625fd66c693bcbca4d9318d69f344e7a46609d0d274036cc50476c4b161a869"
)


def test_frozen_snapshot_reproduces_protected_corpus(tmp_path: Path) -> None:
    config = Path("configs/sites/python-docs.toml").resolve()
    data_root = tmp_path / "frozen"

    result = prepare_dataset(
        config,
        data_root,
        until="corpus",
        device="cpu",
    )
    chunks = data_root / "data/processed/chunks.jsonl"

    assert result.source_page_count == 384
    assert result.section_count == 2_766
    assert result.chunk_count == 8_677
    assert result.corpus_sha256 == PROTECTED_CHUNK_SHA256
    assert hashlib.sha256(chunks.read_bytes()).hexdigest() == PROTECTED_CHUNK_SHA256
    source_urls = [
        json.loads(line)["source_url"]
        for line in chunks.read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        url.startswith("https://docs.python.org/ja/3.13/") for url in source_urls
    )
    assert all("/workspace/" not in url and "file://" not in url for url in source_urls)

    reused = prepare_dataset(config, data_root, until="corpus", device="cpu")
    assert reused.reused_dataset
    assert reused.corpus_sha256 == PROTECTED_CHUNK_SHA256
