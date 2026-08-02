import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from python_doc_rag.dataset_layout import (
    generic_dataset_manifest,
    resolve_dataset_artifacts,
    write_dataset_manifest_atomic,
)
from python_doc_rag.models import SourceDocument
from python_doc_rag.preparation import prepare_dataset
from python_doc_rag.profile_artifacts import profile_artifact_paths, sha256_file
from python_doc_rag.profiles import runtime_profile


def _write_config(
    path: Path,
    *,
    loader: str = "local",
    profile: str = "none",
) -> None:
    if loader == "local":
        loader_table = """type = "local-html-tree"
source_base_url = "https://docs.python.org/ja/3.13/"
include_path_prefixes = ["tutorial/"]"""
        parser_table = """type = "python-sphinx"
python_version = "3.13"
minimum_section_text_length = 20"""
    else:
        loader_table = """type = "bounded-http"
start_urls = ["https://example.com/docs/"]
max_pages = 2
timeout_seconds = 2
request_delay_seconds = 0
max_response_bytes = 10000
user_agent = "fixture/1.0"
respect_robots_txt = true
retain_query = false
max_retries = 0"""
        parser_table = """type = "generic-html"
content_selectors = ["main"]
exclude_selectors = ["nav"]
title_selectors = ["h1", "title"]
heading_levels = [1, 2, 3]
minimum_section_text_length = 5
fallback_to_body = false
include_lead_text = true"""
    path.write_text(
        f"""[dataset]
name = "Fixture docs"
slug = "fixture-docs"
language = "en"
description = "fixture"

[loader]
{loader_table}

[parser]
{parser_table}

[chunking]
chunk_size = 1000
chunk_overlap = 150

[index]
embedding_model = "fixture/model"
embedding_batch_size = 2

[profile]
prepare = "{profile}"
""",
        encoding="utf-8",
    )


def _write_python_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<html><head><title>Fixture</title></head><body><main>
<section id="fixture"><h1>Fixture</h1>
<p>これは検索対象にできる十分な長さの日本語fixture本文です。</p>
</section></main></body></html>""",
        encoding="utf-8",
    )


def _fake_dense_builder(
    input_path: Path,
    index_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    **kwargs: Any,
) -> SimpleNamespace:
    del kwargs
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(b"fixture-index")
    metadata_path.write_bytes(input_path.read_bytes())
    rows = sum(1 for line in metadata_path.read_text(encoding="utf-8").splitlines() if line)
    manifest_path.write_text(
        json.dumps(
            {
                "model_name": "fixture/model",
                "model_revision": None,
                "embedding_dimension": 3,
                "chunk_count": rows,
                "elapsed_seconds": 0.1,
                "input_jsonl": str(input_path),
                "input_jsonl_sha256": sha256_file(input_path),
                "index_sha256": sha256_file(index_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        index_path=index_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        chunk_count=rows,
        embedding_dimension=3,
        elapsed_seconds=0.1,
    )


def test_local_prepare_publishes_generic_layout_and_reuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    monkeypatch.setattr(
        "python_doc_rag.preparation.build_vector_index",
        _fake_dense_builder,
    )

    first = prepare_dataset(
        config,
        data_root,
        source_root=source,
        until="index",
        device="cpu",
    )
    second = prepare_dataset(
        config,
        data_root,
        source_root=source,
        until="index",
        device="cpu",
    )

    assert first.chunk_count == 1
    assert not first.reused_dataset
    assert second.reused_dataset
    assert (data_root / "dataset_manifest.json").is_file()
    assert (data_root / "data/processed/chunks.jsonl").is_file()
    assert (data_root / "indexes/dense.faiss").is_file()
    index_manifest = json.loads(
        (data_root / "indexes/index_manifest.json").read_text(encoding="utf-8")
    )
    assert index_manifest["input_jsonl"] == "data/processed/chunks.jsonl"
    assert str(tmp_path) not in json.dumps(index_manifest)


def test_prepare_failure_leaves_dense_outputs_in_staging_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")

    def fail_builder(
        input_path: Path,
        index_path: Path,
        metadata_path: Path,
        manifest_path: Path,
        **kwargs: Any,
    ) -> None:
        del input_path, metadata_path, manifest_path, kwargs
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_bytes(b"partial")
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(
        "python_doc_rag.preparation.build_vector_index",
        fail_builder,
    )

    with pytest.raises(RuntimeError, match="fixture failure"):
        prepare_dataset(
            config,
            data_root,
            source_root=source,
            until="index",
            device="cpu",
        )

    assert not (data_root / "dataset_manifest.json").exists()
    assert not (data_root / "indexes/dense.faiss").exists()
    assert (data_root / ".prepare-staging/indexes/dense.faiss").is_file()

    monkeypatch.setattr(
        "python_doc_rag.preparation.build_vector_index",
        _fake_dense_builder,
    )
    resumed = prepare_dataset(
        config,
        data_root,
        source_root=source,
        resume=True,
        until="index",
        device="cpu",
    )

    assert resumed.chunk_count == 1
    assert (data_root / "indexes/dense.faiss").is_file()


def test_completed_local_dataset_detects_live_source_mutation(tmp_path: Path) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    _write_config(config)
    page = source / "tutorial/page.html"
    _write_python_page(page)
    prepare_dataset(config, data_root, source_root=source, until="corpus")
    page.write_text(
        "<html><body><main><section id='changed'><h1>Changed</h1>"
        "<p>変更後の検索可能な十分に長いfixture本文です。</p>"
        "</section></main></body></html>",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="local source content"):
        prepare_dataset(config, data_root, source_root=source, until="corpus")


def test_completed_dataset_rejects_cross_artifact_index_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    monkeypatch.setattr(
        "python_doc_rag.preparation.build_vector_index",
        _fake_dense_builder,
    )
    prepare_dataset(
        config,
        data_root,
        source_root=source,
        until="index",
        device="cpu",
    )
    manifest_path = data_root / "indexes/index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_jsonl_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="dense index manifest"):
        prepare_dataset(
            config,
            data_root,
            source_root=source,
            until="index",
            device="cpu",
        )


def test_prepare_rejects_managed_data_symlink_before_external_write(
    tmp_path: Path,
) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    outside = tmp_path / "outside"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    data_root.mkdir()
    outside.mkdir()
    (data_root / "data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        prepare_dataset(config, data_root, source_root=source, until="corpus")

    assert list(outside.iterdir()) == []


def test_prepare_rejects_managed_data_native_junction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    junction = data_root / "data"
    junction.mkdir(parents=True)
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda path: path == junction,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="symlink/junction"):
        prepare_dataset(config, data_root, source_root=source, until="corpus")

    assert list(junction.iterdir()) == []


@pytest.mark.parametrize(
    "relative",
    (
        ".prepare-publication.json",
        ".prepare-rebuild-recovery.json",
        "data/raw/.source-refresh-rollback",
        "data/.raw.backup",
    ),
)
def test_runtime_artifact_resolution_rejects_pending_publication(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / "data-root"
    marker = root / relative
    if marker.suffix == ".json":
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}", encoding="utf-8")
    else:
        marker.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="run prepare to recover"):
        resolve_dataset_artifacts(root)


def test_prepare_resume_rejects_staging_child_symlink_before_external_write(
    tmp_path: Path,
) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    outside = tmp_path / "outside"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    first = prepare_dataset(config, data_root, source_root=source, until="corpus")
    staging = data_root / ".prepare-staging"
    staging.mkdir()
    (staging / "prepare_state.json").write_text(
        json.dumps(
            {
                "schema_revision": "prepare-staging-v1",
                "source_config_sha256": first.dataset_manifest.source_config_sha256,
                "processing_config_sha256": (
                    first.dataset_manifest.processing_config_sha256
                ),
                "until": "corpus",
            }
        ),
        encoding="utf-8",
    )
    outside.mkdir()
    (staging / "data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        prepare_dataset(
            config,
            data_root,
            source_root=source,
            resume=True,
            until="corpus",
        )

    assert list(outside.iterdir()) == []


def test_manifestless_legacy_artifacts_are_preserved_exactly(tmp_path: Path) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    legacy = data_root / "data/processed/python_3_13_ja_chunks.jsonl"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    legacy.parent.mkdir(parents=True)
    original = b'{"legacy": true}\n'
    legacy.write_bytes(original)

    with pytest.raises(RuntimeError, match="legacy dataset artifacts exist"):
        prepare_dataset(config, data_root, source_root=source, until="corpus")

    assert legacy.read_bytes() == original
    assert not (data_root / "dataset_manifest.json").exists()
    assert not (data_root / ".prepare-staging").exists()


def test_partial_staging_precedes_completed_dataset_fast_path(tmp_path: Path) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    first = prepare_dataset(config, data_root, source_root=source, until="corpus")
    staging = data_root / ".prepare-staging"
    staging.mkdir()
    (staging / "prepare_state.json").write_text(
        json.dumps(
            {
                "schema_revision": "prepare-staging-v1",
                "source_config_sha256": first.dataset_manifest.source_config_sha256,
                "processing_config_sha256": (
                    first.dataset_manifest.processing_config_sha256
                ),
                "until": "corpus",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="partial preparation staging"):
        prepare_dataset(config, data_root, source_root=source, until="corpus")

    resumed = prepare_dataset(
        config,
        data_root,
        source_root=source,
        resume=True,
        until="corpus",
    )
    assert not resumed.reused_dataset
    assert not staging.exists()


def test_legacy_v1_dataset_reuses_but_is_never_rebuilt_in_place(
    tmp_path: Path,
) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    prepare_dataset(config, data_root, source_root=source, until="corpus")
    manifest_path = data_root / "dataset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_revision"] = "dataset-artifact-layout-v1"
    payload.pop("source_config_sha256")
    payload.pop("processing_config_sha256")
    payload.pop("source_snapshot_sha256")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy_manifest = manifest_path.read_bytes()
    chunks_path = data_root / "data/processed/chunks.jsonl"
    legacy_chunks = chunks_path.read_bytes()

    reused = prepare_dataset(config, data_root, source_root=source, until="corpus")
    assert reused.reused_dataset

    with pytest.raises(RuntimeError, match="cannot be upgraded in place"):
        prepare_dataset(
            config,
            data_root,
            source_root=source,
            rebuild=True,
            until="corpus",
        )

    assert manifest_path.read_bytes() == legacy_manifest
    assert chunks_path.read_bytes() == legacy_chunks


def test_prepare_reuses_dataset_after_comment_only_config_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "site.toml"
    source = tmp_path / "source"
    data_root = tmp_path / "data-root"
    _write_config(config)
    _write_python_page(source / "tutorial/page.html")
    monkeypatch.setattr(
        "python_doc_rag.preparation.build_vector_index",
        _fake_dense_builder,
    )
    first = prepare_dataset(config, data_root, source_root=source, until="corpus")
    config.write_text(
        config.read_text(encoding="utf-8") + "\n# audit-only comment\n",
        encoding="utf-8",
    )

    second = prepare_dataset(config, data_root, source_root=source, until="corpus")

    assert not first.reused_dataset
    assert second.reused_dataset
    assert (
        second.dataset_manifest.source_config_sha256
        == first.dataset_manifest.source_config_sha256
    )
    assert (
        second.dataset_manifest.processing_config_sha256
        == first.dataset_manifest.processing_config_sha256
    )


def test_http_prepare_continues_after_one_parser_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "http.toml"
    _write_config(config, loader="http")
    good_html = "<html><main><h1 id='ok'>OK</h1><p>enough fixture text</p></main></html>"
    bad_html = "<html><body><nav>only navigation</nav></body></html>"

    def document(url: str, html: str) -> SourceDocument:
        return SourceDocument(
            source_url=url,
            canonical_url=url,
            content=html,
            content_sha256=hashlib.sha256(html.encode()).hexdigest(),
            source_kind="bounded-http",
            logical_path=f"example.com/{url.rsplit('/', 2)[-2]}/index.html",
            category="fixture-docs",
        )

    class FakeLoader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def load(self) -> tuple[SourceDocument, ...]:
            return (
                document("https://example.com/docs/good/", good_html),
                document("https://example.com/docs/bad/", bad_html),
            )

    monkeypatch.setattr(
        "python_doc_rag.preparation.build_document_loader",
        lambda settings, runtime: FakeLoader(settings, runtime),
    )
    raw = tmp_path / "data-root/data/raw"
    raw.mkdir(parents=True)
    (raw / "fetch_manifest.json").write_text("{}", encoding="utf-8")

    result = prepare_dataset(config, tmp_path / "data-root", until="corpus")

    assert result.source_page_count == 2
    assert result.parsed_page_count == 1
    assert result.failed_page_count == 1


def test_generic_dataset_resolves_profile_paths_without_python_hash(
    tmp_path: Path,
) -> None:
    manifest = generic_dataset_manifest(
        dataset_name="Fixture",
        dataset_slug="fixture",
        loader_type="bounded-http",
        parser_type="generic-html",
        site_config_sha256="a" * 64,
        created_at="2026-08-01T00:00:00+00:00",
        source_page_count=1,
        section_count=1,
        chunk_count=1,
    )
    write_dataset_manifest_atomic(manifest, tmp_path / "dataset_manifest.json")

    paths = profile_artifact_paths(runtime_profile("recommended-v2"), tmp_path)

    assert paths.embedding_index_path == (
        tmp_path / "profiles/recommended-v2/bge-m3/index.faiss"
    )
    assert paths.symbol_index_path == (
        tmp_path / "profiles/recommended-v2/symbol_fields.jsonl"
    )
    assert paths.identity_manifest_path == (
        tmp_path / "profiles/recommended-v2/artifact_manifest.json"
    )
    assert len(paths.required_paths) == 5
