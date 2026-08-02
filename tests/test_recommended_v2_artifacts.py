import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from python_doc_rag.embedding_models import retrieval_embedding_spec
from python_doc_rag.models import SearchChunk
from python_doc_rag.profile_artifacts import (
    ProfileArtifactValidation,
    profile_artifact_paths,
    safe_profile_relative_path,
    sha256_file,
    validate_profile_artifacts,
)
from python_doc_rag.profiles import RuntimeProfile, runtime_profile
from python_doc_rag.recommended_v2_artifacts import (
    BASELINE_METADATA_RELATIVE_PATH,
    ArtifactPreparationError,
    prepare_recommended_v2_artifacts,
)
from python_doc_rag.technical_retrieval import write_symbol_sidecar_atomic


def _load_preparation_script() -> Any:
    path = Path(__file__).parent.parent / "scripts/prepare_recommended_v2_artifacts.py"
    spec = importlib.util.spec_from_file_location("recommended_v2_preparation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chunk(index: int) -> SearchChunk:
    return SearchChunk(
        text=f"chunk {index} text",
        page_title="Page",
        section_title=f"section_{index}()",
        source_url=f"https://docs.python.org/ja/3.13/library/x.html#x{index}",
        category="library",
        chunk_index=index,
        start_index=index * 10,
        extra_metadata={"source_path": "library/x.html"},
    )


def _write_chunks(path: Path, chunks: list[SearchChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{json.dumps(chunk.to_dict(), ensure_ascii=False)}\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )


def _validation(
    *,
    errors: tuple[str, ...] = (),
) -> ProfileArtifactValidation:
    spec = retrieval_embedding_spec("bge-m3")
    return ProfileArtifactValidation(
        profile_name="recommended-v2",
        chunk_count=8_677,
        index_count=8_677,
        embedding_dimension=spec.embedding_dimension,
        model_name=spec.model_name,
        model_revision=spec.revision,
        index_sha256="1" * 64,
        metadata_sha256="2" * 64,
        manifest_sha256="4" * 64,
        symbol_sha256="3" * 64,
        errors=errors,
    )


def _create_required_files(root: Path, profile: RuntimeProfile) -> None:
    for path in profile_artifact_paths(profile, root).required_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing", encoding="utf-8")


def _create_baseline_metadata(root: Path) -> Path:
    path = root / BASELINE_METADATA_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("baseline\n", encoding="utf-8")
    return path


def test_profile_paths_derive_exact_recommended_v2_outputs(tmp_path: Path) -> None:
    profile = runtime_profile("recommended-v2")

    paths = profile_artifact_paths(profile, tmp_path)

    assert paths.embedding_index_path == (
        tmp_path
        / "experiments/final_quality_sprint_v2/phase_a/embedding_indexes/"
        "bge-m3/index.faiss"
    )
    assert paths.embedding_metadata_path == paths.embedding_index_path.with_name(
        "metadata.jsonl"
    )
    assert paths.embedding_manifest_path == paths.embedding_index_path.with_name(
        "manifest.json"
    )
    assert paths.symbol_index_path == (
        tmp_path
        / "experiments/final_quality_sprint_v2/phase_a/symbol_fields.jsonl"
    )


@pytest.mark.parametrize("value", ("/tmp/index.faiss", "phase_a/../index.faiss"))
def test_profile_paths_reject_absolute_and_parent_traversal(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="安全な相対path"):
        safe_profile_relative_path(value, "fixture")


def test_profile_paths_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "experiments").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="data-root外"):
        profile_artifact_paths(runtime_profile("recommended-v2"), tmp_path)


def test_validator_checks_manifest_counts_hashes_and_symbol_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_profile = runtime_profile("recommended-v2")
    paths = profile_artifact_paths(original_profile, tmp_path)
    chunks = [_chunk(0), _chunk(1)]
    baseline = tmp_path / BASELINE_METADATA_RELATIVE_PATH
    _write_chunks(baseline, chunks)
    _write_chunks(paths.embedding_metadata_path, chunks)  # type: ignore[arg-type]
    index_path = paths.embedding_index_path
    symbol_path = paths.symbol_index_path
    manifest_path = paths.embedding_manifest_path
    assert index_path is not None
    assert symbol_path is not None
    assert manifest_path is not None
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(b"fixture-faiss")
    symbol_summary = write_symbol_sidecar_atomic(chunks, symbol_path)
    profile = replace(
        original_profile,
        symbol_index_sha256=symbol_summary.sha256,
    )
    spec = retrieval_embedding_spec("bge-m3")
    manifest_path.write_text(
        json.dumps(
            {
                "model_name": spec.model_name,
                "model_revision": spec.revision,
                "embedding_dimension": spec.embedding_dimension,
                "chunk_count": 2,
                "normalized_embeddings": spec.normalize_embeddings,
                "query_prefix": spec.query_prefix,
                "document_prefix": spec.document_prefix,
                "trust_remote_code": spec.trust_remote_code,
                "input_jsonl_sha256": sha256_file(baseline),
                "index_sha256": sha256_file(index_path),
                "metadata_sha256": sha256_file(paths.embedding_metadata_path),  # type: ignore[arg-type]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "python_doc_rag.profile_artifacts._safe_index_summary",
        lambda path, errors: (2, spec.embedding_dimension),
    )

    validation = validate_profile_artifacts(
        profile,
        tmp_path,
        expected_chunk_count=2,
        baseline_metadata_path=baseline,
    )

    assert validation.succeeded
    assert validation.chunk_count == 2
    assert validation.index_count == 2
    assert validation.symbol_sha256 == symbol_summary.sha256


def test_prepare_calls_writers_with_pinned_spec_and_validates_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _create_baseline_metadata(tmp_path)
    chunks = [object()] * 8_677
    calls: dict[str, Any] = {"validation_roots": []}

    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.load_chunks_jsonl",
        lambda path: chunks,
    )

    def fake_symbol_writer(values: list[object], path: Path) -> None:
        calls["symbol"] = (values, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("symbol\n", encoding="utf-8")

    def fake_index_builder(
        input_path: Path,
        index_path: Path,
        metadata_path: Path,
        manifest_path: Path,
        **kwargs: Any,
    ) -> SimpleNamespace:
        calls["builder"] = (input_path, kwargs)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_bytes(b"index")
        metadata_path.write_text("metadata\n", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(embedding_dimension=1024)

    def fake_validate(
        selected: RuntimeProfile,
        root: Path,
        **kwargs: Any,
    ) -> ProfileArtifactValidation:
        calls["validation_roots"].append((selected, root, kwargs))
        return _validation()

    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.write_symbol_sidecar_atomic",
        fake_symbol_writer,
    )
    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.build_vector_index",
        fake_index_builder,
    )
    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.validate_profile_artifacts",
        fake_validate,
    )

    result = prepare_recommended_v2_artifacts(
        tmp_path,
        device="cuda",
        batch_size=17,
    )

    spec = retrieval_embedding_spec("bge-m3")
    _, builder_options = calls["builder"]
    assert builder_options == {
        "model_name": spec.model_name,
        "model_revision": spec.revision,
        "batch_size": 17,
        "device": "cuda",
        "document_prefix": spec.document_prefix,
        "query_prefix": spec.query_prefix,
        "trust_remote_code": False,
    }
    assert calls["symbol"][0] is chunks
    assert len(calls["validation_roots"]) == 2
    assert not result.reused_existing
    assert all(path.is_file() for path in result.paths.required_paths)
    manifest_path = result.paths.embedding_manifest_path
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_jsonl"] == BASELINE_METADATA_RELATIVE_PATH.as_posix()
    assert manifest["input_jsonl_sha256"] == sha256_file(baseline)
    assert manifest["openai_api_used"] is False
    assert manifest["contains_secrets"] is False
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")


def test_prepare_reuses_complete_valid_artifacts_without_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = runtime_profile("recommended-v2")
    _create_baseline_metadata(tmp_path)
    _create_required_files(tmp_path, profile)
    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.validate_profile_artifacts",
        lambda *args, **kwargs: _validation(),
    )
    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.write_symbol_sidecar_atomic",
        lambda *args, **kwargs: pytest.fail("symbol artifact must be reused"),
    )
    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.build_vector_index",
        lambda *args, **kwargs: pytest.fail("BGE index must be reused"),
    )

    result = prepare_recommended_v2_artifacts(tmp_path, validate_only=True)

    assert result.reused_existing


def test_prepare_refuses_invalid_existing_artifacts_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = runtime_profile("recommended-v2")
    _create_baseline_metadata(tmp_path)
    _create_required_files(tmp_path, profile)
    before = {
        path: path.read_bytes()
        for path in profile_artifact_paths(profile, tmp_path).required_paths
    }
    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.validate_profile_artifacts",
        lambda *args, **kwargs: _validation(errors=("hash mismatch",)),
    )

    with pytest.raises(ArtifactPreparationError, match="既存artifact.*検証"):
        prepare_recommended_v2_artifacts(tmp_path)

    assert {path: path.read_bytes() for path in before} == before


def test_prepare_refuses_partial_artifacts_without_implicit_replacement(
    tmp_path: Path,
) -> None:
    profile = runtime_profile("recommended-v2")
    _create_baseline_metadata(tmp_path)
    paths = profile_artifact_paths(profile, tmp_path)
    first = paths.required_paths[0]
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"keep")

    with pytest.raises(ArtifactPreparationError, match="暗黙には上書きしません"):
        prepare_recommended_v2_artifacts(tmp_path)

    assert first.read_bytes() == b"keep"


def test_prepare_cleans_staging_after_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = runtime_profile("recommended-v2")
    _create_baseline_metadata(tmp_path)
    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.load_chunks_jsonl",
        lambda path: [object()] * 8_677,
    )
    monkeypatch.setattr(
        "python_doc_rag.recommended_v2_artifacts.write_symbol_sidecar_atomic",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer failed")),
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        prepare_recommended_v2_artifacts(tmp_path)

    phase = profile_artifact_paths(profile, tmp_path).symbol_index_path
    assert phase is not None
    assert not phase.parent.exists()
    staging_parent = tmp_path / "experiments/final_quality_sprint_v2"
    assert not list(staging_parent.glob(".phase_a.prepare-*"))


def test_prepare_refuses_any_baseline_output_path(tmp_path: Path) -> None:
    _create_baseline_metadata(tmp_path)
    profile = replace(
        runtime_profile("recommended-v2"),
        required_artifacts=(
            "indexes/python_3_13_ja.faiss",
            "experiments/final_quality_sprint_v2/phase_a/symbol_fields.jsonl",
        ),
    )

    with pytest.raises(ArtifactPreparationError, match="baseline artifact"):
        prepare_recommended_v2_artifacts(tmp_path, profile=profile)


def test_script_uses_data_root_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation_script = _load_preparation_script()
    selected: list[Path] = []
    monkeypatch.setenv("PYTHON_DOC_RAG_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(
        preparation_script,
        "prepare_recommended_v2_artifacts",
        lambda root, **kwargs: _script_result(root, selected),
    )

    exit_code = preparation_script.main(["--validate-only"])

    assert exit_code == 0
    assert selected == [tmp_path]
    assert json.loads(capsys.readouterr().out)["status"] == "reused"


def test_script_explicit_data_root_takes_priority_over_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_script = _load_preparation_script()
    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"
    environment_root.mkdir()
    explicit_root.mkdir()
    selected: list[Path] = []
    monkeypatch.setenv("PYTHON_DOC_RAG_DATA_ROOT", str(environment_root))
    monkeypatch.setattr(
        preparation_script,
        "prepare_recommended_v2_artifacts",
        lambda root, **kwargs: _script_result(root, selected),
    )

    exit_code = preparation_script.main(
        ["--data-root", str(explicit_root), "--validate-only"]
    )

    assert exit_code == 0
    assert selected == [explicit_root]


def test_script_import_does_not_load_openai_runtime() -> None:
    previous = sys.modules.pop("openai", None)
    try:
        _load_preparation_script()
        assert "openai" not in sys.modules
    finally:
        if previous is not None:
            sys.modules["openai"] = previous


def _script_result(root: Path, selected: list[Path]) -> SimpleNamespace:
    selected.append(root)
    return SimpleNamespace(
        reused_existing=True,
        paths=profile_artifact_paths(runtime_profile("recommended-v2"), root),
        validation=_validation(),
    )
