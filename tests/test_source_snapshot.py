import json
import re
import stat
import zipfile
from pathlib import Path

import pytest

import python_doc_rag.source_snapshot as source_snapshot_module
from python_doc_rag.source_snapshot import (
    FIXED_ZIP_TIMESTAMP,
    create_deterministic_source_snapshot,
    sha256_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    REPOSITORY_ROOT / "resources" / "source_snapshots" / "python-3.13-ja-2026-07-20.zip"
)
PROVENANCE_PATH = SNAPSHOT_PATH.with_suffix(".provenance.json")
ORIGINAL_RECORDED_SHA256 = (
    "f8ddb3454726cbe34580b4c21723128a1b33b50f1155e9b9184cb790db66d9cb"
)
PROJECT_SNAPSHOT_SHA256 = (
    "1fbc311273f7a4302b2929e483b4dded787d7ea89bdcebf74312732376395777"
)
PROJECT_SNAPSHOT_SIZE = 17_310_566
PROJECT_SNAPSHOT_MEMBER_COUNT = 1_258


def test_snapshot_writer_is_deterministic_and_normalizes_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "copyright.html").write_text("copyright", encoding="utf-8")
    (source / "license.html").write_text("license", encoding="utf-8")
    (source / "nested" / "日本語.html").write_text("本文", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = create_deterministic_source_snapshot(source, first)
    second_result = create_deterministic_source_snapshot(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result == second_result
    assert first_result.file_count == 3
    assert first_result.directory_count == 2
    with zipfile.ZipFile(first) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        assert names == sorted(names, key=lambda name: name.encode("utf-8"))
        assert names == [
            "python-3.13-docs-html/",
            "python-3.13-docs-html/copyright.html",
            "python-3.13-docs-html/license.html",
            "python-3.13-docs-html/nested/",
            "python-3.13-docs-html/nested/日本語.html",
        ]
        assert all(info.date_time == FIXED_ZIP_TIMESTAMP for info in infos)
        assert all(info.create_system == 3 for info in infos)
        for info in infos:
            mode = info.external_attr >> 16
            if info.is_dir():
                assert stat.S_ISDIR(mode)
                assert stat.S_IMODE(mode) == 0o755
            else:
                assert stat.S_ISREG(mode)
                assert stat.S_IMODE(mode) == 0o644


def test_snapshot_writer_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "outside.html"
    target.write_text("outside", encoding="utf-8")
    (source / "escaped.html").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        create_deterministic_source_snapshot(source, tmp_path / "snapshot.zip")


def test_snapshot_writer_rejects_native_junctions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    junction = source / "junction"
    junction.mkdir(parents=True)
    (junction / "page.html").write_text("outside", encoding="utf-8")
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda path: path.name == "junction",
        raising=False,
    )

    with pytest.raises(ValueError, match="symlink"):
        create_deterministic_source_snapshot(source, tmp_path / "snapshot.zip")


def test_snapshot_writer_rejects_file_swapped_to_symlink_after_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    victim = source / "page.html"
    victim.write_text("public", encoding="utf-8")
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("LOCAL-SECRET", encoding="utf-8")
    original_collect = source_snapshot_module._collect_source_entries

    def collect_then_swap(source_root: Path, archive_root: str):  # type: ignore[no-untyped-def]
        entries = original_collect(source_root, archive_root)
        victim.unlink()
        victim.symlink_to(secret)
        return entries

    monkeypatch.setattr(
        source_snapshot_module,
        "_collect_source_entries",
        collect_then_swap,
    )

    with pytest.raises(ValueError, match="changed during snapshot"):
        create_deterministic_source_snapshot(source, tmp_path / "snapshot.zip")


def test_snapshot_writer_rejects_source_tree_changed_during_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "page.html").write_text("content", encoding="utf-8")
    original_collect = source_snapshot_module._collect_source_entries
    calls = 0

    def collect_then_add(source_root: Path, archive_root: str):  # type: ignore[no-untyped-def]
        nonlocal calls
        entries = original_collect(source_root, archive_root)
        calls += 1
        if calls == 1:
            (source / "late.html").write_text("late", encoding="utf-8")
        return entries

    monkeypatch.setattr(
        source_snapshot_module,
        "_collect_source_entries",
        collect_then_add,
    )

    with pytest.raises(ValueError, match="tree changed"):
        create_deterministic_source_snapshot(source, tmp_path / "snapshot.zip")


def test_snapshot_writer_rejects_published_path_swapped_after_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "page.html").write_text("content", encoding="utf-8")
    output = tmp_path / "snapshot.zip"
    replacement = tmp_path / "replacement.zip"
    replacement.write_bytes(b"replacement")
    real_replace = source_snapshot_module.os.replace

    def replace_then_swap(source_path: Path | str, destination: Path | str) -> None:
        real_replace(source_path, destination)
        if Path(destination) == output:
            real_replace(replacement, output)

    monkeypatch.setattr(source_snapshot_module.os, "replace", replace_then_swap)

    with pytest.raises(RuntimeError, match="published snapshot path changed"):
        create_deterministic_source_snapshot(source, output)


def test_snapshot_writer_rejects_output_inside_source_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "page.html").write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="outside source_root"):
        create_deterministic_source_snapshot(source, source / "snapshot.zip")


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "CON.html",
        "page.html:stream",
        "page.html.",
        "a?.html",
        "a*.html",
        "a|b.html",
        "a<b>.html",
        'a"b.html',
        "control\x01.html",
    ),
)
def test_snapshot_writer_rejects_nonportable_member_names(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / unsafe_name).write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="Windows"):
        create_deterministic_source_snapshot(source, tmp_path / "snapshot.zip")


@pytest.mark.parametrize("component", ("COM¹.txt", "LPT³.log", "a" * 256))
def test_snapshot_writer_rejects_additional_windows_component_limits(
    component: str,
) -> None:
    with pytest.raises(ValueError, match="Windows"):
        source_snapshot_module._validate_portable_component(
            component,
            label="source path",
        )


def test_snapshot_writer_rejects_implicit_parent_case_collision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "Foo").mkdir(parents=True)
    (source / "foo").mkdir()
    (source / "Foo/a.html").write_text("a", encoding="utf-8")
    (source / "foo/b.html").write_text("b", encoding="utf-8")

    with pytest.raises(ValueError, match="case-insensitive"):
        create_deterministic_source_snapshot(source, tmp_path / "snapshot.zip")


def test_repository_snapshot_matches_recorded_provenance() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert SNAPSHOT_PATH.is_file()
    assert SNAPSHOT_PATH.stat().st_size == PROJECT_SNAPSHOT_SIZE
    assert sha256_file(SNAPSHOT_PATH) == PROJECT_SNAPSHOT_SHA256
    assert provenance["upstream"]["original_recorded_archive_sha256"] == (
        ORIGINAL_RECORDED_SHA256
    )
    assert provenance["project_snapshot"]["sha256"] == PROJECT_SNAPSHOT_SHA256
    assert provenance["project_snapshot"]["byte_size"] == PROJECT_SNAPSHOT_SIZE
    assert provenance["project_snapshot"]["member_count"] == (
        PROJECT_SNAPSHOT_MEMBER_COUNT
    )
    assert (
        provenance["upstream"]["original_recorded_archive_sha256"]
        != (provenance["project_snapshot"]["sha256"])
    )

    serialized = json.dumps(provenance, ensure_ascii=False)
    assert "/workspace/" not in serialized
    assert "file://" not in serialized
    assert not any(
        isinstance(value, str)
        and (value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value))
        for value in _walk_values(provenance)
    )


def test_repository_snapshot_retains_upstream_license_files() -> None:
    with zipfile.ZipFile(SNAPSHOT_PATH) as archive:
        assert len(archive.infolist()) == PROJECT_SNAPSHOT_MEMBER_COUNT
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert "python-3.13-docs-html/license.html" in names
        assert "python-3.13-docs-html/copyright.html" in names
        assert "python-3.13-docs-html/_sources/license.rst.txt" in names
        assert "python-3.13-docs-html/_sources/copyright.rst.txt" in names
        assert archive.read("python-3.13-docs-html/license.html")
        assert archive.read("python-3.13-docs-html/copyright.html")


def _walk_values(value: object) -> list[object]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_values(child)]
    return [value]
