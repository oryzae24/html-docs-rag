from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import warnings
import zipfile
from pathlib import Path

import pytest

import python_doc_rag.loaders.zip_archive as zip_archive_module
from python_doc_rag.loaders.zip_archive import SafeZipHtmlLoader

Member = tuple[str, bytes, int | None]


def _write_zip(path: Path, members: list[Member]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, content, explicit_mode in members:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                mode = explicit_mode
                if mode is None:
                    mode = (stat.S_IFDIR | 0o755) if name.endswith("/") else (
                        stat.S_IFREG | 0o644
                    )
                info.external_attr = mode << 16
                archive.writestr(info, content)
    return path


def _valid_members(*, text: bytes = b"<html>fixture</html>") -> list[Member]:
    return [
        ("docs/", b"", None),
        ("docs/tutorial/", b"", None),
        ("docs/tutorial/page.html", text, None),
    ]


def _safe_loader(
    archive: Path,
    extraction_root: Path,
    *,
    archive_root: str = "docs",
    max_archive_bytes: int = 1_000_000,
    max_members: int = 20,
    max_member_bytes: int = 10_000,
    max_extracted_bytes: int = 20_000,
) -> SafeZipHtmlLoader:
    return SafeZipHtmlLoader(
        archive,
        extraction_root,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        archive_root=archive_root,
        source_base_url="https://example.test/docs/",
        include_path_prefixes=("tutorial/",),
        source_kind="fixture-archive",
        max_archive_bytes=max_archive_bytes,
        max_members=max_members,
        max_member_bytes=max_member_bytes,
        max_extracted_bytes=max_extracted_bytes,
    )


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../escape.html",
        "docs/../../escape.html",
        "/absolute.html",
        "//server/share.html",
        "C:/windows.html",
        "docs\\..\\escape.html",
        "docs/D:relative/escape.html",
        "docs/tutorial/page.html:stream",
        "docs/tutorial/CON.html",
        "docs/tutorial/page.html.",
        "docs/tutorial/a?.html",
        "docs/tutorial/a*.html",
        "docs/tutorial/a|b.html",
        "docs/tutorial/a<b>.html",
        'docs/tutorial/a"b.html',
        "docs/tutorial/control\x01.html",
        "docs/tutorial/COM¹.html",
        "docs/tutorial/LPT³.html",
        f"docs/tutorial/{'a' * 256}.html",
    ),
)
def test_safe_zip_rejects_escaping_member_paths(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive = _write_zip(
        tmp_path / "unsafe.zip",
        _valid_members() + [(unsafe_name, b"unsafe", None)],
    )

    with pytest.raises(ValueError, match="unsafe|travers|absolute"):
        _safe_loader(archive, tmp_path / "extracted").load()

    assert not (tmp_path / "escape.html").exists()


def test_safe_zip_rejects_nul_member_path(tmp_path: Path) -> None:
    archive = _write_zip(
        tmp_path / "nul.zip",
        [("docs/nul.html", b"unsafe", None)],
    )
    raw = archive.read_bytes()
    assert raw.count(b"docs/nul.html") == 2
    archive.write_bytes(raw.replace(b"docs/nul.html", b"docs/\x00ul.html"))

    with pytest.raises(ValueError, match="NUL"):
        _safe_loader(archive, tmp_path / "extracted").load()


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        (stat.S_IFLNK | 0o777, "symlink"),
        (stat.S_IFCHR | 0o600, "special/device"),
        (stat.S_IFIFO | 0o600, "special/device"),
    ),
)
def test_safe_zip_rejects_links_and_special_entries(
    tmp_path: Path,
    mode: int,
    message: str,
) -> None:
    archive = _write_zip(
        tmp_path / "special.zip",
        _valid_members() + [("docs/tutorial/special", b"target", mode)],
    )

    with pytest.raises(ValueError, match=message):
        _safe_loader(archive, tmp_path / "extracted").load()


@pytest.mark.parametrize(
    "members",
    (
        [
            ("docs/", b"", None),
            ("docs/tutorial/page.html", b"first", None),
            ("docs/tutorial/page.html", b"second", None),
        ],
        [
            ("docs/", b"", None),
            ("docs/Tutorial/page.html", b"first", None),
            ("docs/tutorial/page.html", b"second", None),
        ],
        [
            ("docs/", b"", None),
            ("docs/tutorial/é.html", b"first", None),
            ("docs/tutorial/é.html", b"second", None),
        ],
        [
            ("docs/", b"", None),
            ("docs/tutorial", b"file", None),
            ("docs/tutorial/page.html", b"child", None),
        ],
        [
            ("docs/", b"", None),
            ("docs/Tutorial", b"file", None),
            ("docs/tutorial/page.html", b"child", None),
        ],
        [
            ("docs/Foo/a.html", b"first", None),
            ("docs/foo/b.html", b"second", None),
        ],
        [
            ("docs/é/a.html", b"first", None),
            ("docs/é/b.html", b"second", None),
        ],
    ),
)
def test_safe_zip_rejects_duplicate_casefold_and_file_directory_collisions(
    tmp_path: Path,
    members: list[Member],
) -> None:
    archive = _write_zip(tmp_path / "collision.zip", members)

    with pytest.raises(ValueError, match="duplicate|collision"):
        _safe_loader(archive, tmp_path / "extracted").load()


def test_safe_zip_rejects_member_count_limit(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "many.zip", _valid_members())

    with pytest.raises(ValueError, match="member count"):
        _safe_loader(archive, tmp_path / "extracted", max_members=2).load()


def test_safe_zip_rejects_individual_member_limit(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "large-member.zip", _valid_members(text=b"x" * 21))

    with pytest.raises(ValueError, match="max_member_bytes"):
        _safe_loader(
            archive,
            tmp_path / "extracted",
            max_member_bytes=20,
        ).load()


def test_safe_zip_rejects_total_extracted_limit(tmp_path: Path) -> None:
    archive = _write_zip(
        tmp_path / "large-total.zip",
        _valid_members(text=b"x" * 11)
        + [("docs/tutorial/second.html", b"y" * 11, None)],
    )

    with pytest.raises(ValueError, match="max_extracted_bytes"):
        _safe_loader(
            archive,
            tmp_path / "extracted",
            max_member_bytes=20,
            max_extracted_bytes=21,
        ).load()


def test_safe_zip_rejects_archive_size_limit(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "archive.zip", _valid_members())

    with pytest.raises(ValueError, match="max_archive_bytes"):
        _safe_loader(
            archive,
            tmp_path / "extracted",
            max_archive_bytes=archive.stat().st_size - 1,
        ).load()


def test_safe_zip_requires_exact_configured_archive_root(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "wrong-root.zip", _valid_members())

    with pytest.raises(ValueError, match="archive_root is absent"):
        _safe_loader(
            archive,
            tmp_path / "extracted",
            archive_root="not-docs",
        ).load()


def test_safe_zip_does_not_enumerate_html_outside_archive_root(tmp_path: Path) -> None:
    archive = _write_zip(
        tmp_path / "outside.zip",
        _valid_members()
        + [("other/tutorial/outside.html", b"<html>outside</html>", None)],
    )

    documents = _safe_loader(archive, tmp_path / "extracted").load()

    assert [document.logical_path for document in documents] == [
        "tutorial/page.html"
    ]
    assert all("outside" not in document.source_url for document in documents)


def test_safe_zip_failure_preserves_existing_valid_extraction(tmp_path: Path) -> None:
    extraction = tmp_path / "extracted"
    valid_archive = _write_zip(tmp_path / "valid.zip", _valid_members(text=b"valid"))
    _safe_loader(valid_archive, extraction).load()
    manifest_before = (extraction / "extraction_manifest.json").read_bytes()
    page_before = (extraction / "docs/tutorial/page.html").read_bytes()
    bad_archive = _write_zip(
        tmp_path / "bad.zip",
        [("../escape.html", b"bad", None)],
    )

    with pytest.raises(ValueError):
        _safe_loader(bad_archive, extraction).load()

    assert (extraction / "extraction_manifest.json").read_bytes() == manifest_before
    assert (extraction / "docs/tutorial/page.html").read_bytes() == page_before


def test_safe_zip_rebuilds_cache_with_unrecorded_html(tmp_path: Path) -> None:
    extraction = tmp_path / "extracted"
    archive = _write_zip(tmp_path / "valid.zip", _valid_members(text=b"valid"))
    _safe_loader(archive, extraction).load()
    injected = extraction / "docs/tutorial/injected.html"
    injected.write_text("<html>not from the archive</html>", encoding="utf-8")
    replay = _safe_loader(archive, extraction)

    documents = replay.load()

    assert [document.logical_path for document in documents] == [
        "tutorial/page.html"
    ]
    assert replay.cache_reused is False
    assert not injected.exists()


def test_safe_zip_recovers_extraction_left_between_directory_renames(
    tmp_path: Path,
) -> None:
    extraction = tmp_path / "extracted"
    archive = _write_zip(tmp_path / "valid.zip", _valid_members())
    _safe_loader(archive, extraction).load()
    backup = extraction.with_name(f".{extraction.name}.backup")
    os.rename(extraction, backup)

    replay = _safe_loader(archive, extraction)
    documents = replay.load()

    assert replay.cache_reused
    assert documents[0].content == "<html>fixture</html>"
    assert extraction.is_dir()
    assert not backup.exists()


def test_safe_zip_cleanup_interrupt_keeps_published_extraction_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = tmp_path / "extracted"
    archive = _write_zip(tmp_path / "valid.zip", _valid_members())
    _safe_loader(archive, extraction).load()
    (extraction / "injected.html").write_text("invalid", encoding="utf-8")
    real_rmtree = zip_archive_module.shutil.rmtree

    def interrupt_retired_cleanup(
        path: Path | str,
        *args: object,
        **kwargs: object,
    ) -> None:
        if ".backup.retired-" in Path(path).name:
            raise KeyboardInterrupt("injected extraction cleanup interrupt")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        zip_archive_module.shutil,
        "rmtree",
        interrupt_retired_cleanup,
    )

    rebuilt = _safe_loader(archive, extraction)
    documents = rebuilt.load()

    assert not rebuilt.cache_reused
    assert documents[0].content == "<html>fixture</html>"
    assert not (extraction / "injected.html").exists()


def test_safe_zip_rebuilds_when_cached_member_and_manifest_are_both_tampered(
    tmp_path: Path,
) -> None:
    extraction = tmp_path / "extracted"
    original = b"<html>original</html>"
    tampered = b"<html>tampered</html>"
    assert len(original) == len(tampered)
    archive = _write_zip(tmp_path / "valid.zip", _valid_members(text=original))
    _safe_loader(archive, extraction).load()
    page = extraction / "docs/tutorial/page.html"
    page.write_bytes(tampered)
    manifest_path = extraction / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["members"]:
        if record["path"] == "docs/tutorial/page.html":
            record["sha256"] = hashlib.sha256(tampered).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    replay = _safe_loader(archive, extraction)

    documents = replay.load()

    assert replay.cache_reused is False
    assert documents[0].content == original.decode("utf-8")
    assert page.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("archive_byte_size", -1),
        ("member_count", 999_999),
        ("extracted_byte_size", -1),
        ("archive_format", "evil"),
        ("unexpected_absolute_path", "/secret/path"),
    ),
)
def test_safe_zip_rebuilds_cache_with_non_strict_extraction_manifest(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    extraction = tmp_path / "extracted"
    archive = _write_zip(tmp_path / "valid.zip", _valid_members())
    _safe_loader(archive, extraction).load()
    manifest_path = extraction / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    replay = _safe_loader(archive, extraction)

    documents = replay.load()

    assert replay.cache_reused is False
    assert documents[0].content == "<html>fixture</html>"
    repaired = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(repaired) == zip_archive_module._EXTRACTION_MANIFEST_KEYS


def test_safe_zip_holds_verified_archive_inode_across_zip_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"<html>EXPECTED</html>"
    archive = _write_zip(tmp_path / "source.zip", _valid_members(text=expected))
    expected_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    attacker = _write_zip(
        tmp_path / "replacement.zip",
        _valid_members(text=b"<html>ATTACKER</html>"),
    )
    loader = _safe_loader(archive, tmp_path / "extracted")
    real_preflight = zip_archive_module._preflight_zip_member_count
    swapped = False

    def swap_path_after_open(stream: object, max_members: int) -> None:
        nonlocal swapped
        if not swapped:
            os.replace(attacker, archive)
            swapped = True
        real_preflight(stream, max_members)  # type: ignore[arg-type]

    monkeypatch.setattr(
        zip_archive_module,
        "_preflight_zip_member_count",
        swap_path_after_open,
    )

    try:
        documents = loader.load()
    except ValueError as error:
        assert "archive changed" in str(error)
    else:
        assert documents[0].content == expected.decode("utf-8")

    assert expected_sha256 != hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path = tmp_path / "extracted/extraction_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["archive_sha256"] == expected_sha256


def test_safe_zip_rejects_selected_html_replaced_before_no_follow_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_zip(tmp_path / "valid.zip", _valid_members())
    extraction = tmp_path / "extracted"
    secret = tmp_path / "outside-secret.html"
    secret.write_text("LOCAL-SECRET", encoding="utf-8")
    real_selection = zip_archive_module._selected_html_files

    def select_then_swap(document_root: Path, prefixes: object):  # type: ignore[no-untyped-def]
        selected = real_selection(document_root, prefixes)
        selected[0].unlink()
        selected[0].symlink_to(secret)
        return selected

    monkeypatch.setattr(zip_archive_module, "_selected_html_files", select_then_swap)

    with pytest.raises(ValueError, match="regular file"):
        _safe_loader(archive, extraction).load()


def test_safe_zip_preflights_member_count_before_zipfile_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = [
        (f"docs/tutorial/{index}.html", b"", None)
        for index in range(100)
    ]
    archive = _write_zip(tmp_path / "many.zip", members)

    monkeypatch.setattr(
        zip_archive_module.zipfile,
        "ZipFile",
        lambda *args, **kwargs: pytest.fail("ZipFile parsed entries before preflight"),
    )

    with pytest.raises(ValueError, match="member count"):
        _safe_loader(
            archive,
            tmp_path / "extracted",
            max_members=20,
        ).load()


def test_safe_zip_preflight_counts_central_records_instead_of_trusting_eocd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_zip(
        tmp_path / "forged-count.zip",
        [
            (f"docs/tutorial/{index}.html", b"", None)
            for index in range(100)
        ],
    )
    payload = bytearray(archive.read_bytes())
    eocd = payload.rfind(b"PK\x05\x06")
    assert eocd >= 0
    struct.pack_into("<H", payload, eocd + 8, 1)
    struct.pack_into("<H", payload, eocd + 10, 1)
    archive.write_bytes(payload)

    monkeypatch.setattr(
        zip_archive_module.zipfile,
        "ZipFile",
        lambda *args, **kwargs: pytest.fail("ZipFile parsed forged entry count"),
    )

    with pytest.raises(ValueError, match="member count"):
        _safe_loader(
            archive,
            tmp_path / "extracted",
            max_members=20,
        ).load()


@pytest.mark.parametrize(
    "reserved_name",
    ("extraction_manifest.json", "Extraction_Manifest.json"),
)
def test_safe_zip_rejects_member_reserved_for_extraction_manifest(
    tmp_path: Path,
    reserved_name: str,
) -> None:
    archive = _write_zip(
        tmp_path / "reserved.zip",
        _valid_members() + [(reserved_name, b"{}", None)],
    )

    with pytest.raises(ValueError, match="reserved"):
        _safe_loader(archive, tmp_path / "extracted").load()


def test_safe_zip_rejects_existing_symlink_destination(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "valid.zip", _valid_members())
    outside = tmp_path / "outside"
    outside.mkdir()
    extraction = tmp_path / "extracted"
    extraction.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _safe_loader(archive, extraction).load()

    assert extraction.is_symlink()
    assert list(outside.iterdir()) == []


def test_safe_zip_rejects_existing_symlink_destination_ancestor(
    tmp_path: Path,
) -> None:
    archive = _write_zip(tmp_path / "valid.zip", _valid_members())
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    extraction = linked_parent / "extracted"

    with pytest.raises(ValueError, match="ancestor.*symlink"):
        _safe_loader(archive, extraction).load()

    assert linked_parent.is_symlink()
    assert not (outside / "extracted").exists()
