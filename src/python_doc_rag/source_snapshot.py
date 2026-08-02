"""Create byte-stable ZIP snapshots from validated expanded source trees."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

DEFAULT_ARCHIVE_ROOT = "python-3.13-docs-html"
FIXED_ZIP_TIMESTAMP = (2026, 7, 20, 0, 0, 0)
_TREE_HASH_REVISION = b"python-doc-rag-expanded-source-tree-v1\0"
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
        *(f"com{index}" for index in ("¹", "²", "³")),
        *(f"lpt{index}" for index in ("¹", "²", "³")),
    }
)
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"|?*')


@dataclass(frozen=True)
class SourceSnapshotResult:
    """Identity and inventory of a completed deterministic source snapshot."""

    archive_sha256: str
    archive_size: int
    source_tree_sha256: str
    member_count: int
    file_count: int
    directory_count: int


@dataclass(frozen=True)
class _SourceEntry:
    path: Path
    archive_name: str
    relative_name: str
    is_directory: bool
    device: int
    inode: int
    byte_size: int
    modified_ns: int
    changed_ns: int


class _HashWriter(Protocol):
    def update(self, value: bytes, /) -> None:
        """Add bytes to the digest state."""


def create_deterministic_source_snapshot(
    source_root: Path,
    output_path: Path,
    *,
    archive_root: str = DEFAULT_ARCHIVE_ROOT,
) -> SourceSnapshotResult:
    """Write a deterministic ZIP of ``source_root`` and publish it atomically."""
    root_name = _validate_archive_root(archive_root)
    entries = _collect_source_entries(source_root, root_name)
    _reject_output_inside_source(source_root, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    tree_hasher = hashlib.sha256()
    tree_hasher.update(_TREE_HASH_REVISION)

    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
            strict_timestamps=True,
        ) as archive:
            archive.comment = b""
            for entry in entries:
                info = _zip_info(entry)
                if entry.is_directory:
                    _update_tree_hash(tree_hasher, b"D", entry.relative_name, b"")
                    archive.writestr(info, b"", compress_type=zipfile.ZIP_STORED)
                    continue

                content = _read_stable_source_file(entry)
                _update_tree_hash(tree_hasher, b"F", entry.relative_name, content)
                archive.writestr(
                    info,
                    content,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )

        if _collect_source_entries(source_root, root_name) != entries:
            raise ValueError("source tree changed during snapshot generation")
        archive_sha256, archive_size, archive_identity = _stable_file_sha256(
            temporary_path
        )
        os.replace(temporary_path, output_path)
        _require_published_file_identity(output_path, archive_identity)
    finally:
        temporary_path.unlink(missing_ok=True)

    directory_count = sum(entry.is_directory for entry in entries)
    return SourceSnapshotResult(
        archive_sha256=archive_sha256,
        archive_size=archive_size,
        source_tree_sha256=tree_hasher.hexdigest(),
        member_count=len(entries),
        file_count=len(entries) - directory_count,
        directory_count=directory_count,
    )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_file_sha256(path: Path) -> tuple[str, int, tuple[int, int, int]]:
    """Hash one no-follow file and return the inode identity that was hashed."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("snapshot output staging must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise ValueError("snapshot output staging changed before hashing")
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
        after = os.fstat(source.fileno())
    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != size
        or after.st_mtime_ns != opened.st_mtime_ns
        or after.st_ctime_ns != opened.st_ctime_ns
    ):
        raise ValueError("snapshot output staging changed while hashing")
    return digest.hexdigest(), size, (opened.st_dev, opened.st_ino, size)


def _require_published_file_identity(
    path: Path,
    expected: tuple[int, int, int],
) -> None:
    """Require the published path to name the exact inode hashed before rename."""
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or (observed.st_dev, observed.st_ino, observed.st_size) != expected
    ):
        raise RuntimeError("published snapshot path changed during publication")


def _is_link_like(path: Path) -> bool:
    """Return whether a path is a symlink or native Windows junction."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _validate_archive_root(archive_root: str) -> str:
    if not archive_root or "\\" in archive_root or "\x00" in archive_root:
        raise ValueError("archive_root must be one safe POSIX path component")
    path = PurePosixPath(archive_root)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError("archive_root must be one safe POSIX path component")
    _validate_portable_component(path.parts[0], label="archive_root")
    return path.as_posix()


def _collect_source_entries(source_root: Path, archive_root: str) -> list[_SourceEntry]:
    try:
        root_stat = source_root.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"source root does not exist: {source_root}") from exc
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or _is_link_like(source_root)
        or not stat.S_ISDIR(root_stat.st_mode)
    ):
        raise ValueError("source root must be a real directory, not a symlink")

    entries = [
        _SourceEntry(
            path=source_root,
            archive_name=f"{archive_root}/",
            relative_name="",
            is_directory=True,
            device=root_stat.st_dev,
            inode=root_stat.st_ino,
            byte_size=root_stat.st_size,
            modified_ns=root_stat.st_mtime_ns,
            changed_ns=root_stat.st_ctime_ns,
        )
    ]
    pending = [source_root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            children = list(iterator)
        for child in children:
            child_path = Path(child.path)
            child_stat = child.stat(follow_symlinks=False)
            mode = child_stat.st_mode
            relative = child_path.relative_to(source_root).as_posix()
            _validate_relative_name(relative)
            if stat.S_ISLNK(mode) or _is_link_like(child_path):
                raise ValueError(f"source tree contains a symlink: {relative}")
            if stat.S_ISDIR(mode):
                entries.append(
                    _SourceEntry(
                        path=child_path,
                        archive_name=f"{archive_root}/{relative}/",
                        relative_name=f"{relative}/",
                        is_directory=True,
                        device=child_stat.st_dev,
                        inode=child_stat.st_ino,
                        byte_size=child_stat.st_size,
                        modified_ns=child_stat.st_mtime_ns,
                        changed_ns=child_stat.st_ctime_ns,
                    )
                )
                pending.append(child_path)
            elif stat.S_ISREG(mode):
                entries.append(
                    _SourceEntry(
                        path=child_path,
                        archive_name=f"{archive_root}/{relative}",
                        relative_name=relative,
                        is_directory=False,
                        device=child_stat.st_dev,
                        inode=child_stat.st_ino,
                        byte_size=child_stat.st_size,
                        modified_ns=child_stat.st_mtime_ns,
                        changed_ns=child_stat.st_ctime_ns,
                    )
                )
            else:
                raise ValueError(f"source tree contains a special file: {relative}")

    identities: dict[str, str] = {}
    for entry in entries:
        archive_name = entry.archive_name.rstrip("/")
        identity = _portable_path_identity(archive_name)
        previous = identities.get(identity)
        if previous is not None and previous != archive_name:
            raise ValueError("source tree contains a case-insensitive path collision")
        identities[identity] = archive_name
    return sorted(entries, key=lambda entry: entry.archive_name.encode("utf-8"))


def _validate_relative_name(relative: str) -> None:
    if not relative or "\\" in relative or "\x00" in relative:
        raise ValueError(f"unsafe source path: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe source path: {relative!r}")
    for component in path.parts:
        _validate_portable_component(component, label="source path")


def _validate_portable_component(component: str, *, label: str) -> None:
    if (
        any(character in _WINDOWS_FORBIDDEN_CHARS for character in component)
        or any(ord(character) < 32 for character in component)
        or _WINDOWS_DRIVE_PATTERN.match(component)
        or component.endswith((" ", "."))
        or len(component.encode("utf-16-le")) // 2 > 255
    ):
        raise ValueError(f"unsafe Windows {label}: {component!r}")
    basename = component.split(".", maxsplit=1)[0].casefold()
    if basename in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"unsafe Windows device {label}: {component!r}")


def _portable_path_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return unicodedata.normalize("NFC", normalized.casefold())


def _read_stable_source_file(entry: _SourceEntry) -> bytes:
    """Read the exact file inode scanned earlier without following a final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(entry.path, flags)
    except OSError as error:
        raise ValueError(f"source file changed during snapshot: {entry.relative_name}") from error
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not _same_source_identity(entry, before):
            raise ValueError(f"source file changed during snapshot: {entry.relative_name}")
        content = source.read()
        after = os.fstat(source.fileno())
    if not _same_source_identity(entry, after) or len(content) != entry.byte_size:
        raise ValueError(f"source file changed during snapshot: {entry.relative_name}")
    return content


def _same_source_identity(entry: _SourceEntry, observed: os.stat_result) -> bool:
    return (
        stat.S_ISREG(observed.st_mode)
        and observed.st_dev == entry.device
        and observed.st_ino == entry.inode
        and observed.st_size == entry.byte_size
        and observed.st_mtime_ns == entry.modified_ns
        and observed.st_ctime_ns == entry.changed_ns
    )


def _reject_output_inside_source(source_root: Path, output_path: Path) -> None:
    """Prevent the generated ZIP or its temporary file from entering its input tree."""
    resolved_source = source_root.resolve(strict=True)
    absolute_source = source_root.absolute()
    if output_path.resolve(strict=False).is_relative_to(resolved_source) or (
        output_path.absolute().is_relative_to(absolute_source)
    ):
        raise ValueError("output_path must be outside source_root")


def _zip_info(entry: _SourceEntry) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(entry.archive_name, date_time=FIXED_ZIP_TIMESTAMP)
    info.create_system = 3
    info.comment = b""
    info.extra = b""
    if entry.is_directory:
        info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
        info.compress_type = zipfile.ZIP_STORED
    else:
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _update_tree_hash(
    digest: _HashWriter,
    kind: bytes,
    relative_name: str,
    content: bytes,
) -> None:
    encoded_name = relative_name.encode("utf-8")
    digest.update(kind)
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)
