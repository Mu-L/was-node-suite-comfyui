"""Reading a zip archive: what it holds, and what may safely come out of it.

:class:`Archive` is what a ZIP socket carries. A read is bounded by
:data:`MAX_ENTRY_BYTES`, a session by :data:`MAX_TOTAL_BYTES`, the index by
:data:`MAX_ENTRIES`.
"""

from __future__ import annotations

import io
import os
import stat
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple

from .. import log
from . import kinds

__all__ = [
    "Archive",
    "ArchiveError",
    "BudgetSpent",
    "ENCRYPTED",
    "ESCAPES",
    "Entry",
    "EntryTooLarge",
    "MAX_ENTRIES",
    "MAX_ENTRY_BYTES",
    "MAX_TOTAL_BYTES",
    "NUL_NAME",
    "NotAnArchive",
    "OVERSIZE",
    "REFUSALS",
    "RESERVED",
    "SUFFIX",
    "SYMLINK",
    "is_archive",
    "require_archive",
    "safe_name",
]

logger = log.get_logger("archive.container")

#: The extension an archive this pack writes carries, and the one its menus list.
SUFFIX = ".zip"

#: How many entries are indexed. An archive holding more is listed as far as this and says
#: so, rather than being refused: a 20000-file dataset zip is a real file, and the work of
#: opening one has to be bounded whatever it holds.
MAX_ENTRIES = 4096

#: How many bytes one entry may unpack to. An entry declaring more is refused before it is
#: read, and a read that runs past it stops, since the size in a zip's index is only what the
#: file claims.
MAX_ENTRY_BYTES = 256 * 1024 * 1024

#: How many bytes one :meth:`Archive.read_many` session may unpack in total, so reading a
#: thousand entries is bounded as well as reading one.
MAX_TOTAL_BYTES = 512 * 1024 * 1024

#: Bytes moved per read while an entry is being copied out.
CHUNK = 1024 * 1024

#: Why an entry may not be read or written. Each is reported against the entry that carries
#: it and stops nothing else in the archive.
ESCAPES = "escapes"
NUL_NAME = "nul"
SYMLINK = "symlink"
RESERVED = "reserved"
ENCRYPTED = "encrypted"
OVERSIZE = "oversize"

#: What the user is told for each refusal, in the words the node logs and the report shows.
REFUSALS: dict[str, str] = {
    ESCAPES: (
        "names a place outside the folder it would be written to, so it is not read or "
        "written"
    ),
    NUL_NAME: (
        "holds a null byte in its name, so the name it would be written under is not the "
        "name the archive carries"
    ),
    SYMLINK: (
        "is a symbolic link rather than a file, so it names somewhere else on the machine "
        "instead of holding anything"
    ),
    RESERVED: (
        "is named after a Windows device, so opening it would reach that device rather than "
        "a file"
    ),
    ENCRYPTED: "is encrypted, and nothing here can decrypt it",
    OVERSIZE: "declares more bytes than one entry may unpack to",
}

#: Device names Windows resolves before it looks at a directory, with or without an
#: extension. Refused on every platform, so one archive behaves the same way everywhere.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "clock$"}
    | {f"com{digit}" for digit in "123456789"}
    | {f"lpt{digit}" for digit in "123456789"}
)

#: Entry names listed in a message before it stops naming them.
_NAMES_SHOWN = 10


class ArchiveError(ValueError):
    """An archive could not be read, or an entry could not be taken out of it."""


class NotAnArchive(ArchiveError):
    """The bytes or the file offered are not a readable zip archive."""


class EntryTooLarge(ArchiveError):
    """One entry unpacks past the bytes a read may take out of it."""


class BudgetSpent(ArchiveError):
    """A read reached the total it may unpack before an entry was finished."""


class Entry(NamedTuple):
    """One file an archive holds.

    Attributes:
        name: The relative name a caller may write it under, spelled with ``/``. Empty when
            ``refusal`` says the entry may not be written at all.
        stored: The name exactly as the archive carries it, which is not always ``name``.
        size: Bytes the archive says the entry unpacks to.
        compressed: Bytes it occupies inside the archive.
        kind: :data:`modules.archive.kinds.KINDS` member, or ``None`` where the pack has no
            loader for the file.
        refusal: One of :data:`REFUSALS`, or empty where the entry may be read.
        position: Where the entry sits in the archive's own order, which is how it is read
            back when two entries share a name.
    """

    name: str
    stored: str
    size: int
    compressed: int
    kind: str | None
    refusal: str
    position: int

    @property
    def refused(self) -> bool:
        """Whether this entry may not be read or written."""
        return bool(self.refusal)

    @property
    def supported(self) -> bool:
        """Whether the pack has a loader for this kind of file."""
        return self.kind is not None


class Archive:
    """One zip archive: where it came from, what it holds, and how to read an entry.

    Immutable.

    Attributes:
        source: The absolute path it was opened from, or ``None`` when it came from bytes.
        label: What to call it in a message.
        revision: ``size:mtime_ns`` of the file when it was opened, so a read can tell that
            the file has been rewritten underneath it. Empty for an archive from bytes.
        entries: Every indexed entry, in the archive's own order, refused ones included.
        held: How many entries the archive holds in total, which is above ``len(entries)``
            when the index was capped.
        truncated: Whether the index stopped at :data:`MAX_ENTRIES`.
        directories: How many directory entries were passed over, since a directory is not a
            file and is not counted as one.
    """

    __slots__ = (
        "_data",
        "_entries",
        "directories",
        "held",
        "label",
        "revision",
        "source",
        "truncated",
    )

    def __init__(
        self,
        entries: Iterable[Entry],
        source: str | None = None,
        data: bytes | None = None,
        label: str = "",
        revision: str = "",
        held: int = 0,
        truncated: bool = False,
        directories: int = 0,
    ) -> None:
        """Hold one archive's index.

        Args:
            entries: The indexed entries, in the archive's own order.
            source: Absolute path the archive was opened from.
            data: The archive's bytes, when it did not come from a file.
            label: What to call it in a message. Falls back to ``source``.
            revision: ``size:mtime_ns`` recorded at open time.
            held: Entries the archive holds in total.
            truncated: Whether the index was capped.
            directories: Directory entries passed over.
        """
        self._entries = tuple(entries)
        self._data = data
        self.source = source
        self.label = label or source or "the archive"
        self.revision = revision
        self.held = max(held, len(self._entries))
        self.truncated = truncated
        self.directories = directories

    # ------------------------------------------------------------------ opening

    @classmethod
    def from_path(cls, path: str | os.PathLike) -> "Archive":
        """Read the index of an archive on disk.

        Args:
            path: An absolute path, already resolved through
                ``modules.util.sandbox.resolve_read``.

        Returns:
            The archive. Its bytes stay on disk and each read reopens the file.

        Raises:
            NotAnArchive: The file is not a readable zip, which covers an empty file, a
                damaged central directory and a file that is something else entirely.
        """
        target = Path(path)
        try:
            stamp = target.stat()
        except OSError as error:
            raise NotAnArchive(
                f"{target} could not be opened ({error}).\n"
                f"  Check the file is still there and that this pack may read it."
            ) from error
        # zipfile.is_zipfile is not a gate: it only looks for the end record, so an archive
        # with a damaged central directory passes it and then raises on open.
        with _opened(target, str(target)) as archive:
            indexed = _index(archive)
        return cls(
            indexed.entries,
            source=str(target),
            label=str(target),
            revision=f"{stamp.st_size}:{stamp.st_mtime_ns}",
            held=indexed.held,
            truncated=indexed.truncated,
            directories=indexed.directories,
        )

    @classmethod
    def from_bytes(cls, data: Any, label: str = "the archive") -> "Archive":
        """Read the index of an archive already in memory.

        Args:
            data: The archive's bytes.
            label: What to call it in a message.

        Returns:
            The archive, holding the bytes it was given.

        Raises:
            NotAnArchive: ``data`` is not bytes, or is not a readable zip.
        """
        payload = _as_bytes(data, label)
        with _opened(payload, label) as archive:
            indexed = _index(archive)
        return cls(
            indexed.entries,
            data=payload,
            label=label,
            held=indexed.held,
            truncated=indexed.truncated,
            directories=indexed.directories,
        )

    # ------------------------------------------------------------------ reading

    @property
    def entries(self) -> tuple[Entry, ...]:
        """Every indexed entry, in the archive's own order."""
        return self._entries

    @property
    def files(self) -> tuple[Entry, ...]:
        """The entries that may be read, one per name, the last of a repeated name winning."""
        found: dict[str, Entry] = {}
        for entry in self._entries:
            if not entry.refused:
                found[entry.name] = entry
        return tuple(found.values())

    @property
    def refused(self) -> tuple[Entry, ...]:
        """The entries that may not be read, each carrying its refusal."""
        return tuple(entry for entry in self._entries if entry.refused)

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the readable entries, in the archive's own order."""
        return tuple(entry.name for entry in self.files)

    def supported_files(self) -> tuple[Entry, ...]:
        """The readable entries this pack has a loader for."""
        return tuple(entry for entry in self.files if entry.supported)

    def of_kind(self, kind: str) -> tuple[Entry, ...]:
        """The readable entries of one :data:`modules.archive.kinds.KINDS` member."""
        return tuple(entry for entry in self.files if entry.kind == kind)

    def entry(self, name: str) -> Entry | None:
        """The readable entry one name resolves to, or ``None`` when there is none."""
        for entry in reversed(self.files):
            if entry.name == name:
                return entry
        return None

    def read(self, name: str | Entry, limit: int = MAX_ENTRY_BYTES) -> bytes:
        """Read one entry.

        Args:
            name: The entry's ``name``, or the entry itself.
            limit: Most bytes to take, capped at :data:`MAX_ENTRY_BYTES`.

        Returns:
            The entry's bytes.

        Raises:
            ArchiveError: The name is not one of the readable entries, the entry unpacks past
                the limit, the archive has been rewritten since it was opened, or the entry's
                stream is damaged.
        """
        for got_name, payload in self.read_many([name], limit=limit, total=limit):
            del got_name
            return payload
        raise ArchiveError(self._missing(name))

    def read_many(
        self,
        names: Iterable[str | Entry] | None = None,
        limit: int = MAX_ENTRY_BYTES,
        total: int = MAX_TOTAL_BYTES,
    ) -> Iterator[tuple[str, bytes]]:
        """Read several entries, opening the archive once.

        Args:
            names: The entries to read, by ``name`` or as entries. ``None`` reads every
                readable entry in the archive's own order.
            limit: Most bytes one entry may unpack to, capped at :data:`MAX_ENTRY_BYTES`.
            total: Most bytes the whole session may unpack, capped at
                :data:`MAX_TOTAL_BYTES`.

        Yields:
            ``(name, bytes)`` per entry, in the order asked for.

        Raises:
            ArchiveError: A name is not one of the readable entries, the archive has been
                rewritten since it was opened, or an entry's stream is damaged.
            EntryTooLarge: An entry unpacks past ``limit``.
            BudgetSpent: ``total`` ran out before an entry was finished. That entry is not
                yielded, and the ones already yielded stand.
        """
        wanted = self._wanted(names)
        if not wanted:
            return
        self._require_unchanged()
        per_entry = max(0, min(int(limit), MAX_ENTRY_BYTES))
        budget = max(0, min(int(total), MAX_TOTAL_BYTES))
        with _opened(self._payload(), self.label) as archive:
            infos = archive.infolist()
            for entry in wanted:
                if entry.position >= len(infos):
                    # The index in hand names a position the file no longer holds, which a
                    # changed archive is caught for above and bytes cannot reach at all.
                    raise ArchiveError(
                        f"{entry.stored!r} is no longer at the place {self.label} said it "
                        f"was. Run the prompt again to read the file as it is now."
                    )
                try:
                    payload = _member(
                        archive, infos[entry.position], entry, min(per_entry, budget), self.label
                    )
                except EntryTooLarge:
                    # Whichever of the two bounds is the smaller is the one the entry ran
                    # past, and only one of them says anything about the entry itself.
                    if budget < per_entry:
                        raise BudgetSpent(_budget_spent(entry, budget, self.label)) from None
                    raise
                budget -= len(payload)
                yield entry.name, payload

    def __len__(self) -> int:
        return len(self.files)

    def __repr__(self) -> str:
        return (
            f"<Archive {self.label!r} {len(self.files)} readable file(s), "
            f"{len(self.refused)} refused>"
        )

    # ------------------------------------------------------------------ internals

    def _payload(self):
        """What :func:`_opened` is given: the path on disk, or the bytes in hand."""
        return Path(self.source) if self.source is not None else (self._data or b"")

    def _wanted(self, names: Iterable[str | Entry] | None) -> list[Entry]:
        """The entries a read was asked for, in the order it asked for them.

        Args:
            names: Entry names, entries, or ``None`` for every readable entry.

        Returns:
            The entries.

        Raises:
            ArchiveError: A name is not one of the readable entries.
        """
        if names is None:
            return list(self.files)
        wanted = []
        for name in names:
            entry = name if isinstance(name, Entry) else self.entry(str(name))
            if entry is None or entry.refused:
                raise ArchiveError(self._missing(name))
            wanted.append(entry)
        return wanted

    def _missing(self, name: str | Entry) -> str:
        """The message for a name the archive does not offer."""
        wanted = name.name if isinstance(name, Entry) else str(name)
        refusal = next((e for e in self.refused if wanted in (e.name, e.stored)), None)
        if refusal is not None:
            return f"{wanted!r} in {self.label} {REFUSALS[refusal.refusal]}."
        listed = ", ".join(self.names[:_NAMES_SHOWN]) or "nothing readable"
        if len(self.names) > _NAMES_SHOWN:
            listed += f", and {len(self.names) - _NAMES_SHOWN} more"
        return (
            f"{self.label} holds no readable entry named {wanted!r}. It holds {listed}.\n"
            f"  Names are case sensitive and carry the folders inside the archive, such as "
            f"'images/cat.png'."
        )

    def _require_unchanged(self) -> None:
        """Confirm the file on disk is still the one that was indexed.

        Raises:
            ArchiveError: The archive has been rewritten since it was opened, so the index in
                hand names entries at positions the file no longer holds.
        """
        if self.source is None or not self.revision:
            return
        try:
            stamp = os.stat(self.source)
        except OSError as error:
            raise ArchiveError(
                f"{self.label} could not be read again ({error}). It was opened successfully "
                f"a moment ago, so it has been moved, renamed or deleted since."
            ) from error
        current = f"{stamp.st_size}:{stamp.st_mtime_ns}"
        if current != self.revision:
            raise ArchiveError(
                f"{self.label} has been rewritten since it was opened, so what is inside it "
                f"no longer matches the list that was read.\n"
                f"  Run the prompt again to read the file as it is now."
            )


class _Index(NamedTuple):
    """What one pass over an archive's central directory found.

    Attributes:
        entries: The indexed entries, in the archive's own order.
        held: How many entries the archive holds in total.
        truncated: Whether the index stopped at :data:`MAX_ENTRIES`.
        directories: Directory entries passed over.
    """

    entries: tuple[Entry, ...]
    held: int
    truncated: bool
    directories: int


def is_archive(value: Any) -> bool:
    """Whether a value is an archive, for a node choosing between two branches."""
    return isinstance(value, Archive)


def require_archive(value: Any, label: str = "archive") -> Archive:
    """Read a ZIP socket, or raise naming what arrived instead.

    Args:
        value: Whatever arrived on the socket.
        label: The input's name, used in the message.

    Returns:
        The archive.

    Raises:
        NotAnArchive: ``value`` is not an archive.
    """
    if isinstance(value, Archive):
        return value
    if value is None:
        arrived = "nothing at all"
    elif isinstance(value, (bytes, bytearray, memoryview)):
        arrived = f"{len(bytes(value))} raw byte(s)"
    elif isinstance(value, str):
        arrived = "a string, which is a path rather than an opened archive"
    else:
        arrived = f"a {type(value).__name__}"
    raise NotAnArchive(
        f"the {label} input needs an opened archive and was given {arrived}.\n"
        f"  Connect Zip Open to this input: it opens the file and lists what is inside it."
    )


def safe_name(name: str) -> tuple[str, str]:
    """One entry name as a relative path, or a refusal naming what stopped it.

    Args:
        name: The entry name exactly as the archive carries it.

    Returns:
        ``(name, "")`` with the normalized relative name, or ``("", reason)`` naming a member
        of :data:`REFUSALS`. A name is refused when it is empty, absolute, carries a drive or
        a UNC share, lands above where it started, holds a null byte, or names a Windows
        device.
    """
    if not isinstance(name, str):
        return "", ESCAPES
    if "\x00" in name:
        return "", NUL_NAME
    candidate = name.strip().replace("\\", "/")
    if not candidate or candidate.endswith("/"):
        return "", ESCAPES
    if candidate.startswith("/") or ":" in candidate.split("/")[0]:
        return "", ESCAPES
    parts: list[str] = []
    for part in candidate.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if not parts:
                return "", ESCAPES
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        return "", ESCAPES
    for part in parts:
        if part.split(".")[0].strip().casefold() in _RESERVED_NAMES:
            return "", RESERVED
    return "/".join(parts), ""


# ---------------------------------------------------------------------- helpers


def _as_bytes(data: Any, label: str) -> bytes:
    """An archive's bytes, copied where the original could still be written to.

    Args:
        data: The value offered.
        label: What to call it in the message.

    Returns:
        Immutable bytes.

    Raises:
        NotAnArchive: ``data`` is not bytes at all.
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    raise NotAnArchive(
        f"{label} is read from bytes, and this is a {type(data).__name__}."
    )


def _opened(payload, label: str) -> zipfile.ZipFile:
    """Open an archive from a path or from bytes.

    Args:
        payload: A ``pathlib.Path`` to open, or the archive's bytes.
        label: What to call it in the message.

    Returns:
        The open archive, to be used as a context manager.

    Raises:
        NotAnArchive: The bytes or the file are not a readable zip.
    """
    source = payload if isinstance(payload, Path) else io.BytesIO(payload)
    try:
        return zipfile.ZipFile(source)
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as error:
        raise NotAnArchive(_not_a_zip(payload, label, error)) from error


def _not_a_zip(payload, label: str, error: Exception) -> str:
    """The message for something that is not a readable archive."""
    if isinstance(payload, Path):
        try:
            size = payload.stat().st_size
            with payload.open("rb") as handle:
                opening = handle.read(2)
        except OSError:
            size, opening = 0, b""
    else:
        size, opening = len(payload), bytes(payload[:2])
    if not size:
        hint = "The file holds no bytes at all, so nothing was ever written to it."
    elif opening != b"PK":
        hint = (
            "A zip file starts with the two bytes 'PK' and this one does not, so it is "
            "some other kind of file with a .zip name."
        )
    else:
        hint = (
            "It starts like a zip file, so it is a damaged one: the index at the end of "
            "the file, which says what the archive holds and where, could not be read. A "
            "part-downloaded or part-written file reads like this."
        )
    return f"{label} is not a readable zip archive ({error}).\n  {hint}"


def _index(archive: zipfile.ZipFile) -> _Index:
    """Read an archive's central directory into entries.

    Args:
        archive: The open archive.

    Returns:
        The entries, how many the archive holds, whether the index was capped, and how many
        directory entries were passed over. Every entry is classified here, so nothing else
        has to decide what a name means.
    """
    found: list[Entry] = []
    directories = 0
    infos = archive.infolist()
    for position, info in enumerate(infos):
        if position >= MAX_ENTRIES:
            break
        if info.is_dir():
            # A directory entry holds nothing. It is not a file, is not counted as one, and
            # is created by writing the entries inside it rather than by itself.
            directories += 1
            continue
        # orig_filename is the name the archive carries; ZipInfo rewrites filename, cutting
        # it at a null byte and squaring up separators, so the two differ exactly where a
        # name would be written under something other than what the archive holds.
        stored = getattr(info, "orig_filename", info.filename) or info.filename
        name, refusal = safe_name(stored)
        if not refusal and _is_symlink(info):
            name, refusal = "", SYMLINK
        if not refusal and info.flag_bits & 0x1:
            name, refusal = "", ENCRYPTED
        size = max(0, int(info.file_size))
        if not refusal and size > MAX_ENTRY_BYTES:
            name, refusal = "", OVERSIZE
        found.append(
            Entry(
                name=name,
                stored=stored,
                size=size,
                compressed=max(0, int(info.compress_size)),
                kind=kinds.kind_of(name) if name else None,
                refusal=refusal,
                position=position,
            )
        )
    return _Index(tuple(found), len(infos), len(infos) > MAX_ENTRIES, directories)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """Whether an entry is a symbolic link rather than a file."""
    return stat.S_ISLNK(info.external_attr >> 16)


def _member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    entry: Entry,
    allowed: int,
    label: str,
) -> bytes:
    """Read one entry, bounded whatever the archive claims about it.

    Args:
        archive: The open archive.
        info: The entry's record, taken by position so a repeated name reads the right one.
        entry: The indexed entry, for the message.
        allowed: Most bytes this read may take.
        label: What to call the archive in a message.

    Returns:
        The entry's bytes.

    Raises:
        EntryTooLarge: The entry unpacks past ``allowed``.
        ArchiveError: Its stream could not be read, which covers a truncated member, a
            damaged one and an encrypted one.
    """
    buffer = io.BytesIO()
    taken = 0
    try:
        with archive.open(info) as handle:
            while True:
                # One byte past what is allowed, so an entry larger than the size the
                # archive declares is caught rather than trusted.
                block = handle.read(min(CHUNK, allowed - taken + 1))
                if not block:
                    break
                taken += len(block)
                if taken > allowed:
                    raise EntryTooLarge(_too_large(entry, allowed, label))
                buffer.write(block)
    except (KeyError, zipfile.BadZipFile, RuntimeError, OSError, ValueError, EOFError) as error:
        if isinstance(error, ArchiveError):
            raise
        raise ArchiveError(
            f"{entry.stored!r} could not be read out of {label} ({error}).\n"
            f"  The entry is damaged or was only part written, or it is encrypted, which "
            f"nothing here can open. The rest of the archive may still be readable."
        ) from error
    return buffer.getvalue()


def _too_large(entry: Entry, allowed: int, label: str) -> str:
    """The message for an entry that unpacks past what a read may take."""
    return (
        f"{entry.stored!r} in {label} unpacks to more than {_megabytes(allowed)} and was "
        f"not read. Its own index says {_megabytes(entry.size)}.\n"
        f"  A file that unpacks to far more than it occupies is either damaged or built to "
        f"exhaust memory, so it is refused rather than read."
    )


def _budget_spent(entry: Entry, left: int, label: str) -> str:
    """The message for a read that reached its total before an entry was finished."""
    return (
        f"{entry.stored!r} in {label} unpacks to more than the {_megabytes(left)} left of "
        f"what this read may take in total, so it was not read.\n"
        f"  Nothing is wrong with the entry. It is where the read ran out of its allowance, "
        f"and reading fewer files at a time reaches it."
    )


def _megabytes(count: int) -> str:
    """A byte count as a rounded number of megabytes, for a message."""
    return f"{count / (1024 * 1024):.1f} MB"
