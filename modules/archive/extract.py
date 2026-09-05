"""Writing the entries of an opened archive out onto disk.

:func:`chosen` picks the entries a glob names, :func:`run` writes them, and
:func:`report_text` reads the result back out for a person. One run writes at most
:data:`MAX_FILES` files and :data:`MAX_TOTAL_BYTES` bytes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterator, Sequence

from ..util import sandbox
from . import container, picks
from .summary import size_text

__all__ = [
    "EXISTING",
    "FLATTEN",
    "KEEP",
    "MAX_FILES",
    "MAX_NUMBERED",
    "MAX_TOTAL_BYTES",
    "NAMING",
    "NUMBER",
    "OVERWRITE",
    "Placed",
    "Result",
    "SKIP",
    "chosen",
    "report_text",
    "run",
    "target_name",
]

#: What an entry is called on disk, in the order the widget offers them.
KEEP = "keep folders"
FLATTEN = "file name only"
NAMING: tuple[str, ...] = (KEEP, FLATTEN)

#: What happens where the folder already holds a file of that name.
OVERWRITE = "overwrite"
SKIP = "skip"
NUMBER = "number apart"
EXISTING: tuple[str, ...] = (OVERWRITE, SKIP, NUMBER)

#: How many files one run writes.
MAX_FILES = 4096

#: How many bytes one run writes in total.
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024

#: How many entries one read of the archive asks for before the archive is opened again.
BATCH_FILES = 256

#: How many declared bytes one such read asks for, below the container's own session total.
BATCH_BYTES = container.MAX_TOTAL_BYTES // 2

#: How far a clashing name is numbered before the entry is passed over.
MAX_NUMBERED = 1000

#: How many written files the report names before it stops naming them.
MAX_LINES = 200

#: Why an entry was not written, in the words the report and the log use.
ALREADY_THERE = "a file of that name is already in the folder, and existing is 'skip'"
UNNUMBERABLE = (
    f"a file of that name is already in the folder, and so are the {MAX_NUMBERED} numbered "
    f"names beside it"
)


@dataclasses.dataclass(frozen=True)
class Placed:
    """One entry that reached the folder.

    Attributes:
        name: The entry's name inside the archive.
        path: Where it was written.
        size: Bytes written.
    """

    name: str
    path: Path
    size: int


@dataclasses.dataclass
class Result:
    """What one run wrote, and what it left behind.

    Attributes:
        matched: How many readable entries the glob picked.
        written: One row per file written, in the order they were written.
        skipped: ``{entry name: reason}`` for each picked entry that was not written.
        renamed: ``{file name: entry name}`` for each file numbered apart from another.
        bounds: One line per limit that stopped the run early.
    """

    matched: int = 0
    written: list[Placed] = dataclasses.field(default_factory=list)
    skipped: dict[str, str] = dataclasses.field(default_factory=dict)
    renamed: dict[str, str] = dataclasses.field(default_factory=dict)
    bounds: list[str] = dataclasses.field(default_factory=list)

    @property
    def size(self) -> int:
        """Bytes written across every file."""
        return sum(row.size for row in self.written)

    @property
    def names(self) -> list[str]:
        """The entry names that were written, in the order they were written."""
        return [row.name for row in self.written]

    @property
    def paths(self) -> list[str]:
        """The paths that were written, in the order they were written."""
        return [str(row.path) for row in self.written]

    def bound(self, message: str) -> None:
        """Record that a limit stopped the run, with the line naming which."""
        self.bounds.append(message)


def chosen(archive: container.Archive, pattern: str) -> list[container.Entry]:
    """The readable entries a glob picks, sorted by name.

    Args:
        archive: The opened archive.
        pattern: A glob, as :func:`modules.archive.picks.matches` reads it. Empty picks
            every readable entry.

    Returns:
        The entries, in the order they are written.
    """
    found = [entry for entry in archive.files if picks.matches(entry.name, pattern)]
    found.sort(key=lambda entry: (entry.name.casefold(), entry.name))
    return found


def target_name(entry: container.Entry, naming: str) -> str:
    """What one entry is called on disk.

    Args:
        entry: The entry being written.
        naming: One of :data:`NAMING`. Anything else is read as :data:`KEEP`.

    Returns:
        The entry's name with the folders inside the archive kept, or its last segment alone
        on :data:`FLATTEN`.
    """
    name = entry.name.replace("\\", "/").lstrip("/")
    if naming == FLATTEN:
        return name.rsplit("/", 1)[-1] or name
    return name


def run(
    archive: container.Archive,
    entries: Sequence[container.Entry],
    folder: Path,
    naming: str = KEEP,
    existing: str = OVERWRITE,
) -> Result:
    """Write entries out of an archive into a folder.

    Args:
        archive: The opened archive.
        entries: The entries to write, in the order they are written.
        folder: The destination, already resolved through
            ``modules.util.sandbox.resolve_write``.
        naming: One of :data:`NAMING`.
        existing: One of :data:`EXISTING`.

    Returns:
        What was written, what was passed over and why, and any limit that stopped the run.

    Raises:
        PathNotAllowed: An entry named a place outside ``folder``.
        OSError: A folder could not be created, or a file could not be written.
    """
    result = Result(matched=len(entries))
    wanted = list(entries[:MAX_FILES])
    if len(entries) > MAX_FILES:
        result.bound(
            f"{len(entries)} entry(ies) were picked and the first {MAX_FILES} by name were "
            f"written, which is as many as one run writes; narrow the pattern to choose which"
        )
    taken: set[str] = set()
    budget = MAX_TOTAL_BYTES
    for entry, data in _stream(archive, wanted, result):
        if len(data) > budget:
            result.bound(
                f"the run stopped at {size_text(MAX_TOTAL_BYTES)}, which is as much as one "
                f"run writes, with {len(wanted) - len(result.written)} file(s) left; nothing "
                f"is wrong with them, and a narrower pattern reaches the rest"
            )
            break
        target, note = _place(folder, target_name(entry, naming), existing, taken)
        if target is None:
            result.skipped[entry.name] = note
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        taken.add(str(target).casefold())
        if note:
            result.renamed[target.name] = entry.name
        result.written.append(Placed(entry.name, target, len(data)))
        budget -= len(data)
    return result


def report_text(archive: container.Archive, folder: Path, result: Result) -> str:
    """The whole report on one run.

    Args:
        archive: The archive that was read.
        folder: The destination folder.
        result: What the run did.

    Returns:
        The report: where the files went, a line per file with its size, a line per entry
        that was passed over, and a line per limit that stopped the run.
    """
    lines = [str(folder), _header(archive, result)]
    for row in result.written[:MAX_LINES]:
        lines.append(f"  {row.name:<44} {size_text(row.size):>10}")
    if len(result.written) > MAX_LINES:
        lines.append(f"  ... and {len(result.written) - MAX_LINES} more file(s)")
    if not result.written:
        lines.append("  nothing was written")
    for name, entry in sorted(result.renamed.items()):
        lines.append(f"  {entry!r} went in as '{name}', another file having taken the name")
    if result.skipped:
        lines.append("passed over:")
        for name, reason in sorted(result.skipped.items()):
            lines.append(f"  {name!r} {reason}")
    lines += [f"  {line}" for line in result.bounds]
    return "\n".join(lines)


# ---------------------------------------------------------------------- helpers


def _header(archive: container.Archive, result: Result) -> str:
    """The line counting what the run wrote."""
    parts = [
        f"{len(result.written)} file(s) written, {size_text(result.size)}, "
        f"of {result.matched} the pattern picked"
    ]
    if result.skipped:
        parts.append(f"{len(result.skipped)} passed over")
    if archive.refused:
        parts.append(f"{len(archive.refused)} entry(ies) refused by the archive itself")
    return ", ".join(parts)


def _stream(
    archive: container.Archive, entries: Sequence[container.Entry], result: Result
) -> Iterator[tuple[container.Entry, bytes]]:
    """Read entries one at a time, opening the archive again as a read budget runs out.

    Args:
        archive: The opened archive.
        entries: The entries to read, in the order they are read.
        result: Filled in with each entry that could not be unpacked.

    Yields:
        ``(entry, bytes)`` per entry that was read.
    """
    index = 0
    while index < len(entries):
        batch = _batch(entries[index:])
        read = 0
        try:
            for _, data in archive.read_many(
                batch, limit=container.MAX_ENTRY_BYTES, total=container.MAX_TOTAL_BYTES
            ):
                yield batch[read], data
                read += 1
        except container.BudgetSpent:
            # The next pass opens the archive again and starts on the entry it stopped at.
            pass
        except container.ArchiveError as error:
            result.skipped[batch[read].name] = str(error)
            read += 1
        if read == 0:
            result.skipped[batch[0].name] = (
                "could not be unpacked, and the read made no progress on it"
            )
            read = 1
        index += read


def _batch(entries: Sequence[container.Entry]) -> list[container.Entry]:
    """The entries one read of the archive asks for, by count and by declared size."""
    batch: list[container.Entry] = []
    declared = 0
    for entry in entries[:BATCH_FILES]:
        if batch and declared + entry.size > BATCH_BYTES:
            break
        batch.append(entry)
        declared += entry.size
    return batch


def _place(
    folder: Path, name: str, existing: str, taken: set[str]
) -> tuple[Path | None, str]:
    """Where one entry goes, following the clash rule.

    Args:
        folder: The resolved destination.
        name: What the entry is called on disk.
        existing: One of :data:`EXISTING`.
        taken: Casefolded paths this run has already written.

    Returns:
        ``(path, note)``. ``path`` is ``None`` where the entry is passed over, and ``note``
        then names what stopped it; otherwise ``note`` is non-empty where the name was
        numbered apart from another file.

    Raises:
        PathNotAllowed: ``name`` names a place outside ``folder``.
    """
    target = sandbox.resolve_write_file(folder, name)
    written_here = str(target).casefold() in taken
    if not written_here and not target.exists():
        return target, ""
    if existing == SKIP and not written_here:
        return None, ALREADY_THERE
    if existing == OVERWRITE and not written_here:
        return target, ""
    numbered = _numbered(target, taken)
    if numbered is None:
        return None, UNNUMBERABLE
    return numbered, "numbered apart"


def _numbered(target: Path, taken: set[str]) -> Path | None:
    """The same name with a number before its extension, free on disk and in this run.

    Args:
        target: The path that is already taken.
        taken: Casefolded paths this run has already written.

    Returns:
        ``cat_2.png`` for ``cat.png``, counting up until a free name is found, or ``None``
        where :data:`MAX_NUMBERED` names were taken as well.
    """
    for counter in range(2, MAX_NUMBERED + 2):
        candidate = target.with_name(f"{target.stem}_{counter}{target.suffix}")
        if str(candidate).casefold() not in taken and not candidate.exists():
            return candidate
    return None
