"""What an archive holds, written out for a person to read.

:func:`listing_text` is the report Zip Open shows on the node and emits on its ``listing``
output. :func:`counts` is the same reading as numbers.
"""

from __future__ import annotations

from typing import Mapping

from .container import REFUSALS, Archive
from .kinds import KINDS

__all__ = [
    "MAX_LINES",
    "NO_LOADER",
    "counts",
    "kind_counts_text",
    "listing_text",
    "size_text",
]

#: How many file lines the report holds before it stops naming them.
MAX_LINES = 200

#: How many refused entries the report names before it stops naming them.
MAX_REFUSED = 20

#: The key :func:`counts` files an entry under when the pack has no loader for it.
NO_LOADER = "other"

#: How wide the name column is before the kind and the size are pushed along.
_NAME_WIDTH = 44


def counts(archive: Archive) -> dict[str, int]:
    """How many readable entries of each kind the archive holds.

    Args:
        archive: The opened archive.

    Returns:
        ``{kind: count}`` over :data:`modules.archive.kinds.KINDS` plus :data:`NO_LOADER`,
        every key present even where the count is zero.
    """
    found = {kind: 0 for kind in KINDS}
    found[NO_LOADER] = 0
    for entry in archive.files:
        found[entry.kind or NO_LOADER] += 1
    return found


def size_text(count: int) -> str:
    """A byte count as a number a person reads."""
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    if count < 1024 * 1024 * 1024:
        return f"{count / (1024 * 1024):.1f} MB"
    return f"{count / (1024 * 1024 * 1024):.2f} GB"


def listing_text(archive: Archive) -> str:
    """The whole report on one archive.

    Args:
        archive: The opened archive.

    Returns:
        The report: what the archive is, what it holds, a line per readable file, and a line
        per refused entry naming its refusal. An archive holding nothing readable says so
        rather than answering with a header and no body.
    """
    lines = [archive.label, _header(archive), _kinds(archive)]
    lines = [line for line in lines if line]
    files = archive.files
    for entry in files[:MAX_LINES]:
        name = entry.name
        if len(name) > _NAME_WIDTH:
            name = name[: _NAME_WIDTH - 3] + "..."
        kind = entry.kind or "no loader"
        lines.append(f"  {name:<{_NAME_WIDTH}} {kind:<10} {size_text(entry.size):>10}")
    if len(files) > MAX_LINES:
        lines.append(f"  ... and {len(files) - MAX_LINES} more file(s)")
    if not files:
        lines.append("  nothing in it can be read out")
    lines += _refused(archive)
    return "\n".join(lines)


def _header(archive: Archive) -> str:
    """The line counting what the archive holds."""
    files = len(archive.files)
    parts = [f"{files} readable file(s)"]
    repeated = len([entry for entry in archive.entries if not entry.refused]) - files
    if repeated > 0:
        # Two entries under one name are one file to every zip reader, which resolves the
        # name to the last of them, so the earlier one is not counted as a file.
        parts.append(f"{repeated} replaced by a later entry of the same name")
    if archive.directories:
        folders = (
            "folder entry, which holds nothing"
            if archive.directories == 1
            else "folder entries, which hold nothing"
        )
        parts.append(f"{archive.directories} {folders}")
    if archive.refused:
        parts.append(f"{len(archive.refused)} refused")
    if archive.truncated:
        parts.append(f"listing the first {len(archive.entries)} of {archive.held} entries")
    return ", ".join(parts)


def _kinds(archive: Archive) -> str:
    """The line counting the readable files by kind, empty when there are none."""
    tally = counts(archive)
    named = [f"{tally[kind]} {kind}" for kind in KINDS if tally[kind]]
    if tally[NO_LOADER]:
        named.append(f"{tally[NO_LOADER]} with no loader here")
    return f"  {', '.join(named)}" if named else ""


def _refused(archive: Archive) -> list[str]:
    """The lines naming the refused entries, empty when none were refused."""
    refused = archive.refused
    if not refused:
        return []
    lines = ["refused, and not written anywhere:"]
    for entry in refused[:MAX_REFUSED]:
        lines.append(f"  {entry.stored!r} {REFUSALS.get(entry.refusal, entry.refusal)}")
    if len(refused) > MAX_REFUSED:
        lines.append(f"  ... and {len(refused) - MAX_REFUSED} more")
    return lines


def kind_counts_text(tally: Mapping[str, int]) -> str:
    """One line naming the kinds a tally holds, for a log message."""
    named = [f"{tally.get(kind, 0)} {kind}" for kind in KINDS if tally.get(kind)]
    if tally.get(NO_LOADER):
        named.append(f"{tally[NO_LOADER]} unsupported")
    return ", ".join(named) or "nothing readable"
