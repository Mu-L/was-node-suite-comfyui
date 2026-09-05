"""What a node wrote to disk.

Files and writes publish as figures under the node's own id, with the folder, the format,
the names and the bytes as rows.
"""

from __future__ import annotations

import os

from .. import log
from . import run_result
from .batch_report import readable_bytes

__all__ = ["publish"]

logger = log.get_logger("interface.file_report")

#: What the folder row says where nothing was written, so the row is never blank.
NOWHERE = "nowhere"


def publish(paths, intended=None, kind="", folder=None, facts=None, node_id=None) -> bool:
    """Store what a node wrote to disk, for that node's own interface to fetch.

    Never raises, and never touches the values it is given.

    Args:
        paths: The files the node wrote, in the order it wrote them. One path may appear
            more than once, which is a run whose names collided.
        intended: How many writes the node set out to make. Given and larger than the number
            of paths, the report is a warning naming the shortfall.
        kind: What was written, such as ``png`` or ``mp4``. Left out, the extension of the
            first path.
        folder: The directory the files landed in. Left out, the directory of the first path.
        facts: Anything further worth a row, as a mapping of name to value, merged after the
            report's own.
        node_id: The publishing node's graph id. Left out, the id of the node ComfyUI is
            executing, so a node needs no hidden input to report itself.

    Returns:
        Whatever :func:`run_result.publish` answered, which is False when no browser is
        connected and the readout is not worth building.
    """
    # Ahead of the size reads rather than after them: a report costs one stat per file, and a
    # headless run, an API call and a command line run must pay none of it.
    if not run_result.watching():
        return False
    try:
        written = [str(path) for path in paths or ()]
        # Insertion ordered rather than sorted, so the first and last rows name the first and
        # last file the run wrote and not the first and last alphabetically.
        distinct = list(dict.fromkeys(written))
        home = folder if folder is not None else _folder(distinct)
        own = {
            "folder": str(home) or NOWHERE,
            "format": kind or _extension(distinct),
        }
        # One file has no first and last to tell apart, and the two rows would name it twice.
        if len(distinct) == 1:
            own["file"] = os.path.basename(distinct[0])
        else:
            own["first"] = os.path.basename(distinct[0]) if distinct else NOWHERE
            own["last"] = os.path.basename(distinct[-1]) if distinct else NOWHERE
        own["bytes"] = readable_bytes(_bytes(distinct))
        merged = dict(own)
        merged.update(facts or {})
        if len(merged) > run_result.MAX_FACTS:
            logger.debug(
                "a file report carries %d facts and %d are drawn, so %s is left out",
                len(merged),
                run_result.MAX_FACTS,
                ", ".join(list(merged)[run_result.MAX_FACTS:]),
            )
        short = max(0, int(intended) - len(written)) if intended is not None else 0
        status, summary = _wording(len(written), len(distinct), short, own["folder"])
        return run_result.publish(
            status=status,
            summary=summary,
            counts={"files": len(distinct), "writes": len(written)},
            facts=merged,
            node_id=node_id,
        )
    except Exception as error:
        logger.debug("a file report could not be built (%s)", error)
        return False


def _folder(paths) -> str:
    """The directory the first path sits in, or nothing where there is no path."""
    return os.path.dirname(paths[0]) if paths else ""


def _extension(paths) -> str:
    """The first path's extension without its dot, or what an absent one reads as."""
    if not paths:
        return NOWHERE
    return os.path.splitext(paths[0])[1].lstrip(".").lower() or "no extension"


def _bytes(paths) -> int:
    """How much room the files take on disk, counting a file this cannot stat as nothing.

    Args:
        paths: The distinct paths written.

    Returns:
        The total in bytes.
    """
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            logger.debug("the size of %s could not be read", path)
    return total


def _wording(writes: int, files: int, short: int, home: str) -> tuple[str, str]:
    """The status the report is drawn inside, and the line above the figures.

    Args:
        writes: How many times the node wrote.
        files: How many files those writes left behind.
        short: How many writes the node set out to make and did not.
        home: The folder the files landed in.

    Returns:
        ``(status, summary)``.
    """
    if not writes:
        return run_result.WARNING, "nothing was written"
    if short:
        return run_result.WARNING, (
            f"{writes} of {writes + short} file(s) reached {home}, and {short} did not; the "
            f"reason each one failed is in the log"
        )
    if files < writes:
        return run_result.WARNING, (
            f"{writes} writes landed on {files} file(s) in {home}, so each one replaced the "
            f"one before it"
        )
    return run_result.OK, f"{files} file(s) written to {home}"
