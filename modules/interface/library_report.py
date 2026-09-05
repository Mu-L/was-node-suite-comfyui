"""What a run changed in the terminology pantry or the style library.

Counts publish as figures under the node's own id, the term and the file as rows, and the
entries as one line each in a body.
"""

from __future__ import annotations

from .. import log
from . import run_result

__all__ = ["ENTRIES", "MAX_LINES", "publish"]

logger = log.get_logger("interface.library_report")

#: What the listed lines are called on the panel when the caller names nothing.
ENTRIES = "entries"

#: How many lines a listing hands over. The report carries the first
#: :data:`run_result.MAX_BODY_CHARS` characters of them and the rest are reached a page at a
#: time, so this is what a page can still be asked for. The lines past it are dropped, the
#: heading counting every one the store held.
MAX_LINES = 20000


def publish(
    summary,
    counts=None,
    facts=None,
    lines=(),
    listing=ENTRIES,
    total=None,
    status=run_result.OK,
    node_id=None,
) -> bool:
    """Store what a run changed in a store, for that node's own interface to fetch.

    Never raises, and never touches the values it is given.

    Args:
        summary: One line saying what the run did.
        counts: Named numbers drawn as figures, as a mapping of name to number.
        facts: Named strings drawn as rows, as a mapping of name to value.
        lines: The entries, styles or terms to list, in the order they are stored.
        listing: What the listed lines are called on the panel.
        total: How many lines there were before the listing was cut. Left out, the number
            handed over stands in for it.
        status: One of :data:`run_result.STATUSES`.
        node_id: The publishing node's graph id. Left out, the id of the node ComfyUI is
            executing, so a node needs no hidden input to report itself.

    Returns:
        Whatever :func:`run_result.publish` answered, which is False when no browser is
        connected and the readout is not worth building.
    """
    if not run_result.watching():
        return False
    try:
        # Read once, so an iterator of lines is counted and listed off the same read.
        given = list(lines)
        shown = [str(line) for line in given[:MAX_LINES]]
        held = len(given) if total is None else int(total)
        text = "\n".join(shown)
        # The count is named wherever the report carries fewer lines than the store held,
        # whether they were dropped here or left off the piece the body carries.
        listed = held > len(shown) or len(text) > run_result.MAX_BODY_CHARS
        name = f"{listing} ({held})" if listed else listing
        bodies = run_result.body(name, text) if shown else None
        return run_result.publish(
            status=status,
            summary=summary,
            counts=counts,
            facts=facts,
            bodies=bodies,
            node_id=node_id,
        )
    except Exception as error:
        logger.debug("no library report was published for node %s (%s)", node_id, error)
        return False
