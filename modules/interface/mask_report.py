"""What a mask node did to a mask, as something its node can draw.

Coverage in and out, what was set and cleared, the regions, the box and the value range
publish under the node's own id.
"""

from __future__ import annotations

from .. import log
from ..mask import measure
from . import run_result

__all__ = ["publish"]

logger = log.get_logger("interface.mask_report")


def publish(before, after, source=None, node_id=None) -> bool:
    """Store what a mask node did to a mask, for that node's own interface to fetch.

    Args:
        before: The mask the node was handed, as a ``MASK`` tensor. None for a node with no
            mask to compare against, which publishes the result on its own.
        after: The mask the node answered, as a ``MASK`` tensor.
        source: Which input ``before`` was read from, for a node taking several. Named in
            the summary line. Left out, the summary says "the input".
        node_id: The publishing node's graph id. Left out, the id of the node ComfyUI is
            executing, so a node needs no hidden input to report itself.

    Returns:
        Whatever :func:`run_result.publish` answered, and False when no browser is connected,
        which is decided before either tensor is touched.
    """
    # Ahead of every reduction rather than inside run_result.publish: a reading is a reduce
    # per frame plus a host transfer plus a labelling pass, and a headless run, an API call
    # and a command line run must pay none of it.
    if not run_result.watching():
        return False
    try:
        result = measure.read(after)
        start = measure.read(before) if before is not None else None
        change = measure.compare(before, after) if start is not None else None
        status, summary = _wording(start, result, change, source)
        counts = _counts(start, result, change)
        facts = _facts(start, result, change)
        tallies, tallies_total = _tallies(result)
        _check_room(counts, facts)
        return run_result.publish(
            status=status,
            summary=summary,
            counts=counts,
            tallies=tallies,
            tallies_total=tallies_total,
            facts=facts,
            node_id=node_id,
        )
    except Exception as error:
        logger.debug("no mask report was published for node %s (%s)", node_id, error)
        return False


def _counts(start, result, change) -> dict:
    """The numbers drawn as figures, at most :data:`run_result.MAX_COUNTS` of them.

    Args:
        start: The input's reading, or None.
        result: The result's reading.
        change: What was set and cleared, or None where there was no input to compare.

    Returns:
        A mapping of name to number, in the order they are drawn.
    """
    counts = {"coverage": result.coverage}
    if start is not None:
        counts["before"] = start.coverage
        counts["delta"] = round(result.coverage - start.coverage, 2)
    if change is not None and change.comparable:
        counts["added"] = change.added
        counts["removed"] = change.removed
    if result.regions is not None:
        counts["regions"] = result.regions
    if start is not None and start.regions is not None:
        counts["regions before"] = start.regions
    counts["frames"] = result.frames
    return counts


def _facts(start, result, change) -> dict:
    """The rows drawn under the figures, at most :data:`run_result.MAX_FACTS` of them.

    Args:
        start: The input's reading, or None.
        result: The result's reading.
        change: What was set and cleared, or None where there was no input to compare.

    Returns:
        A mapping of name to text, in the order they are drawn.
    """
    facts = {
        "size": _sizes(start, result),
        "box": _boxes(start, result),
        "set": f"{result.set_pixels:,} px of {result.pixels:,}",
        "soft": _soft(result),
        "range": f"{result.lowest:.3f} to {result.highest:.3f}",
    }
    if result.regions is None:
        facts["regions"] = result.region_note or "region counts were not measured"
    elif result.largest is not None:
        facts["largest"] = f"{result.largest:.2f}% of the set area"
    if result.frames > 1:
        facts["measured"] = f"box and regions from frame {result.measured} of {result.frames}"
    if change is not None and not change.comparable:
        facts["comparable"] = change.reason
    return facts


def _tallies(result) -> tuple[list[dict] | None, int | None]:
    """One coverage row per frame, and how many frames there were to draw one for.

    Args:
        result: The result's reading.

    Returns:
        ``(rows, total)``, both None for a single frame, whose one row is the coverage
        figure again.
    """
    # A total above the rows handed over is what marks a result truncated, so a report
    # carrying no breakdown carries no total either.
    if result.frames < 2:
        return None, None
    rows = [
        {"name": f"frame {index}", "value": value}
        for index, value in enumerate(result.per_frame)
    ]
    return rows, result.frames


def _wording(start, result, change, source) -> tuple[str, str]:
    """The status the report is drawn inside, and the line above the figures.

    Args:
        start: The input's reading, or None.
        result: The result's reading.
        change: What was set and cleared, or None.
        source: Which input the reading came from, or None.

    Returns:
        ``(status, summary)``.
    """
    # Four states are worth a warning: a value a mask cannot hold, an empty result, a fully
    # set result, and nothing set on the way in. A value outside the range is tested first,
    # since it changes what every other figure means.
    named = source or "the input"
    if result.lowest < 0.0 or result.highest > 1.0:
        return run_result.WARNING, (
            f"values run from {result.lowest:.3f} to {result.highest:.3f}, outside the "
            f"0 to 1 a mask holds"
        )
    if result.set_pixels == 0 and (start is None or start.set_pixels > 0):
        return run_result.WARNING, (
            "the mask came out empty" if start is None
            else f"the mask came out empty, and {start.coverage:.2f}% of {named} was set"
        )
    if result.pixels and result.set_pixels == result.pixels and (
        start is None or start.set_pixels < start.pixels
    ):
        return run_result.WARNING, (
            "the mask came out fully set" if start is None
            else f"the mask came out fully set, and {start.coverage:.2f}% of {named} was set"
        )
    if start is not None and start.set_pixels == 0 and result.set_pixels == 0:
        return run_result.WARNING, f"nothing was set in {named}, and nothing is set now"
    line = (
        f"{result.coverage:.2f}% covered"
        f"{_regions(result)}{_shift(start, result, change, named)}"
    )
    return run_result.OK, line


def _regions(result) -> str:
    """How many regions the measured frame holds, or nothing where none were counted."""
    if not result.regions:
        return ""
    word = "region" if result.regions == 1 else "regions"
    return f" in {result.regions} {word}"


def _shift(start, result, change, named) -> str:
    """How the coverage moved, told apart from a mask that came back untouched."""
    if start is None:
        return ""
    if change is not None and change.comparable and not change.added_pixels \
            and not change.removed_pixels:
        return f", unchanged from {named}"
    delta = result.coverage - start.coverage
    if abs(delta) < 0.005:
        return f", the same coverage as {named}"
    return f", {abs(delta):.2f} points {'more' if delta > 0 else 'less'} than {named}"


def _sizes(start, result) -> str:
    """The result's frame size, and the input's alongside it where the two differ."""
    now = f"{result.width}x{result.height}"
    if start is None or (start.width, start.height) == (result.width, result.height):
        return now
    return f"{now}, from {start.width}x{start.height}"


def _boxes(start, result) -> str:
    """The result's bounding box, and the input's alongside it where the frame changed."""
    now = _rectangle(result.box)
    if start is None or (start.width, start.height) == (result.width, result.height):
        return now
    return f"{now}, from {_rectangle(start.box)}"


def _rectangle(box) -> str:
    """One bounding box as its corner and its size, or what an absent box means."""
    if box is None:
        return "nothing set"
    left, top, right, bottom = box
    return f"x{left} y{top} {right - left}x{bottom - top}"


def _soft(result) -> str:
    """The coverage by value, and how much of the frame is neither set nor clear."""
    by_value = f"{result.soft:.2f}% by value"
    if result.partial == 0 or not result.pixels:
        return f"{by_value}, no partial values"
    return f"{by_value}, {100.0 * result.partial / result.pixels:.2f}% part set"


def _check_room(counts: dict, facts: dict) -> None:
    """Say when a report no longer fits, rather than letting a row be dropped in silence.

    Args:
        counts: The figures the report carries.
        facts: The rows the report carries.
    """
    for rows, cap, part in ((counts, run_result.MAX_COUNTS, "figure"),
                            (facts, run_result.MAX_FACTS, "row")):
        if len(rows) > cap:
            logger.warning(
                "a mask report built %s %ss and a run result carries %s, so %s of them are "
                "dropped from what the node draws: %s",
                len(rows), part, cap, len(rows) - cap, ", ".join(list(rows)[cap:]),
            )
