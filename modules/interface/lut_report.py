"""What a LUT node produced, as something its node can draw.

:func:`publish` files the table's name, size and shape, and the strip graded through it.
The strip's neutral band is the transfer response the panel reads its curves off.
"""

from __future__ import annotations

from .. import log
from . import preview, run_result

__all__ = ["SLOT", "publish"]

logger = log.get_logger("interface.lut_report")

#: Slot the graded chart is published under, so a node publishing pictures of its own does
#: not collide with it.
SLOT = "lut_strip"

#: How far a sample may sit from the identity cube before the table is said to change
#: something. A resample through a smaller cube and back leaves rounding of about this size.
IDENTITY_TOLERANCE = 0.002


def _is_identity(table) -> bool:
    """Whether a table maps every colour to itself.

    Args:
        table: A :class:`lut.LUT`.

    Returns:
        True where the largest departure from the identity cube is under
        :data:`IDENTITY_TOLERANCE`, which is the rounding a resample leaves behind.
    """
    import numpy as np

    cube = table.table_3d
    if cube is None:
        return False
    size = cube.shape[0]
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return bool(np.max(np.abs(cube - grid)) <= IDENTITY_TOLERANCE)


def _shape(table) -> str:
    """How a table is stored, in a word.

    Args:
        table: A :class:`lut.LUT`.

    Returns:
        ``3D cube``, ``1D curves`` or ``empty``.
    """
    if table.table_3d is not None:
        return "3D cube"
    if table.table_1d is not None:
        return "1D curves"
    return "empty"


def publish(table, node_id=None, strip=True, detail="") -> bool:
    """Store what a LUT node produced, for that node's own interface to fetch.

    Never raises.

    Args:
        table: A :class:`lut.LUT` the node answered with.
        node_id: The publishing node's graph id. Left out, the node ComfyUI is executing.
        strip: Grade and publish the reference chart. False files the facts alone, for a
            node whose panel states what the table is and draws no chart.
        detail: One extra line for the summary, such as how a blend was mixed.

    Returns:
        True when the facts were stored, and False when no browser is connected or no panel
        is open on the node, which is decided before the chart is graded.
    """
    if not run_result.watching():
        return False
    try:
        size = table.size()
        shape = _shape(table)
        facts = {
            "title": str(table.title or "untitled")[:64],
            "shape": shape,
            "size": f"{size}" if size else "0",
        }
        low = getattr(table, "domain_min", None)
        high = getattr(table, "domain_max", None)
        if low is not None and high is not None:
            span = (float(low[0]), float(high[0]))
            if span != (0.0, 1.0):
                facts["domain"] = f"{span[0]:g} to {span[1]:g}"

        if strip and size:
            from ..image import lut_preview

            preview.publish_output(lut_preview.graded_strip(table), node_id=node_id, slot=SLOT)

        summary = f"{shape} at {size}" if size else "no table"
        if size and _is_identity(table):
            facts["effect"] = "identity, no change"
            summary = f"{summary}, changes nothing"
        if detail:
            summary = f"{summary}, {detail}"
        return run_result.publish(summary=summary, facts=facts, node_id=node_id)
    except Exception as error:
        logger.debug("no lut report was published for node %s (%s)", node_id, error)
        return False
