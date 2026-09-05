"""What a loop is carrying right now, as something its node can draw.

An End node publishes this once per iteration, filed under the original node id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..interface import run_result

#: Longest a summarised value may be before it is cut. Long enough for a prompt fragment to be
#: recognisable, short enough that eight of them fit on a node.
MAX_VALUE_CHARS = 72

#: What a slot holding nothing reads as, rather than an empty line that looks like a bug.
NOT_CONNECTED = "not connected"

#: The socket types a shaped value is named as, so a shape does not have to be decoded to see
#: what the slot is carrying.
IMAGE = "IMAGE"
MASK = "MASK"
LATENT = "LATENT"
TENSOR = "TENSOR"


def describe(value) -> str:
    """One short line naming what a value is, for a readout rather than for a log.

    Args:
        value: Anything a carried slot might hold.

    Returns:
        A line of at most :data:`MAX_VALUE_CHARS` characters.
    """
    try:
        return _cut(_describe(value))
    except Exception:
        return type(value).__name__


def _describe(value) -> str:
    if value is None:
        return NOT_CONNECTED
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value}"
    if isinstance(value, str):
        return f'"{value}"' if value else '"" (empty)'

    described = _describe_tensor(value)
    if described is not None:
        return described

    if isinstance(value, Mapping):
        # A latent is a mapping, so it is read through its samples rather than counted as keys.
        samples = _describe_tensor(value.get("samples"))
        if samples is not None:
            return f"{LATENT} {samples.split(' ', 1)[1]}" if " " in samples else LATENT
        return f"DICT, {len(value)} key(s)"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__.upper()}, {len(value)} item(s)"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return f"{type(value).__name__}, {len(value)} item(s)"
    return type(value).__name__


def _tensor_kind(dims: list[int]) -> str:
    """Which socket type a tensor of this shape is carried on.

    Args:
        dims: The tensor's shape.

    Returns:
        One of :data:`IMAGE`, :data:`MASK` or :data:`TENSOR`. A four dimensional tensor whose
        last axis holds 1, 3 or 4 channels is an image; a three dimensional one is a mask;
        anything else is named for what it is rather than guessed at.
    """
    if len(dims) == 4 and dims[3] in (1, 3, 4):
        return IMAGE
    if len(dims) == 3:
        return MASK
    return TENSOR


def _describe_tensor(value) -> str | None:
    """``value`` as a socket type, a shape and a dtype, or None when it carries no shape.

    Args:
        value: Anything that might be a tensor.

    Returns:
        A line such as ``IMAGE 1x512x512x3 float32``, or None.
    """
    if value is None or isinstance(value, (str, bytes)):
        return None
    # Duck-typed rather than imported: torch must not be imported for a readout, and anything
    # else carrying a shape and a dtype, a numpy array, reads the same way.
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dims = [int(n) for n in shape]
    except (TypeError, ValueError):
        return None
    dtype = getattr(value, "dtype", None)
    spelled = str(dtype).replace("torch.", "") if dtype is not None else type(value).__name__
    if not dims:
        return f"{TENSOR} scalar {spelled}"
    return f"{_tensor_kind(dims)} {'x'.join(str(n) for n in dims)} {spelled}"


def _cut(text: str) -> str:
    """``text`` held to :data:`MAX_VALUE_CHARS`, with an ellipsis where it was cut."""
    flat = " ".join(str(text).split())
    if len(flat) <= MAX_VALUE_CHARS:
        return flat
    return flat[: MAX_VALUE_CHARS - 1] + run_result.ELLIPSIS


def publish_iteration(node_id, summary, counts, slots, status=run_result.OK) -> bool:
    """Store one iteration's state under the loop's original End node id.

    Args:
        node_id: The *original* End node's graph id, stable for the whole run. Passed rather
            than read from the execution context, which on every iteration but the first names the
            ephemeral clone doing the work instead of the node on the canvas.
        summary: One line saying where the loop is up to.
        counts: Named numbers, such as the iteration and the total.
        slots: The carried values in order, as ``(name, value)`` pairs. Only those holding
            something are listed, so an unwired slot costs the readout no room.
        status: One of ``run_result.STATUSES``.

    Returns:
        Whatever :func:`run_result.publish` answered, which is False when no browser is
        connected and the readout is not worth building.
    """
    facts = {}
    for name, value in slots:
        if value is None:
            continue
        facts[name] = describe(value)
    if not facts:
        facts["slots"] = "none wired"
    return run_result.publish(
        status=status,
        summary=summary,
        counts=counts,
        facts=facts,
        node_id=node_id,
    )
