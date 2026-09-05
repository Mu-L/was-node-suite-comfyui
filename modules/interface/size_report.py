"""What a geometry node did to a frame's size.

Both sizes publish under the node's own id, as up to four counts and six facts. A size is
``(width, height)``, in pixels or latent units.
"""

from __future__ import annotations

from .. import log
from . import run_result

__all__ = [
    "IMAGE",
    "LATENT",
    "PIXEL",
    "PLANE",
    "frame_size",
    "publish",
    "spell",
]

logger = log.get_logger("interface.size_report")

#: A tensor whose trailing axis counts channels, so the two axes before it are the height and
#: the width. This is the pack's ``IMAGE`` layout.
IMAGE = "image"

#: A tensor whose last two axes are the height and the width, which is a ``MASK``'s layout and a
#: latent's ``samples``.
PLANE = "plane"

#: A size counted in pixels.
PIXEL = "pixel"

#: A size counted in latent units, which is spelled out wherever the number is written so a
#: latent grid is not read as a pixel size.
LATENT = "latent"

#: How far the two axis factors may differ before they are written separately, as a fraction of
#: the larger of them. Half a percent is wider than one axis rounding to a whole pixel at any
#: size a node in this pack answers.
FACTOR_TOLERANCE = 0.005

#: How far the delivered aspect may sit from the source's before the fact says the shape moved,
#: as a fraction of the source ratio.
ASPECT_TOLERANCE = 0.005

#: The area at which the pixel fact switches from whole pixels to megapixels. Both sides of one
#: fact are written in the same unit, so the two numbers can be compared.
MEGAPIXEL = 1_000_000

#: The smallest area that gets a megapixel figure, which is the area that rounds to 0.01 at two
#: decimal places. A tile of 66 by 65 would read 0.00, and a figure nobody can read is worse
#: than one tile fewer, so a frame under this leaves the figure out.
MEGAPIXEL_FLOOR = 10_000


def frame_size(source, layout=IMAGE):
    """The width and height of whatever a geometry node was handed.

    Args:
        source: A ``(width, height)`` pair, a PIL image, or a tensor. A pair is taken as
            given, so a node reporting a recorded window passes the window rather than a
            picture of it.
        layout: :data:`IMAGE` to read a tensor's trailing axis as its channels, or
            :data:`PLANE` to read its last two axes as the height and the width.

    Returns:
        ``(width, height)`` as two ints, or None when the source carries no size.
    """
    if source is None:
        return None
    if isinstance(source, (tuple, list)) and len(source) == 2:
        try:
            return int(source[0]), int(source[1])
        except (TypeError, ValueError):
            return None
    # A PIL image holds its size as a pair; a torch tensor holds a method of the same name.
    size = getattr(source, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        try:
            return int(size[0]), int(size[1])
        except (TypeError, ValueError):
            return None
    shape = getattr(source, "shape", None)
    try:
        if shape is None or len(shape) < 2:
            return None
        if layout == IMAGE and len(shape) >= 3:
            return int(shape[-2]), int(shape[-3])
        return int(shape[-1]), int(shape[-2])
    except (TypeError, ValueError):
        return None


def spell(source, unit=PIXEL, layout=IMAGE) -> str:
    """A size as the readout writes it.

    Args:
        source: Anything :func:`frame_size` reads a size out of.
        unit: :data:`PIXEL` or :data:`LATENT`.
        layout: :data:`IMAGE` or :data:`PLANE`, passed to :func:`frame_size`.

    Returns:
        ``1024x1536``, or ``64x96 latent`` for a latent grid. Empty where the source
        carries no size, so a fact is dropped rather than carrying a made-up number.
    """
    size = frame_size(source, layout)
    if size is None:
        return ""
    width, height = size
    return f"{width}x{height} latent" if unit == LATENT else f"{width}x{height}"


def _factor(before, after) -> str:
    """The two axis multipliers, written as one when they agree.

    Args:
        before: The source ``(width, height)``.
        after: The delivered ``(width, height)``.

    Returns:
        ``x2.00``, or ``x1.50 wide, x2.00 tall`` when the axes were scaled differently.
    """
    wide = after[0] / before[0] if before[0] else 0.0
    tall = after[1] / before[1] if before[1] else 0.0
    # Two multipliers that print alike are written once: a row reading
    # "x0.67 wide, x0.67 tall" states a difference the reader cannot see.
    if f"{wide:.2f}" == f"{tall:.2f}":
        return f"x{wide:.2f}"
    return f"x{wide:.2f} wide, x{tall:.2f} tall"


def _aspect(before, after) -> str:
    """Whether the shape survived the change, and what it became.

    Args:
        before: The source ``(width, height)``.
        after: The delivered ``(width, height)``.

    Returns:
        ``1.50 kept``, or ``1.50 to 1.00`` when the proportions moved.
    """
    source = before[0] / before[1] if before[1] else 0.0
    result = after[0] / after[1] if after[1] else 0.0
    if abs(result - source) <= ASPECT_TOLERANCE * max(source, 1.0):
        return f"{source:.2f} kept"
    return f"{source:.2f} to {result:.2f}"


def _area(before, after) -> str:
    """How many pixels the frame held and how many it holds now.

    Args:
        before: The source ``(width, height)``.
        after: The delivered ``(width, height)``.

    Returns:
        ``2.07 MP to 0.79 MP``, or a whole pixel count on both sides where neither frame
        reaches a megapixel.
    """
    source = before[0] * before[1]
    result = after[0] * after[1]
    if max(source, result) >= MEGAPIXEL:
        return f"{source / MEGAPIXEL:.2f} MP to {result / MEGAPIXEL:.2f} MP"
    return f"{source:,} px to {result:,} px"


def _scale(before, after) -> float:
    """One number for how much bigger the frame is, area for area.

    Args:
        before: The source ``(width, height)``.
        after: The delivered ``(width, height)``.

    Returns:
        The square root of the area ratio, so a doubling of both sides reads 2.0. 0.0 where
        the source had no area to scale.
    """
    source = before[0] * before[1]
    if source <= 0:
        return 0.0
    return round((after[0] * after[1] / source) ** 0.5, 3)


def _summary(before, after, action, requested, resampled, unit) -> str:
    """What the node did to the size, in one line.

    Args:
        before: The source ``(width, height)``.
        after: The delivered ``(width, height)``.
        action: The past participle naming what the node did.
        requested: The size the node was asked for, or None.
        resampled: True when the source was brought to the delivered size to make it fit.
        unit: :data:`PIXEL` or :data:`LATENT`.

    Returns:
        One sentence, written for the person running the pack.
    """
    source, result = spell(before, unit), spell(after, unit)
    if requested is not None and tuple(requested) != tuple(after):
        return f"{source} {action} to {result}, {spell(requested, unit)} requested"
    if resampled and tuple(before) != tuple(after):
        return f"the {source} input was resampled to the {result} window"
    if tuple(before) == tuple(after):
        return f"{result} {action}, the size is unchanged"
    return f"{source} {action} to {result}"


def publish(before, after, action="resized", requested=None, unit=PIXEL, layout=IMAGE,
            resampled=False, refused=None, facts=None, node_id=None) -> bool:
    """Store what a geometry node did to a frame's size, for its own interface to fetch.

    Args:
        before: The source, as :func:`frame_size` reads one.
        after: What the node delivered, as :func:`frame_size` reads one.
        action: The past participle naming what the node did, such as ``cropped`` or
            ``pasted``, which the summary line is built around.
        requested: The size the node was asked to deliver, where a node is asked for one.
            Given and different from ``after``, the report is a warning and carries an
            ``asked`` fact, which is how a rounded or clamped size is stated rather than
            hidden.
        unit: :data:`PIXEL` for a picture, :data:`LATENT` for a latent grid, which is
            spelled beside every number and drops the pixel facts.
        layout: :data:`IMAGE` or :data:`PLANE`, passed to :func:`frame_size` for both sides.
        resampled: True when the source was brought to the delivered size to make it fit a
            window the node did not choose. With two sizes that differ, the report is a
            warning naming the resample.
        refused: What stopped the node producing anything. Makes the report a warning and
            replaces the summary line.
        facts: Anything further worth a row, as a mapping of name to value, merged after the
            report's own. Two fit; the ones past that are dropped and named in the log.
        node_id: The publishing node's graph id. Left out, the id of the node ComfyUI is
            executing, so a node needs no hidden input to report itself.

    Returns:
        Whatever :func:`run_result.publish` answered, which is False when no browser is
        connected and the readout is not worth building.
    """
    if not run_result.watching():
        return False
    try:
        source = frame_size(before, layout)
        result = frame_size(after, layout)
        if source is None or result is None:
            logger.debug(
                "a size report was published with no size on one side (%r, %r)", before, after
            )
            return False
        want = frame_size(requested, layout) if requested is not None else None

        own = {
            "in": spell(source, unit),
            "out": spell(result, unit),
            "factor": _factor(source, result),
            "aspect": _aspect(source, result),
        }
        if unit != LATENT:
            own["pixels"] = _area(source, result)
        if want is not None and want != result:
            own["asked"] = spell(want, unit)
        merged = dict(own)
        merged.update(facts or {})
        if len(merged) > run_result.MAX_FACTS:
            logger.debug(
                "a size report carries %d facts and %d are drawn, so %s is left out",
                len(merged),
                run_result.MAX_FACTS,
                ", ".join(list(merged)[run_result.MAX_FACTS:]),
            )

        counts = {"width": result[0], "height": result[1], "scale": _scale(source, result)}
        area = result[0] * result[1]
        if unit != LATENT and area >= MEGAPIXEL_FLOOR:
            counts["megapixels"] = round(area / MEGAPIXEL, 2)

        warned = bool(refused) or (want is not None and want != result) \
            or (resampled and source != result)
        return run_result.publish(
            status=run_result.WARNING if warned else run_result.OK,
            summary=refused or _summary(source, result, action, want, resampled, unit),
            counts=counts,
            facts=merged,
            node_id=node_id,
        )
    except Exception as error:
        logger.debug("a size report could not be built (%s)", error)
        return False
