"""Coverage and value figures for a mask, as the numbers a node emits.

Coverage is a fraction of the pixels measured, 0.0 to 1.0. A pixel counts as covered when
its value is above the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .measure import planes

__all__ = [
    "DEFAULT_THRESHOLD",
    "WHOLE_BATCH",
    "Statistics",
    "measure",
    "resolve",
    "summarise",
]

#: Value a pixel is compared against when no threshold is given, matching the level the
#: pack's region operations threshold at.
DEFAULT_THRESHOLD = 0.5

#: Index that measures every mask of the batch together.
WHOLE_BATCH = -1


@dataclass(frozen=True)
class Statistics:
    """What one mask tensor holds, in the units a node emits.

    Attributes:
        coverage: Fraction of the pixels measured that are above the threshold, 0.0 to 1.0.
        covered_pixels: Pixels above the threshold, counted exactly.
        total_pixels: Pixels measured, the frame area times however many frames were.
        lowest: Smallest value measured.
        highest: Largest value measured.
        mean: Average of every value measured, before the threshold.
        is_empty: True where no pixel is above the threshold.
        batch_size: Masks the tensor carries, whatever index selected.
        width: Frame width in pixels.
        height: Frame height in pixels.
        threshold: Value each pixel was compared against.
        measured: Position of the mask measured, or :data:`WHOLE_BATCH` for all of them.
    """

    coverage: float
    covered_pixels: int
    total_pixels: int
    lowest: float
    highest: float
    mean: float
    is_empty: bool
    batch_size: int
    width: int
    height: int
    threshold: float
    measured: int


def resolve(index: int, count: int, out_of_range: str, node: str) -> int:
    """Turn a requested index into a mask's position in the batch.

    Args:
        index: The requested position, counting from 0.
        count: How many masks the batch holds, which is one or more.
        out_of_range: ``wrap``, ``clamp`` or ``error``.
        node: The node's display name, opening the message where the index is refused.

    Returns:
        A position from 0 to ``count - 1``.

    Raises:
        ValueError: The index is outside the batch and ``out_of_range`` is ``error``.
    """
    # Negatives count from the end, as everywhere else in the pack. WHOLE_BATCH never
    # reaches here: the node turns its scope widget into that sentinel before calling.
    if index < 0:
        index = index + count

    if 0 <= index < count:
        return index
    if out_of_range == "wrap":
        return index % count
    if out_of_range == "clamp":
        return 0 if index < 0 else count - 1
    raise ValueError(
        f"{node} was asked for mask {index} of a batch holding {count}, numbered 0 to "
        f"{count - 1}. Set scope to the whole batch, or set out_of_range to wrap or "
        f"clamp."
    )


def measure(
    masks: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    index: int = 0,
    out_of_range: str = "error",
    node: str = "Mask Statistics",
    whole: bool = True,
) -> Statistics:
    """Measure a mask tensor, whole or one mask of its batch.

    Args:
        masks: Mask tensor in any of the layouts a ``MASK`` socket carries.
        threshold: Value a pixel must be above to count as covered.
        index: Which mask to measure, counting from 0, negatives from the end.
        out_of_range: ``wrap``, ``clamp`` or ``error``.
        node: The node's display name, opening any message raised.
        whole: Measure every mask together and ignore ``index``.

    Returns:
        A :class:`Statistics`.

    Raises:
        ValueError: The tensor carries no mask, or the index is outside the batch and
            ``out_of_range`` is ``error``.
    """
    found = planes(masks)
    if not found:
        raise ValueError(
            f"{node} was given a mask batch holding no masks, so there is nothing to "
            f"measure. Check the node feeding mask, and that its batch holds one mask or "
            f"more."
        )

    count = len(found)
    height = int(found[0].shape[0])
    width = int(found[0].shape[1])
    # An explicit flag rather than a sentinel index: -1 is the last mask everywhere in the
    # pack, so a sentinel of -1 would silently answer the whole batch for a valid request.
    if whole:
        chosen, measured = found, WHOLE_BATCH
    else:
        measured = resolve(int(index), count, out_of_range, node)
        chosen = [found[measured]]

    level = float(threshold)
    covered, total, lowest, highest, summed = _reduce(chosen, level)
    return Statistics(
        coverage=covered / total if total else 0.0,
        covered_pixels=covered,
        total_pixels=total,
        lowest=lowest,
        highest=highest,
        mean=summed / total if total else 0.0,
        is_empty=covered == 0,
        batch_size=count,
        width=width,
        height=height,
        threshold=level,
        measured=measured,
    )


def summarise(stats: Statistics) -> str:
    """Write every figure of a reading on one line.

    Args:
        stats: A reading from :func:`measure`.

    Returns:
        The figures as ``name=value`` pairs, opening with which mask was measured.
    """
    where = "all" if stats.measured == WHOLE_BATCH else str(stats.measured)
    return (
        f"index={where}  batch_size={stats.batch_size}  {stats.width}x{stats.height}  "
        f"threshold={stats.threshold:.3f}  coverage={stats.coverage * 100:.2f}%  "
        f"covered={stats.covered_pixels}/{stats.total_pixels}  min={stats.lowest:.4f}  "
        f"max={stats.highest:.4f}  mean={stats.mean:.4f}  "
        f"is_empty={'true' if stats.is_empty else 'false'}"
    )


def _reduce(frames: list[torch.Tensor], threshold: float) -> tuple[int, int, float, float, float]:
    """Every whole-tensor figure, in two host transfers however many frames there are.

    Args:
        frames: The mask planes being measured, in batch order.
        threshold: Value a pixel must be above to count as covered.

    Returns:
        ``(covered pixels, pixels measured, lowest, highest, total value)``, the count exact
        as int64, the total summed in float64 and the rest as python floats.
    """
    total = sum(int(plane.numel()) for plane in frames)
    if not total:
        return 0, 0, 0.0, 0.0, 0.0

    counts, sums, lows, highs = [], [], [], []
    for plane in frames:
        values = plane.to(torch.float32)
        counts.append((values > threshold).sum())
        # Values sum in float64, and the range is cast to it so one stack carries all three.
        sums.append(values.sum(dtype=torch.float64))
        lows.append(values.min().to(torch.float64))
        highs.append(values.max().to(torch.float64))
    # Counts stay int64 through the transfer: a fully covered 8192 square holds 67 million
    # pixels, which float32 cannot represent exactly.
    tallied = torch.stack(counts).cpu()
    gathered = torch.stack([torch.stack(sums), torch.stack(lows), torch.stack(highs)]).cpu()
    return (
        int(tallied.sum().item()),
        total,
        float(gathered[1].min().item()),
        float(gathered[2].max().item()),
        float(gathered[0].sum(dtype=torch.float64).item()),
    )
