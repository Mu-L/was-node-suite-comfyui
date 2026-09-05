"""Measurements of a mask, as the numbers a readout draws.

Coverage is a percentage of the frame area over every frame. A box is ``(left, top, right,
bottom)``, right and bottom exclusive, measured on one frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .. import log
from ..convert.tensors import mask_planes

__all__ = [
    "BINARY_LEVEL",
    "CONNECTIVITY",
    "REGION_MAX_PIXELS",
    "Change",
    "Reading",
    "bounding_box",
    "compare",
    "planes",
    "read",
]

logger = log.get_logger("mask.measure")

#: Value a sample must exceed to count as set, which is the level the pack's own region
#: operations threshold at.
BINARY_LEVEL = 0.5

#: Neighbourhood one connected region is counted over, matching what
#: :func:`~modules.mask.regions.dominant_region`,
#: :func:`~modules.mask.regions.minority_region` and the two crop functions pick a region
#: with. :func:`~modules.mask.regions.arbitrary_region` labels at 8.
CONNECTIVITY = 4

#: Samples in one frame above which the region pass does not run.
REGION_MAX_PIXELS = 4096 * 4096


@dataclass(frozen=True)
class Reading:
    """Everything one mask tensor holds, in the units a readout prints.

    Attributes:
        frames: How many masks the tensor carries.
        height: Frame height in pixels.
        width: Frame width in pixels.
        pixels: Samples in the whole tensor, ``frames x height x width``.
        coverage: Percentage of samples above :data:`BINARY_LEVEL`, to two decimals.
        soft: Percentage the clamped values average to, to two decimals.
        set_pixels: Samples above :data:`BINARY_LEVEL`, counted exactly.
        partial: Samples strictly between 0 and 1, counted exactly.
        lowest: Smallest value in the tensor, to three decimals.
        highest: Largest value in the tensor, to three decimals.
        box: ``(left, top, right, bottom)`` around what frame :attr:`measured` sets, right
            and bottom exclusive, or None where that frame sets nothing.
        regions: Connected regions of frame :attr:`measured`, or None where they were not
            counted.
        largest: Percentage of that frame's set area held by its largest region, to two
            decimals, or None where there is no region to measure.
        region_note: Why :attr:`regions` was not counted, written for the person running the
            pack. Empty where it was.
        per_frame: One binary coverage percentage per frame, in batch order.
        measured: Index of the frame :attr:`box`, :attr:`regions` and :attr:`largest` were
            taken from.
    """

    frames: int
    height: int
    width: int
    pixels: int
    coverage: float
    soft: float
    set_pixels: int
    partial: int
    lowest: float
    highest: float
    box: tuple[int, int, int, int] | None
    regions: int | None
    largest: float | None
    region_note: str
    per_frame: tuple[float, ...]
    measured: int


@dataclass(frozen=True)
class Change:
    """What one mask operation set and cleared.

    Attributes:
        comparable: True when the two tensors share a frame count and a frame size, which is
            what makes a pixelwise difference defined.
        reason: Why they cannot be compared, written for the person running the pack. Empty
            when they can.
        added: Percentage of the frame area set in the result and not in the input, to two
            decimals. 0.0 when the two cannot be compared.
        removed: Percentage set in the input and not in the result, to two decimals.
        added_pixels: Samples set in the result and not in the input, counted exactly.
        removed_pixels: Samples set in the input and not in the result, counted exactly.
    """

    comparable: bool
    reason: str
    added: float
    removed: float
    added_pixels: int
    removed_pixels: int


def planes(masks: torch.Tensor) -> list[torch.Tensor]:
    """Split a mask tensor into one two-axis mask per item of its batch.

    Args:
        masks: Mask tensor in any of the layouts a ``MASK`` socket carries. Split by
            :func:`modules.convert.tensors.mask_planes`, which reads three axes or more as a
            batch, two as a single unbatched mask and fewer as one row.

    Returns:
        One ``(height, width)`` view per mask, in batch order, and none for a batch holding
        no mask. An axis left over after the channel axis is dropped becomes more frames
        rather than a plane with three axes, so every figure measured below is a figure of
        one frame.
    """
    found: list[torch.Tensor] = []
    for plane in mask_planes(masks):
        if plane.ndim == 2:
            found.append(plane)
        else:
            found.extend(planes(plane))
    return found


def bounding_box(plane: torch.Tensor) -> tuple[int, int, int, int] | None:
    """The tightest rectangle around what one mask plane sets.

    Args:
        plane: A ``(height, width)`` mask.

    Returns:
        ``(left, top, right, bottom)``, right and bottom exclusive, or None where nothing in
        the plane is above :data:`BINARY_LEVEL`.
    """
    marked = _floats(plane) > BINARY_LEVEL
    rows = torch.nonzero(torch.any(marked, dim=1)).flatten()
    if rows.numel() == 0:
        return None
    columns = torch.nonzero(torch.any(marked, dim=0)).flatten()
    return (int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)


def read(masks: torch.Tensor, frame: int = 0, components: bool = True) -> Reading:
    """Measure one mask tensor.

    Args:
        masks: Mask tensor in any of the layouts a ``MASK`` socket carries.
        frame: Which frame the box and the region figures are taken from. Out of range is
            held to the last frame.
        components: False to skip the region pass, leaving :attr:`Reading.regions` and
            :attr:`Reading.largest` unset.

    Returns:
        A :class:`Reading`. A tensor carrying no mask reads as zero frames of 0x0, with
        every figure 0, no box and no region count.
    """
    found = planes(masks)
    height = int(found[0].shape[0]) if found else 0
    width = int(found[0].shape[1]) if found else 0
    frames = len(found)
    area = height * width
    pixels = area * frames
    chosen = max(0, min(int(frame), frames - 1))

    marked, summed, lowest, highest, partial = _reduce(found, area)
    set_pixels = int(marked.sum().item())
    regions, largest, note = (None, None, "")
    if components and area:
        regions, largest, note = _regions(found[chosen])

    return Reading(
        frames=frames,
        height=height,
        width=width,
        pixels=pixels,
        coverage=_percent(set_pixels, pixels),
        soft=_percent(summed, pixels),
        set_pixels=set_pixels,
        partial=partial,
        lowest=round(lowest, 3),
        highest=round(highest, 3),
        box=bounding_box(found[chosen]) if area else None,
        regions=regions,
        largest=largest,
        region_note=note,
        per_frame=tuple(_percent(int(count), area) for count in marked.tolist()),
        measured=chosen,
    )


def compare(before: torch.Tensor, after: torch.Tensor) -> Change:
    """Measure what one operation set and cleared, where that is defined.

    Args:
        before: The mask the node was handed.
        after: The mask the node answered.

    Returns:
        A :class:`Change`. Two tensors differing in frame count or frame size, and two
        carrying no mask at all, answer one whose ``comparable`` is False and whose
        ``reason`` says which, since a pixelwise difference between two shapes is not
        defined.
    """
    first = planes(before)
    second = planes(after)
    if len(first) != len(second):
        return _incomparable(
            f"{len(first)} frame(s) in and {len(second)} out, so no pixel was compared"
        )
    if not first:
        return _incomparable("no frame in and none out, so no pixel was compared")
    if tuple(first[0].shape) != tuple(second[0].shape):
        return _incomparable(
            f"{_size(first[0])} in and {_size(second[0])} out, so no pixel was compared"
        )

    rows = []
    for was, now in zip(first, second):
        started = _floats(was) > BINARY_LEVEL
        ended = _floats(now) > BINARY_LEVEL
        rows.append(torch.stack([(ended & ~started).sum(), (started & ~ended).sum()]))
    counted = torch.stack(rows).cpu()
    added = int(counted[:, 0].sum().item())
    removed = int(counted[:, 1].sum().item())
    pixels = int(first[0].numel()) * len(first)
    return Change(
        comparable=True,
        reason="",
        added=_percent(added, pixels),
        removed=_percent(removed, pixels),
        added_pixels=added,
        removed_pixels=removed,
    )


def _floats(plane: torch.Tensor) -> torch.Tensor:
    """A mask plane in a dtype the reductions here neither overflow nor refuse.

    Args:
        plane: One mask, of any dtype a ``MASK`` socket carries.

    Returns:
        The plane itself where it is already float32 or float64, and a float32 copy
        otherwise. A boolean plane refuses comparison against a float level, and a float16
        one overflows its own sum at four million set samples.
    """
    if plane.dtype in (torch.float32, torch.float64):
        return plane
    return plane.to(torch.float32)


def _reduce(found: list[torch.Tensor], area: int) -> tuple[torch.Tensor, float, float, float, int]:
    """Every whole-tensor figure, in two host transfers however many frames there are.

    Args:
        found: One two-axis mask per frame.
        area: Samples in one frame, 0 for a frame with no extent.

    Returns:
        ``(set counts per frame, clamped total, lowest, highest, partial count)``, the counts
        exact as int64, the total summed in float64 and the rest as python floats.
    """
    if not area:
        empty = torch.zeros(len(found), dtype=torch.int64)
        return empty, 0.0, 0.0, 0.0, 0

    counts, totals, lows, highs, parts = [], [], [], [], []
    for plane in found:
        values = _floats(plane)
        counts.append((values > BINARY_LEVEL).sum())
        parts.append(((values > 0) & (values < 1)).sum())
        # Values sum in float64, and the range is cast to it so one stack carries all three.
        totals.append(values.clamp(0.0, 1.0).sum(dtype=torch.float64))
        lows.append(values.min().to(torch.float64))
        highs.append(values.max().to(torch.float64))
    # Counts stay int64 through the transfer: a fully set 8192 square holds 67 million
    # samples, which float32 cannot represent exactly.
    tallied = torch.stack([torch.stack(counts), torch.stack(parts)]).cpu()
    gathered = torch.stack([torch.stack(totals), torch.stack(lows), torch.stack(highs)]).cpu()
    return (
        tallied[0],
        float(gathered[0].sum(dtype=torch.float64).item()),
        float(gathered[1].min().item()),
        float(gathered[2].max().item()),
        int(tallied[1].sum().item()),
    )


def _regions(plane: torch.Tensor) -> tuple[int | None, float | None, str]:
    """How many connected regions one mask plane sets, and the share the largest holds.

    Args:
        plane: A ``(height, width)`` mask.

    Returns:
        ``(regions, largest, note)``, the share as a percentage of the plane's set area and
        the note saying why there is no count. A count that was not taken costs the region
        figures and no other part of a reading.
    """
    if plane.numel() > REGION_MAX_PIXELS:
        side = int(REGION_MAX_PIXELS ** 0.5)
        return None, None, f"region counts stop above {side}x{side}"
    try:
        areas = _areas(_floats(plane) > BINARY_LEVEL)
    except Exception as error:
        logger.debug("no region count was measured (%s)", error)
        return None, None, "region counts could not be measured"
    if not areas.numel():
        return 0, None, ""
    return (
        int(areas.numel()),
        _percent(int(areas.max().item()), int(areas.sum().item())),
        "",
    )


def _areas(marked: torch.Tensor) -> torch.Tensor:
    """The set area of every connected region of one binary plane.

    Args:
        marked: A ``(height, width)`` boolean plane, on any device.

    Returns:
        One area in samples per region, in no order, and an empty tensor for a plane that
        sets nothing. Regions are joined at :data:`CONNECTIVITY`.
    """
    starts = marked.clone()
    starts[:, 1:] &= ~marked[:, :-1]
    total = int(starts.sum())
    if not total:
        return torch.zeros(0, dtype=torch.int64, device=marked.device)
    runs = (torch.cumsum(starts.reshape(-1).to(torch.int64), 0) - 1).reshape(marked.shape)
    parent = _merge(total, *_joins(marked, starts, runs))
    lengths = torch.bincount(runs[marked], minlength=total)
    gathered = torch.zeros(total, dtype=torch.int64, device=marked.device)
    gathered.scatter_reduce_(0, parent, lengths, reduce="sum")
    return gathered[gathered > 0]


def _joins(
    marked: torch.Tensor, starts: torch.Tensor, runs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every pair of horizontal runs that one adjacent sample joins.

    Args:
        marked: A ``(height, width)`` boolean plane.
        starts: True where a run begins, the same shape as ``marked``.
        runs: Index of the run every sample belongs to, the same shape as ``marked``.

    Returns:
        ``(left, right)``, one run index in each per join. A join another join already
        implies is left out, and diagonal joins are added at 8-connectivity.
    """
    under = marked[:-1] & marked[1:] & (starts[:-1] | starts[1:])
    left = [runs[:-1][under]]
    right = [runs[1:][under]]
    if CONNECTIVITY != 4:
        falling = marked[:-1, :-1] & marked[1:, 1:] & ~marked[:-1, 1:] & ~marked[1:, :-1]
        left.append(runs[:-1, :-1][falling])
        right.append(runs[1:, 1:][falling])
        rising = marked[:-1, 1:] & marked[1:, :-1] & ~marked[:-1, :-1] & ~marked[1:, 1:]
        left.append(runs[:-1, 1:][rising])
        right.append(runs[1:, :-1][rising])
    return torch.cat(left), torch.cat(right)


def _merge(total: int, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Gather runs into components by hooking each pair together and compressing.

    Args:
        total: How many runs there are.
        left: One end of every join, as a run index.
        right: The other end of every join, as a run index.

    Returns:
        A ``(total,)`` int64 tensor holding, for every run, the lowest run index of the
        component it belongs to.
    """
    parent = torch.arange(total, dtype=torch.int64, device=left.device)
    while True:
        first = parent.gather(0, left)
        second = parent.gather(0, right)
        if torch.equal(first, second):
            return parent
        parent.scatter_reduce_(
            0, torch.maximum(first, second), torch.minimum(first, second), reduce="amin"
        )
        while True:
            jumped = parent.gather(0, parent)
            if torch.equal(jumped, parent):
                break
            parent = jumped


def _incomparable(reason: str) -> Change:
    """A change carrying only why the two masks could not be compared."""
    return Change(comparable=False, reason=reason, added=0.0, removed=0.0,
                  added_pixels=0, removed_pixels=0)


def _size(plane: torch.Tensor) -> str:
    """One plane's frame size, width first."""
    return f"{int(plane.shape[1])}x{int(plane.shape[0])}"


def _percent(part, whole) -> float:
    """A share of a whole as a percentage to two decimals, 0.0 where the whole is empty."""
    return round(100.0 * float(part) / float(whole), 2) if whole else 0.0
