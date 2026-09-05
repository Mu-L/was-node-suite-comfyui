"""Broadcast-style waveform scopes: one column of the plot per column of the picture.

The horizontal grid is labelled in IRE, the broadcast scale where 0 is black and 100 is
peak white.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..data import paths

__all__ = [
    "PANEL_PLOT_HEIGHT",
    "PANEL_PLOT_WIDTH",
    "STATS_COLUMNS",
    "STATS_FIELD",
    "STATS_LABEL_FIELD",
    "compose_panel_parade",
    "compose_parade",
    "compose_waveform_panel",
    "make_waveform_gray",
    "parade_stats_line",
    "rail_fractions",
    "stats_header",
    "stats_line",
    "stats_row",
    "stats_tensor",
]

#: What each field of :func:`stats_row` holds, in the order it is written.
STATS_COLUMNS = ("min", "max", "mean", "std", "median")

#: Characters each figure of :func:`stats_row` and each name of :func:`stats_header` occupy,
#: right aligned, which is what lines a row up under its header.
STATS_FIELD = 9

#: Characters the channel label of :func:`stats_row` occupies, left aligned.
STATS_LABEL_FIELD = 2

#: IRE levels the horizontal grid is drawn and labelled at.
IRE_LINES = (0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100)

#: Width in pixels reserved on the left of a plot for the IRE labels.
LEFT_PAD = 56

#: Height in pixels reserved under a single-channel plot for its statistics line.
PANEL_PAD = 36

#: Height in pixels reserved under a parade for its three statistics lines.
PARADE_PAD = 72

#: Gap in pixels between the three plots of a parade.
PARADE_GAP = 8

#: Columns each channel of :func:`compose_panel_parade` is resampled to.
PANEL_PLOT_WIDTH = 240

#: Levels :func:`compose_panel_parade` draws over, which is its height in pixels.
PANEL_PLOT_HEIGHT = 256

#: Shortest waveform a plot is drawn at, whatever height was asked for.
MIN_HEIGHT = 128

#: Point size the statistics under a parade are drawn at.
STATS_POINT_SIZE = 14

#: Colour each channel's trace and its statistics line are drawn in.
CHANNEL_COLOURS = {
    "red": ((255, 0, 0), (255, 64, 64)),
    "green": ((0, 255, 0), (64, 255, 64)),
    "blue": ((0, 0, 255), (64, 128, 255)),
}


def stats_tensor(channel: torch.Tensor) -> tuple[float, float, float, float, float]:
    """Summarise one channel of one frame.

    Args:
        channel: Any tensor of samples; it is flattened first.

    Returns:
        ``(minimum, maximum, mean, standard deviation, median)`` as plain floats. The
        deviation is the population one, so a single-pixel image gives 0.0 rather than a
        division by zero.
    """
    values = channel.flatten()
    return (
        float(values.min().item()),
        float(values.max().item()),
        float(values.mean().item()),
        float(values.std(unbiased=False).item()),
        float(values.median().item()),
    )


def stats_line(stats: tuple[float, float, float, float, float]) -> str:
    """Format one channel's statistics as the line drawn under its plot."""
    return (
        f"min {stats[0]:.4f}  max {stats[1]:.4f}  mean {stats[2]:.4f}  "
        f"std {stats[3]:.4f}  median {stats[4]:.4f}"
    )


def parade_stats_line(label: str, stats: tuple[float, float, float, float, float]) -> str:
    """Format one channel's statistics with its ``R``, ``G`` or ``B`` label in front."""
    return f"{label}  {stats_line(stats)}"


def stats_row(label: str, stats: tuple[float, float, float, float, float]) -> str:
    """Format one channel's statistics as a fixed-width row under :func:`stats_header`.

    Args:
        label: What to call the channel, such as ``"R"``.
        stats: A summary from :func:`stats_tensor`.

    Returns:
        The label in :data:`STATS_LABEL_FIELD` characters, then the five values in
        :data:`STATS_COLUMNS` order, each to four decimal places in :data:`STATS_FIELD`
        characters. 47 characters for every value between -999.9999 and 9999.9999.
    """
    figures = "".join(f"{value:{STATS_FIELD}.4f}" for value in stats)
    return f"{label:<{STATS_LABEL_FIELD}}{figures}"


def stats_header() -> str:
    """Format the column names the figures of :func:`stats_row` are written under.

    Returns:
        :data:`STATS_COLUMNS` right aligned over the fields of a row, behind a gap the width
        of the channel label. 47 characters, the width of a row.
    """
    names = "".join(f"{name:>{STATS_FIELD}}" for name in STATS_COLUMNS)
    return f"{'':<{STATS_LABEL_FIELD}}{names}"


def rail_fractions(channel: torch.Tensor) -> tuple[float, float]:
    """How much of one channel sits on the black rail and how much on the white one.

    Args:
        channel: Any tensor of samples; it is flattened first.

    Returns:
        ``(fraction at or below 0, fraction at or above 1)``, each 0.0 to 1.0. An empty
        tensor answers ``(0.0, 0.0)``.
    """
    values = channel.flatten()
    total = values.numel()
    if total == 0:
        return 0.0, 0.0
    return (
        float((values <= 0.0).sum().item()) / total,
        float((values >= 1.0).sum().item()) / total,
    )


def _font():
    """The small bitmap font the IRE labels are drawn in."""
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _stats_font():
    """The font the statistics lines are drawn in."""
    try:
        return ImageFont.truetype(str(paths.font_file()), size=STATS_POINT_SIZE)
    except Exception:
        return _font()


def make_waveform_gray(channel: np.ndarray, out_h: int) -> np.ndarray:
    """Turn one channel of an image into a normalised waveform plot.

    Args:
        channel: ``(height, width)`` array of samples in ``[0, 1]``.
        out_h: Requested plot height in pixels, raised to :data:`MIN_HEIGHT` if lower.

    Returns:
        A ``(out_h, width)`` float array in ``[0, 1]``, row 0 being peak white. Counts are
        smoothed one row up and down before the log compression, which turns the single
        scan lines a flat gradient produces into a readable trace.
    """
    _height, width = channel.shape
    out_h = max(int(out_h), MIN_HEIGHT)
    levels = np.clip((channel * (out_h - 1)).astype(np.int32), 0, out_h - 1)
    plot = np.zeros((out_h, width), dtype=np.float32)
    for column in range(width):
        counts = np.bincount(levels[:, column], minlength=out_h).astype(np.float32)
        if out_h >= 3:
            counts[1:-1] = counts[1:-1] * 0.5 + (counts[:-2] + counts[2:]) * 0.25
        plot[:, column] = counts
    # A flat area puts thousands of samples on one level, which without this compression
    # leaves every other level invisible once the plot is normalised by its peak.
    plot = np.log1p(plot)
    peak = plot.max()
    if peak > 0:
        plot /= peak
    return plot[::-1, :]


def _tint(plot: np.ndarray, colour: str) -> np.ndarray:
    """Colour a single-channel plot as 8-bit RGB."""
    blank = np.zeros_like(plot)
    if colour == "red":
        rgb = np.stack([plot, blank, blank], -1)
    elif colour == "green":
        rgb = np.stack([blank, plot, blank], -1)
    else:
        rgb = np.stack([blank, blank, plot], -1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _draw_grid(
    draw: ImageDraw.ImageDraw, height: int, left: int, right: int, labels: bool = True
) -> None:
    """Draw the IRE grid across a plot area, with each level named unless ``labels`` is off."""
    font = _font()
    for ire in IRE_LINES:
        y = int(round(height - 1 - (ire / 100.0) * (height - 1)))
        draw.line([(left, y), (right, y)], fill=(64, 64, 64), width=1)
        if not labels:
            continue
        draw.text(
            (6, max(0, y - 6)), f"{ire:g}", fill=(200, 200, 200), font=font,
            stroke_width=1, stroke_fill=(0, 0, 0),
        )


def _bucket_max(plot: np.ndarray, count: int, axis: int) -> np.ndarray:
    """Resample one axis of a plot to a fixed length, keeping the brightest sample.

    Args:
        plot: A normalised plot from :func:`make_waveform_gray`.
        count: How many samples that axis is resampled to.
        axis: 0 for the levels, 1 for the columns.

    Returns:
        The same plot with that axis ``count`` long. A shorter axis is repeated and a longer
        one is pooled by its maximum, so a single bright column survives the reduction.
    """
    size = plot.shape[axis]
    if size == count:
        return plot
    if size < count:
        return np.take(plot, (np.arange(count) * size) // count, axis=axis)
    return np.maximum.reduceat(plot, (np.arange(count) * size) // count, axis=axis)


def compose_panel_parade(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Lay the three channel plots side by side at one fixed size, under one IRE grid.

    Args:
        red: Normalised red plot from :func:`make_waveform_gray`.
        green: Normalised green plot, in any shape.
        blue: Normalised blue plot, in any shape.

    Returns:
        A ``(PANEL_PLOT_HEIGHT, PANEL_PLOT_WIDTH * 3 + PARADE_GAP * 2, 3)`` uint8 array,
        whatever the plots measured. The grid carries no level names and nothing is written
        under the plots.
    """
    total_width = PANEL_PLOT_WIDTH * 3 + PARADE_GAP * 2
    canvas = Image.new("RGB", (total_width, PANEL_PLOT_HEIGHT), (0, 0, 0))
    for index, (plot, colour) in enumerate(
        ((red, "red"), (green, "green"), (blue, "blue"))
    ):
        fitted = _bucket_max(_bucket_max(plot, PANEL_PLOT_HEIGHT, 0), PANEL_PLOT_WIDTH, 1)
        canvas.paste(
            Image.fromarray(_tint(fitted, colour)),
            (index * (PANEL_PLOT_WIDTH + PARADE_GAP), 0),
        )
    _draw_grid(ImageDraw.Draw(canvas), PANEL_PLOT_HEIGHT, 0, total_width - 1, labels=False)
    return np.array(canvas, dtype=np.uint8)


def compose_waveform_panel(plot: np.ndarray, colour: str, stats_text: str) -> np.ndarray:
    """Lay out one channel's plot with its IRE grid and its statistics line.

    Args:
        plot: Normalised plot from :func:`make_waveform_gray`.
        colour: ``"red"``, ``"green"`` or ``"blue"``; anything else is drawn as blue.
        stats_text: Line drawn underneath, from :func:`stats_line`.

    Returns:
        An ``(height, width, 3)`` uint8 array.
    """
    panel = Image.fromarray(_tint(plot, colour))
    width, height = panel.size
    canvas = Image.new("RGB", (width + LEFT_PAD, height + PANEL_PAD), (0, 0, 0))
    canvas.paste(panel, (LEFT_PAD, 0))
    draw = ImageDraw.Draw(canvas)
    _draw_grid(draw, height, LEFT_PAD, LEFT_PAD + width - 1)
    draw.text(
        (6, height + 6), stats_text, fill=(255, 255, 255), font=_font(),
        stroke_width=1, stroke_fill=(0, 0, 0),
    )
    return np.array(canvas, dtype=np.uint8)


def compose_parade(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    red_stats: tuple[float, float, float, float, float],
    green_stats: tuple[float, float, float, float, float],
    blue_stats: tuple[float, float, float, float, float],
) -> np.ndarray:
    """Lay the three channel plots side by side under one IRE grid.

    Args:
        red: Normalised red plot from :func:`make_waveform_gray`.
        green: Normalised green plot, the same shape as ``red``.
        blue: Normalised blue plot, the same shape as ``red``.
        red_stats: Red channel summary from :func:`stats_tensor`.
        green_stats: Green channel summary.
        blue_stats: Blue channel summary.

    Returns:
        An ``(height, width, 3)`` uint8 array holding the parade and, beneath it, one
        statistics line per channel aligned under its own plot.
    """
    height, width = red.shape
    total_width = LEFT_PAD + width * 3 + PARADE_GAP * 2
    panel = Image.new("RGB", (total_width, height), (0, 0, 0))
    for index, (plot, colour) in enumerate(
        ((red, "red"), (green, "green"), (blue, "blue"))
    ):
        panel.paste(Image.fromarray(_tint(plot, colour)), (LEFT_PAD + index * (width + PARADE_GAP), 0))
    _draw_grid(ImageDraw.Draw(panel), height, LEFT_PAD, total_width - 1)

    canvas = Image.new("RGB", (total_width, height + PARADE_PAD), (0, 0, 0))
    canvas.paste(panel, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _stats_font()
    for index, (label, stats) in enumerate(
        (("R", red_stats), ("G", green_stats), ("B", blue_stats))
    ):
        colour = CHANNEL_COLOURS[("red", "green", "blue")[index]][1]
        draw.text(
            (LEFT_PAD + index * (width + PARADE_GAP) + 6, height + 6),
            parade_stats_line(label, stats),
            fill=colour, font=font, stroke_width=1, stroke_fill=(0, 0, 0),
        )
    return np.array(canvas, dtype=np.uint8)
