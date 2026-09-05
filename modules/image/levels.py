"""Tonal readouts: histogram charts, the occupied intensity range, and luminance extraction.

:func:`black_white_levels` and :func:`channel_frequency` render a chart as an ``RGBA`` PIL
image. :func:`black_white_points` answers the two intensities as numbers.
"""

from __future__ import annotations

import torch
from PIL import Image, ImageDraw, ImageFont

from ..data import paths
from .histogram import BINS

__all__ = [
    "black_white_levels",
    "black_white_points",
    "channel_frequency",
    "greyscale",
]

#: Pixels across every chart this module renders.
CHART_WIDTH = 1600

#: Pixels down the black and white levels chart.
LEVELS_HEIGHT = 800

#: Pixels down the three channel frequency chart.
FREQUENCY_HEIGHT = 400

#: Pixels reserved around the levels plot: left, top, right, bottom.
LEVELS_MARGINS = (116, 64, 28, 68)

#: Pixels reserved around one channel plot: left, top, right, bottom.
PANEL_MARGINS = (84, 52, 16, 62)

#: Pixels between the three channel plots.
PANEL_GAP = 24

#: Pixels between a plot edge and the figures written beside it.
TICK_GAP = 8

#: Pixels between a marker line and its label.
MARKER_GAP = 7

#: Pixels a dash and the gap after it each occupy on a marker line.
DASH = 9

#: Pixels a channel trace is thickened to.
TRACE_WIDTH = 2

#: Point size a chart title is drawn at.
TITLE_POINTS = 30

#: Point size a channel plot's title is drawn at.
PANEL_TITLE_POINTS = 21

#: Point size an axis name is drawn at.
AXIS_POINTS = 18

#: Point size a tick figure is drawn at.
TICK_POINTS = 15

#: Colour behind every chart.
BACKGROUND = (0.0, 0.0, 0.0)

#: Colour of the grid lines across a plot.
GRID = (0.24, 0.24, 0.24)

#: Colour of the rules down the left and along the bottom of a plot.
AXIS = (0.45, 0.45, 0.45)

#: Colour a tick figure and an axis name are drawn in.
LABEL = (0.78, 0.78, 0.78)

#: Colour a chart title is drawn in.
TITLE = (1.0, 1.0, 1.0)

#: Colour the luminance bars are filled with.
LUMA_TRACE = (0.86, 0.86, 0.86)

#: Colour of the dashed lines marking the black and white points.
MARKER = (1.0, 0.25, 0.25)

#: Colour each channel trace and its title are drawn in, red, green then blue.
CHANNEL_TRACE = ((1.0, 0.25, 0.25), (0.25, 1.0, 0.25), (0.25, 0.5, 1.0))

#: Title over each of the three channel plots, in the same order.
CHANNEL_TITLES = ("Red Channel", "Green Channel", "Blue Channel")

#: Intensities the levels plot draws a grid line and a figure at.
LEVELS_TICKS = (0, 32, 64, 96, 128, 160, 192, 224, 256)

#: Intensities a channel plot draws a grid line and a figure at.
PANEL_TICKS = (0, 64, 128, 192, 255)

#: Fractions of the peak count a horizontal grid line is drawn at.
COUNT_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _font(points: int):
    """The font a chart's text is drawn in.

    Args:
        points: Point size to load the bundled font at.

    Returns:
        A font object, falling back to PIL's built-in bitmap font when the bundled one
        cannot be read.
    """
    try:
        return ImageFont.truetype(str(paths.font_file()), size=points)
    except Exception:
        return ImageFont.load_default()


def _measure(font, text: str) -> tuple[int, int]:
    """Width and height in pixels of the ink one string covers."""
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _codes(image: Image.Image, mode: str) -> torch.Tensor:
    """Read an image's samples as 8-bit codes.

    Args:
        image: Source image, converted to ``mode`` when it is in another.
        mode: PIL mode the samples are read in, such as ``"L"`` or ``"RGB"``.

    Returns:
        A one-dimensional uint8 tensor holding every sample in row order, empty for an
        image with no pixels.
    """
    plane = image if image.mode == mode else image.convert(mode)
    raw = bytearray(plane.tobytes())
    if not raw:
        return torch.zeros(0, dtype=torch.uint8)
    return torch.frombuffer(raw, dtype=torch.uint8)


def _count(codes: torch.Tensor) -> torch.Tensor:
    """Count 8-bit codes into one bin each.

    Args:
        codes: Any uint8 tensor; it is flattened first.

    Returns:
        A ``(BINS,)`` int64 tensor of counts.
    """
    return torch.bincount(codes.reshape(-1).to(torch.int64), minlength=BINS)[:BINS]


def _occupied_range(counts: torch.Tensor) -> tuple[int, int]:
    """First and last occupied bin of a 256 bin histogram.

    Args:
        counts: One count per intensity.

    Returns:
        ``(first, last)`` as intensities 0 to 255. A histogram with no count anywhere
        answers ``(0, 255)``, which is the whole scale.
    """
    occupied = torch.nonzero(counts > 0).reshape(-1)
    if occupied.numel() == 0:
        return 0, 255
    return int(occupied[0].item()), int(occupied[-1].item())


def _canvas(width: int, height: int) -> torch.Tensor:
    """A float32 ``(height, width, 3)`` canvas filled with :data:`BACKGROUND`."""
    return torch.tensor(BACKGROUND, dtype=torch.float32).expand(height, width, 3).clone()


def _paint(
    canvas: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
    colour: tuple[float, float, float],
) -> None:
    """Set every crossing of the given rows and columns to one colour.

    Args:
        canvas: Canvas painted in place.
        rows: Row indices, as any shape.
        columns: Column indices, as any shape.
        colour: Red, green and blue as floats in ``[0, 1]``.
    """
    canvas[rows.reshape(-1, 1), columns.reshape(1, -1)] = torch.tensor(
        colour, dtype=torch.float32
    )


def _fill(
    canvas: torch.Tensor,
    box: tuple[int, int, int, int],
    mask: torch.Tensor,
    colour: tuple[float, float, float],
) -> None:
    """Set every pixel of a box the mask marks to one colour.

    Args:
        canvas: Canvas painted in place.
        box: ``(left, top, width, height)`` of the area the mask covers.
        mask: ``(height, width)`` bool tensor.
        colour: Red, green and blue as floats in ``[0, 1]``.
    """
    left, top, width, height = box
    canvas[top:top + height, left:left + width][mask] = torch.tensor(
        colour, dtype=torch.float32
    )


def _text_plane(width: int, height: int, labels) -> torch.Tensor:
    """Rasterise strings into a coverage plane.

    Args:
        width: Plane width in pixels.
        height: Plane height in pixels.
        labels: One ``(left, top, text, font, turned)`` per string, where ``left`` and
            ``top`` place the ink box and ``turned`` draws the string reading upwards.

    Returns:
        A ``(height, width)`` float32 tensor, 0.0 where nothing was drawn and 1.0 under a
        solid glyph.
    """
    plane = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(plane)
    for left, top, text, font, turned in labels:
        box = font.getbbox(text)
        if turned:
            glyphs = Image.new(
                "L", (max(1, box[2] - box[0]), max(1, box[3] - box[1])), 0
            )
            ImageDraw.Draw(glyphs).text((-box[0], -box[1]), text, fill=255, font=font)
            glyphs = glyphs.rotate(90, expand=True)
            plane.paste(glyphs, (left, top), glyphs)
        else:
            draw.text((left - box[0], top - box[1]), text, fill=255, font=font)
    coverage = torch.frombuffer(bytearray(plane.tobytes()), dtype=torch.uint8)
    return coverage.view(height, width).to(torch.float32) / 255.0


def _draw_text(canvas: torch.Tensor, labels, colour: tuple[float, float, float]) -> None:
    """Composite one colour's strings onto a canvas.

    Args:
        canvas: Canvas painted in place.
        labels: Entries in the form :func:`_text_plane` takes.
        colour: Red, green and blue as floats in ``[0, 1]``.
    """
    if not labels:
        return
    height, width, _ = canvas.shape
    alpha = _text_plane(width, height, labels).unsqueeze(-1)
    tint = torch.tensor(colour, dtype=torch.float32).view(1, 1, 3)
    canvas.mul_(1.0 - alpha).add_(tint * alpha)


def _count_label(value: float) -> str:
    """Format a bin count as the figure written beside a grid line."""
    if value < 1000:
        return str(int(round(value)))
    if value < 1000000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1000000:.1f}M"


def _tick_x(value: float, left: int, width: int, span: int) -> int:
    """Column an intensity falls on.

    Args:
        value: Intensity to place.
        left: Leftmost column of the plot.
        width: Plot width in pixels.
        span: Intensity the rightmost column stands for.

    Returns:
        A column inside the plot.
    """
    return left + min(width - 1, max(0, round(value * width / span)))


def _plot_grid(canvas: torch.Tensor, box, ticks, span: int, peak: float, font) -> tuple:
    """Draw a plot's grid and axis rules, and answer the figures beside them.

    Args:
        canvas: Canvas painted in place.
        box: ``(left, top, width, height)`` of the plot area.
        ticks: Intensities a vertical grid line is drawn at.
        span: Intensity the rightmost column stands for.
        peak: Count the top row stands for.
        font: Font the figures are drawn in.

    Returns:
        ``(labels, gutter)``: one entry per figure in the form :func:`_text_plane` takes,
        and the leftmost column a count figure occupies.
    """
    left, top, width, height = box
    labels = []

    rows = torch.arange(top, top + height)
    for value in ticks:
        x = _tick_x(value, left, width, span)
        _paint(canvas, rows, torch.tensor([x]), GRID)
        text = str(value)
        text_width = _measure(font, text)[0]
        labels.append((
            min(max(left, x - text_width // 2), left + width - text_width),
            top + height + TICK_GAP,
            text,
            font,
            False,
        ))

    columns = torch.arange(left, left + width)
    gutter = left
    for fraction in COUNT_TICKS:
        y = top + height - 1 - round(fraction * (height - 1))
        _paint(canvas, torch.tensor([y]), columns, GRID)
        text = _count_label(peak * fraction)
        text_width, text_height = _measure(font, text)
        start = left - TICK_GAP - text_width
        gutter = min(gutter, start)
        labels.append((start, y - text_height // 2, text, font, False))

    _paint(canvas, rows, torch.tensor([left - 1]), AXIS)
    _paint(canvas, torch.tensor([top + height]), torch.arange(left - 1, left + width), AXIS)
    return labels, gutter


def _axis_names(box, x_name: str, y_name: str, font, tick_font, gutter: int) -> list:
    """Place the two axis names around a plot.

    Args:
        box: ``(left, top, width, height)`` of the plot area.
        x_name: Name written under the plot.
        y_name: Name written up the left of the plot, reading upwards.
        font: Font the names are drawn in.
        tick_font: Font the tick figures were drawn in.
        gutter: Leftmost column a count figure occupies.

    Returns:
        Two entries in the form :func:`_text_plane` takes.
    """
    left, top, width, height = box
    x_width = _measure(font, x_name)[0]
    y_width, y_height = _measure(font, y_name)
    figure_height = _measure(tick_font, "0123456789")[1]
    return [
        (
            left + (width - x_width) // 2,
            top + height + TICK_GAP + figure_height + TICK_GAP,
            x_name,
            font,
            False,
        ),
        (
            max(0, gutter - TICK_GAP - y_height),
            top + (height - y_width) // 2,
            y_name,
            font,
            True,
        ),
    ]


def _bar_mask(counts: torch.Tensor, width: int, height: int, peak: float) -> torch.Tensor:
    """Mark the pixels inside a bar of a bar chart.

    Args:
        counts: One count per bin.
        width: Plot width in pixels.
        height: Plot height in pixels.
        peak: Count the top row stands for.

    Returns:
        A ``(height, width)`` bool tensor.
    """
    if peak <= 0:
        return torch.zeros((height, width), dtype=torch.bool)
    columns = (torch.arange(width) * counts.numel()) // width
    bars = (counts[columns].to(torch.float32) / peak * height).round().clamp(0, height)
    return torch.arange(height).unsqueeze(1) >= (height - bars).unsqueeze(0)


def _trace_mask(
    counts: torch.Tensor, width: int, height: int, peak: float, thickness: int
) -> torch.Tensor:
    """Mark the pixels a line joining one point per bin covers.

    Args:
        counts: One count per bin.
        width: Plot width in pixels.
        height: Plot height in pixels.
        peak: Count the top row stands for.
        thickness: Pixels the line is thickened to where it runs flat.

    Returns:
        A ``(height, width)`` bool tensor.
    """
    bins = counts.numel()
    values = counts.to(torch.float32)
    edges = torch.arange(width + 1, dtype=torch.float32) * (bins - 1) / width
    lower = edges.floor().clamp(0, bins - 1).long()
    upper = (lower + 1).clamp(max=bins - 1)
    fraction = edges - lower
    crossing = values[lower] * (1.0 - fraction) + values[upper] * fraction

    # Every bin is folded into the column its own position falls in.
    columns = torch.arange(bins, dtype=torch.float32) * width / (bins - 1)
    columns = columns.floor().clamp(0, width - 1).long()
    highest = torch.zeros(width, dtype=torch.float32).scatter_reduce(
        0, columns, values, reduce="amax", include_self=False
    )
    lowest = torch.full((width,), float("inf")).scatter_reduce(
        0, columns, values, reduce="amin", include_self=False
    )

    scale = peak if peak > 0 else 1.0
    span = (height - 1) / scale
    top = ((height - 1) - torch.maximum(
        torch.maximum(crossing[:-1], crossing[1:]), highest
    ) * span).floor()
    bottom = ((height - 1) - torch.minimum(
        torch.minimum(crossing[:-1], crossing[1:]), lowest
    ) * span).ceil()
    bottom = torch.maximum(bottom, top + (thickness - 1))
    rows = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    return (rows >= top.unsqueeze(0)) & (rows <= bottom.unsqueeze(0))


def _to_image(canvas: torch.Tensor) -> Image.Image:
    """Convert a float canvas to an opaque image.

    Args:
        canvas: ``(height, width, 3)`` float tensor in ``[0, 1]``.

    Returns:
        An ``RGBA`` PIL image of the same size, its alpha channel solid.
    """
    height, width, _ = canvas.shape
    codes = (canvas.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    alpha = torch.full((height, width, 1), 255, dtype=torch.uint8)
    rgba = torch.cat([codes, alpha], dim=2).contiguous()
    buffer = bytearray(rgba.numel())
    torch.frombuffer(buffer, dtype=torch.uint8).copy_(rgba.reshape(-1))
    return Image.frombytes("RGBA", (width, height), buffer)


def black_white_points(image: Image.Image) -> tuple[int, int]:
    """The darkest and lightest intensities an image's luminance actually occupies.

    Args:
        image: Source image, converted to ``L`` internally.

    Returns:
        ``(black point, white point)`` as intensities 0 to 255. These are the two values
        :func:`black_white_levels` draws as dashed red lines.
    """
    return _occupied_range(_count(_codes(image, "L")))


def black_white_levels(image: Image.Image) -> Image.Image:
    """Plot the greyscale histogram with the black and white points marked.

    Args:
        image: Source image, converted to ``L`` internally.

    Returns:
        An ``RGBA`` PIL image of the rendered chart, not of the source.
    """
    counts = _count(_codes(image, "L"))
    black, white = _occupied_range(counts)
    peak = float(counts.max().item())

    left_pad, top_pad, right_pad, bottom_pad = LEVELS_MARGINS
    width = CHART_WIDTH - left_pad - right_pad
    height = LEVELS_HEIGHT - top_pad - bottom_pad
    box = (left_pad, top_pad, width, height)

    tick_font = _font(TICK_POINTS)
    axis_font = _font(AXIS_POINTS)
    title_font = _font(TITLE_POINTS)

    canvas = _canvas(CHART_WIDTH, LEVELS_HEIGHT)
    labels, gutter = _plot_grid(canvas, box, LEVELS_TICKS, BINS, peak, tick_font)
    labels += _axis_names(box, "Intensity", "Frequency", axis_font, tick_font, gutter)
    _fill(canvas, box, _bar_mask(counts, width, height, peak), LUMA_TRACE)

    dashes = torch.arange(top_pad, top_pad + height)
    dashes = dashes[((dashes - top_pad) // DASH) % 2 == 0]
    marks = []
    for value, name, ahead in ((black, "black", True), (white, "white", False)):
        x = _tick_x(value, left_pad, width, BINS)
        _paint(canvas, dashes, torch.tensor([x]), MARKER)
        text = f"{name} {value}"
        text_width = _measure(tick_font, text)[0]
        start = x + MARKER_GAP if ahead else x - MARKER_GAP - text_width
        marks.append((
            min(max(left_pad, start), left_pad + width - text_width),
            top_pad + MARKER_GAP,
            text,
            tick_font,
            False,
        ))

    title = "Black and White Levels"
    title_width, title_height = _measure(title_font, title)
    _draw_text(canvas, labels, LABEL)
    _draw_text(
        canvas,
        [((CHART_WIDTH - title_width) // 2, (top_pad - title_height) // 2, title,
          title_font, False)],
        TITLE,
    )
    _draw_text(canvas, marks, MARKER)
    return _to_image(canvas)


def channel_frequency(image: Image.Image) -> Image.Image:
    """Plot the per-intensity frequency of each RGB channel side by side.

    Args:
        image: Source image, converted to ``RGB`` internally, so a four-channel picture is
            charted on its colour channels and its alpha is left out.

    Returns:
        An ``RGBA`` PIL image of the rendered chart, not of the source. Each plot is scaled
        to its own busiest bin.
    """
    samples = _codes(image, "RGB").reshape(-1, 3)
    counts = [_count(samples[:, channel]) for channel in range(3)]

    panel_width = (CHART_WIDTH - PANEL_GAP * 2) // 3
    left_pad, top_pad, right_pad, bottom_pad = PANEL_MARGINS
    width = panel_width - left_pad - right_pad
    height = FREQUENCY_HEIGHT - top_pad - bottom_pad

    tick_font = _font(TICK_POINTS)
    axis_font = _font(AXIS_POINTS)
    title_font = _font(PANEL_TITLE_POINTS)

    canvas = _canvas(CHART_WIDTH, FREQUENCY_HEIGHT)
    labels = []
    titles = []
    for channel in range(3):
        origin = channel * (panel_width + PANEL_GAP)
        box = (origin + left_pad, top_pad, width, height)
        peak = float(counts[channel].max().item())
        figures, gutter = _plot_grid(canvas, box, PANEL_TICKS, BINS - 1, peak, tick_font)
        labels += figures
        labels += _axis_names(
            box, "Color Intensity", "Frequency", axis_font, tick_font, gutter
        )
        _fill(
            canvas,
            box,
            _trace_mask(counts[channel], width, height, peak, TRACE_WIDTH),
            CHANNEL_TRACE[channel],
        )
        title = CHANNEL_TITLES[channel]
        title_width, title_height = _measure(title_font, title)
        titles.append([(
            origin + left_pad + (width - title_width) // 2,
            (top_pad - title_height) // 2,
            title,
            title_font,
            False,
        )])

    _draw_text(canvas, labels, LABEL)
    for channel in range(3):
        _draw_text(canvas, titles[channel], CHANNEL_TRACE[channel])
    return _to_image(canvas)


def greyscale(image: torch.Tensor) -> torch.Tensor:
    """Reduce an image tensor to a single luminance channel.

    Args:
        image: Tensor in BGR channel order, either ``(h, w)`` or ``(h, w, channels)``. A
            float tensor is scaled by 255 and truncated to 8-bit codes first.

    Returns:
        A ``(h, w, 1)`` uint8 tensor, or the input unchanged when it already has one
        channel. An alpha channel is not read.
    """
    if image.is_floating_point():
        image = (image * 255).clamp(0, 255).to(torch.uint8)
    channels = image.shape[2] if image.ndim == 3 else 1
    if channels == 1:
        return image
    # The value channel of HSV is the largest of the three colour channels.
    return image[:, :, :3].amax(dim=2).unsqueeze(-1)
