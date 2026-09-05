"""Reading straight line segments out of a line-segment network, and drawing them.

Segments are ``(count, 4)`` of ``x_start, y_start, x_end, y_end`` in the pixels of the frame
they were found in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["decode", "draw"]

#: Peaks read out of the centre map before any are discarded.
PEAKS = 200

#: Width of the window a peak has to be the brightest sample in.
PEAK_WINDOW = 3

#: The centre map is half the side of the frame the network read.
CENTRE_SCALE = 2

#: Half-width of a drawn segment in pixels.
LINE_WIDTH = 1.0


def decode(prediction: torch.Tensor, score: float, length: float) -> torch.Tensor:
    """Turn a network's centre and displacement maps into segments.

    Args:
        prediction: ``(1, 9, height, width)``, channel 0 the centre map and channels 1 to 4
            the displacements to each end.
        score: Confidence a centre must reach, 0 to 1.
        length: Shortest segment kept, in centre-map samples.

    Returns:
        A ``(count, 4)`` tensor of ``x_start, y_start, x_end, y_end`` in frame pixels, empty
        where nothing passed both tests.
    """
    width = prediction.shape[-1]
    displacement = prediction[:, 1:5][0]
    heat = torch.sigmoid(prediction[:, 0])
    # A centre is kept only where it is the brightest sample of its own window.
    highest = F.max_pool2d(heat, PEAK_WINDOW, stride=1, padding=(PEAK_WINDOW - 1) // 2)
    heat = (heat * (highest == heat).to(heat.dtype)).reshape(-1)

    scores, indices = torch.topk(heat, min(PEAKS, heat.numel()), dim=-1, largest=True)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = torch.remainder(indices, width)

    offsets = displacement[:, rows, columns]
    starts = torch.stack([columns + offsets[0], rows + offsets[1]], dim=1)
    ends = torch.stack([columns + offsets[2], rows + offsets[3]], dim=1)
    spans = torch.linalg.vector_norm(starts - ends, dim=1)

    keep = (scores > float(score)) & (spans > float(length))
    if not bool(keep.any()):
        return prediction.new_zeros((0, 4))
    return torch.cat([starts[keep], ends[keep]], dim=1) * CENTRE_SCALE


def draw(segments: torch.Tensor, height: int, width: int, scale_x: float, scale_y: float):
    """Draw segments as white lines on black.

    Args:
        segments: ``(count, 4)`` as :func:`decode` answers, in the network's own pixels.
        height: Canvas height in pixels.
        width: Canvas width in pixels.
        scale_x: Multiplier taking a network x onto the canvas.
        scale_y: Multiplier taking a network y onto the canvas.

    Returns:
        A ``(1, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    device = segments.device
    canvas = torch.zeros((height, width), dtype=torch.float32, device=device)
    if segments.numel() == 0:
        return canvas.expand(3, height, width).unsqueeze(0).clone()

    rows = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
    columns = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)
    half = max(LINE_WIDTH * max(height, width) / 512.0, 0.6)

    for segment in segments:
        x0, y0 = float(segment[0]) * scale_x, float(segment[1]) * scale_y
        x1, y1 = float(segment[2]) * scale_x, float(segment[3]) * scale_y
        canvas = torch.maximum(canvas, _segment(x0, y0, x1, y1, half, rows, columns))
    return (canvas * 255.0).expand(3, height, width).unsqueeze(0).clone()


def _segment(x0, y0, x1, y1, half, rows, columns) -> torch.Tensor:
    """One line's coverage, as samples within ``half`` of the span between two points."""
    span_x, span_y = x1 - x0, y1 - y0
    length_squared = span_x * span_x + span_y * span_y
    dx, dy = columns - x0, rows - y0
    if length_squared <= 1e-9:
        distance = torch.sqrt(dx * dx + dy * dy)
    else:
        # How far along the span the nearest point lies, held to the span's own ends.
        along = ((dx * span_x + dy * span_y) / length_squared).clamp(0.0, 1.0)
        distance = torch.sqrt((dx - along * span_x) ** 2 + (dy - along * span_y) ** 2)
    return (distance <= half).to(torch.float32)
