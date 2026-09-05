"""Colour palette extraction.

K-means over an image's pixels, rendered as a chart of labelled swatches and returned
alongside the hex codes. The clustering is torch and runs on ComfyUI's compute device.
"""

from __future__ import annotations

import math

import torch
from PIL import Image, ImageDraw, ImageFont

from .. import log
from ..data import paths

__all__ = ["compute_device", "default_font_path", "generate_palette", "kmeans", "pixels_of"]

logger = log.get_logger("image.palette")

#: Luminance weights the swatches are ordered by, as ``(red, green, blue)``.
LUMA = (0.299, 0.587, 0.114)

#: Entries of the sample-to-centroid distance table held at once, which caps its memory.
DISTANCE_BUDGET = 1 << 24

#: Runs kept the best of, per initialiser, when a caller names no count.
RUNS = {"k-means++": 1, "random": 10}

#: Candidates drawn per k-means++ step beyond the two every step draws.
_TRIAL_BASE = 2

_ACCUMULATE: dict[str, torch.dtype] = {}


def default_font_path() -> str | None:
    """Path to the bundled label font, or ``None`` when it is not on disk.

    Returns:
        The font path as a string, or ``None``.
    """
    font = paths.font_file()
    return str(font) if font.is_file() else None


def compute_device() -> torch.device:
    """The device the clustering runs on.

    Returns:
        ComfyUI's compute device, or the CPU outside a ComfyUI process.
    """
    try:
        import comfy.model_management as model_management
    except ImportError:
        return torch.device("cpu")
    return model_management.get_torch_device()


def _accumulate_dtype(device: torch.device) -> torch.dtype:
    """The dtype cluster sums are totalled in on ``device``.

    Args:
        device: Device the clustering runs on.

    Returns:
        ``torch.float64``, or ``torch.float32`` where the device carries no float64 type.
    """
    dtype = _ACCUMULATE.get(device.type)
    if dtype is None:
        try:
            torch.zeros(1, dtype=torch.float64, device=device)
            dtype = torch.float64
        except (RuntimeError, TypeError) as error:
            logger.debug("%s has no float64, so cluster sums total in float32: %s", device, error)
            dtype = torch.float32
        _ACCUMULATE[device.type] = dtype
    return dtype


def _assign(samples: torch.Tensor, centroids: torch.Tensor, chunk: int):
    """Nearest centroid of every sample, and the squared distance to it.

    Args:
        samples: ``(n, features)`` float tensor.
        centroids: ``(clusters, features)`` float tensor.
        chunk: Samples measured against every centroid at once.

    Returns:
        ``(labels, errors)``: an ``(n,)`` int64 tensor and an ``(n,)`` float tensor.
    """
    count = samples.shape[0]
    labels = torch.empty(count, dtype=torch.int64, device=samples.device)
    errors = torch.empty(count, dtype=samples.dtype, device=samples.device)
    offsets = centroids.square().sum(1)
    for start in range(0, count, chunk):
        block = samples[start : start + chunk]
        table = torch.addmm(offsets.unsqueeze(0), block, centroids.t(), alpha=-2.0)
        nearest, index = table.min(dim=1)
        labels[start : start + chunk] = index
        errors[start : start + chunk] = (nearest + block.square().sum(1)).clamp_min_(0)
    return labels, errors


def _seed_plus_plus(samples, clusters, generator):
    """Greedy k-means++ centroids, drawn with probability proportional to squared distance.

    Args:
        samples: ``(n, features)`` float tensor.
        clusters: Number of centroids to draw.
        generator: CPU ``torch.Generator`` the draws come from.

    Returns:
        A ``(clusters, features)`` float tensor.
    """
    count = samples.shape[0]
    device = samples.device
    squares = samples.square().sum(1)
    first = int(torch.randint(count, (1,), generator=generator).item())
    centroids = torch.empty((clusters, samples.shape[1]), dtype=samples.dtype, device=device)
    centroids[0] = samples[first]
    closest = (squares - 2.0 * (samples @ samples[first]) + squares[first]).clamp_min_(0)
    trials = _TRIAL_BASE + int(math.log(clusters)) if clusters > 1 else 1
    chunk = max(1, DISTANCE_BUDGET // trials)
    for index in range(1, clusters):
        cumulative = torch.cumsum(closest, dim=0)
        total = float(cumulative[-1])
        draws = torch.rand(trials, generator=generator).to(device=device, dtype=samples.dtype)
        if total <= 0.0:
            picks = (draws * count).to(torch.int64).clamp_(0, count - 1)
        else:
            picks = torch.searchsorted(cumulative, draws * total).clamp_(0, count - 1)
        candidates = samples[picks]
        offsets = candidates.square().sum(1)
        potential = torch.zeros(trials, dtype=samples.dtype, device=device)
        for start in range(0, count, chunk):
            stop = start + chunk
            block = samples[start:stop]
            table = torch.addmm(offsets.unsqueeze(0), block, candidates.t(), alpha=-2.0)
            table.add_(squares[start:stop].unsqueeze(1)).clamp_min_(0)
            potential += torch.minimum(table, closest[start:stop].unsqueeze(1)).sum(0)
        best = int(potential.argmin().item())
        centroids[index] = candidates[best]
        taken = (squares - 2.0 * (samples @ candidates[best]) + offsets[best]).clamp_min_(0)
        closest = torch.minimum(closest, taken)
    return centroids


def _seed_random(samples, clusters, generator):
    """Centroids drawn as distinct sample rows, uniformly.

    Args:
        samples: ``(n, features)`` float tensor.
        clusters: Number of centroids to draw.
        generator: CPU ``torch.Generator`` the draws come from.

    Returns:
        A ``(clusters, features)`` float tensor.
    """
    picks = torch.randperm(samples.shape[0], generator=generator)[:clusters]
    return samples[picks.to(samples.device)].clone()


_SEEDS = {"k-means++": _seed_plus_plus, "random": _seed_random}


def _lloyd(samples, clusters, seeded, max_iter, tolerance, chunk):
    """Refine centroids until they stop moving or ``max_iter`` passes are spent.

    Args:
        samples: ``(n, features)`` float tensor.
        clusters: Number of centroids.
        seeded: ``(clusters, features)`` starting centroids.
        max_iter: Assignment and update passes allowed.
        tolerance: Squared centroid movement at or below which the run stops.
        chunk: Samples measured against every centroid at once.

    Returns:
        ``(centroids, labels, inertia)``.
    """
    totals_dtype = _accumulate_dtype(samples.device)
    wide = samples.to(totals_dtype)
    centroids = seeded
    labels, errors = _assign(samples, centroids, chunk)
    for _ in range(max_iter):
        sums = torch.zeros((clusters, samples.shape[1]), dtype=totals_dtype, device=samples.device)
        sums.index_add_(0, labels, wide)
        counts = torch.bincount(labels, minlength=clusters).to(totals_dtype)
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if empty.numel():
            # An unused centroid restarts on the sample sitting furthest from its own, which
            # that sample's own cluster gives up.
            far = errors.topk(int(empty.numel())).indices
            taken = wide[far]
            sums.index_add_(0, labels[far], -taken)
            counts.index_add_(0, labels[far], torch.full_like(taken[:, 0], -1.0))
            sums[empty] = taken
            counts[empty] = 1
        moved = (sums / counts.clamp_min(1).unsqueeze(1)).to(samples.dtype)
        shift = float((moved - centroids).square().sum())
        centroids = moved
        previous = labels
        labels, errors = _assign(samples, centroids, chunk)
        if shift <= tolerance or bool(torch.equal(labels, previous)):
            break
    return centroids, labels, float(errors.sum())


def kmeans(
    samples: torch.Tensor,
    clusters: int,
    init: str = "k-means++",
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    runs: int | None = None,
    device=None,
):
    """Cluster the rows of ``samples`` by Lloyd's algorithm.

    Args:
        samples: ``(n, features)`` tensor of any numeric dtype.
        clusters: Number of centroids, at most ``n``.
        init: ``"k-means++"`` draws each centroid with probability proportional to its
            squared distance from the ones already chosen. ``"random"`` draws distinct rows
            uniformly.
        max_iter: Assignment and update passes one run is allowed.
        tol: Convergence threshold, multiplied by the mean per-feature variance to give the
            squared centroid movement a run stops at.
        seed: Seed the starting centroids are drawn from. The same seed and the same
            samples give the same centroids.
        runs: Independent runs the lowest-inertia one is kept from. ``None`` reads
            :data:`RUNS`.
        device: Device the clustering runs on, as a ``torch.device`` or a name. ``None``
            selects ComfyUI's compute device, falling back to the CPU.

    Returns:
        ``(centroids, labels, inertia)``: a ``(clusters, features)`` float32 tensor on
        ``samples``' own device, an ``(n,)`` int64 tensor of centroid indices, and the
        summed squared distance from each sample to its centroid.

    Raises:
        ValueError: ``init`` names no initialiser, ``clusters`` is below 1, or ``samples``
            holds fewer rows than ``clusters``.
    """
    if init not in _SEEDS:
        raise ValueError(
            f"unknown k-means initialiser '{init}'. Use one of: {', '.join(sorted(_SEEDS))}"
        )
    flat = samples.reshape(samples.shape[0], -1)
    if clusters < 1:
        raise ValueError(f"k-means needs at least 1 cluster, not {clusters}")
    if flat.shape[0] < clusters:
        raise ValueError(
            f"k-means was asked for {clusters} clusters from {flat.shape[0]} samples. "
            f"Ask for at most {flat.shape[0]}, or give it a larger image."
        )
    home = flat.device
    target = compute_device() if device is None else torch.device(device)
    try:
        centroids, labels, inertia = _cluster(
            flat, clusters, init, max_iter, tol, seed, runs, target
        )
    except RuntimeError as error:
        if target.type == "cpu":
            raise
        logger.warning("k-means could not run on %s, so it ran on the CPU: %s", target, error)
        centroids, labels, inertia = _cluster(
            flat, clusters, init, max_iter, tol, seed, runs, torch.device("cpu")
        )
    return centroids.to(home), labels.to(home), inertia


def _cluster(flat, clusters, init, max_iter, tol, seed, runs, device):
    """Run every initialisation and keep the centroids with the lowest inertia.

    Args:
        flat: ``(n, features)`` tensor.
        clusters: Number of centroids.
        init: Key into the initialisers.
        max_iter: Assignment and update passes one run is allowed.
        tol: Convergence threshold, scaled by the mean per-feature variance.
        seed: Seed of the generator every run draws its starting centroids from.
        runs: Independent runs, or ``None`` to read :data:`RUNS`.
        device: Device the clustering runs on.

    Returns:
        ``(centroids, labels, inertia)``.
    """
    working = flat.to(device=device, dtype=torch.float32)
    attempts = RUNS.get(init, 1) if runs is None else max(1, int(runs))
    chunk = max(1, DISTANCE_BUDGET // clusters)
    tolerance = tol * float(working.var(dim=0, unbiased=False).mean())
    generator = torch.Generator().manual_seed(int(seed))
    best = None
    for _ in range(attempts):
        seeded = _SEEDS[init](working, clusters, generator)
        found = _lloyd(working, clusters, seeded, max(1, int(max_iter)), tolerance, chunk)
        if best is None or found[2] < best[2]:
            best = found
    return best


def pixels_of(img: Image.Image) -> torch.Tensor:
    """Rows of ``(red, green, blue)`` from an ``RGB`` image.

    Args:
        img: Source image, in mode ``RGB``.

    Returns:
        An ``(width * height, 3)`` uint8 tensor.

    Raises:
        ValueError: ``img`` is not in mode ``RGB``.
    """
    if img.mode != "RGB":
        raise ValueError(
            f"an RGB image is needed here, and this one is mode '{img.mode}'. "
            f"Convert it first, with img.convert('RGB')."
        )
    return torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).reshape(-1, 3)


def generate_palette(
    img: Image.Image,
    n_colors: int = 16,
    cell_size: int = 128,
    padding: int = 0,
    font_path: str | None = None,
    font_size: int = 15,
    mode: str = 'chart',
) -> tuple[Image.Image, str]:
    """Render an image's dominant colours as a palette.

    Args:
        img: Source image. Must be ``RGB``: the pixel buffer is read as three channels.
        n_colors: Number of clusters, which is also the number of swatches.
        cell_size: Swatch size in pixels.
        padding: Pixels above and below the label. The band under a swatch is
            ``font_size + 2 * padding`` tall, and ``'back_to_back'`` has no band at all.
        font_path: TrueType font for the labels. ``None`` selects PIL's built-in bitmap
            font, which ignores ``font_size``.
        font_size: Label size in points.
        mode: ``'back_to_back'`` renders one unlabelled row of swatches. Any other value
            renders the labelled grid.

    Returns:
        The palette image, and the swatch colours as newline-separated ``#rrggbb`` codes
        in the order they were drawn.

    Raises:
        ValueError: The halved image has fewer than ``n_colors`` pixels, which leaves
            k-means with fewer samples than clusters.
    """
    # Clustering reads the image at half size.
    img = img.resize((img.width // 2, img.height // 2), resample=Image.BILINEAR)
    centroids, _, _ = kmeans(pixels_of(img), n_colors, seed=0)
    cluster_centers = centroids.clamp(0, 255).to(torch.uint8)

    weights = torch.tensor(LUMA, dtype=torch.float32, device=cluster_centers.device)
    luminance = (cluster_centers.to(torch.float32) @ weights).sqrt()
    dominant = cluster_centers.argmax(dim=1)
    # Swatches are grouped by dominant channel, then run dark to light inside each group.
    by_luminance = torch.argsort(luminance, stable=True)
    order = by_luminance[torch.argsort(dominant[by_luminance], stable=True)]
    sorted_colors = [tuple(row) for row in cluster_centers[order].tolist()]

    if mode == 'back_to_back':
        num_rows = 1
        num_cols = n_colors
        label_height = 0
    else:
        num_rows = math.isqrt(n_colors)
        num_cols = -(-n_colors // num_rows)
        # Every row carries a label band under its swatches.
        label_height = font_size + padding * 2

    row_height = cell_size + label_height
    palette = Image.new('RGB', (num_cols * cell_size, num_rows * row_height), color='white')
    draw = ImageDraw.Draw(palette)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    hex_palette = []
    for i, color in enumerate(sorted_colors):
        if mode == 'back_to_back':
            row, col = 0, i
        else:
            row = i % num_rows
            col = i // num_rows
        cell_x = col * cell_size
        cell_y = row * row_height

        cell = Image.new('RGB', (cell_size, cell_size), color=color)
        palette.paste(cell, (cell_x, cell_y))

        if label_height:
            label = f"R: {color[0]} G: {color[1]} B: {color[2]}"
            # The label is centred from its measured length.
            text_x = cell_x + (cell_size - draw.textlength(label, font=font)) / 2
            draw.text((text_x, cell_y + cell_size + padding), label, font=font, fill='black')

        hex_palette.append('#%02x%02x%02x' % color)

    return palette, '\n'.join(hex_palette)
