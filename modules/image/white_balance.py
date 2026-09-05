"""Estimating what colour the light was, and taking it back out of the picture.

Images are float tensors shaped ``(batch, height, width, channels)`` in ``[0, 1]``. An estimator
answers one colour per frame; dividing by it neutralises the cast.
"""

from __future__ import annotations

import torch

__all__ = ["ESTIMATORS", "balance", "estimate"]

#: The estimators, in the order a node offers them.
ESTIMATORS = ("grey world", "white patch", "shades of grey", "grey edge")

# Grey world assumes the average of a scene is grey, which a large block of one colour breaks.
# White patch assumes the brightest point is white, which a blown highlight breaks. Shades of
# grey sits between them on a Minkowski norm, and grey edge makes the same assumption about the
# average edge rather than the average pixel, which a dominant colour affects far less.

#: The Minkowski power "shades of grey" uses. 1 would be grey world and a large power approaches
#: white patch; 6 is the value the literature settles on.
MINKOWSKI = 6.0

#: The quantile "white patch" treats as the brightest point, rather than the true maximum, which
#: a single blown pixel or a hot pixel would otherwise decide on its own.
WHITE_QUANTILE = 0.99

#: Below this a channel is treated as carrying no information, so it is left alone rather than
#: divided into.
FLOOR = 1e-5


def _spatial(values):
    """Flatten to ``(frames, channels, pixels)`` for a per frame, per channel statistic."""
    return values.permute(0, 3, 1, 2).reshape(values.shape[0], values.shape[3], -1)


def estimate(images, estimator: str = "grey world"):
    """The colour of the light in each frame.

    Args:
        images: An ``IMAGE`` tensor, ``(batch, height, width, channels)``.
        estimator: One of :data:`ESTIMATORS`.

    Returns:
        ``(frames, channels)`` of positive numbers, scaled so they average to one.

    Raises:
        ValueError: ``estimator`` is not one of :data:`ESTIMATORS`.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(
            f"estimator is {estimator!r}, which is not one of {', '.join(ESTIMATORS)}."
        )
    working = images.to(torch.float32).clamp(0.0, 1.0)

    if estimator == "grey edge":
        # The average edge, rather than the average pixel: a scene that is largely one colour
        # still has edges of many, so a dominant colour sways this far less.
        planes = working.permute(0, 3, 1, 2)
        down = (planes[:, :, 1:, :] - planes[:, :, :-1, :]).abs()
        across = (planes[:, :, :, 1:] - planes[:, :, :, :-1]).abs()
        measure = down.mean(dim=(2, 3)) + across.mean(dim=(2, 3))
    else:
        flat = _spatial(working)
        if estimator == "grey world":
            measure = flat.mean(dim=2)
        elif estimator == "white patch":
            measure = torch.quantile(flat, WHITE_QUANTILE, dim=2)
        else:
            measure = (flat.pow(MINKOWSKI).mean(dim=2)).pow(1.0 / MINKOWSKI)

    measure = measure.clamp(min=FLOOR)
    # Scaled to average one, so the estimate says which way the light leaned and nothing about
    # how bright it was.
    return measure / measure.mean(dim=1, keepdim=True)


def smooth_over_time(estimates, radius: int):
    """Average each frame's estimate with its neighbours'.

    Args:
        estimates: ``(frames, channels)`` from :func:`estimate`.
        radius: Frames either side to average over. 0 or a single frame changes nothing.

    Returns:
        ``(frames, channels)``, smoothed along the frame axis.
    """
    frames = int(estimates.shape[0])
    if frames < 2 or int(radius) < 1:
        return estimates
    radius = min(int(radius), frames)
    offsets = torch.arange(-radius, radius + 1, device=estimates.device, dtype=estimates.dtype)
    sigma = max(radius / 2.0, 1e-6)
    weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    weights = weights / weights.sum()
    # Held at the ends rather than faded, so a sequence is not pulled towards grey at its edges.
    padded = torch.nn.functional.pad(
        estimates.transpose(0, 1).unsqueeze(0), (radius, radius), mode="replicate",
    )
    smoothed = torch.nn.functional.conv1d(
        padded.reshape(-1, 1, frames + 2 * radius), weights.reshape(1, 1, -1),
    )
    return smoothed.reshape(estimates.shape[1], frames).transpose(0, 1)


def balance(images, estimator: str = "grey world", strength: float = 1.0, radius: int = 0):
    """Take the estimated cast out of every frame.

    Args:
        images: An ``IMAGE`` tensor, ``(batch, height, width, channels)`` in ``[0, 1]``.
        estimator: One of :data:`ESTIMATORS`.
        strength: How much of the correction to apply, 0 to 1.
        radius: Frames either side the estimate is averaged over, for a sequence. 0 balances
            each frame on its own.

    Returns:
        A tensor of the same shape and dtype.

    Raises:
        ValueError: The input is not a batch of images, or the estimator does not exist.
    """
    if getattr(images, "ndim", 0) != 4:
        raise ValueError(
            "white balancing takes a batch shaped (batch, height, width, channels)"
        )
    strength = float(min(max(strength, 0.0), 1.0))
    if strength <= 0.0 or images.shape[3] < 3:
        return images

    working = images.to(torch.float32).clamp(0.0, 1.0)
    colour = working[..., :3]
    light = smooth_over_time(estimate(colour, estimator), int(radius))

    # Applied as a power so half strength is half the correction in the ratio, not half of a
    # difference: a gain of 2 at half strength is 1.41, which is what undoing half a cast means.
    gains = (1.0 / light).pow(strength).reshape(-1, 1, 1, 3)
    corrected = colour * gains

    # The cast is what is being removed, not the exposure, so brightness is put back where it
    # was rather than left to whatever the gains happened to do to it.
    before = colour.mean(dim=(1, 2, 3), keepdim=True)
    after = corrected.mean(dim=(1, 2, 3), keepdim=True).clamp(min=FLOOR)
    corrected = (corrected * (before / after)).clamp(0.0, 1.0)

    if images.shape[3] > 3:
        # An alpha channel is carried through untouched.
        corrected = torch.cat([corrected, working[..., 3:]], dim=3)
    return corrected.to(images.dtype)
