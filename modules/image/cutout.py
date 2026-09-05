"""Cutting a subject out of a batch, and putting something behind it.

Images are ``(batch, height, width, channels)`` on a 0 to 1 scale. :func:`mattes` answers
one matte per frame, :func:`compose` lays them against a background.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from . import matting

__all__ = ["BACKGROUNDS", "compose", "mattes", "refine", "tidy"]

#: Widget option -> the colour laid behind the subject, as RGBA on a 0 to 255 scale.
BACKGROUNDS = {
    "black": (0, 0, 0, 255),
    "white": (255, 255, 255, 255),
    "magenta": (255, 0, 255, 255),
    "chroma green": (0, 177, 64, 255),
    "chroma blue": (0, 71, 187, 255),
}

#: Kernel width the tidying pass runs at when post processing is on.
TIDY = 3

#: Coverage at or above which a pixel carries the subject's colour rather than the fill.
CLEAR_LEVEL = 1.0 / 255.0


def mattes(model, images: torch.Tensor) -> torch.Tensor:
    """Read one matte per frame, at the side the model was trained for.

    Args:
        model: A :class:`~modules.model.cutout.Cutout`.
        images: ``(batch, height, width, channels)`` on a 0 to 1 scale.

    Returns:
        A ``(batch, height, width)`` tensor on a 0 to 1 scale.
    """
    import comfy.utils

    device = model.backend.load()
    network = model.backend.model
    height, width = int(images.shape[1]), int(images.shape[2])
    side = int(model.side)

    progress = comfy.utils.ProgressBar(len(images))
    answered = []
    for frame in images:
        planes = frame[..., :3].permute(2, 0, 1).unsqueeze(0).to(device=device)
        read = functional.interpolate(
            planes.float(), size=(side, side), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            out = network(read)
        while isinstance(out, (list, tuple)) and out:
            out = out[-1]
        matte = functional.interpolate(
            out.float(), size=(height, width), mode="bilinear", align_corners=False
        )
        answered.append(matte.reshape(1, height, width).clamp(0.0, 1.0).cpu())
        progress.update(1)
    return torch.cat(answered, dim=0)


def refine(
    images: torch.Tensor,
    rough: torch.Tensor,
    foreground_threshold: float,
    background_threshold: float,
    erode: int,
) -> torch.Tensor:
    """Solve the matte again over the band the thresholds leave uncertain.

    Args:
        images: ``(batch, height, width, channels)`` on a 0 to 1 scale.
        rough: ``(batch, height, width)`` matte to take the band from.
        foreground_threshold: Level above which a pixel is certainly the subject, on a
            0 to 1 scale.
        background_threshold: Level below which a pixel is certainly not, on a 0 to 1 scale.
        erode: Pixels the certain regions are pulled back by before solving, 0 for neither.

    Returns:
        A ``(batch, height, width)`` tensor on a 0 to 1 scale.

    Raises:
        ArithmeticError: The solve did not settle. The message names the remedy.
    """
    solved = []
    for frame, one in zip(images, rough):
        band = matting.trimap(one, foreground_threshold, background_threshold, erode)
        solved.append(matting.alpha(frame[..., :3], band).unsqueeze(0))
    return torch.cat(solved, dim=0)


def tidy(mattes: torch.Tensor) -> torch.Tensor:
    """Drop the specks and fill the pinholes of every matte in a batch.

    Args:
        mattes: ``(batch, height, width)`` on a 0 to 1 scale.

    Returns:
        A ``(batch, height, width)`` tensor on a 0 to 1 scale.
    """
    planes = mattes.unsqueeze(1)
    # Opening drops a speck: shrink it away, then grow what is left.
    shrunk = -functional.max_pool2d(-planes, TIDY, stride=1, padding=TIDY // 2)
    opened = functional.max_pool2d(shrunk, TIDY, stride=1, padding=TIDY // 2)
    # Closing fills a pinhole: grow over it, then shrink back.
    grown = functional.max_pool2d(opened, TIDY, stride=1, padding=TIDY // 2)
    closed = -functional.max_pool2d(-grown, TIDY, stride=1, padding=TIDY // 2)
    return closed.squeeze(1)


def compose(
    images: torch.Tensor,
    mattes: torch.Tensor,
    transparency: bool,
    background: str,
    post_processing: bool,
) -> torch.Tensor:
    """Lay the subject against the chosen background, in the subject's own colours.

    Args:
        images: ``(batch, height, width, channels)`` on a 0 to 1 scale.
        mattes: ``(batch, height, width)`` on a 0 to 1 scale.
        transparency: Answer four channels, the matte carried as alpha.
        background: A key of :data:`BACKGROUNDS`, or anything else for none.
        post_processing: Drop the matte's specks and fill its pinholes first.

    Returns:
        A ``(batch, height, width, 3 or 4)`` tensor on a 0 to 1 scale.
    """
    import comfy.utils

    matte = tidy(mattes) if post_processing else mattes
    matte = matte.clamp(0.0, 1.0).to(device=images.device, dtype=images.dtype)
    colour = BACKGROUNDS.get(background)
    planes = images[..., :3]

    progress = comfy.utils.ProgressBar(len(planes))
    estimated = []
    for frame, one in zip(planes, matte):
        estimated.append(matting.foreground(frame, one).unsqueeze(0))
        progress.update(1)
    front = torch.cat(estimated, dim=0).to(images.dtype)

    if colour is None:
        behind = torch.zeros_like(planes)
    else:
        behind = torch.tensor(
            [level / 255.0 for level in colour[:3]], dtype=images.dtype, device=images.device
        ).view(1, 1, 1, 3).expand_as(planes)

    alpha = matte.unsqueeze(-1)
    if not transparency:
        return (front * alpha + behind * (1.0 - alpha)).clamp(0.0, 1.0)
    shown = torch.where(alpha >= CLEAR_LEVEL, front, behind)
    return torch.cat([shown, alpha], dim=-1).clamp(0.0, 1.0)
