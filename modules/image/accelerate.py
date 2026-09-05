"""Running an image operation on ComfyUI's compute device.

An IMAGE arrives and leaves on the CPU. :func:`run_on` moves a batch across, runs the work
and hands it back, falling back to the CPU where the device refuses it.
"""

from __future__ import annotations

__all__ = ["run_on", "working_device"]

import torch

from .. import log

logger = log.get_logger("image.accelerate")


def working_device(prefer_gpu: bool = True) -> torch.device:
    """Where an image operation should run.

    Args:
        prefer_gpu: Whether to ask for ComfyUI's compute device. False answers the CPU.

    Returns:
        A ``torch.device``.
    """
    if not prefer_gpu:
        return torch.device("cpu")
    try:
        from ..model import compute_device

        return compute_device()
    except Exception:
        logger.debug("no compute device is available, so this runs on the CPU", exc_info=True)
        return torch.device("cpu")


def run_on(images: torch.Tensor, work, prefer_gpu: bool = True) -> torch.Tensor:
    """Run one operation over a batch, on the accelerator where there is one.

    Args:
        images: The batch, as ComfyUI hands it over.
        work: A callable taking the batch and answering a tensor of the same shape.
        prefer_gpu: Whether to move the batch to ComfyUI's compute device first.

    Returns:
        The result, on the device the batch arrived on.
    """
    home = images.device
    device = working_device(prefer_gpu)
    if device == home:
        return work(images)

    try:
        return work(images.to(device)).to(home)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        logger.warning(
            "%s could not run this image operation (%s), so it ran on the CPU instead",
            device, type(error).__name__,
        )
        logger.debug("the device reported", exc_info=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return work(images.to(home))
