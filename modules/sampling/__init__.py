"""Implementation shared by the sampling nodes.

:mod:`~modules.sampling.sequence` holds the latent interpolation modes and the reverse
sampling pass, and :mod:`~modules.sampling.conditioning` holds prompt encoding.
"""

from __future__ import annotations


def sampler_names() -> list[str]:
    """The sampler names this ComfyUI offers."""
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def scheduler_names() -> list[str]:
    """The scheduler names this ComfyUI offers."""
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS
