"""Model-driven upscaling, as the pack performs it."""

from __future__ import annotations

import torch

__all__ = ["upscale_with_model"]

#: Square tile an upscale is read in, in pixels, halved on each memory failure.
TILE = 512

#: Pixels neighbouring tiles share, which the blend runs across.
OVERLAP = 32

#: Tile size below which a memory failure is raised rather than retried.
MINIMUM_TILE = 128

#: Bytes an upscale model is assumed to want per byte of tile, per unit of scale.
TILE_MEMORY = 384.0


def upscale_with_model(upscale_model, image: torch.Tensor) -> torch.Tensor:
    """Enlarge an image batch with a loaded upscale model, tiling to fit memory.

    Args:
        upscale_model: The ``UPSCALE_MODEL`` to run.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.

    Returns:
        The enlarged batch in the same layout, clamped to ``[0, 1]``.

    Raises:
        OOM_EXCEPTION: The smallest tile still did not fit.
    """
    import comfy.model_management
    import comfy.utils

    scale = max(float(getattr(upscale_model, "scale", 1.0)), 1.0)
    wanted = (TILE * TILE * 3) * image.element_size() * scale * TILE_MEMORY
    wanted += image.nelement() * image.element_size()
    comfy.model_management.load_models_gpu(
        [upscale_model.patcher], memory_required=wanted, force_full_load=True
    )

    source = image.movedim(-1, -3).to(upscale_model.patcher.load_device)
    output_device = comfy.model_management.intermediate_device()

    tile = TILE
    while True:
        try:
            steps = source.shape[0] * comfy.utils.get_tiled_scale_steps(
                source.shape[3], source.shape[2], tile_x=tile, tile_y=tile, overlap=OVERLAP
            )
            enlarged = comfy.utils.tiled_scale(
                source,
                lambda read: upscale_model(read.float()),
                tile_x=tile,
                tile_y=tile,
                overlap=OVERLAP,
                upscale_amount=scale,
                pbar=comfy.utils.ProgressBar(steps),
                output_device=output_device,
            )
            break
        except comfy.model_management.OOM_EXCEPTION:
            tile //= 2
            if tile < MINIMUM_TILE:
                raise

    return enlarged.movedim(-3, -1).clamp(0.0, 1.0)
