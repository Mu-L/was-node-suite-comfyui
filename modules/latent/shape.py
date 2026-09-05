"""Moving a latent between the 4D image layout and the 5D video layout.

An image latent is ``[B, C, H, W]``. A video latent adds a time axis and flattens to
``[B*T, C, H, W]``.
"""

from __future__ import annotations

import torch

__all__ = ["flatten_5d_to_4d", "is_latent_5d", "unflatten_4d_to_5d"]


def is_latent_5d(samples: torch.Tensor) -> bool:
    """Report whether a latent carries a time axis.

    Args:
        samples: The tensor from a LATENT's ``samples`` key.

    Returns:
        ``True`` for a 5D ``[B, C, T, H, W]`` video latent, ``False`` for anything else,
        including the 4D ``[B, C, H, W]`` image latent.
    """
    return samples.dim() == 5


def flatten_5d_to_4d(samples: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Fold a video latent's time axis into its batch axis.

    Args:
        samples: A 5D ``[B, C, T, H, W]`` latent.

    Returns:
        ``(flat, b, t)``, the latent as a contiguous ``[B*T, C, H, W]`` tensor with frame
        ``t`` of clip ``b`` at row ``b * T + t``, plus the two sizes
        :func:`unflatten_4d_to_5d` needs to undo it.
    """
    b, c, t, h, w = samples.shape
    flat = samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous()
    return flat, b, t


def unflatten_4d_to_5d(samples_btchw: torch.Tensor, b: int, t: int) -> torch.Tensor:
    """Undo :func:`flatten_5d_to_4d`.

    Args:
        samples_btchw: A ``[B*T, C, H, W]`` tensor in the order :func:`flatten_5d_to_4d`
            produced. Its height and width may differ from the ones that went in, which is
            what makes this usable after a resize.
        b: Batch size the flat tensor was built from.
        t: Frame count the flat tensor was built from.

    Returns:
        The latent back in the 5D ``[B, C, T, H, W]`` layout, contiguous.

    Raises:
        ValueError: The leading dimension is not ``b * t``, so the rows cannot be split
            into clips and frames.
    """
    bt, c, h, w = samples_btchw.shape
    if bt != b * t:
        raise ValueError(f"Shape mismatch: bt={bt} != b*t={b * t}")
    return samples_btchw.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
