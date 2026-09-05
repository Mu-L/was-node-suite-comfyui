"""The pixel values a VAE decode produced, before they are folded into 0 to 1.

:func:`unclamped` runs a decode with the VAE's own affine transfer kept and its holding at
0.0 and 1.0 lifted. :func:`measure` answers that transfer.
"""

from __future__ import annotations

import contextlib

import torch

from .. import log

__all__ = ["BEYOND", "PROBE", "measure", "unclamped"]

logger = log.get_logger("image.vae")

#: Decoder values the transfer is measured at, well inside the range it maps.
PROBE = (-0.5, -0.25, 0.0, 0.25, 0.5)

#: Decoder values outside that range, used to see whether the transfer holds them.
BEYOND = (-4.0, 4.0)

#: How far a measured value may sit from the fitted line and still count as affine.
TOLERANCE = 1e-4


def measure(process_output) -> tuple[float, float, bool]:
    """The affine transfer a VAE applies to its decoder output, and whether it holds.

    Args:
        process_output: The VAE's own ``process_output``, which may work in place.

    Returns:
        ``(scale, offset, holds)``. ``holds`` is True where values outside the transfer's
        own range come back pinned rather than mapped. ``(1.0, 0.0, False)`` where the
        transfer is not affine, which is what an identity transfer measures as.
    """
    try:
        sample = torch.tensor(PROBE, dtype=torch.float32).reshape(1, 1, 1, -1).repeat(1, 3, 1, 1)
        mapped = process_output(sample.clone()).float().reshape(3, -1)[0]

        scale = float((mapped[-1] - mapped[0]) / (PROBE[-1] - PROBE[0]))
        offset = float(mapped[0]) - scale * PROBE[0]
        fitted = torch.tensor([scale * value + offset for value in PROBE])
        if float((fitted - mapped).abs().amax()) > TOLERANCE:
            return 1.0, 0.0, False

        far = torch.tensor(BEYOND, dtype=torch.float32).reshape(1, 1, 1, -1).repeat(1, 3, 1, 1)
        held = process_output(far.clone()).float().reshape(3, -1)[0]
        predicted = torch.tensor([scale * value + offset for value in BEYOND])
        holds = float((held - predicted).abs().amax()) > TOLERANCE
    except Exception as error:
        logger.debug("the output transfer could not be measured (%s)", error)
        return 1.0, 0.0, False
    return scale, offset, holds


@contextlib.contextmanager
def unclamped(vae):
    """Run a decode with the VAE's own transfer kept and its holding lifted.

    Args:
        vae: The ComfyUI ``VAE``, whose ``process_output`` is swapped for the duration.

    Yields:
        True while the holding was lifted, False where there was none to lift.
    """
    original = getattr(vae, "process_output", None)
    if original is None:
        yield False
        return

    scale, offset, holds = measure(original)
    if not holds:
        yield False
        return

    def mapped(image):
        return image.mul_(scale).add_(offset)

    vae.process_output = mapped
    logger.debug("decoding with the 0 to 1 hold lifted, transfer %g x + %g", scale, offset)
    try:
        yield True
    finally:
        vae.process_output = original
