"""Reporting how far through a long node is, and what it has produced so far.

Both calls are no-ops outside ComfyUI.
"""

from __future__ import annotations

from .. import log

__all__ = ["progress_bar"]

logger = log.get_logger("modules.interface.progress")

#: Longest side a preview is sent at. Larger costs bandwidth on every step for detail the node's
#: thumbnail cannot show anyway.
PREVIEW_MAX = 512


def progress_bar(total: int) -> "Progress":
    """A progress bar over ``total`` steps.

    Args:
        total: How many steps the work is divided into.

    Returns:
        The bar. Every method is safe to call when ComfyUI is not running.
    """
    return Progress(total)


class Progress:
    """ComfyUI's progress bar where there is one, and nothing where there is not.

    Args:
        total: How many steps the work is divided into.
    """

    def __init__(self, total: int):
        self.total = max(1, int(total))
        self.done = 0
        self.bar = None
        try:
            from comfy.utils import ProgressBar

            self.bar = ProgressBar(self.total)
        except Exception as unavailable:
            logger.debug("no progress bar available: %s", unavailable)

    def update(self, count: int = 1, preview=None) -> None:
        """Advance the bar, optionally showing what has just been produced.

        Args:
            count: Steps completed since the last call.
            preview: An image to draw on the node, as a ``(height, width, channels)`` tensor in
                0 to 1, a PIL image, or ``None`` to leave the node's picture alone.
        """
        self.done = min(self.total, self.done + count)
        if self.bar is None:
            return
        try:
            self.bar.update_absolute(self.done, self.total, self._payload(preview))
        except Exception as failed:
            # A bar that cannot draw must never take the run down with it.
            logger.debug("progress update failed: %s", failed)

    @staticmethod
    def _payload(preview):
        """Turn an image into what ComfyUI's progress hook expects, or ``None``."""
        if preview is None:
            return None
        image = preview
        if hasattr(preview, "detach"):
            import numpy as np
            from PIL import Image

            array = preview.detach().float().clamp(0, 1).cpu().numpy()
            if array.ndim == 4:
                array = array[0]
            image = Image.fromarray((array * 255).round().astype(np.uint8))
        return ("JPEG", image, PREVIEW_MAX)
