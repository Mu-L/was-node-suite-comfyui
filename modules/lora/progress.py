"""One progress bar in the browser and one in the console, driven together.

A failed update is not drawn.
"""

from __future__ import annotations

import sys

__all__ = ["MergeProgress"]


class MergeProgress:
    """A ComfyUI progress bar and a console progress bar under one counter.

    Both are driven from absolute step numbers rather than increments.

    Attributes:
        total: Steps the bar is scaled to.
    """

    def __init__(self, total: int, desc: str):
        """Create both bars.

        Args:
            total: Number of steps the whole run is expected to take. Revisable through
                :meth:`set_total` once the real module count is known.
            desc: Label drawn in front of the console bar.
        """
        from comfy.utils import ProgressBar
        from tqdm import tqdm

        self.total = int(total)
        self.comfy = ProgressBar(self.total)
        self.console = tqdm(
            total=self.total,
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            miniters=1,
            mininterval=0.1,
            file=sys.stderr,
        )

    def set_total(self, total: int) -> None:
        """Rescale both bars to a new step count.

        Args:
            total: The new number of steps.
        """
        from comfy.utils import ProgressBar

        self.total = int(total)
        try:
            self.console.total = self.total
        except Exception:
            pass
        try:
            self.comfy = ProgressBar(self.total)
        except Exception:
            pass

    def update_absolute(self, value: int) -> None:
        """Move both bars to a step number.

        Args:
            value: Steps completed so far, counted from zero.
        """
        step = int(value)
        try:
            if hasattr(self.comfy, "update_absolute"):
                self.comfy.update_absolute(step, self.total)
            else:
                self.comfy.update(1)
        except Exception:
            pass

        try:
            forward = step - int(self.console.n)
            if forward > 0:
                self.console.update(forward)
                self.console.refresh()
        except Exception:
            pass

    def close(self) -> None:
        """Finish the console bar. The ComfyUI bar needs no closing."""
        try:
            try:
                self.console.refresh()
            except Exception:
                pass
            self.console.close()
        except Exception:
            pass
