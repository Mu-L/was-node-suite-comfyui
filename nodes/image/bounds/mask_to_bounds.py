"""Measure the box around everything a mask marks, as image bounds.

A bounds row is ``(rmin, rmax, cmin, cmax)`` with every edge inclusive.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.convert.tensors import mask_planes
from ....modules.log import get_logger

logger = get_logger("nodes.image.bounds")


class MaskToBounds(io.ComfyNode):
    """Emit each mask's bounding box as a row of an ``IMAGE_BOUNDS`` value."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMaskToBounds",
            display_name="Mask to Bounds",
            search_aliases=[
                "WASMaskToBounds",
                "Mask to Bounds",
                "mask to bounds",
                "bounding box",
                "mask extent",
                "region",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Measure the tightest rectangle around everything a mask marks and answer "
                "it as bounds, which is how a mask becomes a window for Bounded Image "
                "Crop, Inset Image Bounds or Draw Image Bounds. A mask marking nothing "
                "answers its whole frame and says so in the console."
            ),
            inputs=[
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "The mask to measure. Every mask of a batch is measured on its own, "
                        "so a moving subject gives a row that follows it. Separate blobs in "
                        "one mask answer a single box covering them all."
                    ),
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How bright a mask pixel must be, from 0.0 to 1.0, to count as "
                        "marked. 0.5 boxes a mask's solid core; 0.0 takes its whole "
                        "feathered edge in."
                    ),
                ),
                io.Int.Input(
                    "padding",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Extra pixels kept on all four sides of the marked area, trimmed "
                        "where the box would run past the frame. 0 = tight against the "
                        "mask; 64 leaves an inpainting pass some surroundings to match "
                        "against."
                    ),
                ),
            ],
            outputs=[
                IMAGE_BOUNDS.Output(
                    display_name="image_bounds",
                    tooltip=(
                        "One row per mask, giving the first and last pixel row and column "
                        "its marked area covers. A mask that marks nothing at this "
                        "threshold covers the whole frame instead, and the console names "
                        "which one did."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, mask, threshold=0.5, padding=0) -> io.NodeOutput:
        """Measure one bounding box per mask of a batch.

        Args:
            mask: Mask tensor in any of the layouts a ``MASK`` socket carries.
            threshold: Level a sample must exceed to count as marked.
            padding: Pixels added to every side before the box is clamped to the frame.

        Returns:
            One ``(rmin, rmax, cmin, cmax)`` row per mask, in batch order.
        """
        bounds = []
        for index, plane in enumerate(mask_planes(mask)):
            height, width = int(plane.shape[0]), int(plane.shape[1])
            marked = plane > threshold
            rows = torch.where(torch.any(marked, dim=1))[0]
            cols = torch.where(torch.any(marked, dim=0))[0]

            if len(rows) == 0:
                logger.warning(
                    "mask %s marks nothing above threshold %s, so its bounds cover the "
                    "whole %sx%s frame", index, threshold, width, height,
                )
                bounds.append((0, height - 1, 0, width - 1))
                continue

            bounds.append((
                max(int(rows[0]) - padding, 0),
                min(int(rows[-1]) + padding, height - 1),
                max(int(cols[0]) - padding, 0),
                min(int(cols[-1]) + padding, width - 1),
            ))

        return io.NodeOutput(bounds)
