"""Crop images to a set of bounds."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.interface import size_report
from ....modules.log import get_logger

logger = get_logger("nodes.image.bounds")


class BoundedImageCrop(io.ComfyNode):
    """Cut each image down to the region named by its bounds."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Bounded Image Crop",
            display_name="Bounded Image Crop",
            search_aliases=["Bounded Image Crop", "crop to bounds", "cut out region"],
            category="WAS Suite/Image/Bound",
            description=(
                "Cut out the part of each image its bounds covers, so a detail can be "
                "worked on at full resolution and later put back with Bounded Image Blend."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The images to cut down. A single image or a batch both work, and "
                        "the pixels are sliced out unchanged, so each crop comes back at the "
                        "size its bounds cover, or smaller where they reach past the edge of "
                        "the image."
                    ),
                ),
                IMAGE_BOUNDS.Input(
                    "image_bounds",
                    tooltip=(
                        "Where to cut. One row per image crops each one separately; any "
                        "other count applies the first row to every image, which is also the "
                        "only way the crops are guaranteed to be the same size."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The cropped regions as a batch. Rows of differing sizes cannot be "
                        "stacked into one batch and raise an error."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, image_bounds) -> io.NodeOutput:
        """Crop each image to its bounds row.

        Raises:
            ValueError: Nothing is connected to the image_bounds input, or a bounds row is
                inside out.
        """
        require_input(
            image_bounds,
            "Bounded Image Crop",
            "image_bounds",
            "bounds",
            "Image Bounds, Inset Image Bounds or Image Crop by Mask",
            "IMAGE_BOUNDS",
        )

        image = image.unsqueeze(0) if image.dim() == 3 else image

        # A bounds count that does not match the batch means one window for every image,
        # so it is read once and reused.
        bounds_len = 1 if len(image_bounds) != len(image) else len(image)

        cropped_images = []
        for idx in range(len(image)):
            if (bounds_len == 1 and idx == 0) or bounds_len > 1:
                rmin, rmax, cmin, cmax = image_bounds[idx]

                if rmin > rmax or cmin > cmax:
                    raise ValueError(
                        f"Bounds row {idx} is inside out: it reads "
                        f"(rmin={rmin}, rmax={rmax}, cmin={cmin}, cmax={cmax}), and rmin "
                        f"must not exceed rmax, nor cmin exceed cmax."
                    )

                # Clamped rather than passed to the slice as given: a negative value counts
                # from the far edge in Python slicing, which would silently crop the wrong
                # region instead of the one asked for.
                height, width = image[idx].shape[0], image[idx].shape[1]
                given = (rmin, rmax, cmin, cmax)
                rmin, rmax = max(rmin, 0), min(rmax, height - 1)
                cmin, cmax = max(cmin, 0), min(cmax, width - 1)

                if rmin > rmax or cmin > cmax:
                    raise ValueError(
                        f"Bounds row {idx} falls outside the image, which is {width} by "
                        f"{height} pixels, so there is nothing to crop."
                    )
                if (rmin, rmax, cmin, cmax) != given:
                    logger.warning(
                        "bounds row %s reads (%s, %s, %s, %s) and was trimmed to (%s, %s, "
                        "%s, %s) to fit a %sx%s image", idx, *given, rmin, rmax, cmin, cmax,
                        width, height,
                    )

            cropped_images.append(image[idx][rmin:rmax + 1, cmin:cmax + 1, :])

        window = torch.stack(cropped_images, dim=0)
        size_report.publish(image, window, action="cropped")
        return io.NodeOutput(window)
