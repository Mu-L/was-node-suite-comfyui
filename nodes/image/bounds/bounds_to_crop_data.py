"""Convert an image bounds row into a crop window.

A bounds row is ``(rmin, rmax, cmin, cmax)`` with every edge inclusive. ``crop_data`` is
``(size, (left, top, right, bottom))`` with exclusive right and bottom edges.
"""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.compat.sockets import require_input
from ....modules.compat.types import CROP_DATA, IMAGE_BOUNDS
from ....modules.image import bounds
from ....modules.log import get_logger

logger = get_logger("nodes.image.bounds")


class BoundsToCropData(io.ComfyNode):
    """Emit one row of an ``IMAGE_BOUNDS`` value as a ``CROP_DATA`` crop window."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBoundsToCropData",
            display_name="Bounds to Crop Data",
            search_aliases=[
                "WASBoundsToCropData",
                "Bounds to Crop Data",
                "bounds to crop data",
                "convert bounds",
                "crop window",
                "uncrop",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Turn a rectangle measured by the bounds nodes into the crop window the "
                "paste nodes read, so a region cut out with Bounded Image Crop can be put "
                "back by Image Paste Crop or Mask Paste Region. A bounds value carries a "
                "row per image while a crop window describes one rectangle, so index says "
                "which row travels on."
            ),
            inputs=[
                IMAGE_BOUNDS.Input(
                    "image_bounds",
                    tooltip=(
                        "The rectangle to convert, from Image Bounds, Inset Image Bounds, "
                        "Mask to Bounds or Image Crop by Mask."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which row to convert. 0 = the first row; -1 = the last, counting "
                        "back from the end. A row that is not there raises, naming how many "
                        "rows the bounds hold."
                    ),
                ),
            ],
            outputs=[
                CROP_DATA.Output(
                    display_name="crop_data",
                    tooltip=(
                        "The same rectangle as a crop window, recorded at the size the "
                        "rectangle covers. A paste node resizes whatever it is handed to "
                        "that size, so a region worked on at a higher resolution lands back "
                        "at the size it was measured at."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image_bounds, index=0) -> io.NodeOutput:
        """Convert one bounds row into a crop window.

        Args:
            image_bounds: Rows of ``(rmin, rmax, cmin, cmax)``, or one bare row.
            index: Which row to convert, negatives counting from the end.

        Returns:
            ``((width, height), (left, top, right, bottom))`` for that row.

        Raises:
            ValueError: Nothing is connected to image_bounds, the value holds no rows,
                index names a row that is not there, or that row is inside out.
        """
        require_input(
            image_bounds,
            "Bounds to Crop Data",
            "image_bounds",
            "bounds",
            "Image Bounds, Inset Image Bounds or Mask to Bounds",
            "IMAGE_BOUNDS",
        )

        rows = bounds.rows(image_bounds)
        if not rows:
            raise ValueError(
                "Bounds to Crop Data was given a bounds value holding no rectangles. Check "
                "the node feeding image_bounds: a region search that matched nothing "
                "produces this."
            )
        if not -len(rows) <= index < len(rows):
            plural = "" if len(rows) == 1 else "s"
            raise ValueError(
                f"Bounds to Crop Data was asked for row {index} of a bounds value holding "
                f"{len(rows)} row{plural}. Rows count from 0, and -1 is the last one."
            )
        if len(rows) > 1:
            logger.info(
                "the bounds hold %s rows and a crop window describes one rectangle, so row "
                "%s is the one converted", len(rows), index,
            )

        rmin, rmax, cmin, cmax = rows[index]
        if rmin > rmax or cmin > cmax:
            raise ValueError(
                f"Bounds row {index} is inside out: it reads (rmin={rmin}, rmax={rmax}, "
                f"cmin={cmin}, cmax={cmax}), and rmin must not exceed rmax, nor cmin exceed "
                f"cmax."
            )

        # A bounds row names its last pixel row and column, a crop window ends one past its
        # last, so both far edges move on by one.
        return io.NodeOutput(
            ((cmax + 1 - cmin, rmax + 1 - rmin), (cmin, rmin, cmax + 1, rmax + 1))
        )
