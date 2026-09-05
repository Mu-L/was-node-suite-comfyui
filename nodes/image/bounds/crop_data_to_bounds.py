"""Convert a crop window into an image bounds row.

``crop_data`` is ``(size, (left, top, right, bottom))`` with exclusive right and bottom
edges. A bounds row is ``(rmin, rmax, cmin, cmax)`` with every edge inclusive.
"""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import CROP_DATA, IMAGE_BOUNDS
from ....modules.image import bounds


class CropDataToBounds(io.ComfyNode):
    """Emit a ``CROP_DATA`` crop window as a one-row ``IMAGE_BOUNDS`` value."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCropDataToBounds",
            display_name="Crop Data to Bounds",
            search_aliases=[
                "WASCropDataToBounds",
                "Crop Data to Bounds",
                "crop data to bounds",
                "convert crop data",
                "bounding box",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Turn the crop window a crop node recorded into a bounds row, so a region "
                "found by Image Crop Location, Image Crop by Mask or Mask Crop Region can "
                "be drawn, inset or cropped through with the bounds nodes. A bounds row "
                "carries no size, so the recorded crop size comes out on its own outputs."
            ),
            inputs=[
                CROP_DATA.Input(
                    "crop_data",
                    tooltip=(
                        "The crop window to convert, from any node with a crop_data output. "
                        "Image Crop Face and Image Crop Face (YuNet) pass False here when "
                        "they found no face, and that raises, since there is no rectangle "
                        "to read."
                    ),
                ),
            ],
            outputs=[
                IMAGE_BOUNDS.Output(
                    display_name="image_bounds",
                    tooltip=(
                        "The window as a single bounds row, giving the first and last pixel "
                        "row and column it covers, for Bounded Image Crop, Inset Image "
                        "Bounds or Draw Image Bounds."
                    ),
                ),
                io.Int.Output(
                    display_name="crop_width",
                    tooltip=(
                        "Width the crop was recorded at, in pixels. It matches the "
                        "rectangle for every crop node but Image Crop Face and Image Crop "
                        "Face (YuNet), which record the padded square they emit. A bounds "
                        "row cannot carry a size, so it is given here instead."
                    ),
                ),
                io.Int.Output(
                    display_name="crop_height",
                    tooltip=(
                        "Height the crop was recorded at, in pixels. Wire it beside "
                        "crop_width wherever the original size is still needed, such as "
                        "resizing a reworked region before it goes back."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, crop_data=None) -> io.NodeOutput:
        """Convert a crop window into a one-row bounds value.

        Args:
            crop_data: ``(size, (left, top, right, bottom))`` from a crop node.

        Returns:
            The bounds row, then the recorded width and height.

        Raises:
            ValueError: Nothing is connected to crop_data, the value is False, it is not a
                size and a rectangle, or the rectangle is empty or inside out.
        """
        require_input(
            crop_data,
            "Crop Data to Bounds",
            "crop_data",
            "crop window",
            "Image Crop Location, Image Crop by Mask or Mask Crop Region",
            "crop_data",
        )

        if crop_data is False:
            raise ValueError(
                "Crop Data to Bounds was given False on its crop_data input, which Image "
                "Crop Face and Image Crop Face (YuNet) pass on when they found no face. "
                "There is no rectangle to convert. Feed it a crop that found something, or "
                "take the bounds from the mask instead with Mask to Bounds."
            )

        (width, height), (left, top, right, bottom) = bounds.crop_window(
            crop_data, "Crop Data to Bounds"
        )
        if right < left or bottom < top:
            raise ValueError(
                f"Crop Data to Bounds was given the inside out window ({left}, {top}, "
                f"{right}, {bottom}). A crop window's right edge cannot sit left of its "
                f"left edge, nor its bottom above its top, so the bounds row would end "
                f"before it started. Whatever produced this crop_data measured the "
                f"rectangle backwards."
            )
        if right == left or bottom == top:
            raise ValueError(
                f"Crop Data to Bounds was given the empty window ({left}, {top}, {right}, "
                f"{bottom}), which Mask Crop Region emits when its mask marks nothing. It "
                f"covers no pixel row or column, so there is no bounds row to read out of "
                f"it. Check the mask feeding that node."
            )

        # A crop window ends one past its last pixel row and column, a bounds row names its
        # last, so both far edges come back by one.
        return io.NodeOutput([(top, bottom - 1, left, right - 1)], width, height)
