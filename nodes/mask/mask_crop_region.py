"""Crop a mask to a square window around its bounding box."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.compat.types import CROP_DATA
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report
from ...modules.mask.regions import crop_region

#: The window crop_region reports for a mask with no bounding box, where it hands the mask
#: back uncropped. The rest of a batch is left uncropped with it.
EMPTY_WINDOW = (0, 0, 0, 0)


class MaskCropRegion(io.ComfyNode):
    """Crop a mask and emit the crop window alongside it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Crop Region",
            display_name="Mask Crop Region",
            search_aliases=["Mask Crop Region", "crop mask", "bounding box"],
            category="WAS Suite/Image/Masking",
            description="Crop a mask to a square window centred on its bounding box, padded on "
            "every side and clipped to the image. crop_data carries the window back to Mask "
            "Paste Region. region_type chooses which region the window is measured on.",
            inputs=[
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "The mask to crop. The tightest box around everything non-black in it "
                        "decides where the crop window sits. Every mask of a batch is cropped to "
                        "the window measured on the first, so one crop_data describes all of them."
                    ),
                ),
                io.Int.Input(
                    "padding",
                    default=24,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Extra room left around the bounding box before it is squared off, in "
                        "pixels per side. 0 crops tight; larger values take in more of the "
                        "surroundings, up to the edges of the image."
                    ),
                ),
                io.Combo.Input(
                    "region_type",
                    options=["dominant", "minority"],
                    tooltip=(
                        "Which region the crop window is measured on. `dominant` takes the "
                        "largest connected area the mask marks, which is the subject in a "
                        "mask that also caught specks; `minority` takes the smallest. A mask "
                        "holding one region crops the same either way."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="cropped_mask",
                    tooltip=(
                        "The mask cut down to the crop window, ready to be worked on at a higher "
                        "effective resolution."
                    ),
                ),
                CROP_DATA.Output(
                    display_name="crop_data",
                    tooltip=(
                        "The window's size and position, to wire into Mask Paste Region so the "
                        "crop can be put back exactly where it came from."
                    ),
                ),
                io.Int.Output(
                    display_name="top_int",
                    tooltip="Y coordinate of the window's top edge in the source mask, in pixels.",
                ),
                io.Int.Output(
                    display_name="left_int",
                    tooltip=(
                        "X coordinate of the window's left edge in the source mask, in pixels."
                    ),
                ),
                io.Int.Output(
                    display_name="right_int",
                    tooltip=(
                        "X coordinate of the window's right edge in the source mask, in pixels."
                    ),
                ),
                io.Int.Output(
                    display_name="bottom_int",
                    tooltip=(
                        "Y coordinate of the window's bottom edge in the source mask, in pixels."
                    ),
                ),
                io.Int.Output(
                    display_name="width_int",
                    tooltip=(
                        "Width of cropped_mask in pixels. Smaller than the padded square when the "
                        "window ran off the side of the image."
                    ),
                ),
                io.Int.Output(
                    display_name="height_int",
                    tooltip=(
                        "Height of cropped_mask in pixels. Smaller than the padded square when "
                        "the window ran off the top or bottom of the image."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, mask, padding, region_type) -> io.NodeOutput:
        from PIL import ImageOps

        images = mask_images(mask)
        region_mask, crop_data = crop_region(images[0], region_type, padding)
        (width, height), window = crop_data
        left, top, right, bottom = window

        if window == EMPTY_WINDOW:
            rest = images[1:]
        else:
            rest = [image.crop(window) for image in images[1:]]
        regions = [pil2mask(ImageOps.invert(image)) for image in (region_mask, *rest)]
        cropped = stack_masks(regions)
        mask_report.publish(mask, cropped, source="mask")

        return io.NodeOutput(
            cropped, crop_data, top, left, right, bottom, width, height
        )
