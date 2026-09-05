"""Lay a batch of images out as a contact sheet."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.interface import size_report


class CreateGridImageFromBatch(io.ComfyNode):
    """Arrange every image in a batch into one grid image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Create Grid Image from Batch",
            display_name="Create Grid Image from Batch",
            search_aliases=[
                "Create Grid Image from Batch",
                "contact sheet",
                "montage",
                "image grid",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Lay a batch of images out as a single grid image, for comparing a run of "
                "results side by side."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The batch to lay out. Every image in it gets a cell.",
                ),
                io.Int.Input(
                    "border_width",
                    default=3,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "Gap between cells in pixels, filled with the border colour. 0 puts the "
                        "images flush against each other."
                    ),
                ),
                io.Int.Input(
                    "number_of_columns",
                    default=6,
                    min=1,
                    max=24,
                    step=1,
                    tooltip=(
                        "How many images per row. The number of rows follows from the batch "
                        "size: 12 images in 6 columns give 2 rows. The grid is always this many "
                        "columns wide, even when there are fewer images."
                    ),
                ),
                io.Int.Input(
                    "max_cell_size",
                    default=256,
                    min=32,
                    max=2048,
                    step=1,
                    tooltip=(
                        "Largest side of one cell in pixels. Images are scaled down to fit and "
                        "keep their proportions; an image already smaller than this is left as "
                        "it is."
                    ),
                ),
                io.Int.Input(
                    "border_red",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Red level of the background showing between cells, 0 to 255.",
                ),
                io.Int.Input(
                    "border_green",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Green level of the background showing between cells, 0 to 255.",
                ),
                io.Int.Input(
                    "border_blue",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Blue level of the background showing between cells, 0 to 255. All three "
                        "at 0 gives a black background, all three at 255 a white one."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="One image holding the whole grid, as a batch of one.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        border_width=3,
        number_of_columns=6,
        max_cell_size=256,
        border_red=0,
        border_green=0,
        border_blue=0,
    ) -> io.NodeOutput:
        folded = dynamic.fold(images)
        images = folded.images
        from PIL import Image

        cols = number_of_columns
        border_color = (border_red, border_green, border_blue)

        images_resized = []
        max_row_height = 0

        for tensor_img in images:
            img = tensor2pil(tensor_img)
            img_w, img_h = img.size
            aspect_ratio = img_w / img_h

            if img_w > img_h:
                cell_w = min(img_w, max_cell_size)
                cell_h = int(cell_w / aspect_ratio)
            else:
                cell_h = min(img_h, max_cell_size)
                cell_w = int(cell_h * aspect_ratio)

            images_resized.append(img.resize((cell_w, cell_h)))
            max_row_height = max(max_row_height, cell_h)

        # Row height follows the tallest image rather than max_cell_size, so a batch of
        # landscape pictures produces a grid with no wasted height.
        max_row_height = int(max_row_height)
        rows = math.ceil(len(images_resized) / cols)

        grid_width = cols * max_cell_size + (cols - 1) * border_width
        grid_height = rows * max_row_height + (rows - 1) * border_width

        new_image = Image.new("RGB", (grid_width, grid_height), border_color)

        for i, img in enumerate(images_resized):
            x = (i % cols) * (max_cell_size + border_width)
            y = (i // cols) * (max_row_height + border_width)

            img_w, img_h = img.size
            paste_x = x + (max_cell_size - img_w) // 2
            paste_y = y + (max_row_height - img_h) // 2

            new_image.paste(img, (paste_x, paste_y, paste_x + img_w, paste_y + img_h))

        size_report.publish(
            images,
            new_image.size,
            action="laid out as a grid",
            facts={
                "grid": f"{cols} by {rows}",
                "cell": f"{max_cell_size}x{max_row_height}",
            },
        )
        return io.NodeOutput(dynamic.unfold(pil2tensor(new_image), folded))
