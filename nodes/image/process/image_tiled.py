"""Cut one image into a grid of tiles and return them as a batch."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import image_planes, pil2tensor, tensor2pil
from ....modules.interface import size_report


class ImageTiled(io.ComfyNode):
    """Slice an image into a grid of tiles, emitted as a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Tiled",
            display_name="Image Tiled",
            search_aliases=["Image Tiled", "split image", "tiles", "grid slice"],
            category="WAS Suite/Image/Process",
            description=(
                "Cut an image into a grid of smaller tiles and hand them on as a batch, "
                "for processing a large picture a piece at a time."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to cut up. Only the first image of a batch is used."
                    ),
                ),
                io.Int.Input(
                    "num_tiles",
                    default=4,
                    max=64,
                    min=2,
                    step=1,
                    tooltip=(
                        "How many tiles to aim for. Square numbers divide evenly, 4 gives a "
                        "2x2 grid and 16 a 4x4 one, while other counts leave a part-row or "
                        "part-column and so return a few tiles more than requested."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGES",
                    tooltip=(
                        "The tiles as one batch, ordered left to right then top to bottom. "
                        "Every tile is the same size; edge tiles are padded with black where "
                        "the image ran out."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, num_tiles=6) -> io.NodeOutput:
        folded = dynamic.fold(image)
        image = folded.images
        # The output batch is the tiles, so only one image is cut up: tiling several would
        # mix two meanings into one batch.
        image = tensor2pil(image_planes(image)[0])
        img_width, img_height = image.size

        num_rows = int(num_tiles**0.5)
        num_cols = (num_tiles + num_rows - 1) // num_rows
        tile_width = img_width // num_cols
        tile_height = img_height // num_rows

        tiles = []
        for y in range(0, img_height, tile_height):
            for x in range(0, img_width, tile_width):
                tile = image.crop((x, y, x + tile_width, y + tile_height))
                tiles.append(pil2tensor(tile))

        cut = torch.stack(tiles, dim=0).squeeze(1)
        size_report.publish(
            image.size,
            (tile_width, tile_height),
            action="cut into tiles",
            facts={"tiles": f"{len(tiles)} of {num_cols} by {num_rows}"},
        )
        return io.NodeOutput(dynamic.unfold(cut, folded))
