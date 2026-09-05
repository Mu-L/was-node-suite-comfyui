"""Mirror a batch of images."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


class ImageFlip(io.ComfyNode):
    """Mirror every image in a batch along one axis."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Flip",
            display_name="Image Flip",
            search_aliases=["Image Flip", "mirror", "flop", "reverse image"],
            category="WAS Suite/Image/Transform",
            description=(
                "Mirror every image in the batch. The size does not change and no pixels "
                "are resampled, so nothing is lost."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The images to mirror.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["horizontal", "vertical"],
                    tooltip=(
                        "Which axis to mirror across. `horizontal` swaps left and right, "
                        "as a mirror does; `vertical` swaps top and bottom, as still water "
                        "does."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The mirrored images, in the order they arrived.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, mode) -> io.NodeOutput:
        from PIL import Image

        def turned(image):
            if mode == "horizontal":
                return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mode == "vertical":
                return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            return image

        return io.NodeOutput(filtered_planes(images, turned))
