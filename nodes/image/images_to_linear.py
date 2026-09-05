"""Force a batch of images to a single luminance channel."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.convert.tensors import filtered_planes


class ImagesToLinear(io.ComfyNode):
    """Convert every image in a batch to single-channel greyscale."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Images to Linear",
            display_name="Images to Linear",
            search_aliases=["Images to Linear", "greyscale", "grayscale", "luminance"],
            category="WAS Suite/Image",
            description=(
                "Flatten every image in the batch to one greyscale channel, weighted the "
                "way the eye sees brightness, which is the form the depth and mask nodes "
                "expect. Colour is discarded and cannot be recovered afterwards. 'Linear' "
                "is PIL's name for this single-channel mode; no gamma conversion is applied."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The images to flatten to brightness only.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The same images with one channel each. Nodes that insist on three "
                        "channels need Images to RGB after this."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        return io.NodeOutput(filtered_planes(images, lambda plane: plane.convert("L")))
