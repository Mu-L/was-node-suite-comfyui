"""Outline extraction with two convolution kernels."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes

#: Laplacian kernel: an 8-weighted centre against its eight neighbours, which cancels flat
#: areas to black and leaves a bright line wherever the brightness changes.
LAPLACIAN_KERNEL = (-1, -1, -1, -1, 8, -1, -1, -1, -1)


class ImageEdgeDetectionFilter(io.ComfyNode):
    """Reduce an image to the outlines of whatever it contains."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Edge Detection Filter",
            display_name="Image Edge Detection Filter",
            search_aliases=[
                "Image Edge Detection Filter",
                "edge detect",
                "outline",
                "laplacian",
                "find edges",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Turn an image into an outline drawing: black where the picture is flat, "
                "bright where the brightness changes. Useful as a control image or as a "
                "line-art layer."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to outline. A batch is handled one image at a time.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["normal", "laplacian"],
                    tooltip=(
                        "Which outline to draw. `normal` finds edges from the difference "
                        "between neighbouring pixels and gives thin, soft lines; `laplacian` "
                        "uses a sharper kernel that responds harder to fine detail and noise, "
                        "so its lines are brighter and busier."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The outlines, bright on a near-black background."),
            ],
        )

    @classmethod
    def execute(cls, image, mode) -> io.NodeOutput:
        from PIL import ImageFilter

        def outline(edges):
            if mode == "normal":
                return edges.filter(ImageFilter.FIND_EDGES)
            if mode == "laplacian":
                return edges.filter(ImageFilter.Kernel((3, 3), LAPLACIAN_KERNEL, 1, 0))
            return edges

        return io.NodeOutput(filtered_planes(image, outline))
