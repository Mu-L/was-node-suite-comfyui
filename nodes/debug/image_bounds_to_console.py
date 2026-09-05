"""Print image bounds to the console and pass them through."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat.types import IMAGE_BOUNDS
from ...modules.log import get_logger

logger = get_logger("nodes.debug")


class ImageBoundsToConsole(io.ComfyNode):
    """Log every ``(rmin, rmax, cmin, cmax)`` row on an IMAGE_BOUNDS wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Bounds to Console",
            display_name="Image Bounds to Console",
            search_aliases=["Image Bounds to Console", "print bounds", "debug bounds"],
            category="WAS Suite/Debug",
            description="Print image bounds to the console and pass them through unchanged.",
            inputs=[
                IMAGE_BOUNDS.Input(
                    "image_bounds",
                    tooltip=(
                        "Bounding boxes to print, from a node such as Image Bounds or Inset "
                        "Image Bounds. One line is printed per box, giving its first and "
                        "last pixel row and its first and last pixel column as "
                        "(rmin, rmax, cmin, cmax)."
                    ),
                ),
                io.String.Input(
                    "label",
                    default="Debug to Console",
                    multiline=False,
                    tooltip=(
                        "Heading printed on the line above the boxes, so several of these "
                        "nodes can be told apart in the console. Left empty, the heading is "
                        "'Debug to Console'."
                    ),
                ),
            ],
            outputs=[
                IMAGE_BOUNDS.Output(
                    tooltip=(
                        "The same bounding boxes that came in, unchanged, so the node can "
                        "be dropped between a bounds producer and a crop."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image_bounds, label) -> io.NodeOutput:
        heading = label if label.strip() != "" else "Debug to Console"
        rows = ", \n    ".join(
            "\t(rmin={}, rmax={}, cmin={}, cmax={})".format(a, b, c, d)
            for a, b, c, d in image_bounds
        )
        rendered = f"[\n{rows}\n]"
        logger.info("%s:\n%s", heading, rendered)
        return io.NodeOutput(image_bounds, ui=ui.PreviewText(rendered))

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """NaN never compares equal to itself, so the bounds are printed on every run."""
        return float("NaN")
