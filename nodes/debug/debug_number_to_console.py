"""Print a number to the console and pass it through."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat.types import NUMBER
from ...modules.log import get_logger

logger = get_logger("nodes.debug")


class DebugNumberToConsole(io.ComfyNode):
    """Log the value on a NUMBER wire under a user-supplied heading."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Debug Number to Console",
            display_name="Debug Number to Console",
            search_aliases=["Debug Number to Console", "print number", "log number"],
            category="WAS Suite/Debug",
            description="Print a number to the console and pass it through unchanged.",
            inputs=[
                io.MultiType.Input(
                    "number",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The value to print. As well as this pack's NUMBER wire it accepts "
                        "a plain INT or FLOAT, so a core node's numeric output can be "
                        "inspected without a conversion node in between."
                    ),
                ),
                io.String.Input(
                    "label",
                    default="Debug to Console",
                    multiline=False,
                    tooltip=(
                        "Heading printed on the line above the value, so several of these "
                        "nodes can be told apart in the console. Left empty, the heading "
                        "is 'Debug to Console'."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The same value that came in, unchanged, so the node can sit in "
                        "the middle of a chain instead of ending it."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, number, label) -> io.NodeOutput:
        heading = label if label.strip() != "" else "Debug to Console"
        logger.info("%s:\n%s", heading, number)
        return io.NodeOutput(number, ui=ui.PreviewText(str(number)))

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """NaN never compares equal to itself, so the value is printed on every run."""
        return float("NaN")
