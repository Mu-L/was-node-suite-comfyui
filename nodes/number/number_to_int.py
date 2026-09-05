"""Truncate a NUMBER to an integer."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class NumberToInt(io.ComfyNode):
    """Convert a NUMBER to an INT, discarding any fractional part."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number to Int",
            display_name="Number to Int",
            search_aliases=["Number to Int", "int", "integer", "convert"],
            category="WAS Suite/Number/Operations",
            description=(
                "Hand a value on as a whole INT, so a NUMBER wire from this pack can reach a "
                "core node's integer input such as steps, width or a seed."
            ),
            inputs=[
                io.MultiType.Input(
                    "number",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The value to make whole. Any fraction is cut off rather than "
                        "rounded, so 2.9 gives 2 and -2.9 gives -2."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(tooltip="The value with its fractional part removed."),
            ],
        )

    @classmethod
    def execute(cls, number) -> io.NodeOutput:
        return io.NodeOutput(int(number))
