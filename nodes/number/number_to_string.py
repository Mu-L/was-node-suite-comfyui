"""Render a NUMBER as a string."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class NumberToString(io.ComfyNode):
    """Convert a NUMBER to its ``str`` representation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number to String",
            display_name="Number to String",
            search_aliases=["Number to String", "Number to Text", "str", "text", "convert"],
            category="WAS Suite/Number/Operations",
            description=(
                "Convert a NUMBER to a STRING. The value keeps the type it arrives with, so "
                "an integer renders as '8' and a float as '8.0'."
            ),
            inputs=[
                io.MultiType.Input(
                    "number",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The value to write out. Put a Number to Int in front of it to lose "
                        "a trailing '.0'."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The number written out in full, with no padding, thousands "
                        "separators or currency signs: 8, 8.0, -1.25."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, number) -> io.NodeOutput:
        return io.NodeOutput(str(number))
