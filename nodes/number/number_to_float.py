"""Widen a NUMBER to a float."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class NumberToFloat(io.ComfyNode):
    """Convert a NUMBER to a FLOAT."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number to Float",
            display_name="Number to Float",
            search_aliases=["Number to Float", "float", "convert"],
            category="WAS Suite/Number/Operations",
            description=(
                "Hand a value on as a decimal FLOAT, so a NUMBER wire from this pack can "
                "reach a core node's float input such as a CFG or a denoise."
            ),
            inputs=[
                io.MultiType.Input(
                    "number",
                    [NUMBER, io.Int, io.Float],
                    tooltip="The value to hand on. Nothing is rounded or clamped.",
                ),
            ],
            outputs=[
                io.Float.Output(
                    tooltip="The same value as a decimal, so 8 leaves here as 8.0.",
                ),
            ],
        )

    @classmethod
    def execute(cls, number) -> io.NodeOutput:
        return io.NodeOutput(float(number))
