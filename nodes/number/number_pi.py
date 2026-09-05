"""The constant pi as a NUMBER."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class NumberPI(io.ComfyNode):
    """Emit ``math.pi``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number PI",
            display_name="Number PI",
            search_aliases=["Number PI", "pi", "constant", "math"],
            category="WAS Suite/Number",
            description=(
                "Emit the constant pi, 3.141592653589793, for maths built out of the Number "
                "Operation nodes. It has no settings."
            ),
            inputs=[],
            outputs=[
                NUMBER.Output(
                    tooltip="Pi on a NUMBER socket, for a Number Operation input.",
                ),
                io.Float.Output(
                    tooltip="The same value on a FLOAT socket, for a core node's float widget.",
                ),
            ],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(math.pi, math.pi)
