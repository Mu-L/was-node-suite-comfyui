"""Turn a 0-1 float into a boolean and its numeric equivalents."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER


class LogicBoolean(io.ComfyNode):
    """Round a 0-1 float to a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Logic Boolean",
            display_name="Logic Boolean",
            search_aliases=["Logic Boolean", "boolean", "bool"],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Turn a value between 0.0 and 1.0 into a true/false switch and the numbers "
                "that stand for it, which is how the Input Switch nodes and any node with a "
                "reset or toggle input get fed from one control."
            ),
            inputs=[
                io.Float.Input(
                    "boolean",
                    default=1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "The value to decide on, between 0.0 and 1.0. Anything above 0.5 "
                        "counts as true and anything at or below 0.5 as false, so 0.5 "
                        "itself is false."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    tooltip="True or false, for the boolean input of an Input Switch node.",
                ),
                NUMBER.Output(
                    tooltip="The same decision as 1 or 0, for a NUMBER input such as a reset.",
                ),
                io.Int.Output(tooltip="The same decision as 1 or 0, on an INT socket."),
                io.Float.Output(
                    tooltip=(
                        "The widget value itself, not rounded, so 0.35 leaves here as 0.35 "
                        "while the other three outputs read false."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, boolean=1.0) -> io.NodeOutput:
        rounded = int(round(boolean))
        return io.NodeOutput(bool(rounded), rounded, rounded, boolean)
