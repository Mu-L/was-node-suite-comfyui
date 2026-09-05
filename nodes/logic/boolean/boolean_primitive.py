"""A standalone boolean value."""

from __future__ import annotations

from comfy_api.latest import io


class LogicBooleanPrimitive(io.ComfyNode):
    """Emit the boolean set on its widget."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Logic Boolean Primitive",
            display_name="Logic Boolean Primitive",
            search_aliases=["Logic Boolean Primitive", "boolean primitive", "bool"],
            category="WAS Suite/Logic/Boolean",
            description=(
                "A single true/false checkbox on a node of its own, so one switch can drive "
                "the boolean input of several nodes at once."
            ),
            inputs=[
                io.Boolean.Input(
                    "boolean",
                    default=False,
                    tooltip="The value to send on: ticked is true, unticked is false.",
                ),
            ],
            outputs=[
                io.Boolean.Output(tooltip="The state of the checkbox, true or false."),
            ],
        )

    @classmethod
    def execute(cls, boolean) -> io.NodeOutput:
        return io.NodeOutput(boolean)
