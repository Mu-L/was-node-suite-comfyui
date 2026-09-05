"""Boolean negation."""

from __future__ import annotations

from comfy_api.latest import io


class LogicNOT(io.ComfyNode):
    """Emit the logical negation of a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Logic NOT",
            display_name="Logic NOT",
            search_aliases=["Logic NOT", "not", "invert boolean"],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Flip a true/false value over, which turns one condition into its opposite "
                "without a second comparison node."
            ),
            inputs=[
                io.Boolean.Input(
                    "boolean",
                    default=False,
                    tooltip="The value to invert. True comes out false, false comes out true.",
                ),
            ],
            outputs=[
                io.Boolean.Output(tooltip="The opposite of the input value."),
            ],
        )

    @classmethod
    def execute(cls, boolean) -> io.NodeOutput:
        return io.NodeOutput(not boolean)
