"""Boolean AND of two inputs."""

from __future__ import annotations

from comfy_api.latest import io


class LogicComparisonAND(io.ComfyNode):
    """Emit the logical AND of two booleans."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Logic Comparison AND",
            display_name="Logic Comparison AND",
            search_aliases=["Logic Comparison AND", "and", "boolean and"],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Combine two true/false values so the result is true only when both of them "
                "are, which is how two conditions are required at once."
            ),
            inputs=[
                io.Boolean.Input(
                    "boolean_a",
                    default=False,
                    tooltip="The first condition. Usually linked from another logic node.",
                ),
                io.Boolean.Input(
                    "boolean_b",
                    default=False,
                    tooltip="The second condition, which also has to hold for a true result.",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    tooltip="True when both inputs are true, false if either one is false.",
                ),
            ],
        )

    @classmethod
    def execute(cls, boolean_a, boolean_b) -> io.NodeOutput:
        return io.NodeOutput(boolean_a and boolean_b)
