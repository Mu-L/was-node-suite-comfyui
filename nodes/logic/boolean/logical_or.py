"""Boolean OR of two inputs."""

from __future__ import annotations

from comfy_api.latest import io


class LogicComparisonOR(io.ComfyNode):
    """Emit the logical OR of two booleans."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Logic Comparison OR",
            display_name="Logic Comparison OR",
            search_aliases=["Logic Comparison OR", "or", "boolean or"],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Combine two true/false values so the result is true when at least one of "
                "them is, which is how either of two conditions is accepted."
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
                    tooltip="The second condition, which can carry the result on its own.",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    tooltip=(
                        "True when either input is true, or both are; false only when "
                        "neither is."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, boolean_a, boolean_b) -> io.NodeOutput:
        return io.NodeOutput(boolean_a or boolean_b)
