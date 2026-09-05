"""Boolean XOR of two inputs."""

from __future__ import annotations

from comfy_api.latest import io


class LogicComparisonXOR(io.ComfyNode):
    """Emit the logical XOR of two booleans."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Logic Comparison XOR",
            display_name="Logic Comparison XOR",
            search_aliases=["Logic Comparison XOR", "xor", "boolean xor"],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Compare two true/false values and report whether they disagree: true when "
                "exactly one of them is true, false when they match."
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
                    tooltip="The second condition, compared against the first.",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    tooltip=(
                        "True when exactly one input is true; false when both are true or "
                        "both are false."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, boolean_a, boolean_b) -> io.NodeOutput:
        return io.NodeOutput(boolean_a != boolean_b)
