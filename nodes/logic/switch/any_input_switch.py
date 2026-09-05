"""Route one of two values of any type onward."""

from __future__ import annotations

from comfy_api.latest import io


class AnyInputSwitch(io.ComfyNode):
    """Select between two inputs of any type with a boolean.

    Both inputs are lazy, so the unselected branch is never evaluated.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("any_switch")
        return io.Schema(
            node_id="WASAnyInputSwitch",
            display_name="Any Input Switch",
            search_aliases=[
                "WASAnyInputSwitch", "Any Input Switch",
                "switch",
                "any switch",
                "boolean switch",
                "route",
                "branch",
                "if else",
            ],
            category="WAS Suite/Logic/Switch",
            description=(
                "Pass one of two values on, chosen by a boolean, whatever type they are. "
                "The unselected input is not evaluated, so the work behind it is skipped."
            ),
            inputs=[
                io.MatchType.Input(
                    "input_a",
                    template=template,
                    lazy=True,
                    tooltip=(
                        "Passed on when boolean is true. Any type. The first connection "
                        "fixes the type; input_b and output then take that type only."
                    ),
                ),
                io.MatchType.Input(
                    "input_b",
                    template=template,
                    lazy=True,
                    tooltip="Passed on when boolean is false. Must match input_a's type.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip="Selects the input. true = input_a, false = input_b.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="output",
                    tooltip="The selected input, typed to whatever was connected.",
                ),
            ],
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, input_a=None, input_b=None) -> list[str]:
        """Ask only for the branch that is about to be used.

        Args:
            boolean: Which input the run will select.
            input_a: The true branch, present once it has been evaluated.
            input_b: The false branch, present once it has been evaluated.

        Returns:
            The input still needed, or an empty list once it has arrived.
        """
        if boolean and input_a is None:
            return ["input_a"]
        if not boolean and input_b is None:
            return ["input_b"]
        return []

    @classmethod
    def execute(cls, input_a=None, input_b=None, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(input_a if boolean else input_b)
