"""Answer whether a wire is carrying nothing, and pass it on."""

from __future__ import annotations

from comfy_api.latest import io


class AnyIsEmpty(io.ComfyNode):
    """Test whatever is connected for emptiness."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAnyIsEmpty",
            display_name="Any Is Empty",
            search_aliases=[
                "WASAnyIsEmpty",
                "Any Is Empty",
                "is empty",
                "is blank",
                "has value",
                "null check",
            ],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Answer whether a wire is carrying nothing, whatever type it is, and pass "
                "the value straight through. An empty mask, a blank line of text, an empty "
                "list and a batch of no frames all read as empty, so a graph can branch on "
                "a stage that produced nothing."
            ),
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=io.MatchType.Template("any_is_empty"),
                    tooltip="Anything. The wire is read and passed on unchanged.",
                ),
                io.Boolean.Input(
                    "zero_is_empty",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Count a mask or an image that is entirely black as empty. Off, only "
                        "a batch of no frames counts."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    display_name="is_empty",
                    tooltip="true when nothing is being carried. Wire it to a switch.",
                ),
                io.MatchType.Output(
                    template=io.MatchType.Template("any_is_empty"),
                    display_name="value",
                    tooltip="The same value, unchanged, so the node sits in the middle of a chain.",
                ),
                io.String.Output(
                    display_name="reason",
                    tooltip="Why it reads as empty, or what it holds when it does not.",
                ),
            ],
        )

    @classmethod
    def execute(cls, value=None, zero_is_empty=False) -> io.NodeOutput:
        """Answer whether the value is empty.

        Args:
            value: Whatever is wired in.
            zero_is_empty: Count an all-zero tensor as empty.

        Returns:
            Whether it is empty, the value unchanged, and why.
        """
        from ....modules.logic.describe import describe_value

        found = describe_value(value)
        empty, reason = found["is_empty"], ""
        if empty:
            reason = f"{found['type_name']} carrying nothing"
        elif zero_is_empty and hasattr(value, "shape"):
            try:
                if not bool(value.any()):
                    empty, reason = True, f"{found['type_name']} {found['shape']} is entirely zero"
            except Exception:
                reason = ""
        if not empty and not reason:
            reason = f"{found['type_name']} {found['shape']}".strip()
        return io.NodeOutput(empty, value, reason)
