"""Put a LIST back onto a plain STRING socket, as a list."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LIST


class TextListToStrings(io.ComfyNode):
    """Unpack a ``LIST`` onto a ``STRING`` socket declared ``is_output_list``.

    Text List to Text is the other direction, collapsing a list into one string.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextListToStrings",
            display_name="Text List to Strings",
            search_aliases=[
                "WASTextListToStrings", "Text List to Strings",
                "list to strings",
                "unpack list",
                "iterate list",
                "output is list",
                "number list",
                "fan out",
                "one per run",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Turn a LIST into a STRING list, which runs every node downstream once per "
                "entry. The way to feed a list into nodes that take plain text."
            ),
            inputs=[
                LIST.Input(
                    "text_list",
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "The list to unpack. Any LIST output does: Text Split to List, Text "
                        "List, Text Dictionary Keys, Image Color Palette."
                    ),
                ),
                io.Boolean.Input(
                    "remove_empty",
                    default=False,
                    tooltip=(
                        "Whether entries holding nothing are dropped before the list is "
                        "handed on. Each empty entry would otherwise cost a full run of "
                        "everything downstream to produce a result from an empty prompt."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="strings",
                    is_output_list=True,
                    tooltip=(
                        "The entries, one per run. A node reading this executes once for "
                        "each of them, so a list of six prompts renders six images. An "
                        "empty list stops the prompt, because a graph cannot be run zero "
                        "times."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many entries were handed on, which is how many times the graph "
                        "below this node runs."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text_list, remove_empty=False) -> io.NodeOutput:
        from ....modules.compat.lists import as_list, require_values

        values = [str(entry) for entry in as_list(text_list)]
        if remove_empty:
            values = [value for value in values if value.strip()]
        require_values(
            values,
            "Text List to Strings was given a list with nothing in it to unpack, so the "
            "graph below it cannot be run. Check the node feeding text_list, or turn "
            "remove_empty off if every entry was blank.",
        )
        return io.NodeOutput(values, len(values))
