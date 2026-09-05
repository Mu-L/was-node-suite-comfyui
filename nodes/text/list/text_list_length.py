"""Count the entries of a list."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LIST, NUMBER


class TextListLength(io.ComfyNode):
    """Count the entries of a ``LIST``, on the NUMBER, INT and FLOAT sockets.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextListLength",
            display_name="Text List Length",
            search_aliases=[
                "WASTextListLength", "Text List Length",
                "list length",
                "list size",
                "count list",
                "number list",
                "list of numbers",
                "count numbers",
                "how many values",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Count the entries in a list, as a NUMBER, an INT and a FLOAT. Also reports "
                "whether the list is empty, for a switch that has to handle that case."
            ),
            inputs=[
                LIST.Input(
                    "text_list",
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "The list to count, such as the LIST output of Text Split to List, "
                        "Text List, Text Dictionary Keys or Image Color Palette."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The entry count, for any node taking a NUMBER, Number Operation, "
                        "or the index of Text List Get."
                    ),
                ),
                io.Int.Output(
                    display_name="length",
                    tooltip="The same count as a whole number.",
                ),
                io.Float.Output(
                    display_name="length_float",
                    tooltip=(
                        "The same count as a decimal, for the division a progress fraction "
                        "needs without a conversion node in between."
                    ),
                ),
                io.Boolean.Output(
                    display_name="is_empty",
                    tooltip=(
                        "True when the list holds nothing. Wire it into a switch to route "
                        "around the nodes that would fail on an empty list."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text_list) -> io.NodeOutput:
        entries = list(text_list) if isinstance(text_list, (list, tuple)) else [text_list]
        length = len(entries)
        return io.NodeOutput(length, length, float(length), length == 0)
