"""Emit a dictionary's keys as a list."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import DICT, LIST


class DictionaryKeys(io.ComfyNode):
    """Emit the keys of a dictionary."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Dictionary Keys",
            display_name="Text Dictionary Keys",
            search_aliases=["Text Dictionary Keys", "dictionary keys", "dict keys"],
            category="WAS Suite/Text/Dictionary",
            description="Emit the keys of a dictionary, in insertion order.",
            inputs=[
                DICT.Input(
                    "dictionary",
                    tooltip="The dictionary whose entry names are wanted.",
                ),
            ],
            outputs=[
                LIST.Output(
                    tooltip=(
                        "The names of the dictionary's entries, in the order they were added. "
                        "Text List to Text joins them into one string; Text List Concatenate "
                        "skips them, because they arrive as a live view of the dictionary "
                        "rather than as a plain list."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, dictionary) -> io.NodeOutput:
        """Emit the keys.

        Raises:
            ValueError: Nothing is connected to the dictionary input.
        """
        require_input(
            dictionary,
            "Text Dictionary Keys",
            "dictionary",
            "dictionary",
            "Text Dictionary New or Text Dictionary Convert",
            "DICT",
        )
        # A keys view, not a list. Text List to Text joins it; Text List Concatenate
        # tests isinstance(value, list) and drops it, which is the behaviour every saved
        # workflow was built against.
        return io.NodeOutput(dictionary.keys())
