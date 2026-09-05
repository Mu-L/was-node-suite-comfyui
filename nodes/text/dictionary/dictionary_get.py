"""Read one key out of a dictionary."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import DICT


class DictionaryGet(io.ComfyNode):
    """Look a key up in a dictionary and emit its value as text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Dictionary Get",
            display_name="Text Dictionary Get",
            search_aliases=["Text Dictionary Get", "dictionary lookup", "dict get"],
            category="WAS Suite/Text/Dictionary",
            description=(
                "Look up a key in a dictionary and emit its value as text. A key that is "
                "not in the dictionary emits the default value."
            ),
            inputs=[
                DICT.Input(
                    "dictionary",
                    tooltip="The dictionary to read from.",
                ),
                io.String.Input(
                    "key",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Name of the entry to fetch, for example 'subject'. Matching is "
                        "exact, so case and spaces have to line up with the key as it was "
                        "stored."
                    ),
                ),
                io.String.Input(
                    "default_value",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "Emitted when the dictionary has no such key. Left empty, a missing "
                        "key gives an empty string rather than failing the prompt."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The entry's value as text. A value that is a list or a number is "
                        "rendered the way python prints it, so a list arrives as "
                        "['a', 'b']."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, dictionary, key, default_value="") -> io.NodeOutput:
        """Look the key up.

        Raises:
            ValueError: Nothing is connected to the dictionary input.
        """
        require_input(
            dictionary,
            "Text Dictionary Get",
            "dictionary",
            "dictionary",
            "Text Dictionary New or Text Dictionary Convert",
            "DICT",
        )
        return io.NodeOutput(str(dictionary.get(key, default_value)))
