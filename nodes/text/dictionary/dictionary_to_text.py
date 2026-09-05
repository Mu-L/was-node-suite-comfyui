"""Render a dictionary as text."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import DICT


class DictionaryToText(io.ComfyNode):
    """Render a dictionary as its Python representation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Dictionary To Text",
            display_name="Text Dictionary To Text",
            search_aliases=["Text Dictionary To Text", "dictionary to string", "dict repr"],
            category="WAS Suite/Text/Dictionary",
            description=(
                "Render a dictionary as text. The result uses single quotes and reads "
                "back through Text Dictionary Convert."
            ),
            inputs=[
                DICT.Input(
                    "dictionary",
                    tooltip="The dictionary to write out as text.",
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The dictionary written out on one line, for example "
                        "{'subject': 'a cat'}. Useful for a preview or a saved text file, "
                        "and Text Dictionary Convert reads it back."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, dictionary) -> io.NodeOutput:
        return io.NodeOutput(str(dictionary))
