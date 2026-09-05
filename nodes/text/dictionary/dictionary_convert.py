"""Read a dictionary out of its text representation."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import DICT


class DictionaryConvert(io.ComfyNode):
    """Parse a Python dictionary literal into a DICT socket."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Dictionary Convert",
            display_name="Text Dictionary Convert",
            search_aliases=["Text Dictionary Convert", "text to dictionary", "parse dict"],
            category="WAS Suite/Text/Dictionary",
            description=(
                "Convert text holding a dictionary literal into a dictionary. The text is "
                "read with ast.literal_eval, which accepts single quotes as well as JSON's "
                "double quotes and executes nothing."
            ),
            inputs=[
                io.String.Input(
                    "dictionary_text",
                    multiline=True,
                    placeholder="Eg: {'subject': 'a cat'}",
                    tooltip=(
                        "Dictionary literal; STRING. Single or double quotes both work. "
                        "Eg: {'subject': 'a cat'}"
                    ),
                ),
            ],
            outputs=[
                DICT.Output(
                    tooltip="The dictionary the text described, for the other DICT nodes.",
                ),
            ],
        )

    @classmethod
    def execute(cls, dictionary_text) -> io.NodeOutput:
        # literal_eval rather than json.loads: the text is not guaranteed to use double
        # quotes, and str(dict), what Text Dictionary To Text emits, never does.
        import ast

        return io.NodeOutput(ast.literal_eval(dictionary_text))
