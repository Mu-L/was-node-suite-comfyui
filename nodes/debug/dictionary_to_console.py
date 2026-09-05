"""Pretty-print a dictionary to the console and pass it through."""

from __future__ import annotations

from pprint import pformat

from comfy_api.latest import io, ui

from ...modules.compat.types import DICT
from ...modules.log import get_logger

logger = get_logger("nodes.debug")


class DictionaryToConsole(io.ComfyNode):
    """Log a dictionary under a user-supplied heading and return it unchanged."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Dictionary to Console",
            display_name="Dictionary to Console",
            search_aliases=["Dictionary to Console", "print dict", "debug dictionary"],
            category="WAS Suite/Debug",
            description="Pretty-print a dictionary to the console and pass it through unchanged.",
            inputs=[
                DICT.Input(
                    "dictionary",
                    tooltip=(
                        "The dictionary to print. It is laid out over several indented "
                        "lines rather than crammed onto one, so a nested structure such as "
                        "the one Load Text File returns stays readable."
                    ),
                ),
                io.String.Input(
                    "label",
                    default="Dictionary Output",
                    multiline=False,
                    tooltip=(
                        "Heading printed on the line above the dictionary, so several of "
                        "these nodes can be told apart in the console. Left empty, the "
                        "heading is 'Dictionary Output'."
                    ),
                ),
            ],
            outputs=[
                DICT.Output(
                    tooltip=(
                        "The same dictionary that came in, unchanged, so the node can sit "
                        "in the middle of a chain instead of ending it."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, dictionary, label) -> io.NodeOutput:
        heading = label if label.strip() != "" else "Dictionary Output"
        rendered = pformat(dictionary, indent=4)
        logger.info("%s:\n%s", heading, rendered)
        return io.NodeOutput(dictionary, ui=ui.PreviewText(rendered))
