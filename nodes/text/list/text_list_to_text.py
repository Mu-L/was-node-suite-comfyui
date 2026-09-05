"""Join a list of text into one string."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LIST


class TextListToText(io.ComfyNode):
    """Join a list into one string with a delimiter."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text List to Text",
            display_name="Text List to Text",
            search_aliases=["Text List to Text", "join list", "list to string"],
            category="WAS Suite/Text/List",
            description=(
                "Join a list into one string, separated by the delimiter. Type \\n as the "
                "delimiter to join with newlines."
            ),
            inputs=[
                io.String.Input(
                    "delimiter",
                    default=", ",
                    tooltip=(
                        "Placed between the entries. The default ', ' builds a "
                        "comma-separated prompt; type \\n to put each entry on its own line; "
                        "leave it empty to run them together with nothing between."
                    ),
                ),
                LIST.Input(
                    "text_list",
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "The list to flatten, for example the lines of a file from Text Load "
                        "Line From File or the entries of Text List. Every entry has to be "
                        "text; a list holding a number fails the prompt."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip="The list's entries as one string, separated by the delimiter.",
                ),
            ],
        )

    @classmethod
    def execute(cls, delimiter, text_list) -> io.NodeOutput:
        if delimiter == "\\n":
            delimiter = "\n"
        return io.NodeOutput(delimiter.join(text_list))
