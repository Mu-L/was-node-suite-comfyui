"""Convert a STRING widget value to the suite's TEXT socket type."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "text_type"


class StringToText(io.ComfyNode):
    """Pass a string through unchanged."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="String to Text",
            display_name="String to Text",
            search_aliases=["String to Text", "string", "text", "convert"],
            category="WAS Suite/Text/Operations",
            description=(
                "Deprecated. Nothing replaces it: every text socket is a plain STRING, so a "
                "string wires straight into a text input and this node does nothing. Delete "
                "it from a workflow and join the two wires it sat between."
            ),
            inputs=[
                io.String.Input(
                    "string",
                    tooltip=(
                        "The string to pass on. Nothing is done to it: text inputs accept a "
                        "string directly, so this node can be deleted from a workflow and "
                        "its two wires joined."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(tooltip="The input string, unchanged."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, string) -> io.NodeOutput:
        return io.NodeOutput(string)
