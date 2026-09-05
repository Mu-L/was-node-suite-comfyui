"""Convert the suite's TEXT socket type to a plain STRING."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "text_type"


class TextToString(io.ComfyNode):
    """Pass text through unchanged."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text to String",
            display_name="Text to String",
            search_aliases=["Text to String", "text", "string", "convert"],
            category="WAS Suite/Text/Operations",
            description=(
                "Deprecated. Nothing replaces it: every text socket is a plain STRING, so a "
                "text output wires straight into a string input and this node does nothing. "
                "Delete it from a workflow and join the two wires it sat between."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: any text",
                    tooltip=(
                        "Text to pass through unchanged; STRING, as `a tabby cat`. String inputs take text "
                        "outputs directly."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(tooltip="The input text, unchanged."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        return io.NodeOutput(text)
