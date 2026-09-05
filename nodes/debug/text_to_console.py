"""Print a string to the console and pass it through."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.log import get_logger

logger = get_logger("nodes.debug")


class TextToConsole(io.ComfyNode):
    """Log the text arriving on the socket under a user-supplied heading."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text to Console",
            display_name="Text to Console",
            search_aliases=["Text to Console", "print", "debug text", "log"],
            category="WAS Suite/Debug",
            description="Print connected text to the console and pass it through unchanged.",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: a cat on a mat",
                    tooltip=(
                        "Text to print in the terminal; STRING, as `a tabby cat`. Passed through "
                        "unchanged."
                    ),
                ),
                io.String.Input(
                    "label",
                    default="Text Output",
                    multiline=False,
                    tooltip=(
                        "Heading printed on the line above the text, so several of these "
                        "nodes can be told apart in the console. Left empty, the heading "
                        "is 'Text to Console'."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The same text that came in, unchanged, so the node can sit in the "
                        "middle of a chain instead of ending it."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, text, label) -> io.NodeOutput:
        heading = label if label.strip() != "" else "Text to Console"
        rendered = str(text)
        logger.info("%s:\n%s", heading, rendered)
        return io.NodeOutput(text, ui=ui.PreviewText(rendered))
