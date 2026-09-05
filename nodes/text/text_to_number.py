"""Parse a string into a number."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class TextToNumber(io.ComfyNode):
    """Convert ``text`` to a number."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text to Number",
            display_name="Text to Number",
            search_aliases=["Text to Number", "parse number", "string to number", "atoi"],
            category="WAS Suite/Text/Operations",
            description=(
                "Parse a string into a number: a float when it contains a decimal point, "
                "an int otherwise."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: 42 or 3.5",
                    tooltip=(
                        "Number written as text; STRING. A decimal point gives a "
                        "decimal, otherwise a whole number. Non-numeric text fails the "
                        "prompt. Eg: 3.5"
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The parsed value, still a whole number or a decimal depending on "
                        "the text. For the NUMBER inputs of the suite's own maths nodes."
                    ),
                ),
                io.Float.Output(tooltip="The same value as a decimal, for example 42.0."),
                io.Int.Output(
                    tooltip=(
                        "The same value as a whole number, with anything after the decimal "
                        "point dropped: 3.9 becomes 3 and -3.9 becomes -3."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        if "." in text:
            number = float(text)
        else:
            number = int(text)
        return io.NodeOutput(number, float(number), int(number))
