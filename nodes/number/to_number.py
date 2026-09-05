"""Put a whole number, a decimal or a switch onto the pack's own number wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class ToNumber(io.ComfyNode):
    """Convert an INT, a FLOAT or a BOOLEAN to a NUMBER."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASToNumber",
            display_name="To Number",
            search_aliases=[
                "WASToNumber",
                "To Number",
                "int to number",
                "float to number",
                "boolean to number",
                "convert number",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Put a whole number, a decimal or a switch onto NUMBER, the wire this pack's "
                "own arithmetic runs on. Anything answering an INT, a FLOAT or a BOOLEAN can "
                "then feed a node that takes only a NUMBER, so a size, a count or a flag from "
                "elsewhere joins a chain of number nodes without a node in between."
            ),
            inputs=[
                io.MultiType.Input(
                    "value",
                    [io.Int, io.Float, io.Boolean],
                    tooltip=(
                        "What to convert. A whole number and a decimal pass through as they "
                        "are, and a switch becomes 1 for true and 0 for false."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    display_name="number",
                    tooltip=(
                        "The same value on the NUMBER wire. A decimal keeps its fraction, so "
                        "feed it to a node that rounds where a whole number is wanted."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        """Answer the value as a number.

        Args:
            value: An int, a float or a bool, as the connected socket answered it.

        Returns:
            The value, with a bool read as 1 or 0.

        Raises:
            ValueError: The value is not something that reads as a number.
        """
        if isinstance(value, bool):
            return io.NodeOutput(int(value))
        if isinstance(value, (int, float)):
            return io.NodeOutput(value)
        try:
            text = str(value).strip()
            return io.NodeOutput(int(text) if text.lstrip("+-").isdigit() else float(text))
        except (TypeError, ValueError) as unreadable:
            raise ValueError(
                f"To Number was given {value!r}, which is not a whole number, a decimal or a "
                f"switch. Connect an INT, a FLOAT or a BOOLEAN"
            ) from unreadable
