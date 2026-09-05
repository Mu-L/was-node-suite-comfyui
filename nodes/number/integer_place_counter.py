"""Count the digits of an integer."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log

logger = log.get_logger("nodes.number")


class IntegerPlaceCounter(io.ComfyNode):
    """Count the decimal places an integer occupies.

    The count is the length of the decimal spelling, so 0 has one place and 1000 has four.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Integer place counter",
            display_name="Integer Place Counter",
            search_aliases=["Integer place counter", "digits", "places", "length"],
            category="WAS Suite/Number/Operations",
            description=(
                "Count how many decimal digits an integer has, which is the padding width a "
                "zero-padded frame or batch number needs."
            ),
            inputs=[
                io.Int.Input(
                    "int_input",
                    default=0,
                    min=0,
                    max=10000000,
                    step=1,
                    tooltip=(
                        "The whole number to measure. Usually linked from a frame or batch "
                        "count rather than typed in."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="INT_PLACES",
                    tooltip=(
                        "How many digits the number is written with: 1 for 0 through 9, 3 "
                        "for 100, 4 for 1000."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, int_input) -> io.NodeOutput:
        places = len(str(int_input))
        logger.info("Integer places count: %s", places)
        return io.NodeOutput(places)
