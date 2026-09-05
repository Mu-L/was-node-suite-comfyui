"""Round a NUMBER up to a multiple."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class NumberMultipleOf(io.ComfyNode):
    """Round a number up to the next multiple of ``multiple``.

    A number that is already a multiple is passed through unchanged.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number Multiple Of",
            display_name="Number Multiple Of",
            search_aliases=["Number Multiple Of", "multiple", "round", "snap"],
            category="WAS Suite/Number/Operations",
            description=(
                "Round a number up to the next multiple of the given value, which is how a "
                "free-typed dimension is snapped to the multiple of 8 a latent needs. A "
                "number that already divides evenly is returned as it is."
            ),
            inputs=[
                io.MultiType.Input(
                    "number",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The value to snap upward, such as a width or height that was typed "
                        "or computed freely."
                    ),
                ),
                io.Int.Input(
                    "multiple",
                    default=8,
                    min=-18446744073709551615,
                    max=18446744073709551615,
                    tooltip=(
                        "The spacing to snap to. 8 turns 500 into 504, which is the grid a "
                        "latent needs; 64 turns 500 into 512. A value of 0 stops with a "
                        "division error."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The snapped value, or the original when it already divided evenly."
                    ),
                ),
                io.Float.Output(tooltip="The same snapped value, on a FLOAT socket."),
                io.Int.Output(
                    tooltip=(
                        "The snapped value as a whole number, cut off rather than rounded."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, number, multiple=8) -> io.NodeOutput:
        if number % multiple != 0:
            number = (number // multiple) * multiple + multiple
        return io.NodeOutput(number, number, int(number))
