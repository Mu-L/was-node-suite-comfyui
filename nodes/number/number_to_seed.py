"""Wrap a NUMBER in the pack's SEED socket."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER, SEED


class NumberToSeed(io.ComfyNode):
    """Convert a NUMBER to a SEED, the single-key mapping ``{"seed": value}``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number to Seed",
            display_name="Number to Seed",
            search_aliases=["Number to Seed", "seed", "convert"],
            category="WAS Suite/Number/Operations",
            description=(
                "Repackage a number as a SEED, the socket KSampler (WAS) takes its seed on. "
                "Core samplers want a plain INT instead, use Number to Int for those."
            ),
            inputs=[
                io.MultiType.Input(
                    "number",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The value to use as the seed, typically from Number Counter or "
                        "Random Number. It is passed through as it is, not rounded."
                    ),
                ),
            ],
            outputs=[
                SEED.Output(
                    tooltip=(
                        "The seed in the shape KSampler (WAS) expects, for its seed input."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, number) -> io.NodeOutput:
        return io.NodeOutput({"seed": number})
