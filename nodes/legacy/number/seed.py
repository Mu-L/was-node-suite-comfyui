"""A seed value on four sockets at once."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER, SEED

REQUIRES = "core_dupes"


class Seed(io.ComfyNode):
    """Emit one seed as SEED, the mapping ``{"seed": value}``, and as NUMBER, FLOAT and INT."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Seed",
            display_name="Seed (Number Outputs)",
            search_aliases=["Seed", "seed", "noise seed"],
            category="WAS Suite/Number",
            description=(
                "Deprecated: use ComfyUI's own Seed node instead. Emits one seed value on a "
                "SEED socket and on NUMBER, FLOAT and INT sockets. The SEED socket is read "
                "only by the deprecated KSampler (WAS); every core sampler takes a plain "
                "INT."
            ),
            inputs=[
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "The seed to hand out. The same seed reproduces the same noise, so an "
                        "image can be repeated exactly; change it for a different one. Any "
                        "whole number; `0` is as good a seed as any."
                    ),
                ),
            ],
            outputs=[
                SEED.Output(
                    display_name="seed",
                    tooltip="The seed in the shape KSampler (WAS) expects on its seed input.",
                ),
                NUMBER.Output(
                    display_name="number",
                    tooltip="The bare seed on a NUMBER socket, for this pack's number nodes.",
                ),
                io.Float.Output(
                    display_name="float",
                    tooltip="The same seed as a float, so 42 leaves here as 42.0.",
                ),
                io.Int.Output(
                    display_name="int",
                    tooltip="The same seed as an INT, for a core sampler's seed widget.",
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, seed) -> io.NodeOutput:
        return io.NodeOutput({"seed": seed}, seed, float(seed), int(seed))
