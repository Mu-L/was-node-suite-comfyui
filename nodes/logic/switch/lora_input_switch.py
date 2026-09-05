"""Route one of two model/CLIP pairs onward."""

from __future__ import annotations

from comfy_api.latest import io


class LoraInputSwitch(io.ComfyNode):
    """Select between two model and CLIP pairs with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Lora Input Switch",
            display_name="Lora Input Switch",
            search_aliases=["Lora Input Switch", "lora switch", "boolean switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Pass one model and CLIP pair of two on, chosen by a boolean, which "
                "switches a whole LoRA branch with a single control. The model_a and "
                "clip_a pair is sent when the boolean is true, the model_b and clip_b pair "
                "when it is false."
            ),
            inputs=[
                io.Model.Input(
                    "model_a",
                    tooltip="The diffusion model sent on when boolean is true, with clip_a.",
                ),
                io.Clip.Input(
                    "clip_a",
                    tooltip=(
                        "The text encoder sent on when boolean is true. Wire it from the "
                        "same LoRA loader as model_a, so a patched model keeps the CLIP it "
                        "was patched with."
                    ),
                ),
                io.Model.Input(
                    "model_b",
                    tooltip="The diffusion model sent on when boolean is false, with clip_b.",
                ),
                io.Clip.Input(
                    "clip_b",
                    tooltip=(
                        "The text encoder sent on when boolean is false. Wire it from the "
                        "same LoRA loader as model_b."
                    ),
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which pair passes; BOOLEAN. true = model_a and clip_a, false = "
                        "model_b and clip_b. Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(tooltip="The model half of the selected pair."),
                io.Clip.Output(tooltip="The text encoder half of the same selected pair."),
            ],
        )

    @classmethod
    def execute(cls, model_a, clip_a, model_b, clip_b, boolean=True) -> io.NodeOutput:
        if boolean:
            return io.NodeOutput(model_a, clip_a)
        return io.NodeOutput(model_b, clip_b)
