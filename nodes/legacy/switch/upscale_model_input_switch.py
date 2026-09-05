"""Route one of two upscale models onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class UpscaleModelSwitch(io.ComfyNode):
    """Select between two upscale models with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Upscale Model Switch",
            display_name="Upscale Model Switch",
            search_aliases=[
                "Upscale Model Switch",
                "upscale model input switch",
                "boolean switch",
            ],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Model Switch instead. It takes the type of whatever is "
                "connected and skips the branch it does not select. This node passes one of two "
                "upscale models on, chosen by a boolean: upscale_model_a when the boolean is "
                "true, upscale_model_b when it is false."
            ),
            inputs=[
                io.UpscaleModel.Input(
                    "upscale_model_a",
                    tooltip="The upscale model sent on when boolean is true.",
                ),
                io.UpscaleModel.Input(
                    "upscale_model_b",
                    tooltip="The upscale model sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = upscale_model_a, false = "
                        "upscale_model_b. Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.UpscaleModel.Output(
                    tooltip="Whichever of the two upscale models was selected.",
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, upscale_model_a, upscale_model_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(upscale_model_a if boolean else upscale_model_b)
