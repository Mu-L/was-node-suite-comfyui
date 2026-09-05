"""Route one of two CLIP models onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class CLIPInputSwitch(io.ComfyNode):
    """Select between two CLIP models with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIP Input Switch",
            display_name="CLIP Input Switch",
            search_aliases=["CLIP Input Switch", "clip switch", "boolean switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Model Switch instead. It takes the type of whatever is "
                "connected and skips the branch it does not select. This node passes one of two "
                "CLIP text encoders on, chosen by a boolean: clip_a when the boolean is true, "
                "clip_b when it is false."
            ),
            inputs=[
                io.Clip.Input(
                    "clip_a",
                    tooltip="The text encoder sent on when boolean is true.",
                ),
                io.Clip.Input(
                    "clip_b",
                    tooltip="The text encoder sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = clip_a, false = clip_b. "
                        "Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.Clip.Output(tooltip="Whichever of the two text encoders was selected."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, clip_a, clip_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(clip_a if boolean else clip_b)
