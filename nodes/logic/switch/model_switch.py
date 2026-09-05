"""Route one of two loaded models onward, with the socket typed to what it may carry."""

from __future__ import annotations

from comfy_api.latest import io


class ModelSwitch(io.ComfyNode):
    """Select between two loaded models with a boolean.

    Both inputs are lazy, so the unselected branch is never evaluated.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template(
            "model_switch",
            [
                io.Model, io.Vae, io.Clip, io.ClipVision, io.ControlNet,
                io.UpscaleModel, io.LatentUpscaleModel, io.StyleModel, io.Gligen,
                io.Photomaker, io.LoraModel, io.AudioEncoder, io.ModelPatch,
                io.BackgroundRemoval,
            ],
        )
        return io.Schema(
            node_id="WASModelSwitch",
            display_name="Model Switch",
            search_aliases=[
                "WASModelSwitch",
                "Model Switch",
                "switch",
                "model switch",
                "vae switch",
                "clip switch",
                "route",
                "branch",
            ],
            category="WAS Suite/Logic/Switch",
            description=(
                "Pass one of two loaded models on, chosen by a boolean. A model here is "
                "anything a loader answers: a diffusion model, a VAE, a text encoder, a "
                "CLIP vision model, a ControlNet, an upscale model, a style model and the "
                "rest. The socket refuses anything that is not one, and the branch it does "
                "not pick is never loaded."
            ),
            inputs=[
                io.MatchType.Input(
                    "input_a",
                    template=template,
                    lazy=True,
                    tooltip=(
                        "Passed on when boolean is true. Takes MODEL, VAE, CLIP, CLIP_VISION, CONTROL_NET, UPSCALE_MODEL, STYLE_MODEL and the other loader types. The first connection "
                        "fixes the type; input_b and output then take that type only."
                    ),
                ),
                io.MatchType.Input(
                    "input_b",
                    template=template,
                    lazy=True,
                    tooltip="Passed on when boolean is false. Must match input_a's type.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip="Selects the input. true = input_a, false = input_b.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="output",
                    tooltip="The selected input, typed to whatever was connected.",
                ),
            ],
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, input_a=None, input_b=None) -> list[str]:
        """Ask only for the branch that is about to be used.

        Args:
            boolean: Which input the run will select.
            input_a: The true branch, present once it has been evaluated.
            input_b: The false branch, present once it has been evaluated.

        Returns:
            The input still needed, or an empty list once it has arrived.
        """
        if boolean and input_a is None:
            return ["input_a"]
        if not boolean and input_b is None:
            return ["input_b"]
        return []

    @classmethod
    def execute(cls, input_a=None, input_b=None, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(input_a if boolean else input_b)
