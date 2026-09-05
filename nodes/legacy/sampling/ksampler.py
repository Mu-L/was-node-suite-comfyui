"""Core KSampler fed by a SEED wire instead of a seed widget."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import SEED

REQUIRES = "sampling"


def sampler_names() -> list[str]:
    """The sampler names this ComfyUI offers."""
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def scheduler_names() -> list[str]:
    """The scheduler names this ComfyUI offers. Read live, as in :func:`sampler_names`."""
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS


class KSampler(io.ComfyNode):
    """Sample a latent with core's sampler, taking the seed from a SEED socket."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KSampler (WAS)",
            display_name="KSampler (Seed Socket)",
            search_aliases=["KSampler (WAS)", "KSampler", "sampler", "sample", "seed sampler"],
            category="WAS Suite/Sampling",
            description=(
                "Deprecated: use ComfyUI's KSampler instead, with a Seed node feeding its "
                "seed input where the seed arrives on a wire. Samples a latent with the core "
                "sampler, taking the seed from a SEED socket rather than from a widget."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model doing the sampling."),
                SEED.Input(
                    "seed",
                    tooltip=(
                        "The noise seed, arriving on a wire from a Seed or Number to Seed "
                        "node rather than as a widget. This socket is the only thing that "
                        "sets this node apart from ComfyUI's own KSampler."
                    ),
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10000,
                    tooltip=(
                        "How many sampling steps to run. More steps take longer and resolve "
                        "more detail, with little to gain past about 30 for most models."
                    ),
                ),
                io.Float.Input(
                    "cfg",
                    default=8.0,
                    min=0.0,
                    max=100.0,
                    tooltip=(
                        "How closely the image is held to the prompt. Around 7-8 suits most "
                        "models; lower is looser and softer, much higher tends to burn "
                        "contrast and flatten detail."
                    ),
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=sampler_names(),
                    tooltip=(
                        "The sampling algorithm. 'euler' is the plain, predictable choice; "
                        "the 'ancestral' and 'sde' variants add fresh noise as they go; the "
                        "'dpmpp' family converges in fewer steps. The list is whatever this "
                        "ComfyUI offers."
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=scheduler_names(),
                    tooltip=(
                        "How the noise level is stepped down over the run. 'normal' and "
                        "'karras' are the usual choices, karras spending more steps at low "
                        "noise where fine detail is decided."
                    ),
                ),
                io.Conditioning.Input(
                    "positive",
                    tooltip="Encoded prompt describing what the image should contain.",
                ),
                io.Conditioning.Input(
                    "negative",
                    tooltip="Encoded prompt describing what to keep out of the image.",
                ),
                io.Latent.Input(
                    "latent_image",
                    tooltip=(
                        "The latent to sample: an empty one to generate from scratch, or an "
                        "encoded image to work from. Its size sets the output size."
                    ),
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the latent is redrawn. 1.0 ignores its content and "
                        "generates from noise; around 0.5 keeps the composition and changes "
                        "the detail; 0.0 changes nothing."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    tooltip="The sampled latent. Decode it with a VAE Decode to see the picture.",
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
    ) -> io.NodeOutput:
        """Sample the latent.

        Raises:
            ValueError: Nothing is connected to the model, seed, positive, negative or
                latent_image input.
        """
        # ComfyUI's root nodes.py. This pack's own nodes/ package is reachable only as a
        # submodule of the pack, so the bare name resolves to ComfyUI's module.
        from nodes import common_ksampler

        for value, socket, thing, source, source_output in (
            (model, "model", "model", "checkpoint loader", "MODEL"),
            (seed, "seed", "seed", "Seed (WAS) or Number to Seed", "SEED"),
            (positive, "positive", "conditioning", "CLIP Text Encode", "CONDITIONING"),
            (negative, "negative", "conditioning", "CLIP Text Encode", "CONDITIONING"),
            (latent_image, "latent_image", "latent", "Empty Latent Image", "LATENT"),
        ):
            require_input(value, "KSampler (WAS)", socket, thing, source, source_output)

        sampled = common_ksampler(
            model,
            seed["seed"],
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )
        return io.NodeOutput(sampled[0])
