"""Carry a model, a clip, a vae and both conditionings on a single wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import BUS


class BusNode(io.ComfyNode):
    """Pack five sockets into one BUS wire and unpack them again."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Bus Node",
            display_name="Bus Node",
            search_aliases=["Bus Node", "bus", "reroute", "pipe"],
            category="WAS Suite/Utilities",
            description=(
                "Bundle model, clip, vae, positive and negative onto one wire. Connected "
                "inputs override what arrives on the bus."
            ),
            inputs=[
                BUS.Input(
                    "bus",
                    optional=True,
                    tooltip=(
                        "An incoming bundle from an earlier Bus Node, carrying all five "
                        "values on one wire. Leave it disconnected on the first node of a "
                        "chain and connect the five inputs directly instead."
                    ),
                ),
                io.Model.Input(
                    "model",
                    optional=True,
                    tooltip=(
                        "Diffusion model to put on the bus. Connected, it replaces whatever "
                        "model arrived on the bus input; disconnected, the bus keeps its "
                        "own. A model has to reach the node one way or the other."
                    ),
                ),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip=(
                        "Text encoder to put on the bus. Connected, it replaces whatever "
                        "clip arrived on the bus input; disconnected, the bus keeps its "
                        "own. A clip has to reach the node one way or the other."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip=(
                        "VAE to put on the bus. Connected, it replaces whatever vae arrived "
                        "on the bus input; disconnected, the bus keeps its own. A vae has "
                        "to reach the node one way or the other."
                    ),
                ),
                io.Conditioning.Input(
                    "positive",
                    optional=True,
                    tooltip=(
                        "Positive conditioning to put on the bus. Connected, it replaces "
                        "whatever arrived on the bus input. Unlike the model, clip and vae "
                        "this one is not required, and stays empty if neither side has it."
                    ),
                ),
                io.Conditioning.Input(
                    "negative",
                    optional=True,
                    tooltip=(
                        "Negative conditioning to put on the bus. Connected, it replaces "
                        "whatever arrived on the bus input. Unlike the model, clip and vae "
                        "this one is not required, and stays empty if neither side has it."
                    ),
                ),
            ],
            outputs=[
                BUS.Output(
                    display_name="bus",
                    tooltip=(
                        "All five values bundled onto one wire, to carry across the graph "
                        "and unpack at the next Bus Node."
                    ),
                ),
                io.Model.Output(
                    display_name="model",
                    tooltip="The model now on the bus: the model input, or the incoming bus's.",
                ),
                io.Clip.Output(
                    display_name="clip",
                    tooltip="The clip now on the bus: the clip input, or the incoming bus's.",
                ),
                io.Vae.Output(
                    display_name="vae",
                    tooltip="The vae now on the bus: the vae input, or the incoming bus's.",
                ),
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip=(
                        "The positive conditioning now on the bus: the positive input, or "
                        "the incoming bus's."
                    ),
                ),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip=(
                        "The negative conditioning now on the bus: the negative input, or "
                        "the incoming bus's."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        bus=(None, None, None, None, None),
        model=None,
        clip=None,
        vae=None,
        positive=None,
        negative=None,
    ) -> io.NodeOutput:
        bus_model, bus_clip, bus_vae, bus_positive, bus_negative = bus

        out_model = model or bus_model
        out_clip = clip or bus_clip
        out_vae = vae or bus_vae
        out_positive = positive or bus_positive
        out_negative = negative or bus_negative

        out_bus = (out_model, out_clip, out_vae, out_positive, out_negative)

        if not out_model:
            raise ValueError("Either model or bus containing a model should be supplied")
        if not out_clip:
            raise ValueError("Either clip or bus containing a clip should be supplied")
        if not out_vae:
            raise ValueError("Either vae or bus containing a vae should be supplied")

        return io.NodeOutput(out_bus, out_model, out_clip, out_vae, out_positive, out_negative)
