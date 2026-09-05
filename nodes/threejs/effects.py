"""Passes a finished frame is put through before it is shown."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_EFFECT
from ...modules.threejs.spec import compact_deps, create_spec, require_spec

REQUIRES = "threejs"

#: What every effect says about the socket it chains from.
INPUT_HINT = (
    "The effect before this one, if any. Left unwired this is the first pass after the "
    "scene is drawn."
)

#: What every effect says about the socket it hands on.
OUTPUT_HINT = "The chain so far, for the next effect or for Three App's effects socket."


class ThreeBloom(io.ComfyNode):
    """Bleed light out of the brightest parts of a frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeBloom",
            display_name="Three Bloom",
            search_aliases=["WASThreeBloom", "Three Bloom", "glow", "bloom", "post"],
            category="WAS Suite/Three",
            description=(
                "Bleed a glow out of everything brighter than a threshold, which is what makes "
                "an emissive material or a bright highlight read as a light source rather than "
                "as a pale patch. Chain it into Three App's effects socket. It costs a pass "
                "over the frame, so a large render is slower with it than without."
            ),
            inputs=[
                THREE_EFFECT.Input("input", optional=True, tooltip=INPUT_HINT),
                io.Float.Input(
                    "strength",
                    default=0.6,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip="How much glow is added. 0.3 is a hint, 0.6 usual, 2.0 heavy.",
                ),
                io.Float.Input(
                    "radius",
                    default=0.4,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="How far the glow spreads. 0.1 is tight around a highlight, 1.0 a wide haze.",
                ),
                io.Float.Input(
                    "threshold",
                    default=0.85,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "How bright a pixel has to be before it glows. 0.85 catches highlights "
                        "alone, 0.0 makes the whole frame glow."
                    ),
                ),
            ],
            outputs=[THREE_EFFECT.Output(display_name="effects", tooltip=OUTPUT_HINT)],
        )

    @classmethod
    def execute(cls, strength, radius, threshold, input=None) -> io.NodeOutput:
        """Describe the pass.

        Raises:
            ValueError: The input is not an effect descriptor.
        """
        if input is not None:
            require_spec(input, "effect")
        return io.NodeOutput(
            create_spec(
                "effect",
                "Bloom",
                params={
                    "strength": float(strength),
                    "radius": float(radius),
                    "threshold": float(threshold),
                },
                deps=compact_deps(input=input),
            )
        )


class ThreeDepthOfField(io.ComfyNode):
    """Throw everything but one distance out of focus."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeDepthOfField",
            display_name="Three Depth Of Field",
            search_aliases=[
                "WASThreeDepthOfField",
                "Three Depth Of Field",
                "bokeh",
                "focus",
                "blur",
                "post",
            ],
            category="WAS Suite/Three",
            description=(
                "Keep one distance from the camera sharp and blur everything nearer and "
                "further, the way a real lens does. Chain it into Three App's effects socket. "
                "Focus is measured in scene units from the camera, so a camera 9 units back "
                "from an object at the origin focuses on it at 9.0."
            ),
            inputs=[
                THREE_EFFECT.Input("input", optional=True, tooltip=INPUT_HINT),
                io.Float.Input(
                    "focus",
                    default=10.0,
                    min=0.0,
                    max=10000.0,
                    step=0.1,
                    tooltip=(
                        "Distance from the camera that stays sharp, in scene units. 10.0 for a "
                        "camera 10 units from its subject."
                    ),
                ),
                io.Float.Input(
                    "aperture",
                    default=0.0002,
                    min=0.0,
                    max=0.05,
                    step=0.0001,
                    tooltip=(
                        "How fast the blur comes on either side of the focus. 0.0002 is gentle, "
                        "0.01 is a shallow portrait lens."
                    ),
                ),
                io.Float.Input(
                    "max_blur",
                    default=0.01,
                    min=0.0,
                    max=0.2,
                    step=0.001,
                    tooltip="How far the blur is allowed to go. 0.01 is soft, 0.05 is a wash.",
                ),
            ],
            outputs=[THREE_EFFECT.Output(display_name="effects", tooltip=OUTPUT_HINT)],
        )

    @classmethod
    def execute(cls, focus, aperture, max_blur, input=None) -> io.NodeOutput:
        """Describe the pass.

        Raises:
            ValueError: The input is not an effect descriptor.
        """
        if input is not None:
            require_spec(input, "effect")
        return io.NodeOutput(
            create_spec(
                "effect",
                "DepthOfField",
                params={
                    "focus": float(focus),
                    "aperture": float(aperture),
                    "maxBlur": float(max_blur),
                },
                deps=compact_deps(input=input),
            )
        )


class ThreeAntialias(io.ComfyNode):
    """Smooth the stepped edges a render leaves."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeAntialias",
            display_name="Three Antialias",
            search_aliases=[
                "WASThreeAntialias",
                "Three Antialias",
                "smaa",
                "aliasing",
                "jaggies",
                "post",
            ],
            category="WAS Suite/Three",
            description=(
                "Smooth the stepped edges left along a silhouette. Three App's own antialias "
                "setting is turned off once any effect is in the chain, since the passes draw "
                "into their own buffers, so this is what puts it back. Put it last in the "
                "chain, after any glow or blur."
            ),
            inputs=[THREE_EFFECT.Input("input", optional=True, tooltip=INPUT_HINT)],
            outputs=[THREE_EFFECT.Output(display_name="effects", tooltip=OUTPUT_HINT)],
        )

    @classmethod
    def execute(cls, input=None) -> io.NodeOutput:
        """Describe the pass.

        Raises:
            ValueError: The input is not an effect descriptor.
        """
        if input is not None:
            require_spec(input, "effect")
        return io.NodeOutput(
            create_spec("effect", "Antialias", deps=compact_deps(input=input))
        )
