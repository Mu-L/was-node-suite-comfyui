"""Paint one layer of a stack with a flat colour or a two-stop gradient."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import layer_fx
from ....modules.log import get_logger

logger = get_logger("nodes.image.layerfx")

#: What the node calls itself in a message.
NODE_NAME = "Layer Overlay"


class LayerOverlay(io.ComfyNode):
    """Bake a colour or gradient fill into one layer of a stack."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerOverlay",
            display_name="Layer Overlay",
            search_aliases=[
                "WASLayerOverlay",
                "Layer Overlay",
                "color overlay",
                "gradient overlay",
                "tint",
                "layer effect",
                "layer style",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Paint one layer in a stack with a flat colour or a two-stop linear "
                "gradient, held inside whatever that layer already covers. It recolours a "
                "cut-out subject, tints a title or grades one element of a composite "
                "without touching the rest. The layer's shape, size and placement are left "
                "exactly as they were."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to paint. Wire in Add Layer, Layers "
                        "From Bounding Boxes, or another layer effect to stack effects up."
                    ),
                ),
                io.Int.Input(
                    "layer_index",
                    default=-1,
                    min=-1000,
                    max=1000,
                    step=1,
                    tooltip=(
                        "Which layer to paint when layer_name is empty. -1 = the top of the "
                        "stack, -2 = the one below it, 0 = the bottom, 1 = the next one up."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to paint, matched ignoring case and spare space. "
                        "Empty reads layer_index instead. 'subject' picks the layer Add "
                        "Layer was given that name."
                    ),
                ),
                io.Combo.Input(
                    "fill",
                    options=list(layer_fx.OVERLAY_FILLS),
                    default=layer_fx.OVERLAY_FILLS[0],
                    tooltip=(
                        "What the layer is painted with. `flat` uses color everywhere, "
                        "`gradient` runs from color to color_b across the layer at the "
                        "angle set below."
                    ),
                ),
                io.String.Input(
                    "color",
                    default="#ff8800",
                    tooltip=(
                        "The flat colour, or the first stop of a gradient, as hexadecimal "
                        "digits. #ff8800 = orange, #ffffff = white. Three digits such as "
                        "#f80 work too."
                    ),
                ),
                io.String.Input(
                    "color_b",
                    default="#0088ff",
                    tooltip=(
                        "The second stop of a gradient, as hexadecimal digits. #0088ff = "
                        "blue, #000000 = black. Read only when fill is `gradient`."
                    ),
                ),
                io.Float.Input(
                    "angle",
                    default=90.0,
                    min=-360.0,
                    max=360.0,
                    step=1.0,
                    tooltip=(
                        "Degrees the gradient runs at, counted counter-clockwise from "
                        "pointing right. 0 = color at the left and color_b at the right, 90 "
                        "= color at the bottom and color_b at the top."
                    ),
                ),
                io.Float.Input(
                    "opacity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the fill is laid down. 1.0 = the layer's own colours "
                        "are replaced, 0.4 = a tint over them, 0.0 = nothing."
                    ),
                ),
                io.Combo.Input(
                    "blend_mode",
                    options=list(layer_fx.BLEND_MODES),
                    default=layer_fx.BLEND_MODES[0],
                    tooltip=(
                        "How the fill mixes with the layer's own pixels. `normal` replaces "
                        "them, `color` keeps their light and takes the fill's hue, "
                        "`multiply` darkens, `overlay` keeps contrast."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the fill baked into that layer's picture. Wire it "
                        "into Create Layered Image or the next effect."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, layer_index=-1, layer_name="",
                fill=layer_fx.OVERLAY_FILLS[0], color="#ff8800", color_b="#0088ff",
                angle=90.0, opacity=1.0,
                blend_mode=layer_fx.BLEND_MODES[0]) -> io.NodeOutput:
        found = layer_fx.stack(layers)
        place, entry = layer_fx.chosen(found, layer_index, layer_name, NODE_NAME)
        tint = layer_fx.colour(color, "color")
        second = layer_fx.colour(color_b, "color_b")

        def render(colours, alpha):
            return layer_fx.overlay(colours, alpha, fill, tint, second, angle, opacity,
                                    blend_mode)

        document, grown = layer_fx.applied(layers, entry, render, 0, 0)
        layer_fx.report(NODE_NAME, document, grown, place, len(found), 0)
        logger.info(
            "%s painted a %s fill on layer %d of %d", NODE_NAME, fill, place, len(found)
        )
        return io.NodeOutput(document)
