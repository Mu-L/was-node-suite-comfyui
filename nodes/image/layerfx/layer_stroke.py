"""Draw an outline along the edge of one layer in a stack."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import layer_fx
from ....modules.log import get_logger

logger = get_logger("nodes.image.layerfx")

#: What the node calls itself in a message.
NODE_NAME = "Layer Stroke"


class LayerStroke(io.ComfyNode):
    """Bake an outline into one layer of a stack."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerStroke",
            display_name="Layer Stroke",
            search_aliases=[
                "WASLayerStroke",
                "Layer Stroke",
                "outline",
                "border",
                "layer effect",
                "layer style",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Draw an outline along the edge of one layer in a stack. The band follows "
                "what the layer actually covers rather than its rectangle, so a cut-out "
                "subject gets an outline around the subject. An outer band grows that "
                "layer's picture and shifts it back, so the layer stays where it was on "
                "the canvas."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to outline. Wire in Add Layer, Layers "
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
                        "Which layer to outline when layer_name is empty. -1 = the top of "
                        "the stack, -2 = the one below it, 0 = the bottom, 1 = the next one "
                        "up."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to outline, matched ignoring case and spare "
                        "space. Empty reads layer_index instead. 'subject' picks the layer "
                        "Add Layer was given that name."
                    ),
                ),
                io.Combo.Input(
                    "alignment",
                    options=list(layer_fx.STROKE_POSITIONS),
                    default=layer_fx.STROKE_POSITIONS[0],
                    tooltip=(
                        "Where the band sits against the edge. `outer` sits wholly outside "
                        "and grows the layer, `inner` sits wholly inside and grows nothing, "
                        "`centre` straddles the edge and grows the layer by half the width."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=4,
                    min=0,
                    max=128,
                    step=1,
                    tooltip=(
                        "Pixels the band spans. 2 = a hairline, 8 = a clear outline, 24 = a "
                        "heavy border, 0 = nothing drawn. An outer band of 12 makes the "
                        "layer 24 wider and 24 taller."
                    ),
                ),
                io.String.Input(
                    "color",
                    default="#000000",
                    tooltip=(
                        "Colour of the band, as hexadecimal digits. #000000 = black, "
                        "#ffffff = white, #ff8800 = orange. Three digits such as #f80 work "
                        "too."
                    ),
                ),
                io.Float.Input(
                    "opacity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the band is laid down. 1.0 = solid, 0.5 = half "
                        "strength, 0.0 = nothing."
                    ),
                ),
                io.Combo.Input(
                    "blend_mode",
                    options=list(layer_fx.BLEND_MODES),
                    default=layer_fx.BLEND_MODES[0],
                    tooltip=(
                        "How the band mixes with the layer's own pixels where the two "
                        "overlap. `normal` lays the colour down as it is, `multiply` "
                        "darkens, `screen` lightens, `difference` inverts."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the outline baked into that layer's picture and "
                        "transparency. Wire it into Create Layered Image or the next effect."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, layer_index=-1, layer_name="",
                alignment=layer_fx.STROKE_POSITIONS[0], width=4, color="#000000",
                opacity=1.0, blend_mode=layer_fx.BLEND_MODES[0]) -> io.NodeOutput:
        found = layer_fx.stack(layers)
        place, entry = layer_fx.chosen(found, layer_index, layer_name, NODE_NAME)
        tint = layer_fx.colour(color, "color")
        work, keep = layer_fx.stroke_margin(alignment, width)

        def render(colours, alpha):
            return layer_fx.stroke(colours, alpha, alignment, width, tint, opacity, blend_mode)

        document, grown = layer_fx.applied(layers, entry, render, work, keep)
        layer_fx.report(NODE_NAME, document, grown, place, len(found), keep)
        logger.info(
            "%s drew a %dpx %s band on layer %d of %d", NODE_NAME, width, alignment, place,
            len(found),
        )
        return io.NodeOutput(document)
