"""Change one layer's placement, size, angle, opacity, blend mode, name or order."""

from __future__ import annotations

import math

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: Widget value leaving a setting as the layer already had it.
KEEP = "keep"

#: What a three way switch offers, in menu order.
SWITCHES = (KEEP, "on", "off")


class LayerEdit(io.ComfyNode):
    """Rewrite one layer of a ``LAYERS`` document in place."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerEdit",
            display_name="Layer Edit",
            search_aliases=[
                "WASLayerEdit",
                "Layer Edit",
                "layer properties",
                "move layer",
                "layer opacity",
                "blend mode",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Change one layer of a stack: where it sits, the size and angle it is drawn "
                "at, its opacity, blend mode, mirroring, name, visibility and place in the "
                "stack. Every setting has a keep value, so one node changes the one thing it "
                "is set for and leaves the rest of the layer exactly as it was."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to change. Wire in Add Layer, Layer "
                        "Duplicate or anything else answering a LAYERS document."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer, counting 0 from the back of the stack. -1 is the front "
                        "layer and -2 the one under it. Ignored while layer_name names one."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to change instead of index. Blank uses index. "
                        "'sky' finds a layer called Sky, and finds Sky Backdrop where nothing "
                        "is called exactly Sky."
                    ),
                ),
                io.Boolean.Input(
                    "set_position",
                    default=False,
                    tooltip=(
                        "Whether x and y are written. 'false' leaves the layer where it is; "
                        "'true' moves it to x, y."
                    ),
                ),
                io.Int.Input(
                    "x",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Left edge on the canvas in pixels, written only while set_position "
                        "is true. 0 is the canvas edge, 100 is 100px in, -40 hangs the layer "
                        "off the left."
                    ),
                ),
                io.Int.Input(
                    "y",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Top edge on the canvas in pixels, written only while set_position is "
                        "true. 0 is the top of the canvas, 100 is 100px down."
                    ),
                ),
                io.Int.Input(
                    "w",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Width the layer is drawn at, in pixels. 0 = keep, 512 = drawn 512 "
                        "across whatever its picture measures."
                    ),
                ),
                io.Int.Input(
                    "h",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Height the layer is drawn at, in pixels. 0 = keep, 512 = drawn 512 "
                        "tall whatever its picture measures."
                    ),
                ),
                io.Boolean.Input(
                    "set_rotation",
                    default=False,
                    tooltip=(
                        "Whether rotation is written. 'false' leaves the layer at the angle "
                        "it already has; 'true' turns it to rotation."
                    ),
                ),
                io.Float.Input(
                    "rotation",
                    default=0.0,
                    min=-360.0,
                    max=360.0,
                    step=0.1,
                    tooltip=(
                        "Turn in degrees, written only while set_rotation is true. 0 = "
                        "upright, 90 = a quarter turn clockwise, -15 = a slight lean the "
                        "other way."
                    ),
                ),
                io.Float.Input(
                    "opacity",
                    default=-1.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How solid the layer is. -1 = keep, 0.0 = drawn but invisible, 0.5 = "
                        "half strength, 1.0 = solid."
                    ),
                ),
                io.Combo.Input(
                    "blend_mode",
                    options=[KEEP, *layer_ops.BLEND_MODES],
                    tooltip=(
                        "How the layer mixes with what is under it. 'keep' leaves the mode it "
                        "has; 'normal' covers; 'multiply' darkens; 'screen' lightens; "
                        "'luminosity' keeps the colour below and takes only the brightness."
                    ),
                ),
                io.Combo.Input(
                    "visible",
                    options=list(SWITCHES),
                    tooltip=(
                        "Whether the compositor draws the layer. 'keep' leaves it, 'on' shows "
                        "it, 'off' hides it while leaving it in the stack."
                    ),
                ),
                io.Combo.Input(
                    "flip_h",
                    options=list(SWITCHES),
                    tooltip=(
                        "Whether the layer is mirrored left to right. 'keep' leaves it, 'on' "
                        "mirrors it, 'off' draws it the way round its picture is."
                    ),
                ),
                io.Combo.Input(
                    "flip_v",
                    options=list(SWITCHES),
                    tooltip=(
                        "Whether the layer is mirrored top to bottom. 'keep' leaves it, 'on' "
                        "mirrors it, 'off' draws it the way up its picture is."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="",
                    tooltip=(
                        "What the layer is called. Blank keeps the name it has; 'sky' renames "
                        "it to sky, which is what Layer Select and Layer Order then match on."
                    ),
                ),
                io.Int.Input(
                    "z_index",
                    default=-1,
                    min=-1,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the layer sits in the stack, 0 at the back. -1 = keep. 2 puts "
                        "it third from the back and renumbers the rest around it."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The whole stack with that one layer changed, numbered from 0 at the "
                        "back, for Create Layered Image or another edit."
                    ),
                ),
                io.Int.Output(
                    display_name="index",
                    tooltip="Where the changed layer now sits, counting 0 from the back.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        layers,
        index=0,
        layer_name="",
        set_position=False,
        x=0,
        y=0,
        w=0,
        h=0,
        set_rotation=False,
        rotation=0.0,
        opacity=-1.0,
        blend_mode=KEEP,
        visible=KEEP,
        flip_h=KEEP,
        flip_v=KEEP,
        name="",
        z_index=-1,
    ) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Edit was handed a stack with no layer in it, so there is nothing to "
                "change. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        position = layer_ops.found(stack, index, layer_name, where="Layer Edit")
        entry = dict(stack[position])
        if set_position:
            entry["x"], entry["y"] = int(x), int(y)
        if w > 0:
            entry["w"] = int(w)
        if h > 0:
            entry["h"] = int(h)
        if set_rotation:
            entry["rotation"] = math.radians(float(rotation))
        if opacity >= 0.0:
            entry["opacity"] = float(opacity)
        if blend_mode != KEEP:
            entry["blend_mode"] = blend_mode
        if visible != KEEP:
            entry["visible"] = visible == SWITCHES[1]
        if flip_h != KEEP:
            entry["flip_h"] = flip_h == SWITCHES[1]
        if flip_v != KEEP:
            entry["flip_v"] = flip_v == SWITCHES[1]
        if name.strip():
            entry["name"] = name.strip()
        stack[position] = entry

        if z_index >= 0:
            landing = min(int(z_index), len(stack) - 1)
            stack.insert(landing, stack.pop(position))
            position = landing

        document = layer_ops.rebuilt(layers, stack)
        line = (
            f"changed '{entry.get('name') or 'unnamed'}', now index {position} "
            f"of {len(stack)} layer(s)"
        )
        layer_ops.report(
            "Layer Edit", line, document,
            counts={"layer": position},
            facts={
                "name": str(entry.get("name") or "unnamed"),
                "drawn": "{}x{}".format(*layer_ops.drawn_size(entry)),
                "placement": f"{entry.get('x', 0)}, {entry.get('y', 0)}",
                "blend": str(entry.get("blend_mode") or "normal"),
            },
        )
        logger.info("Layer Edit %s", line)
        return io.NodeOutput(document, position, ui=ui.PreviewText(line))
