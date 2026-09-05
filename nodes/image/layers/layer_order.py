"""Move one layer through a stack, or sort the whole stack by name or by area."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: Prefix of every move that reorders the whole stack rather than one layer.
SORTS = "sort by"


class LayerOrder(io.ComfyNode):
    """Restack a ``LAYERS`` document, renumbering it from 0 at the back."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerOrder",
            display_name="Layer Order",
            search_aliases=[
                "WASLayerOrder",
                "Layer Order",
                "bring to front",
                "send to back",
                "restack",
                "reorder layers",
                "sort layers",
                "z index",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Move one layer through the stack, to the front, to the back, one step either "
                "way or to an exact place, or sort every layer by name or by the area it "
                "covers. The result is renumbered from 0 at the back, so the next node sees a "
                "stack with no gaps in it."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack to restack. Wire in Add Layer, Layer Edit or anything else "
                        "answering a LAYERS document."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer moves, counting 0 from the back of the stack. -1 is the "
                        "front layer. Ignored while layer_name names one, and by the sort "
                        "moves."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to move instead of index. Blank uses index. 'sky' "
                        "finds a layer called Sky, and finds Sky Backdrop where nothing is "
                        "called exactly Sky."
                    ),
                ),
                io.Combo.Input(
                    "move",
                    options=list(layer_ops.MOVES),
                    tooltip=(
                        "'to front' draws the layer over everything, 'to back' under "
                        "everything, 'up one' and 'down one' swap it with its neighbour, 'to "
                        "index' drops it at target, and a sort reorders the whole stack."
                    ),
                ),
                io.Int.Input(
                    "target",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where 'to index' puts the layer, counting 0 from the back. -1 is the "
                        "front. Read by no other move."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The restacked document, numbered from 0 at the back, for Create "
                        "Layered Image or another edit."
                    ),
                ),
                io.Int.Output(
                    display_name="index",
                    tooltip=(
                        "Where that layer landed, counting 0 from the back. -1 after a sort "
                        "that had no layer to follow."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, index=0, layer_name="", move=layer_ops.MOVES[0], target=0) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Order was handed a stack with no layer in it, so there is nothing to "
                "reorder. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        sorting = move.startswith(SORTS)
        position = layer_ops.found(
            stack, index, layer_name, where="Layer Order", required=not sorting
        )
        ordered, landed = layer_ops.moved(stack, position, move, int(target))
        document = layer_ops.rebuilt(layers, ordered)

        line = f"{move}: {len(ordered)} layer(s), tracked layer now at index {landed}"
        layer_ops.report(
            "Layer Order", line, document,
            counts={"landed at": landed},
            facts={"move": move, "order": ", ".join(
                str(item.get("name") or "unnamed") for item in ordered[:6]
            )},
        )
        logger.info("Layer Order %s", line)
        return io.NodeOutput(document, landed, ui=ui.PreviewText(line))
