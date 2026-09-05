"""Drop the layers a range, a name and a visibility pick out."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")


class LayerRemove(io.ComfyNode):
    """Take the layers a filter matches out of a ``LAYERS`` document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerRemove",
            display_name="Layer Remove",
            search_aliases=[
                "WASLayerRemove",
                "Layer Remove",
                "delete layer",
                "drop layer",
                "discard layers",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Drop the layers that match and keep the rest: a run of the stack by index, a "
                "name, whether the compositor draws them, or all three at once. Use it to "
                "throw away a guide layer before the render, or to clear every hidden layer "
                "out of a working document."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack to thin out. Wire in Add Layer, Layer Edit or anything else "
                        "answering a LAYERS document."
                    ),
                ),
                io.Int.Input(
                    "first_index",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Lowest layer removed, counting 0 from the back of the stack. -1 is "
                        "the front layer, -2 the one under it."
                    ),
                ),
                io.Int.Input(
                    "last_index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Highest layer removed, counted the same way. -1 = the front of the "
                        "stack, so 0 and -1 together reach every layer."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="",
                    tooltip=(
                        "Text a layer's name has to carry to be removed. Blank reaches every "
                        "name in the range. Case is ignored, so 'guide' matches Guide."
                    ),
                ),
                io.Combo.Input(
                    "match",
                    options=list(layer_ops.MATCHES),
                    tooltip=(
                        "How name is compared. 'contains' removes Guide Grid for 'guide'; "
                        "'exact' removes only a layer called exactly that."
                    ),
                ),
                io.Combo.Input(
                    "visibility",
                    options=list(layer_ops.VISIBILITIES),
                    tooltip=(
                        "Which layers the filter reaches by their switch. 'any' ignores it, "
                        "'visible only' removes what the compositor draws, 'hidden only' "
                        "clears out what it skips."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "What is left, numbered from 0 at the back, for Create Layered Image "
                        "or another edit."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many layers are left, for a switch that handles one.",
                ),
                io.Int.Output(
                    display_name="dropped",
                    tooltip="How many were removed, so a filter that took too much shows.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        layers,
        first_index=0,
        last_index=-1,
        name="",
        match=layer_ops.MATCHES[0],
        visibility=layer_ops.VISIBILITIES[0],
    ) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Remove was handed a stack with no layer in it, so there is nothing to "
                "remove. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        flags = layer_ops.matching(
            stack, int(first_index), int(last_index), name, match, visibility
        )
        kept = [entry for entry, hit in zip(stack, flags) if not hit]
        if not kept:
            raise ValueError(
                f"Layer Remove matched all {len(stack)} layer(s), and a document with no layer "
                f"in it cannot be composited. Narrow first_index and last_index, or set name "
                f"to the one layer to drop."
            )

        document = layer_ops.rebuilt(layers, kept)
        line = f"removed {len(stack) - len(kept)} of {len(stack)} layer(s)"
        layer_ops.report(
            "Layer Remove", line, document,
            counts={"kept": len(kept), "removed": len(stack) - len(kept)},
            facts={"kept names": ", ".join(
                str(item.get("name") or "unnamed") for item in kept[:6]
            )},
        )
        logger.info("Layer Remove %s", line)
        return io.NodeOutput(
            document, len(kept), len(stack) - len(kept), ui=ui.PreviewText(line)
        )
