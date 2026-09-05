"""Keep only the layers a range, a name and a visibility pick out."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")


class LayerSelect(io.ComfyNode):
    """Reduce a ``LAYERS`` document to the layers a filter matches."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerSelect",
            display_name="Layer Select",
            search_aliases=[
                "WASLayerSelect",
                "Layer Select",
                "filter layers",
                "keep layers",
                "isolate layer",
                "solo layer",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Keep only the layers that match, and drop the rest: a run of the stack by "
                "index, a name, whether the compositor draws them, or all three at once. Use "
                "it to isolate one layer for a filter, to pull the visible layers out of a "
                "working document, or to cut a stack down before flattening it."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack to filter. Wire in Add Layer, Layer Edit or anything else "
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
                        "Lowest layer kept, counting 0 from the back of the stack. -1 is the "
                        "front layer, -2 the one under it."
                    ),
                ),
                io.Int.Input(
                    "last_index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Highest layer kept, counted the same way. -1 = the front of the "
                        "stack, so 0 and -1 together keep every layer."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="",
                    tooltip=(
                        "Text a layer's name has to carry. Blank keeps every name. Case is "
                        "ignored, so 'sky' matches Sky."
                    ),
                ),
                io.Combo.Input(
                    "match",
                    options=list(layer_ops.MATCHES),
                    tooltip=(
                        "How name is compared. 'contains' keeps Sky Backdrop for 'sky'; "
                        "'exact' keeps only a layer called exactly that."
                    ),
                ),
                io.Combo.Input(
                    "visibility",
                    options=list(layer_ops.VISIBILITIES),
                    tooltip=(
                        "Which layers survive by their switch. 'any' ignores it, 'visible "
                        "only' keeps what the compositor draws, 'hidden only' keeps what it "
                        "skips."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The layers that matched, numbered from 0 at the back, for Create "
                        "Layered Image or another edit."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many layers came through, for a switch that handles one.",
                ),
                io.Int.Output(
                    display_name="dropped",
                    tooltip="How many the filter left behind, so a filter that took too much shows.",
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
                "Layer Select was handed a stack with no layer in it, so there is nothing to "
                "filter. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        flags = layer_ops.matching(
            stack, int(first_index), int(last_index), name, match, visibility
        )
        kept = [entry for entry, hit in zip(stack, flags) if hit]
        if not kept:
            raise ValueError(
                f"Layer Select matched none of the {len(stack)} layer(s), and a document with "
                f"no layer in it cannot be composited. Widen first_index and last_index, clear "
                f"name, or set visibility to '{layer_ops.VISIBILITIES[0]}'."
            )

        document = layer_ops.rebuilt(layers, kept)
        line = f"kept {len(kept)} of {len(stack)} layer(s)"
        layer_ops.report(
            "Layer Select", line, document,
            counts={"kept": len(kept), "dropped": len(stack) - len(kept)},
            facts={"kept names": ", ".join(
                str(item.get("name") or "unnamed") for item in kept[:6]
            )},
        )
        logger.info("Layer Select %s", line)
        return io.NodeOutput(
            document, len(kept), len(stack) - len(kept), ui=ui.PreviewText(line)
        )
