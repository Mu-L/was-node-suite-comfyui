"""Copy one layer, offset the copy and stack it straight above the original."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")


class LayerDuplicate(io.ComfyNode):
    """Add a copy of one layer to a ``LAYERS`` document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerDuplicate",
            display_name="Layer Duplicate",
            search_aliases=[
                "WASLayerDuplicate",
                "Layer Duplicate",
                "copy layer",
                "clone layer",
                "drop shadow",
                "offset copy",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Copy one layer and stack the copy directly above the original, offset by dx "
                "and dy. The copy carries the same picture, mask, size, angle and blend mode, "
                "so it is the start of a drop shadow, an outline or a repeated element that "
                "Layer Edit then changes on its own."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to copy. Wire in Add Layer, Layer Edit or "
                        "anything else answering a LAYERS document."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer is copied, counting 0 from the back of the stack. -1 is "
                        "the front layer. Ignored while layer_name names one."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to copy instead of index. Blank uses index. 'sky' "
                        "finds a layer called Sky, and finds Sky Backdrop where nothing is "
                        "called exactly Sky."
                    ),
                ),
                io.Int.Input(
                    "dx",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the copy sits right of the original. 0 = straight on top, 8 = "
                        "8px right, -8 = 8px left."
                    ),
                ),
                io.Int.Input(
                    "dy",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the copy sits below the original. 0 = straight on top, 8 = 8px "
                        "down, -8 = 8px up."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="",
                    tooltip=(
                        "What the copy is called. Blank adds ' copy' to the original's name, "
                        "so Sky becomes Sky copy and the two stay tellable apart by name."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the copy in it, numbered from 0 at the back, for "
                        "Create Layered Image or another edit."
                    ),
                ),
                io.Int.Output(
                    display_name="index",
                    tooltip="Where the copy landed, counting 0 from the back.",
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, index=-1, layer_name="", dx=0, dy=0, name="") -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Duplicate was handed a stack with no layer in it, so there is nothing "
                "to copy. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        position = layer_ops.found(stack, index, layer_name, where="Layer Duplicate")
        copy = layer_ops.duplicated(stack[position], int(dx), int(dy), name)
        stack.insert(position + 1, copy)

        document = layer_ops.rebuilt(layers, stack)
        line = f"copied to '{copy['name']}' at index {position + 1} of {len(stack)} layer(s)"
        logger.info("Layer Duplicate %s", line)
        return io.NodeOutput(document, position + 1, ui=ui.PreviewText(line))
