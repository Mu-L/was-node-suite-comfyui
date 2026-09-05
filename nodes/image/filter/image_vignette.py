"""Darken or lighten a frame away from its centre."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic, optics

logger = log.get_logger("nodes.image.filter")


class ImageVignette(io.ComfyNode):
    """Apply a radial falloff to a batch of images."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageVignette",
            display_name="Image Vignette",
            search_aliases=[
                "WASImageVignette",
                "Image Vignette",
                "vignette",
                "corner falloff",
                "edge darkening",
                "spotlight",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Fall the frame off towards its corners, the way a wide lens does, or the "
                "other way to lift them. A small amount pulls the eye to the middle of a "
                "shot; a negative amount takes an existing vignette back out. The centre "
                "moves, so the falloff can sit on a subject that is not in the middle."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to shade. Each one gets the same falloff and comes back "
                        "at the size it went in at."
                    ),
                ),
                io.Float.Input(
                    "amount",
                    default=0.5,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How far the corners move. 0.0 = no change, 0.5 = corners at half "
                        "brightness, 1.0 = corners black, -0.5 = corners lifted instead."
                    ),
                ),
                io.Float.Input(
                    "size",
                    default=0.75,
                    min=0.05,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "How far out the falloff finishes. 0.75 = clear of the middle and "
                        "full at the corners, 1.0 = only the very corners, 0.3 = a tight "
                        "spotlight."
                    ),
                ),
                io.Float.Input(
                    "feather",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the way in the falloff is spread over. 0.0 = a hard "
                        "edged circle, 0.5 = a gradual one, 1.0 = falling off from the "
                        "centre out."
                    ),
                ),
                io.Combo.Input(
                    "shape",
                    options=list(optics.SHAPES),
                    tooltip=(
                        "`to the frame` stretches the falloff into an oval matching the "
                        "aspect ratio, which is what a lens does; `circular` keeps it round, "
                        "so a wide frame darkens at its left and right first."
                    ),
                ),
                io.Float.Input(
                    "centre_x",
                    default=0.5,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Where the falloff is centred across the frame. 0.5 = the middle, "
                        "0.0 = the left edge, 1.0 = the right edge."
                    ),
                ),
                io.Float.Input(
                    "centre_y",
                    default=0.5,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Where the falloff is centred down the frame. 0.5 = the middle, 0.0 "
                        "= the top edge, 1.0 = the bottom edge."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The frames with the falloff applied.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, amount=0.5, size=0.75, feather=0.5, shape=optics.SHAPES[0],
        centre_x=0.5, centre_y=0.5,
    ) -> io.NodeOutput:
        folded = dynamic.fold(images)
        shaded = optics.vignette(
            folded.images, float(amount), float(size), float(feather), shape,
            float(centre_x), float(centre_y),
        )
        logger.info("Image Vignette shaded %d frame(s) by %.2f", int(images.shape[0]), amount)
        return io.NodeOutput(dynamic.unfold(shaded, folded))
