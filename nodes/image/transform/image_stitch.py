"""Join two images along one edge."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import broadcast_image_planes, stack_images, tensor2pil
from ....modules.image.stitch import stitch_image
from ....modules.interface import size_report


#: Every slot the series grows through, in the order they are stitched. image_a and image_b
#: are the frozen v2 pair; the rest are revealed by web/was_growing_inputs.js as each fills.
SLOT_NAMES = tuple(f"image_{letter}" for letter in "abcdefghijklmnopqrstuvwxyz")


class ImageStitch(io.ComfyNode):
    """Place one image beside another and blend the seam."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Stitch",
            display_name="Image Stitch (Advanced)",
            search_aliases=["Image Stitch", "join images", "panorama", "concatenate images"],
            category="WAS Suite/Image/Transform",
            description=(
                "Put image_b against one edge of image_a on a single canvas, fading the two "
                "together across the join. The images overlap by the feathering width, so "
                "the canvas is that much shorter than the two laid end to end. The canvas "
                "takes its other side from image_a alone, which leaves black where a "
                "smaller image_b does not reach and crops a larger one."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip=(
                        "The image that stays in place, and the one whose other dimension "
                        "sets the canvas size."
                    ),
                ),
                io.Image.Input(
                    "image_b",
                    tooltip="The image placed against image_a.",
                ),
                io.Combo.Input(
                    "stitch",
                    options=["top", "left", "bottom", "right"],
                    tooltip=(
                        "Which side of image_a image_b goes on. `left` and `right` build a "
                        "wide canvas, `top` and `bottom` a tall one."
                    ),
                ),
                io.Int.Input(
                    "feathering",
                    default=50,
                    min=0,
                    max=2048,
                    step=1,
                    tooltip=(
                        "Width of the blended overlap in pixels. 0 butts the images "
                        "together with a hard edge; 50 fades them over 50 pixels, which "
                        "hides the seam between two views of the same scene. The seam is cut "
                        "out of the pair, so a feather wider than a slot narrows the result, "
                        "and stitching many slots that way narrows it further."
                    ),
                ),
                io.Image.Input(
                    "image_c",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_d",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_e",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_f",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_g",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_h",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_i",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_j",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_k",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_l",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_m",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_n",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_o",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_p",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_q",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_r",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_s",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_t",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_u",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_v",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_w",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_x",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_y",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
                io.Image.Input(
                    "image_z",
                    optional=True,
                    tooltip=(
                        "A further image, stitched on after the one before it in the same "
                        "direction. The interface reveals the next slot as this one is filled."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="One image holding both inputs, blended across the join.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b, stitch, feathering, **extra) -> io.NodeOutput:
        """Stitch every connected slot together, in slot order.

        Args:
            image_a: The first picture.
            image_b: The second, stitched onto it.
            stitch: Which side the next picture goes on.
            feathering: How far the seam is blended.
            extra: The optional ``image_c`` to ``image_z`` slots, connected or not.

        Returns:
            One picture per frame, every connected slot stitched on in order.
        """
        scale = dynamic.peak(
            image_a, image_b, *(extra.get(name) for name in SLOT_NAMES[2:])
        )
        folded = dynamic.fold(image_a, scale)
        joined = [
            tensor2pil(first)
            for first, _ in broadcast_image_planes(folded.images, folded.images)
        ]
        for name in SLOT_NAMES[1:]:
            following = image_b if name == "image_b" else extra.get(name)
            if following is None:
                continue
            following = dynamic.fold(following, scale).images
            joined = [
                stitch_image(carried, tensor2pil(plane), stitch, feathering)
                for carried, (plane, _) in zip(
                    joined, broadcast_image_planes(following, following)
                )
            ]

        canvas = stack_images(joined)
        filled = [n for n in SLOT_NAMES[2:] if extra.get(n) is not None]
        size_report.publish(
            image_a,
            canvas,
            action=f"stitched with {len(filled) + 1} more on the {stitch}",
            facts={"image_b": size_report.spell(image_b)},
        )
        return io.NodeOutput(dynamic.unfold(canvas, folded))
