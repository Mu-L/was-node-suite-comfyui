"""Batching images of different sizes by bringing them to one size first."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.image import fit
from ...modules.image.batching import (
    IMAGE_SLOTS,
    MAX_SLOTS,
    as_batch,
    check_image_dimensions,
    image_slot_template,
)
from ...modules.interface import batch_report
from ...modules.util.slots import connected_in_order


class ImageBatchAdvanced(io.ComfyNode):
    """Concatenate images from a growing slot list, fitting them to one size on request."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        # The settings are declared before the slot list, so they keep their place as the list
        # grows and are read before the sockets they govern.
        return io.Schema(
            node_id="WASImageBatchAdvanced",
            display_name="Image Batch Advanced",
            search_aliases=[
                "WASImageBatchAdvanced", "Image Batch Advanced",
                "image batch",
                "batch different sizes",
                "resize to match",
                "crop to match",
                "pad to match",
                "combine images",
            ],
            category="WAS Suite/Image",
            description=(
                "Join any number of images into one batch, on a slot list that grows a socket "
                "each time one is filled. Turn enforce_aspect_ratio on and images of different "
                "sizes are brought to the first slot's size first, by stretching, cropping or "
                "padding, so they can be batched without matching them by hand."
            ),
            inputs=[
                io.Boolean.Input(
                    "enforce_aspect_ratio",
                    default=False,
                    tooltip=(
                        "Bring every image to the size of the first connected slot; BOOLEAN. "
                        "Off, images of differing sizes are refused, which is what the plain "
                        "Image Batch does."
                    ),
                ),
                io.Combo.Input(
                    "resize_method",
                    list(fit.METHODS),
                    tooltip=(
                        "How an image is brought to size; COMBO. 'resize' stretches it, "
                        "'crop' keeps its shape and takes the middle, 'pad' keeps its shape "
                        "and fills the rest with black. Ignored while "
                        "enforce_aspect_ratio is off."
                    ),
                ),
                io.Autogrow.Input(
                    "images",
                    template=image_slot_template(),
                    tooltip=(
                        "The images to join, in slot order. The list grows as slots are "
                        f"filled, up to {MAX_SLOTS}. An unconnected slot contributes "
                        "nothing rather than a blank frame."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "Every connected image as one batch, in slot order. A slot holding "
                        "a batch contributes all of its frames."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many frames the batch holds, which is the total across the "
                        "slots rather than the number of slots."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, enforce_aspect_ratio=False, resize_method="resize") -> io.NodeOutput:
        """Fit the connected slots to one size if asked, then concatenate them.

        Raises:
            ValueError: No slot holds an image, a slot holds something that is not one, or
                the sizes differ while ``enforce_aspect_ratio`` is off.
        """
        names = connected_in_order(images, IMAGE_SLOTS)
        tensors = [as_batch(images[name], name) for name in names]
        if not tensors:
            raise ValueError(
                "Image Batch Advanced has no images connected. Connect at least one slot."
            )

        fitted = 0
        if enforce_aspect_ratio:
            height, width = fit.target_size(tensors[0])
            resized = []
            for tensor in tensors:
                before = fit.target_size(tensor)
                brought = fit.fit_to(tensor, height, width, str(resize_method))
                fitted += before != (height, width)
                resized.append(brought)
            tensors = resized

        size, mode = batch_report.describe_images(tensors[0])
        try:
            # Channels still have to agree, and so do sizes when nothing was fitted, so the
            # same check answers for both and names the slot either way.
            check_image_dimensions(
                tensors,
                names,
                node="Image Batch Advanced",
                advice=(
                    "Turn enforce_aspect_ratio on to bring every slot to the size of the "
                    "first, or resize the odd one yourself."
                ),
            )
        except ValueError as refused:
            batch_report.publish(
                frames=sum(int(tensor.shape[0]) for tensor in tensors),
                slots=len(tensors),
                size=size,
                mode=mode,
                memory=sum(batch_report.memory_of(tensor) for tensor in tensors),
                refused=str(refused),
            )
            raise
        batched = torch.cat(tensors, dim=0)
        batch_report.publish(
            frames=int(batched.shape[0]),
            slots=len(tensors),
            size=size,
            mode=mode,
            memory=batch_report.memory_of(batched),
            fitted=fitted or None,
        )
        return io.NodeOutput(batched, int(batched.shape[0]))
