"""Segment Anything masking from point prompts."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.sockets import require_input
from ...modules.compat.types import SAM_MODEL, SAM_PARAMETERS
from ...modules.convert.tensors import image_planes, tensor2sam
from ...modules.interface import mask_report

REQUIRES = "sam"


class SamImageMask(io.ComfyNode):
    """Segment an image at the points `SAM Parameters` describes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM Image Mask",
            display_name="SAM Image Mask",
            search_aliases=["SAM Image Mask", "segment anything", "point mask", "sam mask"],
            category="WAS Suite/Image/Masking",
            description=(
                "Select part of an image by pointing at it: Segment Anything works out where "
                "the object under each point begins and ends and returns it as a mask. Enable "
                "features.sam to load this node."
            ),
            inputs=[
                SAM_MODEL.Input(
                    "sam_model",
                    tooltip="The model from SAM Model Loader.",
                ),
                SAM_PARAMETERS.Input(
                    "sam_parameters",
                    tooltip=(
                        "The points to segment from, out of SAM Parameters or SAM Parameters "
                        "Combine. Their coordinates are read against this image, so they have "
                        "to be inside it."
                    ),
                ),
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to segment. Every image of a batch is segmented against the "
                        "same points, so the points have to be inside all of them."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The selection as a black and white image, white where the object is. "
                        "Ready to preview, or to use as a matte."
                    ),
                ),
                io.Mask.Output(
                    tooltip=(
                        "The same selection as a mask, for an inpainting or compositing node. "
                        "1.0 inside the object and 0.0 outside it."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, sam_model, sam_parameters, image) -> io.NodeOutput:
        """Segment the image at the given points.

        Raises:
            ValueError: Nothing is connected to the sam_model or sam_parameters input.
        """
        import numpy as np
        import torch

        require_input(
            sam_model,
            "SAM Image Mask",
            "sam_model",
            "model",
            "SAM Model Loader",
            "SAM_MODEL",
        )
        require_input(
            sam_parameters,
            "SAM Image Mask",
            "sam_parameters",
            "points",
            "SAM Parameters or SAM Parameters Combine",
            "SAM_PARAMETERS",
        )

        points = sam_parameters["points"]
        labels = sam_parameters["labels"]

        device = sam_model.load()
        processor = sam_model.processor
        model = sam_model.model

        mattes = []
        masks = []
        for plane in image_planes(image):
            inputs = processor(
                tensor2sam(plane),
                input_points=[points.tolist()],
                input_labels=[labels.tolist()],
                return_tensors="pt",
            ).to(device)

            # The points together describe one thing to select, so multimask_output=False
            # asks for a single mask over the whole set rather than one per point.
            with torch.no_grad():
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    multimask_output=False,
                )

            # post_process_masks removes the padding the processor added and scales the mask
            # back to the source resolution, returning one (point_batch, masks, h, w) tensor
            # per image. One image and one mask leaves a leading axis on each.
            predicted = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0][0].numpy()

            selection = np.expand_dims(predicted, axis=-1)

            mattes.append(torch.from_numpy(np.repeat(selection, 3, axis=-1)))

            mask = torch.from_numpy(selection)
            mask = mask.squeeze(2)
            masks.append(mask.squeeze().to(torch.float32))

        # The leading axis post_process_masks leaves on each matte is the batch axis, so
        # the mattes concatenate. A single mask keeps the unbatched 2D shape it has always
        # returned and more than one takes a batch axis.
        stacked = masks[0] if len(masks) == 1 else torch.stack(masks, dim=0)
        mask_report.publish(None, stacked)
        return io.NodeOutput(torch.cat(mattes, dim=0), stacked)
