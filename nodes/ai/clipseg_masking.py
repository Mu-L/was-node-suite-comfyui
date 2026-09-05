"""Text-prompted masking with CLIPSeg."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import CLIPSEG_MODEL
from ...modules.interface import mask_report

REQUIRES = "clipseg"


class ClipsegMasking(io.ComfyNode):
    """Mask whatever a phrase describes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPSeg Masking",
            display_name="CLIPSeg Masking",
            search_aliases=["CLIPSeg Masking", "text to mask", "clipseg", "segment by prompt"],
            category="WAS Suite/Image/Masking",
            description=(
                "Make a mask from a word for what to select, such as `person`, `sky` or "
                "`the red car`. Enable features.clipseg to load this node."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to search. A batch is handled as a batch, one mask per "
                        "image, all against the same text."
                    ),
                ),
                io.String.Input(
                    "text",
                    default="",
                    multiline=False,
                    tooltip=(
                        "What to select, in plain words: 'the sky', 'a red car', 'hair'. Short "
                        "noun phrases work best. An empty string still runs and matches "
                        "nothing in particular."
                    ),
                ),
                CLIPSEG_MODEL.Input(
                    "clipseg_model",
                    tooltip=(
                        "The segmentation model, from CLIPSeg Model Loader. One loader can "
                        "feed several nodes so the weights are built once."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASK",
                    tooltip=(
                        "The match as a mask, for an inpainting or compositing node. Brighter "
                        "means a stronger match."
                    ),
                ),
                io.Image.Output(
                    display_name="MASK_IMAGE",
                    tooltip=(
                        "The same match as a black and white image, to preview or to feed a "
                        "node that takes an image rather than a mask."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, text, clipseg_model) -> io.NodeOutput:
        """Score every pixel against the text and answer the mask."""
        import torch

        batch, height, width, _ = image.shape
        backend = clipseg_model
        device = backend.load()

        inputs = backend.processor(
            text=[text] * batch,
            images=image.permute(0, 3, 1, 2) * 255,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            scores = torch.sigmoid(backend.model(**inputs)[0])

        # A batch of one comes back without its leading axis.
        if scores.dim() == 2:
            scores = scores.unsqueeze(0)

        low = scores.amin()
        spread = (scores.amax() - low).clamp(min=1e-8)
        mask = ((scores - low) / spread).clamp(0.0, 1.0)

        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(1).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        matte = mask.to(device=image.device, dtype=image.dtype)
        mask_report.publish(None, matte)
        return io.NodeOutput(matte, matte.unsqueeze(-1).repeat(1, 1, 1, 3))
