"""Text-prompted masking of several images at once with CLIPSeg."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ...modules.compat.types import CLIPSEG_MODEL
from ...modules.convert.tensors import image_planes, tensor2pil
from ...modules.interface import mask_report

REQUIRES = "clipseg"


#: The letter of each picture and prompt pair, in socket order.
SLOTS = tuple("abcdefghijklmnopqrstuvwx")


class ClipsegBatchMasking(io.ComfyNode):
    """Mask up to six image sockets against up to six phrases in one pass."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPSeg Batch Masking",
            display_name="CLIPSeg Batch Masking",
            search_aliases=[
                "CLIPSeg Batch Masking",
                "text to mask batch",
                "clipseg",
                "segment by prompt",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Mask several images at once, each against its own description, and return the "
                "images, the masks and the masks as images as three matching batches. Enable "
                "features.clipseg to load this node."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip=(
                        "First image. Its size decides the size of every output, and the other "
                        "images have to match it. A batch here is masked image by image, all "
                        "against text_a."
                    ),
                ),
                io.Image.Input(
                    "image_b",
                    tooltip=(
                        "Second image. Masked against text_b, and a batch here is masked image "
                        "by image against it."
                    ),
                ),
                io.String.Input(
                    "text_a",
                    default="",
                    multiline=True,
                    tooltip=(
                        "What to select in image_a, in plain words: 'the sky', 'a red car'. "
                        "Short noun phrases work best. This box and text_b are always used, "
                        "empty or not."
                    ),
                ),
                io.String.Input(
                    "text_b",
                    default="",
                    multiline=True,
                    tooltip="What to select in image_b.",
                ),
                CLIPSEG_MODEL.Input(
                    "clipseg_model",
                    tooltip=(
                        "The segmentation model, from CLIPSeg Model Loader. One loader can "
                        "feed several nodes so the weights are built once."
                    ),
                ),
                io.Image.Input(
                    "image_c",
                    optional=True,
                    tooltip="Third image, if there is one. Same size as image_a.",
                ),
                io.Image.Input(
                    "image_d",
                    optional=True,
                    tooltip="Fourth image, if there is one. Same size as image_a.",
                ),
                io.Image.Input(
                    "image_e",
                    optional=True,
                    tooltip="Fifth image, if there is one. Same size as image_a.",
                ),
                io.Image.Input(
                    "image_f",
                    optional=True,
                    tooltip="Sixth image, if there is one. Same size as image_a.",
                ),
                io.Image.Input(
                    "image_g",
                    optional=True,
                    tooltip="Picture 7, segmented by text_g. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_h",
                    optional=True,
                    tooltip="Picture 8, segmented by text_h. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_i",
                    optional=True,
                    tooltip="Picture 9, segmented by text_i. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_j",
                    optional=True,
                    tooltip="Picture 10, segmented by text_j. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_k",
                    optional=True,
                    tooltip="Picture 11, segmented by text_k. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_l",
                    optional=True,
                    tooltip="Picture 12, segmented by text_l. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_m",
                    optional=True,
                    tooltip="Picture 13, segmented by text_m. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_n",
                    optional=True,
                    tooltip="Picture 14, segmented by text_n. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_o",
                    optional=True,
                    tooltip="Picture 15, segmented by text_o. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_p",
                    optional=True,
                    tooltip="Picture 16, segmented by text_p. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_q",
                    optional=True,
                    tooltip="Picture 17, segmented by text_q. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_r",
                    optional=True,
                    tooltip="Picture 18, segmented by text_r. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_s",
                    optional=True,
                    tooltip="Picture 19, segmented by text_s. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_t",
                    optional=True,
                    tooltip="Picture 20, segmented by text_t. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_u",
                    optional=True,
                    tooltip="Picture 21, segmented by text_u. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_v",
                    optional=True,
                    tooltip="Picture 22, segmented by text_v. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_w",
                    optional=True,
                    tooltip="Picture 23, segmented by text_w. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "image_x",
                    optional=True,
                    tooltip="Picture 24, segmented by text_x. Unconnected is skipped.",
                ),
                io.String.Input(
                    "text_c",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to select in image_c. An empty box is left out of the list.",
                ),
                io.String.Input(
                    "text_d",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to select in image_d. An empty box is left out of the list.",
                ),
                io.String.Input(
                    "text_e",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to select in image_e. An empty box is left out of the list.",
                ),
                io.String.Input(
                    "text_f",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to select in image_f. An empty box is left out of the list.",
                ),
                io.String.Input(
                    "text_g",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_g. Empty is skipped.",
                ),
                io.String.Input(
                    "text_h",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_h. Empty is skipped.",
                ),
                io.String.Input(
                    "text_i",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_i. Empty is skipped.",
                ),
                io.String.Input(
                    "text_j",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_j. Empty is skipped.",
                ),
                io.String.Input(
                    "text_k",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_k. Empty is skipped.",
                ),
                io.String.Input(
                    "text_l",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_l. Empty is skipped.",
                ),
                io.String.Input(
                    "text_m",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_m. Empty is skipped.",
                ),
                io.String.Input(
                    "text_n",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_n. Empty is skipped.",
                ),
                io.String.Input(
                    "text_o",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_o. Empty is skipped.",
                ),
                io.String.Input(
                    "text_p",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_p. Empty is skipped.",
                ),
                io.String.Input(
                    "text_q",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_q. Empty is skipped.",
                ),
                io.String.Input(
                    "text_r",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_r. Empty is skipped.",
                ),
                io.String.Input(
                    "text_s",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_s. Empty is skipped.",
                ),
                io.String.Input(
                    "text_t",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_t. Empty is skipped.",
                ),
                io.String.Input(
                    "text_u",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_u. Empty is skipped.",
                ),
                io.String.Input(
                    "text_v",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_v. Empty is skipped.",
                ),
                io.String.Input(
                    "text_w",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_w. Empty is skipped.",
                ),
                io.String.Input(
                    "text_x",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="What to find in image_x. Empty is skipped.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGES_BATCH",
                    tooltip=(
                        "Every image that was given, as one batch in input order, so the masks "
                        "line up with the pictures they came from."
                    ),
                ),
                io.Mask.Output(
                    display_name="MASKS_BATCH",
                    tooltip=(
                        "One mask per image, brighter where the phrase matched, for an "
                        "inpainting or compositing node."
                    ),
                ),
                io.Image.Output(
                    display_name="MASK_IMAGES_BATCH",
                    tooltip=(
                        "The same masks as black and white images, to preview or to feed a node "
                        "that takes an image rather than a mask."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many frames each batch holds, which is the total across the "
                        "slots rather than the number of slots."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_a,
        image_b,
        text_a,
        text_b,
        clipseg_model,
        **extra,
    ) -> io.NodeOutput:
        import torch

        frames_a = image_planes(image_a)
        frames_b = image_planes(image_b)
        images_pil = [tensor2pil(plane) for plane in frames_a + frames_b]
        prompts = [text_a] * len(frames_a) + [text_b] * len(frames_b)

        # shape[-2:] is the width and the channel count, which is the comparison the
        # inherited guard makes; two images of the same width and depth pass it whatever
        # their heights are.
        for letter in SLOTS[2:]:
            name = f"image_{letter}"
            picture, text = extra.get(name), extra.get(f"text_{letter}")
            frames = []
            if picture is not None:
                if picture.shape[-2:] != image_a.shape[-2:]:
                    raise ValueError(f"Size of {name} is different from image_a.")
                frames = image_planes(picture)
                images_pil += [tensor2pil(plane) for plane in frames]
            # An empty phrase drops out of the list whether or not its socket is connected,
            # and a phrase on an unconnected socket still enters it once. CLIPSeg needs as
            # many phrases as images, so a gap in the middle raises rather than being guessed.
            if text:
                prompts += [text] * max(len(frames), 1)

        images_tensor = torch.cat(
            [
                torch.from_numpy(
                    np.array(img.convert("RGB")).astype(np.float32) / 255.0
                ).unsqueeze(0)
                for img in images_pil
            ],
            dim=0,
        )

        backend = clipseg_model
        device = backend.load()

        inputs = backend.processor(
            text=prompts, images=images_pil, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            result = backend.model(**inputs)

        scores = torch.sigmoid(result.logits)
        # A batch of one comes back without its leading axis.
        if scores.dim() == 2:
            scores = scores.unsqueeze(0)

        low = scores.amin()
        spread = (scores.amax() - low).clamp(min=1e-8)
        mask = ((scores - low) / spread).clamp(0.0, 1.0)

        width, height = images_pil[0].size
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(1).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Two sockets of four images give eight images and eight phrases, so all three
        # outputs are one batch of that same length.
        matte = mask.to(device=images_tensor.device, dtype=images_tensor.dtype)
        mask_report.publish(None, matte)
        return io.NodeOutput(
            images_tensor, matte, matte.unsqueeze(-1).repeat(1, 1, 1, 3), int(matte.shape[0])
        )
