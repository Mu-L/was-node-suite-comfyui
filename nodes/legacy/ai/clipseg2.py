"""Tiled text-prompted masking with CLIPSeg, thresholded to black and white."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ....modules.compat.types import CLIPSEG_MODEL
from ....modules.model import clipseg

REQUIRES = "dupes"

#: Side of the square CLIPSeg reads, in pixels. The model's own input resolution, so a tile
#: this size needs no resampling before it is scored.
SLICE_SIZE = 352

#: How far neighbouring tiles overlap, in pixels. Half a tile, so every pixel away from the
#: edges is covered twice and the seam is crossfaded rather than cut.
OVERLAP = SLICE_SIZE // 2


def tile_window(row, column, height, width):
    """Where one tile sits in an image.

    Args:
        row: Tile row index.
        column: Tile column index.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(top, left, bottom, right)`` in pixels.
    """
    start_h = row * (SLICE_SIZE - OVERLAP)
    start_w = column * (SLICE_SIZE - OVERLAP)

    end_h = min(start_h + SLICE_SIZE, height)
    end_w = min(start_w + SLICE_SIZE, width)

    start_h = max(0, end_h - SLICE_SIZE)
    start_w = max(0, end_w - SLICE_SIZE)

    return start_h, start_w, end_h, end_w


def blend_mask():
    """Build the crossfade weights two neighbouring tiles are averaged with.

    Returns:
        A ``(1, SLICE_SIZE, SLICE_SIZE, 1)`` float32 tensor: 1.0 across the middle, ramping
        linearly to 0.0 over the outermost ``OVERLAP`` pixels on each of the four sides.
    """
    import torch

    weights = np.ones((SLICE_SIZE, SLICE_SIZE))
    weights[:OVERLAP, :] *= np.linspace(0, 1, OVERLAP)[:, None]
    weights[-OVERLAP:, :] *= np.linspace(1, 0, OVERLAP)[:, None]
    weights[:, :OVERLAP] *= np.linspace(0, 1, OVERLAP)[None, :]
    weights[:, -OVERLAP:] *= np.linspace(1, 0, OVERLAP)[None, :]
    return torch.tensor(weights, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


def score_tile(image, text, processor, model, device):
    """Score one tile against a phrase and return the match as a mask.

    Args:
        image: Tile as a ``(batch, height, width, channels)`` float tensor in 0-1.
        text: Phrase every image in the tile batch is scored against.
        processor: CLIPSeg processor.
        model: CLIPSeg segmentation model.
        device: Device the model's weights are resident on.

    Returns:
        ``(mask, mask_image)``: the match as ``(batch, height, width, 1)`` and the same
        values repeated across three channels, both on the CPU and scaled to 0-1 against
        this tile's own strongest match.
    """
    import torch
    import torchvision

    B, H, W, C = image.shape

    with torch.no_grad():
        pixels = image.permute(0, 3, 1, 2).to(torch.float32) * 255
        inputs = processor(
            text=[text] * B, images=pixels, padding=True, return_tensors="pt"
        ).to(device)

        result = model(**inputs)
        scores = torch.sigmoid(result[0])
        mask = (scores - scores.min()) / scores.max()
        mask = torchvision.transforms.functional.resize(mask, (H, W))
        mask = mask.unsqueeze(-1)
        mask_image = mask.repeat(1, 1, 1, 3)

    return mask.cpu(), mask_image.cpu()


class Clipseg2(io.ComfyNode):
    """Mask what a phrase describes, tiled and then hard-thresholded."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPSEG2",
            display_name="CLIPSeg Tiled Masking",
            search_aliases=["CLIPSEG2", "clipseg tiled", "text to mask", "segment by prompt"],
            category="WAS Suite/Image/Masking",
            is_deprecated=True,
            description=(
                "Deprecated: use CLIPSeg Masking, which does the same job in one pass and also "
                "returns a MASK. This one scores the image in overlapping tiles and returns a "
                "hard black and white result. Enable legacy.dupes to load it."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to search. One image at a time; a batch of more than one is "
                        "refused."
                    ),
                ),
                io.String.Input(
                    "text",
                    default="",
                    multiline=False,
                    tooltip=(
                        "What to select, in plain words: 'the sky', 'a red car', 'hair'. Short "
                        "noun phrases work best."
                    ),
                ),
                io.Boolean.Input(
                    "use_cuda",
                    default=False,
                    tooltip=(
                        "On, the model runs on the graphics card, which is much faster on a "
                        "large image because every tile is a separate pass. Off, it runs on the "
                        "processor. A machine with no graphics card runs on the processor "
                        "either way."
                    ),
                ),
                CLIPSEG_MODEL.Input(
                    "clipseg_model",
                    optional=True,
                    tooltip=(
                        "An already-loaded model from CLIPSeg Model Loader. Wire one in to load "
                        "the weights once and share them between several nodes; leave it empty "
                        "to load the default model here."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The match as a pure black and white image: white where the phrase was "
                        "found, black everywhere else, with no soft edge."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, text, use_cuda, clipseg_model=None) -> io.NodeOutput:
        import torch
        import torch.nn.functional as F

        B, H, W, C = image.shape

        if B != 1:
            raise NotImplementedError("Batch size must be 1")

        rows = (H - OVERLAP) // (SLICE_SIZE - OVERLAP) + 1
        columns = (W - OVERLAP) // (SLICE_SIZE - OVERLAP) + 1

        # Tiling keeps detail smaller than the model's input resolution, which a single
        # whole-image pass loses to the downscale.
        tiles = []
        for row in range(rows):
            for column in range(columns):
                top, left, bottom, right = tile_window(row, column, H, W)
                tiles.append(image[:, top:bottom, left:right, :])

        backend = clipseg_model if clipseg_model is not None else clipseg.load()
        if use_cuda:
            backend = backend.on(None)
        device = backend.load()
        processor = backend.processor
        model = backend.model

        # The processor is shared with every other CLIPSeg node in the process, so the two
        # flags set here are put back before returning: the tiles are already the model's
        # input resolution and must not be resized, and they arrive scaled to 0-255.
        rescale = processor.image_processor.do_rescale
        resize = processor.image_processor.do_resize
        processor.image_processor.do_rescale = True
        processor.image_processor.do_resize = False
        try:
            # The whole-image pass is blended in so the tiles agree on what the phrase meant.
            global_pass = image.permute(0, 3, 1, 2)
            global_pass = F.interpolate(
                global_pass, size=(SLICE_SIZE, SLICE_SIZE), mode="bilinear", align_corners=False
            )
            global_pass = global_pass.permute(0, 2, 3, 1)
            _, global_pass = score_tile(global_pass.float(), text, processor, model, device)
            global_pass = global_pass.permute(0, 3, 1, 2)
            global_pass = F.interpolate(
                global_pass, size=(H, W), mode="bilinear", align_corners=False
            )
            global_pass = global_pass.permute(0, 2, 3, 1)

            scored_tiles = []
            for tile in tiles:
                _, scored = score_tile(tile, text, processor, model, device)
                scored_tiles.append(scored)
        finally:
            processor.image_processor.do_rescale = rescale
            processor.image_processor.do_resize = resize

        scored_tiles = torch.cat(scored_tiles)

        reconstructed = torch.zeros((B, H, W, C))
        count_map = torch.zeros((B, H, W, C))
        weights = blend_mask()

        for index in range(scored_tiles.shape[0]):
            top, left, bottom, right = tile_window(index // columns, index % columns, H, W)
            reconstructed[:, top:bottom, left:right, :] += scored_tiles[index] * weights
            count_map[:, top:bottom, left:right, :] += weights

        count_map[count_map == 0] = 1
        tiled = reconstructed / count_map

        total_power = (tiled + global_pass) / 2
        just_black = global_pass < 0.01

        condition = (total_power > 0.5) | (tiled > 0.5) | (global_pass > 0.5)
        condition = condition & ~just_black

        return io.NodeOutput(torch.where(condition, 1.0, 0.0))
