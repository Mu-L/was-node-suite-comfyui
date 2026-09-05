"""Canny edge detection."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image.convolve import (
    gaussian_blur,
    gradients,
    hysteresis,
    luminance,
    thin_edges,
)

#: Fraction of the strongest gradient an edge has to reach when no threshold is given.
AUTO_THRESHOLD = 0.1

#: Width in pixels of the kernel the image is smoothed with before its gradients are read.
BLUR_SIZE = 5

#: Standard deviation of that kernel, in pixels.
BLUR_SIGMA = 1.4

#: Value of the upper threshold at which no edge tracing runs.
NO_TRACING = 1.0


def canny_detector(images, weak_th: float | None = None, strong_th: float | None = None):
    """Find the edges of a batch of images and thin them to single-pixel lines.

    Args:
        images: ``(batch, height, width, channels)`` tensor in ``[0, 1]``.
        weak_th: Fraction of the strongest gradient in an image below which an edge is
            dropped. ``None`` or 0 selects a tenth of it.
        strong_th: Fraction of the strongest gradient at which an edge starts, from where
            it is followed down to ``weak_th``. ``None`` or 1.0 traces nothing and leaves
            ``weak_th`` alone deciding.

    Returns:
        A ``(batch, height, width, 3)`` tensor in ``[0, 1]``, carrying gradient strength
        along each edge and zero everywhere else.
    """
    x = images.permute(0, 3, 1, 2).float() * 255.0
    gray = gaussian_blur(luminance(x), size=BLUR_SIZE, sigma=BLUR_SIGMA)
    magnitude, gx, gy = gradients(gray)
    peak = magnitude.amax(dim=(1, 2, 3), keepdim=True)

    thinned = thin_edges(magnitude, gx, gy)
    low = peak * (weak_th if weak_th else AUTO_THRESHOLD)
    if strong_th is not None and strong_th < NO_TRACING:
        kept = hysteresis(thinned, low, peak * strong_th)
    else:
        kept = thinned >= low

    edges = (thinned * kept / 255.0).clamp(0, 1)
    return edges.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)


class ImageCannyFilter(io.ComfyNode):
    """Reduce an image to thin bright edge lines on black."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Canny Filter",
            display_name="Image Canny Filter",
            search_aliases=["Image Canny Filter", "canny", "edge detect", "outline", "lineart"],
            category="WAS Suite/Image/Filter",
            description=(
                "Trace the edges in an image as thin bright lines on a black background, "
                "the usual input for a Canny ControlNet. The whole batch is traced at once, "
                "on the same device the images are already on."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The images to trace. Every image in a batch is traced.",
                ),
                io.Boolean.Input(
                    "enable_threshold",
                    default=False,
                    tooltip=(
                        "Whether threshold_low and threshold_high are read at all. Off = a cut-off "
                        "worked out from each image itself, at a tenth of its strongest edge, "
                        "which holds line density steady across a batch; on = the two values "
                        "below, for matching a reference render."
                    ),
                ),
                io.Float.Input(
                    "threshold_low",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strong an edge has to be to survive, where 1.0 is the strongest "
                        "edge found. Raising it drops faint detail and leaves only the main "
                        "outlines. 0.0 means automatic, the same tenth-of-the-strongest cut-off "
                        "used when enable_threshold is off."
                    ),
                ),
                io.Float.Input(
                    "threshold_high",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strong an edge has to be to start a line, which is then followed "
                        "through anything above threshold_low. 0.3 starts lines on the firm "
                        "outlines and carries them into fainter detail joined to them; 1.0 "
                        "follows nothing and leaves threshold_low alone deciding."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The traced edges: bright lines on black, one image per image in the "
                        "input batch."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, enable_threshold, threshold_low, threshold_high) -> io.NodeOutput:
        if not enable_threshold:
            threshold_low = None
            threshold_high = None

        return io.NodeOutput(canny_detector(images, threshold_low, threshold_high))
