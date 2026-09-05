"""The tensor dimensions of a latent, as numbers."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.sockets import require_input
from ...modules.compat.types import NUMBER


class LatentSizeToNumber(io.ComfyNode):
    """Emit a latent's width and height on NUMBER, FLOAT and INT sockets.

    Latent dimensions are an eighth of the pixel size.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Latent Size to Number",
            display_name="Latent Size to Number",
            search_aliases=["Latent Size to Number", "width", "height", "dimensions", "size"],
            category="WAS Suite/Number/Operations",
            description=(
                "Read a latent's tensor width and height, on one pair of sockets per numeric "
                "type. These are latent units: an eighth of the pixel dimensions the latent "
                "decodes to."
            ),
            inputs=[
                io.Latent.Input(
                    "samples",
                    tooltip=(
                        "The latent to measure, from an Empty Latent Image, a VAE Encode or "
                        "a sampler. It is passed over untouched."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    display_name="tensor_w_num",
                    tooltip=(
                        "Latent width, on a NUMBER socket. 64 here means 512 pixels once "
                        "decoded; multiply by 8 for the pixel width."
                    ),
                ),
                NUMBER.Output(
                    display_name="tensor_h_num",
                    tooltip=(
                        "Latent height, on a NUMBER socket. Multiply by 8 for the pixel "
                        "height."
                    ),
                ),
                io.Float.Output(
                    display_name="tensor_w_float",
                    tooltip="The same latent width as a float, so 64 leaves here as 64.0.",
                ),
                io.Float.Output(
                    display_name="tensor_h_float",
                    tooltip="The same latent height as a float, so 64 leaves here as 64.0.",
                ),
                io.Int.Output(
                    display_name="tensor_w_int",
                    tooltip="The same latent width as an INT.",
                ),
                io.Int.Output(
                    display_name="tensor_h_int",
                    tooltip="The same latent height as an INT.",
                ),
            ],
        )

    @classmethod
    def execute(cls, samples) -> io.NodeOutput:
        """Read the latent's size.

        Raises:
            ValueError: Nothing is connected to the samples input.
        """
        require_input(
            samples,
            "Latent Size to Number",
            "samples",
            "latent",
            "latent source such as Empty Latent Image or VAE Encode",
            "LATENT",
        )
        height, width = (int(size) for size in samples["samples"].shape[-2:])
        return io.NodeOutput(width, height, float(width), float(height), width, height)
