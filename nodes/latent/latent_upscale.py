"""Scale a latent by a factor with a selectable interpolation mode."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.sockets import require_input
from ...modules.interface import size_report

VALID_MODES = ["area", "bicubic", "bilinear", "nearest"]

#: Interpolation modes ``torch.nn.functional.interpolate`` accepts ``align_corners`` for.
#: Passing it to any other mode raises, so ``align`` is dropped for ``area`` and
#: ``nearest``.
ALIGNABLE_MODES = ["linear", "bilinear", "bicubic", "trilinear"]


class LatentUpscaleByFactor(io.ComfyNode):
    """Resize a LATENT's spatial dimensions by a multiplier."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Latent Upscale by Factor (WAS)",
            display_name="Latent Upscale by Factor",
            search_aliases=[
                "Latent Upscale by Factor (WAS)",
                "latent upscale",
                "scale latent",
                "resize latent",
            ],
            category="WAS Suite/Latent/Transform",
            description=(
                "A latent resized by a multiplier, with a choice of how the values in "
                "between are worked out."
            ),
            inputs=[
                io.Latent.Input("samples", tooltip="The latent to resize."),
                io.Combo.Input(
                    "mode",
                    options=VALID_MODES,
                    tooltip=(
                        "How new values are worked out between the existing ones. `nearest` "
                        "copies the closest value and is blocky; `bilinear` and `bicubic` "
                        "interpolate and are progressively smoother; `area` averages over "
                        "the source region and suits shrinking rather than enlarging."
                    ),
                ),
                io.Float.Input(
                    "factor",
                    default=2.0,
                    min=0.1,
                    max=8.0,
                    step=0.01,
                    tooltip=(
                        "Multiplier applied to both the height and the width. 2.0 doubles "
                        "the size, 0.5 halves it, 1.0 leaves it as it is. A factor small "
                        "enough to shrink an axis away leaves one latent block of it, so "
                        "the result is never empty."
                    ),
                ),
                io.Boolean.Input(
                    "align",
                    default=True,
                    tooltip=(
                        "Whether the outermost values are pinned to the edges of the result "
                        "instead of to the centres of the corner samples, which shifts the "
                        "image very slightly. Only the bilinear and bicubic modes use this; "
                        "`area` and `nearest` ignore it."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(tooltip="The resized latent."),
            ],
        )

    @classmethod
    def execute(cls, samples, mode, factor, align) -> io.NodeOutput:
        """Resize the latent.

        Raises:
            ValueError: Nothing is connected to the samples input, the mode is not one of
                :data:`VALID_MODES`, or the factor is not a positive number.
        """
        require_input(
            samples,
            "Latent Upscale by Factor (WAS)",
            "samples",
            "latent",
            "latent source such as Empty Latent Image or VAE Encode",
            "LATENT",
        )
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid interpolation mode `{mode}` selected. "
                f"Valid modes are: {', '.join(VALID_MODES)}."
            )
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
            raise ValueError(f"The input `factor` is `{factor}`, but should be a positive number.")

        align_corners = align
        scaled = samples.copy()
        shape = scaled["samples"].shape
        # A latent has to keep at least one block on each axis: interpolate refuses a zero
        # output size on every mode but area, and area answers with an empty latent.
        size = tuple(max(1, int(round(dim * factor))) for dim in shape[-2:])
        if mode in ALIGNABLE_MODES:
            scaled["samples"] = torch.nn.functional.interpolate(
                scaled["samples"], size=size, mode=mode, align_corners=align_corners
            )
        else:
            scaled["samples"] = torch.nn.functional.interpolate(
                scaled["samples"], size=size, mode=mode
            )
        size_report.publish(
            samples["samples"],
            scaled["samples"],
            action="scaled",
            unit=size_report.LATENT,
            layout=size_report.PLANE,
            facts={"mode": mode},
        )
        return io.NodeOutput(scaled)
