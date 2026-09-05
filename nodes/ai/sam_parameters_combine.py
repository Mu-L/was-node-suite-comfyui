"""Merging two sets of Segment Anything point prompts."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ...modules.compat.sockets import require_input
from ...modules.compat.types import SAM_PARAMETERS

REQUIRES = "sam"


class SamParametersCombine(io.ComfyNode):
    """Join two SAM_PARAMETERS into one."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM Parameters Combine",
            display_name="SAM Parameters Combine",
            search_aliases=["SAM Parameters Combine", "sam points merge", "join sam points"],
            category="WAS Suite/Image/Masking",
            description=(
                "Merge two sets of Segment Anything points into a single set, so several SAM "
                "Parameters nodes can describe one selection. Enable features.sam to load "
                "this node."
            ),
            inputs=[
                SAM_PARAMETERS.Input(
                    "sam_parameters_a",
                    tooltip="The points and labels that come first in the merged set.",
                ),
                SAM_PARAMETERS.Input(
                    "sam_parameters_b",
                    tooltip="The points and labels appended after the first set.",
                ),
            ],
            outputs=[
                SAM_PARAMETERS.Output(
                    tooltip=(
                        "Every point of both inputs, for the sam_parameters input of SAM "
                        "Image Mask or a further SAM Parameters Combine."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, sam_parameters_a, sam_parameters_b) -> io.NodeOutput:
        """Join the two point sets.

        Raises:
            ValueError: Nothing is connected to one of the two point inputs.
        """
        for value, socket in (
            (sam_parameters_a, "sam_parameters_a"),
            (sam_parameters_b, "sam_parameters_b"),
        ):
            require_input(
                value,
                "SAM Parameters Combine",
                socket,
                "points",
                "SAM Parameters or another SAM Parameters Combine",
                "SAM_PARAMETERS",
            )

        parameters = {
            "points": np.concatenate(
                (sam_parameters_a["points"], sam_parameters_b["points"]), axis=0
            ),
            "labels": np.concatenate((sam_parameters_a["labels"], sam_parameters_b["labels"])),
        }
        return io.NodeOutput(parameters)
