"""MiDaS depth-estimation model loading."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import MIDAS_MODEL
from ...modules.model import midas

REQUIRES = "midas"


class MidasModelLoader(io.ComfyNode):
    """Load the depth model the MiDaS nodes run on.

    The MIDAS_MODEL socket carries a :class:`~modules.model.Backend` of processor and model.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiDaS Model Loader",
            display_name="MiDaS Model Loader",
            search_aliases=["MiDaS Model Loader", "midas", "depth", "dpt"],
            category="WAS Suite/Loaders",
            description=(
                "Load a MiDaS depth model for MiDaS Depth Approximation and MiDaS Mask "
                "Image. Enable features.midas to load this node."
            ),
            inputs=[
                io.Combo.Input(
                    "midas_model",
                    options=["DPT_Large", "DPT_Hybrid", "DPT_Small"],
                    tooltip=(
                        "Which depth model to load. `DPT_Large` is the most accurate and the "
                        "slowest, `DPT_Hybrid` is roughly half the size and close behind it, "
                        "and `DPT_Small` is the quickest and the roughest."
                    ),
                ),
            ],
            outputs=[
                MIDAS_MODEL.Output(
                    display_name="midas_model",
                    tooltip=(
                        "The loaded model, for the midas_model input of MiDaS Depth "
                        "Approximation and MiDaS Mask Image."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, midas_model) -> io.NodeOutput:
        # Legacy ``.pt`` files cached under models/midas/checkpoints are not readable by
        # this backend, and the error raised here names the repository that replaces them.
        return io.NodeOutput(midas.load(midas_model))
