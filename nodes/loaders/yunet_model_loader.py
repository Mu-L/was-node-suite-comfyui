"""YuNet face detection model loading."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import YUNET_MODEL
from ...modules.model import yunet

REQUIRES = "yunet"


class YuNetModelLoader(io.ComfyNode):
    """Open the YuNet detector for Image Crop Face (YuNet)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASYuNetModelLoader",
            display_name="YuNet Model Loader",
            search_aliases=[
                "WASYuNetModelLoader",
                "YuNet Model Loader",
                "yunet",
                "face detection",
                "face detector",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Load the YuNet face detector for Image Crop Face (YuNet). The weights "
                "ship with the pack, so there is nothing to download and nothing to "
                "install: connect this to Image Crop Face (YuNet) and run it. The detector "
                "runs on whatever device ComfyUI is using."
            ),
            inputs=[],
            outputs=[
                YUNET_MODEL.Output(
                    display_name="yunet_model",
                    tooltip=(
                        "The loaded detector, for the yunet_model input of Image Crop Face "
                        "(YuNet)."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        """Answer the detector, built once and kept for the process."""
        return io.NodeOutput(yunet.load())
