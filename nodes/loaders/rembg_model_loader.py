"""Cutout model loading."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import REMBG_MODEL
from ...modules.model import cutout

REQUIRES = "preprocessors"

#: The cutout models the ``model`` widget offers, in the order they are listed.
MODELS = tuple(cutout.MODELS)


class RembgModelLoader(io.ComfyNode):
    """Build the cutout network Image Remove Background runs on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASRembgModelLoader",
            display_name="Image Remove Background Model Loader",
            search_aliases=[
                "WASRembgModelLoader",
                "Image Remove Background Model Loader",
                "Cutout Model Loader",
                "Rembg Model Loader",
                "rembg",
                "remove background",
                "cutout",
                "birefnet",
                "ben2",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Build a cutout network for Image Remove Background. Building one takes a "
                "moment and holds a few hundred megabytes, so it is kept for the life of "
                "the process and one loader can feed several nodes. Weights go in "
                "ComfyUI/models/birefnet and ComfyUI/models/ben2, and are downloaded there "
                "on first use when features.network is on."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=list(MODELS),
                    tooltip=(
                        "Which cutout network to build. `BiRefNet General` suits most "
                        "pictures. `BiRefNet Portrait` is trained on people and `BiRefNet "
                        "Matting HR` on fine edges like hair, both read at 2048 across. "
                        "`BEN2` is a second opinion from another family. docs/MODELS.md "
                        "lists what each one suits and what it weighs."
                    ),
                ),
            ],
            outputs=[
                REMBG_MODEL.Output(
                    display_name="rembg_model",
                    tooltip=(
                        "The built network, for the rembg_model input of Image Remove "
                        "Background."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(cutout.load(model))
