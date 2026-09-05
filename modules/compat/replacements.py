"""Node replacement table and its registration entry point.

An ``io.NodeReplace`` entry migrates a saved workflow from a retired node onto its
successor with no rewiring.
"""

from __future__ import annotations

from comfy_api.latest import ComfyAPI, io

from .. import log

logger = log.get_logger("compat")

#: Every migration the pack offers, keyed on ``(old_node_id, new_node_id)`` at registration.
REPLACEMENTS: tuple[io.NodeReplace, ...] = (
    io.NodeReplace(
        new_node_id="WASImageCropFaceNative",
        old_node_id="Image Crop Face",
        old_widget_ids=["crop_padding_factor", "cascade_xml"],
        input_mapping=[
            {"new_id": "image", "old_id": "image"},
            {"new_id": "crop_padding_factor", "old_id": "crop_padding_factor"},
            {"new_id": "cascade", "old_id": "cascade_xml"},
        ],
        output_mapping=[{"new_idx": 0, "old_idx": 0}, {"new_idx": 1, "old_idx": 1}],
    ),
    io.NodeReplace(
        new_node_id="WASImageGradientMapNative",
        old_node_id="Image Gradient Map",
        old_widget_ids=["flip_left_right"],
        input_mapping=[
            {"new_id": "image", "old_id": "image"},
            {"new_id": "flip_left_right", "old_id": "flip_left_right"},
            {"new_id": "gradient_image", "old_id": "gradient_image"},
        ],
        output_mapping=[{"new_idx": 0, "old_idx": 0}],
    ),
    io.NodeReplace(
        new_node_id="Image Batch",
        old_node_id="WASImageBatchAutogrow",
        old_widget_ids=[],
        input_mapping=[
            {"new_id": "images_a", "old_id": "images.images_a"},
            {"new_id": "images_b", "old_id": "images.images_b"},
            {"new_id": "images_c", "old_id": "images.images_c"},
            {"new_id": "images_d", "old_id": "images.images_d"},
            {"new_id": "images_e", "old_id": "images.images_e"},
            {"new_id": "images_f", "old_id": "images.images_f"},
            {"new_id": "images_g", "old_id": "images.images_g"},
            {"new_id": "images_h", "old_id": "images.images_h"},
            {"new_id": "images_i", "old_id": "images.images_i"},
            {"new_id": "images_j", "old_id": "images.images_j"},
            {"new_id": "images_k", "old_id": "images.images_k"},
            {"new_id": "images_l", "old_id": "images.images_l"},
            {"new_id": "images_m", "old_id": "images.images_m"},
            {"new_id": "images_n", "old_id": "images.images_n"},
            {"new_id": "images_o", "old_id": "images.images_o"},
            {"new_id": "images_p", "old_id": "images.images_p"},
            {"new_id": "images_q", "old_id": "images.images_q"},
            {"new_id": "images_r", "old_id": "images.images_r"},
            {"new_id": "images_s", "old_id": "images.images_s"},
            {"new_id": "images_t", "old_id": "images.images_t"},
            {"new_id": "images_u", "old_id": "images.images_u"},
            {"new_id": "images_v", "old_id": "images.images_v"},
            {"new_id": "images_w", "old_id": "images.images_w"},
            {"new_id": "images_x", "old_id": "images.images_x"},
            {"new_id": "images_y", "old_id": "images.images_y"},
            {"new_id": "images_z", "old_id": "images.images_z"},
        ],
        output_mapping=[{"new_idx": 0, "old_idx": 0}, {"new_idx": 1, "old_idx": 1}],
    ),
    io.NodeReplace(
        new_node_id="Lora Loader",
        old_node_id="Load Lora",
        old_widget_ids=["lora_name", "strength_model", "strength_clip"],
        input_mapping=[
            {"new_id": "model", "old_id": "model"},
            {"new_id": "clip", "old_id": "clip"},
            {"new_id": "lora_name", "old_id": "lora_name"},
            {"new_id": "strength_model", "old_id": "strength_model"},
            {"new_id": "strength_clip", "old_id": "strength_clip"},
        ],
        output_mapping=[
            {"new_idx": 0, "old_idx": 0},
            {"new_idx": 1, "old_idx": 1},
            {"new_idx": 2, "old_idx": 2},
        ],
    ),
    io.NodeReplace(
        new_node_id="Number to String",
        old_node_id="Number to Text",
        old_widget_ids=[],
        input_mapping=[{"new_id": "number", "old_id": "number"}],
        output_mapping=[{"new_idx": 0, "old_idx": 0}],
    ),
    # The old menu is not carried across. Its default named every terminology, which the
    # empty pick box already answers with, and any other value is a term name typed again.
    io.NodeReplace(
        new_node_id="WASNoodleSoupPick",
        old_node_id="WASNoodleSoupTermList",
        old_widget_ids=["term", "limit", "term_name"],
        input_mapping=[
            {"new_id": "limit", "old_id": "limit"},
            {"new_id": "term", "old_id": "term_name"},
        ],
        output_mapping=[
            {"new_idx": 0, "old_idx": 0},
            {"new_idx": 1, "old_idx": 1},
            {"new_idx": 2, "old_idx": 2},
            {"new_idx": 3, "old_idx": 3},
        ],
    ),
)


async def register_replacements() -> None:
    """Register every entry in :data:`REPLACEMENTS`. Called once from ``on_load``.

    Never raises: a failure is logged and the entries after it are left unregistered.
    """
    api = ComfyAPI()
    for index, replacement in enumerate(REPLACEMENTS):
        try:
            # register reaches into PromptServer.instance, absent outside a running server.
            await api.node_replacement.register(replacement)
        except Exception as error:
            logger.warning(
                "%s -> %s was not registered (%s), and %d further replacement(s) were not "
                "attempted; a workflow saved against one of those old node ids opens with "
                "that node missing and no swap offered for it",
                replacement.old_node_id, replacement.new_node_id, error,
                len(REPLACEMENTS) - index - 1,
            )
            return
    if REPLACEMENTS:
        logger.debug("registered %d node replacement(s)", len(REPLACEMENTS))
