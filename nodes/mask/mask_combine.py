"""Combine masks into one."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from . import same_size_or_refuse
from ...modules.interface import mask_report

#: Shape of the placeholder mask a loader with no mask of its own emits, such as an image
#: file with no alpha channel. A mask of this shape holding nothing is dropped before the
#: sum, which leaves an unused slot unable to whiten the result.
EMPTY_MASK_SHAPE = (1, 64, 64)

#: Input ids in slot order, so a size mismatch names the socket it came from.
SLOTS = tuple(f"mask_{letter}" for letter in "abcdefghijklmnopqrstuvwx")


def is_empty_slot(mask) -> bool:
    """Whether a mask is the blank placeholder rather than a mask to combine.

    Args:
        mask: Mask tensor from one of the inputs.

    Returns:
        True when the mask is shaped like the placeholder *and* holds nothing. A genuine
        64x64 mask with anything set in it is a mask, not an unused slot, and the shape
        alone does not tell the two apart.
    """
    return mask.shape == EMPTY_MASK_SHAPE and not mask.any()


class MasksCombineRegions(io.ComfyNode):
    """Sum the connected masks and clamp the result to the mask range."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Masks Combine Regions",
            display_name="Masks Combine Regions",
            search_aliases=["Masks Combine Regions", "combine masks", "merge masks", "union"],
            category="WAS Suite/Image/Masking",
            description="Sum every connected mask and clamp the total to 0-1. An empty 64x64 "
            "mask, which is what an unused slot carries, is skipped; if that leaves nothing, "
            "mask_a is returned unchanged.",
            inputs=[
                io.Mask.Input(
                    "mask_a",
                    tooltip=(
                        "The first mask to merge, and the fallback: if every slot turns out to "
                        "hold an empty 64x64 placeholder, this is what comes back out."
                    ),
                ),
                io.Mask.Input(
                    "mask_b",
                    tooltip=(
                        "The second mask to merge. All the masks wired in must be the same width "
                        "and height."
                    ),
                ),
                io.Mask.Input(
                    "mask_c",
                    optional=True,
                    tooltip="A third mask to merge in. Leave it unconnected to skip it.",
                ),
                io.Mask.Input(
                    "mask_d",
                    optional=True,
                    tooltip="A fourth mask to merge in. Leave it unconnected to skip it.",
                ),
                io.Mask.Input(
                    "mask_e",
                    optional=True,
                    tooltip="A fifth mask to merge in. Leave it unconnected to skip it.",
                ),
                io.Mask.Input(
                    "mask_f",
                    optional=True,
                    tooltip="A sixth mask to merge in. Leave it unconnected to skip it.",
                ),
                io.Mask.Input(
                    "mask_g",
                    optional=True,
                    tooltip="Mask 7, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_h",
                    optional=True,
                    tooltip="Mask 8, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_i",
                    optional=True,
                    tooltip="Mask 9, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_j",
                    optional=True,
                    tooltip="Mask 10, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_k",
                    optional=True,
                    tooltip="Mask 11, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_l",
                    optional=True,
                    tooltip="Mask 12, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_m",
                    optional=True,
                    tooltip="Mask 13, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_n",
                    optional=True,
                    tooltip="Mask 14, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_o",
                    optional=True,
                    tooltip="Mask 15, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_p",
                    optional=True,
                    tooltip="Mask 16, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_q",
                    optional=True,
                    tooltip="Mask 17, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_r",
                    optional=True,
                    tooltip="Mask 18, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_s",
                    optional=True,
                    tooltip="Mask 19, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_t",
                    optional=True,
                    tooltip="Mask 20, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_u",
                    optional=True,
                    tooltip="Mask 21, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_v",
                    optional=True,
                    tooltip="Mask 22, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_w",
                    optional=True,
                    tooltip="Mask 23, drawn onto the result. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "mask_x",
                    optional=True,
                    tooltip="Mask 24, drawn onto the result. Unconnected is skipped.",
                ),
            ],
            outputs=[
                io.Mask.Output(
                    tooltip=(
                        "One mask covering every area set in any of the inputs. The values are "
                        "added and then clamped to 0-1, so overlaps come out solid white rather "
                        "than overflowing."
                    )
                )
            ],
        )

    @classmethod
    def execute(cls, mask_a, mask_b, **extra) -> io.NodeOutput:
        supplied = {"mask_a": mask_a, "mask_b": mask_b, **extra}
        named = [
            (name, supplied[name])
            for name in SLOTS
            if supplied.get(name) is not None
        ]
        valid_masks = [(name, m) for name, m in named if not is_empty_slot(m)]

        same_size_or_refuse(valid_masks, "Masks Combine Regions")

        if len(valid_masks) == 0:
            mask_report.publish(mask_a, mask_a, source="mask_a")
            return io.NodeOutput(mask_a)
        if len(valid_masks) == 1:
            mask_report.publish(valid_masks[0][1], valid_masks[0][1], source=valid_masks[0][0])
            return io.NodeOutput(valid_masks[0][1])

        combined_mask = torch.sum(torch.stack([m for _, m in valid_masks], dim=0), dim=0)
        combined_mask = torch.clamp(combined_mask, 0, 1)
        mask_report.publish(valid_masks[0][1], combined_mask, source=valid_masks[0][0])
        return io.NodeOutput(combined_mask)
