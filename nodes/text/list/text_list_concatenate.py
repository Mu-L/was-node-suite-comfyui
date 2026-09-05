"""Concatenate lists of text."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LIST

#: Input ids in socket order.
SLOTS = tuple(f"list_{letter}" for letter in "abcdefghijklmnopqrstuvwx")


class TextListConcatenate(io.ComfyNode):
    """Join lists end to end, in socket order."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text List Concatenate",
            display_name="Text List Concatenate",
            search_aliases=[
                "Text List Concatenate",
                "merge lists",
                "join lists",
                "number list",
                "list of numbers",
                "join schedules",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Join up to 24 lists end to end, in socket order. Unconnected inputs "
                "are skipped."
            ),
            inputs=[
                LIST.Input(
                    "list_a",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "First list, whose entries come first in the result. An unconnected "
                        "input contributes nothing."
                    ),
                ),
                LIST.Input(
                    "list_b",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="Second list, appended after list_a.",
                ),
                LIST.Input(
                    "list_c",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="Third list, appended after list_b.",
                ),
                LIST.Input(
                    "list_d",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="Fourth list, appended last.",
                ),
                LIST.Input(
                    "list_e",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 5, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_f",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 6, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_g",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 7, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_h",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 8, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_i",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 9, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_j",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 10, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_k",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 11, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_l",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 12, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_m",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 13, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_n",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 14, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_o",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 15, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_p",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 16, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_q",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 17, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_r",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 18, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_s",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 19, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_t",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 20, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_u",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 21, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_v",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 22, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_w",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 23, joined on after the one before it. Unconnected is skipped.",
                ),
                LIST.Input(
                    "list_x",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip="List 24, joined on after the one before it. Unconnected is skipped.",
                ),
            ],
            outputs=[
                LIST.Output(
                    tooltip=(
                        "One list holding the entries of every connected input, in socket "
                        "order. Duplicates are kept."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, **extra) -> io.NodeOutput:
        merged_list: list[str] = []
        for name in SLOTS:
            value = extra.get(name)
            if isinstance(value, list):
                merged_list += value
        return io.NodeOutput(merged_list)
