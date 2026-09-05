"""Collect text inputs into a list."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LIST

#: Input ids in socket order.
SLOTS = tuple(f"text_{letter}" for letter in "abcdefghijklmnopqrstuvwx")


class TextList(io.ComfyNode):
    """Gather text inputs into one list."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text List",
            display_name="Text List",
            search_aliases=["Text List", "strings to list", "collect text"],
            category="WAS Suite/Text/List",
            description=(
                "Collect up to 24 texts into a list, in socket order. Each entry is "
                "typed in or wired in, and empty entries are skipped."
            ),
            inputs=[
                io.String.Input(
                    "text_a",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a cinematic photograph",
                    tooltip=(
                        "First entry of the LIST; STRING. Empty entries are skipped. "
                        "Eg: a cinematic photograph"
                    ),
                ),
                io.String.Input(
                    "text_b",
                    multiline=True,
                    optional=True,
                    placeholder="Second entry of the list",
                    tooltip=(
                        "Entry 2; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_c",
                    multiline=True,
                    optional=True,
                    placeholder="Third entry of the list",
                    tooltip=(
                        "Entry 3; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_d",
                    multiline=True,
                    optional=True,
                    placeholder="Fourth entry of the list",
                    tooltip=(
                        "Entry 4; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_e",
                    multiline=True,
                    optional=True,
                    placeholder="Fifth entry of the list",
                    tooltip=(
                        "Entry 5; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_f",
                    multiline=True,
                    optional=True,
                    placeholder="Sixth entry of the list",
                    tooltip=(
                        "Entry 6; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_g",
                    multiline=True,
                    optional=True,
                    placeholder="Seventh entry of the list",
                    tooltip=(
                        "Entry 7; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_h",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 8",
                    tooltip=(
                        "Entry 8 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_i",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 9",
                    tooltip=(
                        "Entry 9 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_j",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 10",
                    tooltip=(
                        "Entry 10 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_k",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 11",
                    tooltip=(
                        "Entry 11 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_l",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 12",
                    tooltip=(
                        "Entry 12 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_m",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 13",
                    tooltip=(
                        "Entry 13 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_n",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 14",
                    tooltip=(
                        "Entry 14 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_o",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 15",
                    tooltip=(
                        "Entry 15 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_p",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 16",
                    tooltip=(
                        "Entry 16 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_q",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 17",
                    tooltip=(
                        "Entry 17 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_r",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 18",
                    tooltip=(
                        "Entry 18 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_s",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 19",
                    tooltip=(
                        "Entry 19 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_t",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 20",
                    tooltip=(
                        "Entry 20 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_u",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 21",
                    tooltip=(
                        "Entry 21 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_v",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 22",
                    tooltip=(
                        "Entry 22 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_w",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 23",
                    tooltip=(
                        "Entry 23 of the list; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_x",
                    multiline=True,
                    optional=True,
                    placeholder="Entry 24",
                    tooltip=(
                        "Entry 24 of the list; STRING. Empty is skipped."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    tooltip=(
                        "The texts that hold something, as one list, in socket order. Text "
                        "List to Text turns it back into a string."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        **extra,
    ) -> io.NodeOutput:
        text_list: list[str] = []
        for name in SLOTS:
            value = extra.get(name)
            # An empty box and an unconnected socket both mean nothing was given here, and
            # they are indistinguishable at this end: a widget always sends its value, so
            # reading "" as an entry would make a freshly dropped node emit seven of them.
            if isinstance(value, str) and value != "":
                text_list.append(value)
        return io.NodeOutput(text_list)
