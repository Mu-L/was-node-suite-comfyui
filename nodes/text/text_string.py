"""Four single-line text fields on one node."""

from __future__ import annotations

from comfy_api.latest import io



#: Input ids in socket order, the first box being the unsuffixed one.
SLOTS = ("text", *(f"text_{letter}" for letter in "bcdefghijklmnopqrstuvwx"))


class TextString(io.ComfyNode):
    """Emit up to four single-line strings, each with tokens substituted."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text String",
            display_name="Text String",
            search_aliases=["Text String", "string", "four strings"],
            category="WAS Suite/Text",
            description=(
                "Four text fields on one node, each with its own output. Tokens such as "
                "[time] and [user] are substituted in each of them."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Text for the first output. Tokens such as `[time]`, `[user]` and "
                        "`[hostname]` are replaced with their values, so this is a convenient "
                        "source for file name prefixes and captions."
                    ),
                ),
                io.String.Input(
                    "text_b",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text for the second output, expanded the same way.",
                ),
                io.String.Input(
                    "text_c",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text for the third output, expanded the same way.",
                ),
                io.String.Input(
                    "text_d",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text for the fourth output, expanded the same way.",
                ),
                io.String.Input(
                    "text_e",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 5, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_f",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 6, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_g",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 7, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_h",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 8, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_i",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 9, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_j",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 10, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_k",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 11, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_l",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 12, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_m",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 13, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_n",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 14, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_o",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 15, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_p",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 16, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_q",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 17, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_r",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 18, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_s",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 19, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_t",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 20, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_u",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 21, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_v",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 22, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_w",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 23, emitted on its own output. Tokens are substituted.",
                ),
                io.String.Input(
                    "text_x",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Text 24, emitted on its own output. Tokens are substituted.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="TEXT",
                    tooltip="The text field, with its tokens replaced.",
                ),
                io.String.Output(
                    display_name="TEXT_B",
                    tooltip="The text_b field, with its tokens replaced.",
                ),
                io.String.Output(
                    display_name="TEXT_C",
                    tooltip="The text_c field, with its tokens replaced.",
                ),
                io.String.Output(
                    display_name="TEXT_D",
                    tooltip="The text_d field, with its tokens replaced.",
                ),
                io.String.Output(
                    display_name="TEXT_E",
                    tooltip="Text 5, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_F",
                    tooltip="Text 6, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_G",
                    tooltip="Text 7, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_H",
                    tooltip="Text 8, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_I",
                    tooltip="Text 9, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_J",
                    tooltip="Text 10, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_K",
                    tooltip="Text 11, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_L",
                    tooltip="Text 12, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_M",
                    tooltip="Text 13, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_N",
                    tooltip="Text 14, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_O",
                    tooltip="Text 15, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_P",
                    tooltip="Text 16, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_Q",
                    tooltip="Text 17, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_R",
                    tooltip="Text 18, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_S",
                    tooltip="Text 19, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_T",
                    tooltip="Text 20, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_U",
                    tooltip="Text 21, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_V",
                    tooltip="Text 22, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_W",
                    tooltip="Text 23, with its tokens substituted.",
                ),
                io.String.Output(
                    display_name="TEXT_X",
                    tooltip="Text 24, with its tokens substituted.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text="", **extra) -> io.NodeOutput:
        return io.NodeOutput(text, *(extra.get(name, "") for name in SLOTS[1:]))
