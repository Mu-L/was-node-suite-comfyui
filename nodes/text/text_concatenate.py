"""Join text inputs with a delimiter."""

from __future__ import annotations

from comfy_api.latest import io

#: Input ids in socket order.
SLOTS = tuple(f"text_{letter}" for letter in "abcdefghijklmnopqrstuvwx")


class TextConcatenate(io.ComfyNode):
    """Join connected text inputs in socket order, skipping the empty ones."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Concatenate",
            display_name="Text Concatenate",
            search_aliases=["Text Concatenate", "join text", "combine strings"],
            category="WAS Suite/Text",
            description=(
                "Join up to 24 text inputs with a delimiter, in socket order. Empty "
                "inputs are skipped. Type \\n as the delimiter to join with newlines."
            ),
            inputs=[
                io.String.Input(
                    "delimiter",
                    default=", ",
                    tooltip=(
                        "Put between the joined pieces; STRING. Eg: ', ' for a prompt, \n for "
                        "one per line, empty to run them together."
                    ),
                ),
                io.Boolean.Input(
                    "clean_whitespace",
                    default=True,
                    tooltip=(
                        "Trim whitespace off each piece before joining. `on` also drops a "
                        "piece holding only spaces."
                    ),
                ),
                io.String.Input(
                    "text_a",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: masterpiece",
                    tooltip=(
                        "First piece to join; STRING. Joined with delimiter; empty "
                        "pieces are skipped. Eg: masterpiece"
                    ),
                ),
                io.String.Input(
                    "text_b",
                    multiline=True,
                    optional=True,
                    placeholder="Second piece, joined after the first",
                    tooltip=(
                        "Piece 2; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_c",
                    multiline=True,
                    optional=True,
                    placeholder="Third piece, joined after the second",
                    tooltip=(
                        "Piece 3; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_d",
                    multiline=True,
                    optional=True,
                    placeholder="Fourth piece, joined last",
                    tooltip=(
                        "Piece 4; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_e",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 5, joined in socket order",
                    tooltip=(
                        "Piece 5; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_f",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 6, joined in socket order",
                    tooltip=(
                        "Piece 6; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_g",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 7, joined in socket order",
                    tooltip=(
                        "Piece 7; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_h",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 8, joined in socket order",
                    tooltip=(
                        "Piece 8; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_i",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 9, joined in socket order",
                    tooltip=(
                        "Piece 9; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_j",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 10, joined in socket order",
                    tooltip=(
                        "Piece 10; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_k",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 11, joined in socket order",
                    tooltip=(
                        "Piece 11; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_l",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 12, joined in socket order",
                    tooltip=(
                        "Piece 12; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_m",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 13, joined in socket order",
                    tooltip=(
                        "Piece 13; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_n",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 14, joined in socket order",
                    tooltip=(
                        "Piece 14; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_o",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 15, joined in socket order",
                    tooltip=(
                        "Piece 15; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_p",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 16, joined in socket order",
                    tooltip=(
                        "Piece 16; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_q",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 17, joined in socket order",
                    tooltip=(
                        "Piece 17; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_r",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 18, joined in socket order",
                    tooltip=(
                        "Piece 18; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_s",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 19, joined in socket order",
                    tooltip=(
                        "Piece 19; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_t",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 20, joined in socket order",
                    tooltip=(
                        "Piece 20; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_u",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 21, joined in socket order",
                    tooltip=(
                        "Piece 21; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_v",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 22, joined in socket order",
                    tooltip=(
                        "Piece 22; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_w",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 23, joined in socket order",
                    tooltip=(
                        "Piece 23; STRING. Empty is skipped."
                    ),
                ),
                io.String.Input(
                    "text_x",
                    multiline=True,
                    optional=True,
                    placeholder="Piece 24, joined in socket order",
                    tooltip=(
                        "Piece 24; STRING. Empty is skipped."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The connected inputs joined in socket order, separated by the "
                        "delimiter."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        delimiter,
        clean_whitespace,
        **extra,
    ) -> io.NodeOutput:
        if delimiter in ("\n", "\\n"):
            delimiter = "\n"

        text_inputs: list[str] = []
        for name in SLOTS:
            value = extra.get(name)
            if not isinstance(value, str):
                continue
            if clean_whitespace:
                value = value.strip()
            if value != "":
                text_inputs.append(value)

        return io.NodeOutput(delimiter.join(text_inputs))
