"""Shuffle the separated terms of a string."""

from __future__ import annotations

import random

from comfy_api.latest import io


class TextShuffle(io.ComfyNode):
    """Split ``text`` on ``separator``, shuffle the parts and join them back."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Shuffle",
            display_name="Text Shuffle",
            search_aliases=["Text Shuffle", "shuffle", "randomize", "prompt terms"],
            category="WAS Suite/Text/Operations",
            description="Randomly reorder the separated terms of a string.",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: cat, forest, sunset",
                    tooltip=(
                        "List on one line; STRING. Reordered randomly and rejoined with "
                        "separator. Eg: `cat, forest, sunset`"
                    ),
                ),
                io.String.Input(
                    "separator",
                    default=",",
                    multiline=False,
                    tooltip=(
                        "The character the text is cut apart on, and the character the "
                        "shuffled parts are rejoined with. The default ',' shuffles a "
                        "comma-separated prompt term by term; a single space shuffles it "
                        "word by word."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "Which order comes out. The same seed and the same text always give "
                        "the same order; change it to shuffle differently. Any whole number; "
                        "`0` is as good a seed as any."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The same terms in a new order, rejoined with the separator. "
                        "Whitespace that sat next to a separator moves with its term."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text, separator, seed) -> io.NodeOutput:
        if seed is not None:
            random.seed(seed)

        text_list = text.split(separator)
        random.shuffle(text_list)
        new_text = separator.join(text_list)

        return io.NodeOutput(new_text)
