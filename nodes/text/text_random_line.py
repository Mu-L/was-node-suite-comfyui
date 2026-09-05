"""Pick one line of text at random."""

from __future__ import annotations

import random

from comfy_api.latest import io


class TextRandomLine(io.ComfyNode):
    """Choose one line of the incoming text, seeded reproducibly."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Random Line",
            display_name="Text Random Line",
            search_aliases=["Text Random Line", "random line", "pick line"],
            category="WAS Suite/Text",
            description="Pick one line of the incoming text at random, chosen by the seed.",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="one candidate per line",
                    tooltip=(
                        "Candidates, one per line; STRING, as `a tabby cat`. One line is returned. Blank "
                        "lines count as candidates."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Which line comes out. The same seed and the same text always give the "
                        "same line; change it to draw a different one. Any whole number; `0` "
                        "is as good a seed as any."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(tooltip="The single line the seed selected."),
            ],
        )

    @classmethod
    def execute(cls, text, seed) -> io.NodeOutput:
        lines = text.split("\n")
        random.seed(seed)
        return io.NodeOutput(random.choice(lines))
