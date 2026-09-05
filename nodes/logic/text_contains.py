"""Substring test over two strings."""

from __future__ import annotations

from comfy_api.latest import io


class TextContains(io.ComfyNode):
    """Report whether one string occurs inside another."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Contains",
            display_name="Text Contains",
            search_aliases=["Text Contains", "substring", "text search"],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Search one piece of text for another and report whether it is in there, "
                "which is how a prompt can be tested for a word before a branch is taken."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    default="",
                    multiline=False,
                    tooltip="The text to search through, such as a prompt or a file name, as `a tabby cat`.",
                ),
                io.String.Input(
                    "sub_text",
                    default="",
                    multiline=False,
                    tooltip=(
                        "The word or phrase to look for. It has to appear as written, "
                        "spaces included, and it can sit anywhere in the text rather than "
                        "only at the start. Left empty, it matches everything."
                    ),
                ),
                io.Boolean.Input(
                    "case_insensitive",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Whether capitals are ignored. On, 'Cat' finds 'cat'; off, only an "
                        "exact match of upper and lower case counts."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    tooltip="True when the phrase was found somewhere in the text.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text, sub_text, case_insensitive=True) -> io.NodeOutput:
        if case_insensitive:
            sub_text = sub_text.lower()
            text = text.lower()
        return io.NodeOutput(sub_text in text)
