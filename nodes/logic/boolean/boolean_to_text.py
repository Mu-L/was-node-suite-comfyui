"""Spell a boolean as text."""

from __future__ import annotations

from comfy_api.latest import io


class BooleanToText(io.ComfyNode):
    """Emit ``"True"`` or ``"False"`` for a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Boolean To Text",
            display_name="Boolean To Text",
            search_aliases=["Boolean To Text", "bool to string", "boolean to string"],
            category="WAS Suite/Logic/Boolean",
            description=(
                'Spell a true/false value out as the text "True" or "False", so it can go '
                "into a prompt, a file name or a debug string."
            ),
            inputs=[
                io.Boolean.Input(
                    "boolean",
                    default=False,
                    tooltip=(
                        "The value to spell out. Usually linked from a comparison node "
                        "such as Logic Comparison AND or Text Contains."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        'The words "True" or "False", capitalised, with no surrounding '
                        "spaces or quotes."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, boolean) -> io.NodeOutput:
        return io.NodeOutput("True" if boolean else "False")
