"""Expand suite tokens such as ``[time]`` and ``[user]`` inside a string."""

from __future__ import annotations

from comfy_api.latest import io


class TextParseTokens(io.ComfyNode):
    """Replace every known token in ``text`` with its value."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Parse Tokens",
            display_name="Text Parse Tokens",
            search_aliases=["Text Parse Tokens", "tokens", "variables", "substitute"],
            category="WAS Suite/Text/Tokens",
            description=(
                "Replace suite tokens such as [time], [hostname] and [user], plus any custom "
                "tokens, with their current values."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: render_[time(%Y-%m-%d)]",
                    tooltip=(
                        "Text holding tokens; STRING. [time], [time(%Y-%m-%d)], "
                        "[hostname], [user], [cuda_device], [cuda_name], plus custom "
                        "tokens. Eg: render_[time]"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The text with every known token replaced. An unrecognised token is "
                        "left as written."
                    ),
                ),
            ],
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        # Every string input arrives with its tokens already replaced, so the value is
        # handed on as it stands.
        return io.NodeOutput(text)
