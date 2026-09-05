"""Split a string into a list of strings."""

from __future__ import annotations

import re

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LIST

logger = log.get_logger("nodes.text.list")


class TextSplitToList(io.ComfyNode):
    """Split a string into a ``LIST`` and into a ``STRING`` list."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextSplitToList",
            display_name="Text Split to List",
            search_aliases=[
                "WASTextSplitToList", "Text Split to List",
                "split",
                "explode",
                "string to list",
                "lines to list",
                "delimiter",
                "number list",
                "list of numbers",
                "numbers from text",
                "csv",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Split text into a list, on a delimiter, on line breaks, on whitespace or "
                "on a regular expression. The pieces come out twice: as one LIST, and as a "
                "STRING list that runs everything downstream once per piece. `delimiter` "
                "cuts on the exact text in the delimiter field, which is how a "
                "comma-separated prompt becomes one entry per tag, while `regex` reads that "
                "field as a pattern, so `[,;]` cuts on either mark. `characters` ignores "
                "the delimiter field."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    tooltip=(
                        "The text to cut up, as `cat, forest, sunset`. Typed here, or wired in from any node "
                        "with a text output, such as a prompt built by Text Concatenate."
                    ),
                ),
                io.Combo.Input(
                    "split_by",
                    options=["delimiter", "lines", "whitespace", "regex", "characters"],
                    tooltip=(
                        "Where the cuts go. `delimiter` and `regex` read the delimiter "
                        "field; `lines` cuts on line breaks, `whitespace` on runs of space, "
                        "`characters` on every character."
                    ),
                ),
                io.String.Input(
                    "delimiter",
                    default=",",
                    multiline=False,
                    tooltip=(
                        "What to cut on, read only by `delimiter` and `regex`. Type \\n for "
                        "a line break or \\t for a tab. An unreadable regular expression "
                        "stops with the error the pattern produced, naming the position in "
                        "it that failed."
                    ),
                ),
                io.Boolean.Input(
                    "trim_whitespace",
                    default=True,
                    tooltip=(
                        "Whether each piece has its surrounding space removed. On, "
                        "'a, b, c' gives 'a', 'b', 'c'; off it gives 'a', ' b', ' c', and "
                        "the leading spaces travel into whatever reads the list."
                    ),
                ),
                io.Boolean.Input(
                    "remove_empty",
                    default=True,
                    tooltip=(
                        "Whether pieces holding nothing are dropped. Two delimiters in a "
                        "row produce an empty piece, which is what a trailing comma on a "
                        "prompt leaves behind. Turn this off when the position of every "
                        "entry matters and an empty slot has to stay a slot."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    tooltip=(
                        "Every piece on one wire, for Text List Get, Text List Concatenate "
                        "and Text List to Text."
                    ),
                ),
                io.String.Output(
                    display_name="strings",
                    is_output_list=True,
                    tooltip=(
                        "The same pieces as a STRING list. Because this is a list, a node "
                        "reading it runs once per piece and produces one result per piece, "
                        "wire it into a sampler's prompt to render every entry in turn. "
                        "Text that splits into nothing stops the prompt, because a graph "
                        "cannot be run zero times."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many pieces the split produced.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, text, split_by="delimiter", delimiter=",", trim_whitespace=True, remove_empty=True
    ) -> io.NodeOutput:
        from ....modules.compat.lists import require_values

        parts = cls.split(text, split_by, delimiter)
        if trim_whitespace:
            parts = [part.strip() for part in parts]
        if remove_empty:
            parts = [part for part in parts if part]
        logger.debug("Text Split to List produced %d entries by %s", len(parts), split_by)
        require_values(
            parts,
            f"Text Split to List was given text that splits by {split_by} into nothing, so "
            f"there is no list to hand on and the graph below it cannot be run. The text "
            f"input is empty or holds only separators. Check what feeds it, or turn "
            f"remove_empty off to keep the empty pieces as entries.",
        )
        # A list of its own per slot, so a node that edits the one it was handed does not
        # change what the other slot emits or how many times the graph below it runs.
        return io.NodeOutput(parts, list(parts), len(parts))

    @staticmethod
    def split(text: str, split_by: str, delimiter: str) -> list[str]:
        """Cut ``text`` into pieces by one of the five modes.

        Args:
            text: The string to cut.
            split_by: ``delimiter``, ``lines``, ``whitespace``, ``regex`` or ``characters``.
            delimiter: Read by ``delimiter`` and ``regex``. The two-character escapes
                ``\\n`` and ``\\t`` stand for a line break and a tab, since a single-line
                widget cannot hold either.

        Returns:
            The pieces, untrimmed and including empty ones.

        Raises:
            re.error: ``split_by`` is ``regex`` and ``delimiter`` is not a valid pattern.
        """
        if split_by == "lines":
            return text.splitlines()
        if split_by == "whitespace":
            return text.split()
        if split_by == "characters":
            return list(text)

        separator = delimiter.replace("\\n", "\n").replace("\\t", "\t")
        if split_by == "regex":
            return re.split(separator, text)
        # An empty delimiter is a ValueError from str.split, and the whitespace split is
        # the only reading of "cut on nothing" that produces anything usable.
        return text.split(separator) if separator else text.split()
