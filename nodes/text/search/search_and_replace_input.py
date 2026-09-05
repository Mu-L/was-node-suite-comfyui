"""Literal search and replace where the search and replacement terms arrive as links."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER


class TextFindAndReplaceInput(io.ComfyNode):
    """Replace every literal occurrence of ``find`` in ``text`` with ``replace``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Find and Replace Input",
            display_name="Text Find and Replace Input",
            search_aliases=[
                "Text Find and Replace Input",
                "search and replace",
                "literal replace",
            ],
            category="WAS Suite/Text/Search",
            description=(
                "Replace every literal occurrence of the find text and report how many "
                "replacements were made. All three text terms come from links."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: a cat on a mat",
                    tooltip=(
                        "Text to search; STRING, as `a tabby cat`. Every occurrence of find becomes "
                        "replace."
                    ),
                ),
                io.String.Input(
                    "find",
                    multiline=True,
                    placeholder="Exact text to look for, taken literally",
                    tooltip=(
                        "Exact text to look for, taken literally and matched case "
                        "sensitively, no pattern characters, unlike Text Find and "
                        "Replace. An empty search term leaves the text untouched and "
                        "reports 0 replacements."
                    ),
                ),
                io.String.Input(
                    "replace",
                    multiline=True,
                    placeholder="What each occurrence becomes. Empty deletes them",
                    tooltip=(
                        "What each occurrence becomes, again taken literally. An empty "
                        "value deletes the occurrences."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="result_text",
                    tooltip="The text with every occurrence of the search term replaced.",
                ),
                NUMBER.Output(
                    display_name="replacement_count_number",
                    tooltip=(
                        "How many occurrences were replaced, for the NUMBER inputs of the "
                        "suite's own maths and logic nodes. 0 means the term was not present."
                    ),
                ),
                io.Float.Output(
                    display_name="replacement_count_float",
                    tooltip="The same count as a decimal, for example 3.0.",
                ),
                io.Int.Output(
                    display_name="replacement_count_int",
                    tooltip="The same count as a whole number, for a core INT input.",
                ),
            ],
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, text, find, replace) -> io.NodeOutput:
        if not find:
            return io.NodeOutput(text, 0, 0.0, 0)

        # A replacement that reintroduces the search term, cat becoming (cat), would match
        # itself forever in the loop below, so it gets one pass over the original text.
        if find in replace:
            count = text.count(find)
            return io.NodeOutput(text.replace(find, replace), count, float(count), int(count))

        count = 0
        new_text = text
        while find in new_text:
            new_text = new_text.replace(find, replace, 1)
            count += 1
        return io.NodeOutput(new_text, count, float(count), int(count))
