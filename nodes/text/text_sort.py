"""Sort the separated terms of a prompt, keeping bracketed groups intact."""

from __future__ import annotations

from comfy_api.latest import io


class TextSort(io.ComfyNode):
    """Split ``text`` on ``separator``, sort the terms and join them back."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Sort",
            display_name="Text Sort",
            search_aliases=["Text Sort", "sort", "alphabetical", "prompt terms"],
            category="WAS Suite/Text/Operations",
            description=(
                "Sort the separated terms of a prompt alphabetically, leaving parenthesised "
                "attention groups intact."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: sunset, cat, forest",
                    tooltip=(
                        "List on one line; STRING. Sorted alphabetically and rejoined "
                        "with separator. Leading brackets and weights are ignored. Eg: "
                        "`sunset, cat, forest`"
                    ),
                ),
                io.String.Input(
                    "separator",
                    default=", ",
                    multiline=False,
                    tooltip=(
                        "The character the text is cut apart on, and the string the sorted "
                        "terms are rejoined with. The default ', ' sorts a comma-separated "
                        "prompt and puts a comma and a space back between each term."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The terms in alphabetical order, rejoined with the separator. "
                        "Leading parentheses are ignored while sorting, so '((sunset))' "
                        "files under s."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text, separator) -> io.NodeOutput:
        tokens = cls.split_using_protected_groups(
            text.strip(separator + " \t\n\r"), separator.strip()
        )
        sorted_tokens = sorted(tokens, key=cls.token_without_leading_brackets)
        return io.NodeOutput(separator.join(sorted_tokens))

    @staticmethod
    def token_without_leading_brackets(token):
        """Sort key: the token with its unescaped parentheses removed and trimmed."""
        return token.replace("\\(", "\0\1").replace("(", "").replace("\0\1", "(").strip()

    @staticmethod
    def split_using_protected_groups(text, separator):
        """Split ``text`` on ``separator`` except where nesting depth is above zero.

        Args:
            text: The prompt to split.
            separator: A single separator character.

        Returns:
            The trimmed parts, with separators that sat inside parentheses restored.
        """
        protected_groups = ""
        nesting_level = 0
        for char in text:
            if char == "(":
                nesting_level += 1
            if char == ")":
                nesting_level -= 1

            if char == separator and nesting_level > 0:
                protected_groups += "\0"
            else:
                protected_groups += char

        return [part.replace("\0", separator).strip() for part in protected_groups.split(separator)]
