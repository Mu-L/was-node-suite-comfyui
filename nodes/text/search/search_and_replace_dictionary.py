"""Replace delimited keys in text with the matching values of a dictionary."""

from __future__ import annotations

import random

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import DICT


class TextFindAndReplaceByDictionary(io.ComfyNode):
    """Substitute every ``__term__`` in ``text`` with ``dictionary["term"]``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Find and Replace by Dictionary",
            display_name="Text Find and Replace by Dictionary",
            search_aliases=[
                "Text Find and Replace by Dictionary",
                "search and replace",
                "dictionary",
                "template",
            ],
            category="WAS Suite/Text/Search",
            description=(
                "Replace each delimited term in the text with its value from a dictionary, "
                "for example __subject__ with the dictionary's subject entry."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: a __animal__ in a __place__",
                    tooltip=(
                        "Text to fill in; STRING. Blanks written __key__ are swapped "
                        "for that key's dictionary value; a LIST value is redrawn per "
                        "blank. Eg: a __animal__ in a __place__"
                    ),
                ),
                DICT.Input(
                    "dictionary",
                    tooltip=(
                        "The replacements; DICT. Each key is a term to look for, its value the "
                        "replacement. A LIST value has one item drawn per occurrence. Unmatched "
                        "terms are left as written."
                    ),
                ),
                io.String.Input(
                    "replacement_key",
                    default="__",
                    multiline=False,
                    tooltip=(
                        "Marker put either side of a key; STRING. Eg: __ matches __subject__, "
                        "not subject."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "Seed for drawing from LIST values; INT. The same seed rewrites the "
                        "same text. Ignored for single-value entries. Any whole number; `0` is "
                        "as good a seed as any."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip="The text with every marked term replaced by its entry's value.",
                ),
            ],
            not_idempotent=True,
        )

    @staticmethod
    def replacement(value) -> str:
        """Return the text one entry of the dictionary substitutes.

        Args:
            value: The entry's value. A list or tuple has one item drawn from it with the
                shared :mod:`random` module, seeded by the caller before each call, so two
                occurrences of one term can differ and both are reproducible. Any other
                value is used as its text.

        Returns:
            The text to substitute. An empty list substitutes nothing.
        """
        if isinstance(value, (list, tuple)):
            return str(random.choice(value)) if value else ""
        return value if isinstance(value, str) else str(value)

    @classmethod
    def execute(cls, text, dictionary, replacement_key, seed) -> io.NodeOutput:
        """Substitute every term the dictionary holds.

        Raises:
            ValueError: Nothing is connected to the dictionary input.
        """
        require_input(
            dictionary,
            "Text Find and Replace by Dictionary",
            "dictionary",
            "dictionary",
            "Text Dictionary New or Text Dictionary Convert",
            "DICT",
        )
        random.seed(seed)
        new_text = text

        for term in dictionary.keys():
            tkey = f"{replacement_key}{term}{replacement_key}"
            tcount = new_text.count(tkey)
            for _ in range(tcount):
                new_text = new_text.replace(tkey, cls.replacement(dictionary[term]), 1)
                if seed != 0:
                    seed += 1
                    random.seed(seed)

        return io.NodeOutput(new_text)
