"""Dynamic prompts: pick one option out of ``<a|b|c>``.

Every angle-bracketed group is replaced with one of its ``|``-separated options, chosen at
random from a seeded RNG, and the pass repeats while any group is still standing.
"""

from __future__ import annotations

import random
import re

__all__ = ["parse_dynamic_prompt"]


def parse_dynamic_prompt(prompt, seed):
    """Resolve every ``<a|b|c>`` group in ``prompt`` to one of its options.

    Args:
        prompt: The prompt to parse.
        seed: Seed for the shared :mod:`random` module, applied once before the first
            choice. Unlike the other parsers here, every value seeds, ``0`` included.

    Returns:
        The parsed prompt. A group with no ``|`` resolves to its own contents, which is
        what strips the brackets from a single-option group.
    """
    random.seed(seed)

    def replace_match(match):
        options = match.group(1).split("|")
        return random.choice(options)

    parse_prompt = re.sub(r"\<(.*?)\>", replace_match, prompt)
    while re.search(r"\<(.*?)\>", parse_prompt):
        parse_prompt = re.sub(r"\<(.*?)\>", replace_match, parse_prompt)

    return parse_prompt
