"""Prompt variables: capture a phrase once, repeat it by number.

``$|a stormy sky|$`` captures the phrase it wraps and names it ``$1``, ``$2`` and so on.
Every occurrence of a name expands back to its phrase.
"""

from __future__ import annotations

import re

__all__ = ["parse_prompt_vars"]


def parse_prompt_vars(input_string, optional_vars=None):
    """Capture ``$|...|$`` phrases, number them, and expand every reference.

    Args:
        input_string: The prompt to parse.
        optional_vars: An existing ``{"$1": phrase}`` table to expand against and add to.
            It is updated in place, and numbering continues after the entries it already
            holds.

    Returns:
        A ``(text, variables)`` pair: the parsed text, and the table of every variable
        known after the parse. A phrase is inserted literally, so one holding a backslash
        is not read as an escape.
    """
    variables = optional_vars or {}
    pattern = r"\$\|(.*?)\|\$"
    variable_count = len(variables) + 1

    def replace_variable(match):
        nonlocal variable_count
        variable_name = f"${variable_count}"
        variables[variable_name] = match.group(1)
        variable_count += 1
        return variable_name

    output_string = re.sub(pattern, replace_variable, input_string)

    for variable_name, phrase in variables.items():
        variable_pattern = re.escape(variable_name)
        # The replacement is a callable, which re.sub inserts verbatim. A string
        # replacement is a template instead: a phrase holding a Windows path is a bad \U
        # escape and a phrase holding \1 is a group reference, and the phrase cannot be
        # escaped the way the pattern is, being the text to insert.
        output_string = re.sub(
            variable_pattern, lambda match, replacement=phrase: replacement, output_string
        )

    return output_string, variables
