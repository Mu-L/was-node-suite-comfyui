"""Wildcard substitution from a directory of text files.

Every file under the wildcards directory becomes one key: ``animals/birds.txt`` is
``__animals/birds__``. Each occurrence of a key is replaced with one random non-comment
line from that file, and substitution runs twice.
"""

from __future__ import annotations

import os
import random
import re

from .. import log

__all__ = ["replace_wildcards"]

logger = log.get_logger("prompt.wildcards")

#: Multiple of a file's line count that bounds one draw. Lines are sampled uniformly over
#: the whole file, comments included, so a file whose usable lines are rare needs
#: proportionally more attempts: this is a hundred times the expected number of attempts
#: when exactly one line is usable, which is the worst case for a file that has one at all.
DRAW_BUDGET_FACTOR = 100

#: Floor under the draw budget, so a short file still gets a generous number of attempts.
MIN_DRAW_BUDGET = 1000


def _usable(line: str) -> bool:
    """Whether a line can be drawn: not blank, and not a ``#`` or ``//`` comment."""
    line = line.strip()
    return bool(line) and not line.startswith("#") and not line.startswith("//")


def _draw_line(lines: list[str], file_path: str) -> str:
    """One random usable line from ``lines``.

    Args:
        lines: The file's lines, as read.
        file_path: Path the lines came from, named in the error.

    Returns:
        The drawn line, stripped.

    Raises:
        ValueError: The file holds no line that can be drawn, or the draw did not find
            one within its budget.
    """
    if not any(_usable(line) for line in lines):
        raise ValueError(
            f"the wildcard file `{file_path}` has no line to draw: every one of its "
            f"{len(lines)} line(s) is blank or a comment starting with # or //. Add a line "
            f"to it, or move it out of the wildcards directory"
        )
    budget = max(MIN_DRAW_BUDGET, len(lines) * DRAW_BUDGET_FACTOR)
    for _ in range(budget):
        line = random.choice(lines).strip()
        if _usable(line):
            return line
    raise ValueError(
        f"the wildcard file `{file_path}` did not yield a usable line in {budget} draws, "
        f"so almost every one of its {len(lines)} line(s) is blank or a comment"
    )


def _substituted(text: str, key: str, file_path: str) -> str:
    """Replace every occurrence of one wildcard key with a single line drawn from its file.

    Args:
        text: The text as it stands.
        key: The delimited key this file answers to.
        file_path: The file to draw from.

    Returns:
        The text with every occurrence of ``key`` replaced by one drawn line. Unchanged
        when the file holds no line at all, and when the text does not mention the key.

    Raises:
        OSError: The file could not be read.
        UnicodeDecodeError: The file is not UTF-8.
        ValueError: The text mentions the key and the file has no line to draw.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    if not lines:
        return text
    if key not in text:
        # Raising here would let one comments-only file refuse every prompt in the install,
        # including prompts that name only files that work.
        if not any(_usable(line) for line in lines):
            logger.warning(
                "the wildcard file `%s` has no line to draw: every one of its %s line(s) is "
                "blank or a comment starting with # or //. No wildcard in this prompt names "
                "it, so it is skipped; a prompt that does name it fails.",
                file_path, len(lines),
            )
            return text
        _draw_line(lines, file_path)
        return text
    return text.replace(key, _draw_line(lines, file_path))


def replace_wildcards(text, seed=None, noodle_key="__", wildcard_dir=None):
    """Replace each wildcard key in ``text`` with a random line from its file.

    Args:
        text: The prompt to parse.
        seed: Seed for the shared :mod:`random` module. Any falsy value, ``None`` and
            ``0`` alike, leaves the module's existing state alone, so the parse is not
            reproducible. Callers pass ``None if seed == 0 else seed``.
        noodle_key: Delimiter placed either side of a wildcard's relative path.
        wildcard_dir: Directory of wildcard files. Defaults to the ``paths.wildcards``
            config key, and is created when it does not exist.

    Returns:
        The parsed text. A key with no matching file is left as it is, and so is a key
        whose file is empty.

    Raises:
        OSError: The wildcards directory could not be created, or one of its files could
            not be read.
        UnicodeDecodeError: A wildcard file is not UTF-8.
        ValueError: A wildcard file this text names holds no line that can be drawn. A file
            the text does not name is logged and skipped instead.
    """
    # The delimiter is a widget value and reaches a pattern here, so it is escaped: a
    # noodle_key of '*', '+' or '(' is not a valid pattern on its own.
    delimiter = re.escape(noodle_key)

    def replace_nested(text, key_path_dict):
        if re.findall(f"{delimiter}(.+?){delimiter}", text):
            for key, file_path in key_path_dict.items():
                text = _substituted(text, key, file_path)
        return text

    if wildcard_dir is None:
        from .. import config

        wildcard_dir = str(config.wildcards_directory())
    if not os.path.exists(wildcard_dir):
        os.makedirs(wildcard_dir, exist_ok=True)

    logger.info("wildcard path: %s", wildcard_dir)

    # Set the random seed for reproducibility
    if seed:
        random.seed(seed)

    # Create a dictionary of key to file path pairs
    key_path_dict = {}
    for root, dirs, files in os.walk(wildcard_dir):
        for file in files:
            file_path = os.path.join(root, file)
            key = os.path.relpath(file_path, wildcard_dir).replace(os.path.sep, "/").rsplit(".", 1)[0]
            key_path_dict[f"{noodle_key}{key}{noodle_key}"] = os.path.abspath(file_path)

    # Replace keys in text with random lines from corresponding files
    for key, file_path in key_path_dict.items():
        text = _substituted(text, key, file_path)

    # Replace sub-wildcards in result
    text = replace_nested(text, key_path_dict)

    return text
