"""Numbered output file names, and the directory a save node writes them into.

:func:`generate_filename` answers the next unused name in a directory, built from a
prefix, a delimiter, a zero-padded number, a suffix and an extension.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from . import sandbox

__all__ = ["OUTPUT_TOKENS", "generate_filename", "resolve_output_directory"]

#: Widget values that mean the ComfyUI output directory itself.
OUTPUT_TOKENS = ("", "none", "None", ".")


def resolve_output_directory(value: str) -> Path:
    """Resolve the user-supplied directory to write into.

    Args:
        value: The widget value, already token-expanded.

    Returns:
        An absolute directory path inside a permitted write root. It need not exist yet.

    Raises:
        PathNotAllowed: The value resolved outside every permitted write root.
    """
    value = value.strip()
    if value in OUTPUT_TOKENS:
        # Imported here, which leaves a widget naming a directory outright resolvable
        # without ComfyUI's own module being present.
        import folder_paths

        value = folder_paths.get_output_directory()
    # The widget value is read as written, as the v2 node read it, so a relative path lands
    # where a workflow carried over from v2 expects it.
    return sandbox.resolve_write(value)


def generate_filename(path, prefix, delimiter, number_padding, extension, suffix) -> str:
    """The next unused file name in ``path``.

    Args:
        path: Directory the file is written to. Scanned for existing numbers.
        prefix: Leading name part.
        delimiter: Separator between the prefix and the number. May be empty.
        number_padding: Digits the number is padded to. ``0`` drops the number entirely,
            so the name is the prefix, the suffix and the extension.
        extension: File extension, leading dot included.
        suffix: Trailing name part, between the number and the extension.

    Returns:
        A file name, not a path.

    Raises:
        PathNotAllowed: The name built from ``prefix`` names a file outside ``path``. The
            prefix is a widget value, so the test for an existing file goes through the
            containment layer rather than joining the two.
    """
    if number_padding == 0:
        return f"{prefix}{suffix}{extension}"

    if delimiter:
        pattern = f"{re.escape(prefix)}{re.escape(delimiter)}(\\d{{{number_padding}}}){re.escape(suffix)}{re.escape(extension)}"
    else:
        pattern = f"{re.escape(prefix)}(\\d{{{number_padding}}}){re.escape(suffix)}{re.escape(extension)}"

    existing_counters = [
        int(re.search(pattern, filename).group(1))
        for filename in os.listdir(path)
        if re.match(pattern, filename) and filename.endswith(extension)
    ]
    existing_counters.sort()
    counter = existing_counters[-1] + 1 if existing_counters else 1

    def _numbered(number: int) -> str:
        return f"{prefix}{delimiter}{number:0{number_padding}}{suffix}{extension}"

    filename = _numbered(counter)
    while sandbox.resolve_write_file(path, filename).exists():
        counter += 1
        filename = _numbered(counter)
    return filename
