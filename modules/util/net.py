"""Downloading a file over HTTP.

Reached only from nodes gated on ``features.network``, so ``requests`` and ``tqdm`` are
resolved through :mod:`modules.deps` at call time rather than imported here: a default
install has the network group off and never needs either package.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import deps, log

__all__ = ["download_file"]

logger = log.get_logger("util.net")

#: Config key that enables every caller of this module. Named in dependency errors.
FEATURE = "features.network"


def download_file(url: str, filename: str | None = None, path: str | None = None) -> bool:
    """Fetch ``url`` and write the response body to ``path``/``filename``.

    Streams the body in 1 KiB chunks and renders a progress bar on the console.

    Args:
        url: Address to fetch.
        filename: Destination file name. Defaults to the last ``/``-separated segment of
            ``url``.
        path: Destination directory. Defaults to the working directory.

    Returns:
        True when the server answered 200 and the body was written; False on 404 or on
        any other status code, both of which are logged.

    Raises:
        DependencyError: ``requests`` or ``tqdm`` is not installed.
        ValueError: The destination lands outside ``path``. A file name defaulted from a
            URL is remote input, and ``os.path.join`` drops the directory entirely when
            the name it is given is absolute.
        OSError: The destination cannot be opened for writing.
    """
    requests = deps.require("requests", feature=FEATURE)
    tqdm = deps.require("tqdm", feature=FEATURE).tqdm

    if not filename:
        filename = url.split("/")[-1]
    if not path:
        path = "."
    save_path = os.path.join(path, filename)
    _check_contained(path, save_path)
    response = requests.get(url, stream=True)
    if response.status_code == requests.codes.ok:
        file_size = int(response.headers.get("Content-Length", 0))
        with open(save_path, "wb") as file:
            with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as progress:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    progress.update(len(chunk))
        logger.info("downloaded file saved at: %s", save_path)
        return True
    elif response.status_code == requests.codes.not_found:
        logger.error("file not found: %s", url)
    else:
        logger.error("failed to download %s, status code %s", url, response.status_code)
    return False


def _check_contained(directory: str, save_path: str) -> None:
    """Confirm a destination stays inside the directory it was meant for.

    Args:
        directory: The intended destination directory.
        save_path: The joined destination path.

    Raises:
        ValueError: The resolved destination is the directory itself, or lies outside it.
            Both symlinks and ``..`` segments are covered: each side is fully resolved
            before the comparison.
    """
    root = Path(directory).expanduser().resolve()
    target = Path(save_path).expanduser().resolve()
    if root not in target.parents:
        raise ValueError(f"refusing to write {target}, which is not a file inside {root}")
