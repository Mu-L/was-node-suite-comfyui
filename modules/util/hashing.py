"""Deterministic seeds and file digests.

Every function here is pure: no configuration, no paths of its own, and the only I/O is
:func:`get_sha256` reading the file it was handed.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["get_sha256", "image2seed", "seed_batch"]


def image2seed(image) -> int:
    """Derive a seed from an image's pixel data.

    Args:
        image: A PIL image.

    Returns:
        The first four bytes of the SHA-256 digest of the image buffer, read big-endian,
        in the range 0 to 2**32 - 1.
    """
    image_data = image.tobytes()
    hash_object = hashlib.sha256(image_data)
    hash_digest = hash_object.digest()
    seed = int.from_bytes(hash_digest[:4], byteorder="big")
    return seed


def get_sha256(file_path) -> str:
    """Hash a file's contents.

    Read in 4 KiB chunks, so the file size is bounded by the disk rather than by memory.

    Args:
        file_path: Path to an existing readable file.

    Returns:
        The SHA-256 digest as a lowercase hex string.

    Raises:
        OSError: The file cannot be opened or read.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def seed_batch(seed, batches, seeds) -> list[list[int]]:
    """Draw distinct seeds, one row per batch.

    Args:
        seed: Seed for the generator that produces the rows.
        batches: Number of rows to draw.
        seeds: Number of seeds per row. Drawn without replacement, so a row never holds
            the same seed twice; separate rows may still overlap.

    Returns:
        ``batches`` lists of ``seeds`` integers, each below 2**32 - 1.
    """
    rng = np.random.default_rng(seed)
    btch = [rng.choice(2**32 - 1, seeds, replace=False).tolist() for _ in range(batches)]
    return btch
