"""One intrinsic decomposition, served to every map read out of it.

:func:`answer` answers one map of a picture batch on a 0 to 255 scale. Runs are dropped
least recently used first, to :data:`RUN_LIMIT` of them and :data:`MEMORY_LIMIT` bytes.
"""

from __future__ import annotations

import hashlib

import torch

from .. import log

__all__ = ["MEMORY_LIMIT", "RUN_LIMIT", "Run", "answer"]

logger = log.get_logger("image.intrinsic_maps")

#: Decomposition runs kept before the least recently used one is dropped.
RUN_LIMIT = 4

#: Bytes of latents and decoded maps kept across every run. The newest run is always kept,
#: whatever it holds.
MEMORY_LIMIT = 512 * 1024 * 1024

_runs: dict[tuple, "Run"] = {}


class Run:
    """One sampling run and the maps decoded from it so far.

    Attributes:
        prediction: The :class:`~modules.model.marigold.Prediction` that was sampled.
        maps: Target name -> a ``(batch, 3, height, width)`` float32 CPU tensor in
            ``[0, 1]``.
    """

    def __init__(self, prediction):
        """Hold a prediction with nothing decoded from it yet.

        Args:
            prediction: What :meth:`~modules.model.marigold.Marigold.predict` answered.
        """
        self.prediction = prediction
        self.maps: dict[str, torch.Tensor] = {}

    @property
    def held(self) -> int:
        """Bytes the latents and the decoded maps take together."""
        return self.prediction.latents.nbytes + sum(
            picture.nbytes for picture in self.maps.values()
        )


def answer(loaded, image, resolution: int, steps: int, seed: int, channel: str):
    """One intrinsic map of a picture batch.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived, holding the checkpoint.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the networks read at, held to the picture's own.
        steps: Denoising steps per frame.
        seed: Chooses the noise every frame of the batch starts from.
        channel: One of the names in the model's entry of
            :data:`~modules.model.marigold.MAPS`.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.

    Raises:
        ValueError: The model does not answer ``channel``.
    """
    import comfy.utils

    from ..model import marigold

    answers = marigold.MAPS.get(loaded.name, ())
    if channel not in answers:
        raise ValueError(
            f"{loaded.name} answers {', '.join(answers)}, not {channel!r}. "
            f"Choose a model that reads it."
        )
    target = marigold.target_of(channel)
    frames = int(image.shape[0])
    edge = min(int(resolution), max(int(image.shape[1]), int(image.shape[2])))
    reading = image[..., :3]

    key = (loaded.name, edge, int(steps), int(seed), _digest(reading))
    run = _lookup(key)
    decoded = run is not None and target in run.maps
    if run is None:
        total = marigold.passes(frames, int(steps), 1)
    else:
        total = 0 if decoded else frames
    progress = comfy.utils.ProgressBar(total).update if total else None
    if total:
        loaded.backend.load()

    model = loaded.backend.model
    if run is None:
        sampled = model.predict(
            reading.permute(0, 3, 1, 2), int(steps), edge, int(seed), progress
        )
        run = _store(key, Run(sampled))
    if not decoded:
        run.maps[target] = model.picture(run.prediction, target, progress)
        _trim()
    return marigold.view(loaded.name, channel, run.maps[target]) * 255.0


def _digest(image: torch.Tensor) -> str:
    """A digest of a picture batch's contents.

    Args:
        image: ``(batch, height, width, channels)``.

    Returns:
        A 32 character hex string covering the shape and every sample.
    """
    reading = image.detach().to(device="cpu", dtype=torch.float32).contiguous()
    stamp = hashlib.blake2b(digest_size=16)
    stamp.update(str(tuple(reading.shape)).encode())
    stamp.update(memoryview(reading.numpy()).cast("B"))
    return stamp.hexdigest()


def _lookup(key: tuple):
    """The run cached under ``key``, marked as most recently used, or ``None``."""
    run = _runs.pop(key, None)
    if run is None:
        return None
    _runs[key] = run
    return run


def _store(key: tuple, run: Run) -> Run:
    """Cache ``run`` under ``key``, dropping whatever no longer fits.

    Returns:
        ``run``.
    """
    _runs[key] = run
    _trim()
    return run


def _trim() -> None:
    """Drop least recently used runs until the cache is inside both limits."""
    while len(_runs) > 1 and (len(_runs) > RUN_LIMIT or _held() > MEMORY_LIMIT):
        oldest = next(iter(_runs))
        logger.debug("dropping the intrinsic run %s, which no longer fits", oldest)
        del _runs[oldest]


def _held() -> int:
    """Bytes every cached run is holding."""
    return sum(run.held for run in _runs.values())
