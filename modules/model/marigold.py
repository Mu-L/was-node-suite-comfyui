"""Marigold intrinsic image decomposition, on the published safetensors.

:func:`load` answers a :class:`~modules.model.Backend` holding a :class:`Marigold`, whose
:meth:`Marigold.decompose` reads a picture batch and answers one map per target.
:data:`MAPS` names every map a checkpoint answers and :func:`view` shows one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from .. import log
from ..data import paths
from . import managed_module, published_files
from .autoencoder_kl import SCALING_FACTOR, AutoencoderKL
from .unet2d import UNet2DCondition

__all__ = [
    "CHANNELS",
    "DEFAULT_RESOLUTION",
    "DEFAULT_STEPS",
    "EMPTY_PROMPT",
    "FEATURE",
    "FOLDER",
    "GAMMA",
    "LATENT_SCALE",
    "MAPS",
    "MODELS",
    "PROMPT_KEY",
    "SCALING_FACTOR",
    "SCHEDULER",
    "SPACES",
    "SUB_TARGETS",
    "TARGETS",
    "UNET_FILE",
    "VAE_FILE",
    "Marigold",
    "Prediction",
    "Schedule",
    "empty_prompt",
    "load",
    "passes",
    "target_of",
    "view",
]

logger = log.get_logger("model.marigold")

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoints.
FOLDER = "intrinsics"

#: Widget option -> repository.
MODELS = {
    "Marigold IID Appearance": "prs-eth/marigold-iid-appearance-v1-1",
    "Marigold IID Lighting": "prs-eth/marigold-iid-lighting-v1-1",
}

#: Widget option -> the targets it answers, in the order its UNet stacks them.
TARGETS = {
    "Marigold IID Appearance": ("albedo", "material"),
    "Marigold IID Lighting": ("albedo", "shading", "residual"),
}

#: Target -> the reading each channel of a stacked target carries, ``None`` for a channel
#: the checkpoint declares undefined.
SUB_TARGETS = {"material": ("roughness", "metallicity", None)}

#: ``(model, map)`` -> the space the map is in and whether it carries an arbitrary scale.
#: Albedo is sRGB from the appearance checkpoint and linear from the lighting one.
SPACES = {
    ("Marigold IID Appearance", "albedo"): ("srgb", False),
    ("Marigold IID Appearance", "material"): ("stack", False),
    ("Marigold IID Appearance", "roughness"): ("linear", False),
    ("Marigold IID Appearance", "metallicity"): ("linear", False),
    ("Marigold IID Lighting", "albedo"): ("linear", False),
    ("Marigold IID Lighting", "shading"): ("linear", True),
    ("Marigold IID Lighting", "residual"): ("linear", True),
}

#: Map -> the stacked target it is read out of and the channel it occupies there.
CHANNELS = {
    sub: (target, index)
    for target, subs in SUB_TARGETS.items()
    for index, sub in enumerate(subs)
    if sub is not None
}


def _answered() -> dict[str, tuple[str, ...]]:
    """Every map each checkpoint answers, a stacked target's channels under their own names.

    Returns:
        Widget option -> map names, in the order the targets are stacked.
    """
    named = {}
    for model, targets in TARGETS.items():
        maps: list[str] = []
        for target in targets:
            maps += [sub for sub in SUB_TARGETS.get(target, ()) if sub is not None]
            maps.append(target)
        named[model] = tuple(maps)
    return named


#: Widget option -> every map its checkpoint answers, in stacking order.
MAPS = _answered()

#: Path inside a repository to the weights of each half, in the half precision variant.
UNET_FILE = "unet/diffusion_pytorch_model.fp16.safetensors"
VAE_FILE = "vae/diffusion_pytorch_model.fp16.safetensors"

#: The prompt embedding, which ships with the pack, and the tensor it holds.
EMPTY_PROMPT = "marigold_empty_prompt.safetensors"
PROMPT_KEY = "empty_prompt"

#: Denoising steps and longest processing edge the checkpoints publish as their defaults.
DEFAULT_STEPS = 4
DEFAULT_RESOLUTION = 768

#: How much smaller each latent side is than the picture it came from.
LATENT_SCALE = 8

#: Power a linear reading is raised to for showing.
GAMMA = 1.0 / 2.2

#: Smallest divisor a map carrying an arbitrary scale is held to.
SCALE_FLOOR = 1e-6

#: Values the UNet holds for every sample of latent in the batch it reads.
UNET_WORKING = 8600

#: Values the autoencoder holds for every sample of latent it carries either way.
CODEC_WORKING = 78000

#: Share of the memory free on the device that one chunk of a batch may want.
SPARE = 0.6

#: Most frames carried through one pass at once.
MOST_AT_ONCE = 8

#: The DDIM schedule both checkpoints publish.
SCHEDULER = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "prediction_type": "v_prediction",
    "rescale_betas_zero_snr": True,
    "set_alpha_to_one": False,
    "steps_offset": 1,
    "timestep_spacing": "trailing",
    "clip_sample": False,
    "thresholding": False,
}


class Schedule:
    """The DDIM schedule the intrinsics checkpoints publish.

    Attributes:
        train_timesteps: Timesteps the schedule was trained over.
        alphas_cumprod: Share of the signal left at each training timestep.
        final_alpha_cumprod: What stands in for the timestep before the first.
    """

    def __init__(self, settings: dict | None = None):
        """Build the schedule.

        Args:
            settings: Scheduler settings, or ``None`` for :data:`SCHEDULER`.
        """
        held = SCHEDULER if settings is None else settings
        self.train_timesteps = int(held["num_train_timesteps"])
        betas = (
            torch.linspace(
                held["beta_start"] ** 0.5,
                held["beta_end"] ** 0.5,
                self.train_timesteps,
                dtype=torch.float32,
            )
            ** 2
        )
        if held["rescale_betas_zero_snr"]:
            betas = _zero_terminal_snr(betas)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0)
            if held["set_alpha_to_one"]
            else self.alphas_cumprod[0].clone()
        )

    def walk(self, steps: int) -> list[tuple[int, int]]:
        """The timesteps a run visits, each paired with the one it lands on.

        Args:
            steps: Denoising steps, from 1 to :attr:`train_timesteps`.

        Returns:
            ``(timestep, previous)`` pairs, largest timestep first. A ``previous`` below
            zero stands for the timestep before the schedule begins.

        Raises:
            ValueError: ``steps`` is below one or above :attr:`train_timesteps`.
        """
        if steps < 1 or steps > self.train_timesteps:
            raise ValueError(
                f"Denoising steps must be from 1 to {self.train_timesteps}, not {steps}"
            )
        spacing = self.train_timesteps / steps
        gap = self.train_timesteps // steps
        visited = [round(self.train_timesteps - index * spacing) - 1 for index in range(steps)]
        return [(timestep, timestep - gap) for timestep in visited]

    def step(
        self,
        residual: torch.Tensor,
        sample: torch.Tensor,
        timestep: int,
        previous: int,
    ) -> torch.Tensor:
        """Answer the sample one timestep further along the schedule.

        Args:
            residual: What the network read out of ``sample``, in velocity form.
            sample: The latent at ``timestep``.
            timestep: The timestep ``sample`` stands at.
            previous: The timestep to land on, below zero for the one before the first.

        Returns:
            A tensor shaped like ``sample``.
        """
        alpha = self.alphas_cumprod[timestep]
        landing = (
            self.alphas_cumprod[previous] if previous >= 0 else self.final_alpha_cumprod
        )
        beta = 1 - alpha
        original = (alpha**0.5) * sample - (beta**0.5) * residual
        noise = (alpha**0.5) * residual + (beta**0.5) * sample
        return (landing**0.5) * original + ((1 - landing) ** 0.5) * noise


def _zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """Rescale a beta schedule so the last timestep keeps none of the signal.

    Args:
        betas: One beta per training timestep.

    Returns:
        A tensor shaped like ``betas``.
    """
    kept = torch.cumprod(1.0 - betas, dim=0).sqrt()
    first, last = kept[0].clone(), kept[-1].clone()
    kept -= last
    kept *= first / (first - last)
    bar = kept**2
    alphas = torch.cat([bar[0:1], bar[1:] / bar[:-1]])
    return 1 - alphas


@dataclass(frozen=True)
class Prediction:
    """What one sampling run answered, before any map is decoded out of it.

    Attributes:
        latents: ``(batch, targets * 4, height, width)`` prediction latents on the CPU.
        padding: The ``(down, right)`` samples the picture was padded by.
        size: ``(height, width)`` a map is answered at.
        targets: The maps the latents hold, in the order they are stacked.
    """

    latents: torch.Tensor
    padding: tuple[int, int]
    size: tuple[int, int]
    targets: tuple[str, ...]


class Marigold(nn.Module):
    """One intrinsics checkpoint: its UNet, its autoencoder and its prompt embedding.

    Attributes:
        unet: Reads a picture latent stacked with a prediction latent, answers a residual.
        vae: Carries pictures to latents and back.
        empty_prompt: The ``(1, 2, 1024)`` embedding the cross attention reads.
        schedule: The DDIM schedule the denoising walks.
        targets: The maps this checkpoint answers, in the order the UNet stacks them.
        name: The widget option this checkpoint was loaded for.
    """

    def __init__(
        self,
        unet: UNet2DCondition,
        vae: AutoencoderKL,
        prompt: torch.Tensor,
        targets: tuple[str, ...],
        name: str,
    ):
        """Hold the two networks together.

        Args:
            unet: The conditioned UNet.
            vae: The autoencoder.
            prompt: The ``(1, 2, 1024)`` prompt embedding.
            targets: The maps this checkpoint answers.
            name: The widget option this checkpoint was loaded for.
        """
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.register_buffer("empty_prompt", prompt, persistent=False)
        self.schedule = Schedule()
        self.targets = tuple(targets)
        self.name = name

    @property
    def target_count(self) -> int:
        """How many maps the UNet stacks into one prediction."""
        return self.unet.out_channels // self.vae.latent_channels

    def scale(self, latent: torch.Tensor) -> torch.Tensor:
        """Answer an encoder latent on the scale the UNet reads."""
        return latent * SCALING_FACTOR

    def unscale(self, latent: torch.Tensor) -> torch.Tensor:
        """Answer a UNet latent on the scale the decoder reads."""
        return latent / SCALING_FACTOR

    def decompose(
        self,
        images: torch.Tensor,
        steps: int = DEFAULT_STEPS,
        resolution: int = DEFAULT_RESOLUTION,
        seed: int = 0,
        targets: Sequence[str] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decompose a picture batch into the maps this checkpoint answers.

        Args:
            images: ``(batch, 3, height, width)`` in ``[0, 1]``.
            steps: Denoising steps every frame takes.
            resolution: Longest edge the networks read at, or 0 for the picture's own size.
            seed: Chooses the noise the first step starts from, up to ``2 ** 64 - 1``.
            targets: Which of :attr:`targets` to answer, or ``None`` for all of them.
            progress: Called with the number of frame passes just finished.

        Returns:
            Target name -> a ``(batch, 3, height, width)`` float32 CPU tensor in ``[0, 1]``,
            at the size ``images`` arrived at.

        Raises:
            ValueError: ``images`` is not a picture batch in ``[0, 1]``, ``resolution`` is
                negative, or ``targets`` names a map this checkpoint does not answer.
        """
        wanted = self._asked(targets)
        predicted = self.predict(images, steps, resolution, seed, progress)
        return {name: self.picture(predicted, name, progress) for name in wanted}

    def predict(
        self,
        images: torch.Tensor,
        steps: int = DEFAULT_STEPS,
        resolution: int = DEFAULT_RESOLUTION,
        seed: int = 0,
        progress: Callable[[int], None] | None = None,
    ) -> "Prediction":
        """Sample a picture batch once, answering the latents every map is decoded from.

        Args:
            images: ``(batch, 3, height, width)`` in ``[0, 1]``.
            steps: Denoising steps every frame takes.
            resolution: Longest edge the networks read at, or 0 for the picture's own size.
            seed: Chooses the noise the first step starts from, up to ``2 ** 64 - 1``.
            progress: Called with the number of frame passes just finished.

        Returns:
            A :class:`Prediction` holding every target this checkpoint answers.

        Raises:
            ValueError: ``images`` is not a picture batch in ``[0, 1]``, or ``resolution``
                is negative.
        """
        _in_range(images)
        if resolution < 0:
            raise ValueError(f"Processing resolution cannot be negative, and is {resolution}")

        device = self.unet.conv_in.weight.device
        dtype = self.unet.conv_in.weight.dtype
        size = (int(images.shape[-2]), int(images.shape[-1]))
        read = None if resolution == 0 else _reading(size, int(resolution))
        channels = self.vae.latent_channels

        with torch.no_grad():
            source, padding = _prepared(images, read, device, dtype)
            samples = (source.shape[-2] // LATENT_SCALE) * (source.shape[-1] // LATENT_SCALE)
            latent = torch.cat(
                _in_chunks(
                    source,
                    _at_once(device, dtype, samples, CODEC_WORKING),
                    lambda chunk, real: self._latent(chunk, real, progress),
                ),
                dim=0,
            )
            del source

            tile = self._noise(
                (1, self.target_count * channels, *latent.shape[-2:]),
                int(seed),
                device,
                latent.dtype,
            )
            walk = self.schedule.walk(int(steps))
            prompt = self.empty_prompt.to(device=device, dtype=dtype)
            predicted = torch.cat(
                _in_chunks(
                    latent,
                    _at_once(device, dtype, samples, UNET_WORKING),
                    lambda chunk, real: self._denoise(
                        chunk, real, tile, prompt, walk, progress
                    ),
                ),
                dim=0,
            )
            del latent, tile
        return Prediction(predicted.cpu(), padding, size, self.targets)

    def picture(
        self,
        predicted: "Prediction",
        target: str,
        progress: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        """Decode one target out of a prediction, at the size the picture arrived at.

        Args:
            predicted: What :meth:`predict` answered.
            target: One of the names in ``predicted.targets``.
            progress: Called with the number of frame passes just finished.

        Returns:
            ``(batch, 3, height, width)`` float32 on the CPU, in ``[0, 1]``.

        Raises:
            ValueError: ``target`` is not one of ``predicted.targets``.
        """
        if target not in predicted.targets:
            raise ValueError(
                f"{self.name} answers {', '.join(predicted.targets)}, not {target!r}. "
                f"Choose a model that reads it."
            )
        device = self.unet.conv_in.weight.device
        dtype = self.unet.conv_in.weight.dtype
        channels = self.vae.latent_channels
        index = predicted.targets.index(target)
        band = predicted.latents[:, index * channels : (index + 1) * channels]
        samples = int(band.shape[-2]) * int(band.shape[-1])
        with torch.no_grad():
            return torch.cat(
                _in_chunks(
                    band,
                    _at_once(device, dtype, samples, CODEC_WORKING),
                    lambda chunk, real: self._picture(
                        chunk.to(device=device, dtype=dtype),
                        real,
                        predicted.padding,
                        predicted.size,
                        progress,
                    ),
                ),
                dim=0,
            )

    def _asked(self, targets: Sequence[str] | None) -> tuple[str, ...]:
        """The targets to decode, in stacking order.

        Args:
            targets: Target names, or ``None`` for every one this checkpoint answers.

        Returns:
            The names in the order the UNet stacks them.

        Raises:
            ValueError: A name is not one this checkpoint answers.
        """
        if targets is None:
            return self.targets
        unknown = [name for name in targets if name not in self.targets]
        if unknown:
            raise ValueError(
                f"{self.name} answers {', '.join(self.targets)}, not "
                f"{', '.join(unknown)}. Choose a model that reads them."
            )
        return tuple(name for name in self.targets if name in set(targets))

    def _noise(
        self, shape: tuple[int, ...], seed: int, device, dtype: torch.dtype
    ) -> torch.Tensor:
        """The one noise tile every frame of a batch starts its first step from.

        Args:
            shape: ``(1, targets * 4, height, width)``.
            seed: Chooses the tile, from 0 to ``2 ** 64 - 1``.
            device: Where the tile is drawn.
            dtype: Precision it is drawn in.

        Returns:
            A tensor of ``shape``.
        """
        return torch.randn(
            shape,
            generator=torch.Generator(device=device).manual_seed(int(seed)),
            device=device,
            dtype=dtype,
        )

    def _latent(
        self, source: torch.Tensor, real: int, progress: Callable[[int], None] | None
    ) -> torch.Tensor:
        """Encode one chunk of pictures into the latents the UNet reads.

        Args:
            source: ``(chunk, 3, height, width)`` on a minus 1 to 1 scale.
            real: How many leading frames of ``source`` came from the batch.
            progress: Called with the number of frame passes just finished.

        Returns:
            ``(chunk, 4, height / 8, width / 8)`` on the scale the UNet reads.
        """
        latent = self.scale(self.vae.encode(source))
        if progress is not None:
            progress(real)
        return latent

    def _denoise(
        self,
        latent: torch.Tensor,
        real: int,
        tile: torch.Tensor,
        prompt: torch.Tensor,
        walk: list[tuple[int, int]],
        progress: Callable[[int], None] | None,
    ) -> torch.Tensor:
        """Walk one chunk of picture latents down the schedule.

        Args:
            latent: ``(chunk, 4, height, width)`` scaled picture latents.
            real: How many leading frames of ``latent`` came from the batch.
            tile: ``(1, targets * 4, height, width)`` noise every frame starts from.
            prompt: The prompt embedding, on the device and in the precision of the run.
            walk: The ``(timestep, previous)`` pairs to visit.
            progress: Called with the number of frame passes just finished.

        Returns:
            ``(chunk, targets * 4, height, width)`` prediction latents.
        """
        count = int(latent.shape[0])
        prediction = tile.expand(count, -1, -1, -1)
        text = prompt.repeat(count, 1, 1)
        for timestep, previous in walk:
            residual = self.unet(torch.cat([latent, prediction], dim=1), timestep, text)
            prediction = self.schedule.step(residual, prediction, timestep, previous)
            if progress is not None:
                progress(real)
        return prediction

    def _picture(
        self,
        latent: torch.Tensor,
        real: int,
        padding: tuple[int, int],
        size: tuple[int, int],
        progress: Callable[[int], None] | None,
    ) -> torch.Tensor:
        """Decode one chunk of prediction latents into a map at the picture's own size.

        Args:
            latent: ``(chunk, 4, height, width)`` prediction latents.
            real: How many leading frames of ``latent`` came from the batch.
            padding: The ``(down, right)`` samples the picture was padded by.
            size: ``(height, width)`` the map is answered at.
            progress: Called with the number of frame passes just finished.

        Returns:
            ``(chunk, 3, height, width)`` float32 on the CPU, in ``[0, 1]``.
        """
        picture = torch.clip(self.vae.decode(self.unscale(latent)), -1.0, 1.0)
        picture = (picture + 1.0) / 2.0
        down, right = padding
        picture = picture[:, :, : -down or None, : -right or None]
        picture = functional.interpolate(picture, size, mode="bilinear", antialias=False)
        if progress is not None:
            progress(real)
        return picture.float().cpu()


def view(name: str, channel: str, prediction: torch.Tensor) -> torch.Tensor:
    """One raw prediction as the picture its map stands for.

    Args:
        name: A key of :data:`MODELS`.
        channel: A map name from that model's entry of :data:`MAPS`.
        prediction: ``(batch, 3, height, width)`` raw prediction of the target the map is
            read from, in ``[0, 1]``.

    Returns:
        ``(batch, 3, height, width)`` in ``[0, 1]``. A map holding one reading carries it
        in all three channels; a stacked map carries each reading in its own channel and
        zero where the checkpoint declares none.

    Raises:
        ValueError: The checkpoint answers no map of that name.
    """
    if channel not in MAPS.get(name, ()):
        raise ValueError(
            f"{name} answers {', '.join(MAPS.get(name, ()))}, not {channel!r}. "
            f"Choose a model that reads it."
        )
    plane = prediction.float()
    if channel in SUB_TARGETS:
        packed = torch.zeros_like(plane)
        for index, sub in enumerate(SUB_TARGETS[channel]):
            if sub is not None:
                packed[:, index : index + 1] = _shown(
                    name, sub, plane[:, index : index + 1]
                )
        return packed
    if channel in CHANNELS:
        index = CHANNELS[channel][1]
        return _shown(name, channel, plane[:, index : index + 1]).repeat(1, 3, 1, 1)
    return _shown(name, channel, plane)


def passes(frames: int, steps: int, maps: int = 1) -> int:
    """How many frame passes one decomposition takes.

    Args:
        frames: Frames in the batch.
        steps: Denoising steps each frame takes.
        maps: Maps decoded out of each frame.

    Returns:
        One encode pass, ``steps`` denoising passes and one decode pass per map, for every
        frame.
    """
    return int(frames) * (1 + int(steps) + int(maps))


def target_of(channel: str) -> str:
    """The target a map is decoded from.

    Args:
        channel: A map name.

    Returns:
        ``channel`` itself for a target, or the stacked target it is one channel of.
    """
    return CHANNELS[channel][0] if channel in CHANNELS else channel


def _shown(name: str, channel: str, plane: torch.Tensor) -> torch.Tensor:
    """One raw plane on the scale a picture is shown at.

    Args:
        name: A key of :data:`MODELS`.
        channel: A map name.
        plane: ``(batch, channels, height, width)`` in ``[0, 1]``.

    Returns:
        A tensor shaped like ``plane``.
    """
    space, arbitrary = SPACES[(name, channel)]
    if space != "linear":
        return plane
    if arbitrary:
        plane = plane / plane.amax(dim=(1, 2, 3), keepdim=True).clamp(min=SCALE_FLOOR)
    return plane**GAMMA


def _reading(size: tuple[int, int], resolution: int) -> tuple[int, int]:
    """The size a picture is read at.

    Args:
        size: ``(height, width)`` of the picture.
        resolution: Longest edge to read at.

    Returns:
        ``(height, width)``, neither side under :data:`LATENT_SCALE`.
    """
    longest = max(size)
    return (
        max(LATENT_SCALE, size[0] * resolution // longest),
        max(LATENT_SCALE, size[1] * resolution // longest),
    )


def _in_range(images: torch.Tensor) -> None:
    """Refuse a batch the networks cannot read.

    Args:
        images: The batch that arrived.

    Raises:
        ValueError: It is not a four dimensional float batch of at least one frame and
            three channels holding values from 0 to 1.
    """
    if images.ndim != 4 or images.shape[1] < 3:
        raise ValueError(
            f"The intrinsic maps read a batch shaped (frames, 3, height, width), and this "
            f"one is {tuple(images.shape)}."
        )
    if images.shape[0] < 1 or images.shape[-2] < 1 or images.shape[-1] < 1:
        raise ValueError(
            f"The intrinsic maps read at least one frame with a side of at least one "
            f"sample, and this batch is {tuple(images.shape)}. Feed a picture in."
        )
    if not torch.is_floating_point(images):
        raise ValueError(
            f"The intrinsic maps read a float batch, and this one is {images.dtype}."
        )
    low, high = float(images.min()), float(images.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(
            f"The intrinsic maps read a picture from 0 to 1, and this batch runs from "
            f"{low:.4f} to {high:.4f}. Bring it into range first with Image Tone Map, "
            f"then feed it in."
        )


def _prepared(
    images: torch.Tensor,
    read: tuple[int, int] | None,
    device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """A picture batch as the networks read it.

    Args:
        images: ``(batch, 3, height, width)`` in ``[0, 1]``.
        read: ``(height, width)`` to read at, or ``None`` for the size it arrived at.
        device: Where inference runs.
        dtype: Precision inference runs in.

    Returns:
        The batch on a minus 1 to 1 scale with both sides taken up to a multiple of
        :data:`LATENT_SCALE`, and the ``(down, right)`` samples added to reach it.
    """
    planes = images[:, :3].to(device=device, dtype=dtype) * 2.0 - 1.0
    if read is not None:
        planes = functional.interpolate(planes, read, mode="bilinear", antialias=True)
    down = -planes.shape[-2] % LATENT_SCALE
    right = -planes.shape[-1] % LATENT_SCALE
    if down or right:
        planes = functional.pad(planes, (0, right, 0, down), mode="replicate")
    return planes, (down, right)


def _at_once(device, dtype: torch.dtype, samples: int, working: int) -> int:
    """How many frames one pass carries at once.

    Args:
        device: Where inference runs.
        dtype: Precision inference runs in.
        samples: Samples in one frame's latent.
        working: Values the pass holds per sample of latent.

    Returns:
        From one to :data:`MOST_AT_ONCE`.
    """
    free = _free(device)
    if free is None:
        return MOST_AT_ONCE
    wanted = samples * working * dtype.itemsize
    return max(1, min(MOST_AT_ONCE, int(free * SPARE / max(wanted, 1))))


def _free(device) -> int | None:
    """Bytes free on the device, or ``None`` where nothing can say.

    Args:
        device: Where inference runs.

    Returns:
        The free byte count, or ``None``.
    """
    try:
        import comfy.model_management as model_management
    except ImportError:
        if device.type == "cuda":
            return int(torch.cuda.mem_get_info(device)[0])
        return None
    return int(model_management.get_free_memory(device))


def _chunk(count: int, at_once: int) -> int:
    """The frames every chunk of a batch carries.

    Args:
        count: Frames in the batch.
        at_once: Most frames one chunk may carry.

    Returns:
        One length for every chunk, from 1 to ``at_once``.
    """
    at_once = max(1, int(at_once))
    if count <= at_once:
        return max(1, count)
    chunks = -(-count // at_once)
    return -(-count // chunks)


def _in_chunks(source: torch.Tensor, at_once: int, run) -> list[torch.Tensor]:
    """Run over a batch a chunk at a time, every chunk the same length.

    Args:
        source: The batch, one frame per leading index.
        at_once: Most frames one chunk may carry.
        run: Callable taking a chunk and how many of its leading frames came from
            ``source``, answering one result per frame of the chunk.

    Returns:
        The chunk results, each trimmed to the frames it came from, in order.

    Raises:
        OutOfMemoryError: One frame on its own still did not fit.
    """
    count = int(source.shape[0])
    while True:
        size = _chunk(count, at_once)
        done = []
        try:
            for start in range(0, count, size):
                piece = source[start : start + size]
                real = int(piece.shape[0])
                if real < size:
                    # The last chunk is filled out with copies of its own last frame.
                    filler = piece[-1:].expand(size - real, *piece.shape[1:])
                    piece = torch.cat([piece, filler], dim=0)
                done.append(run(piece, real)[:real])
            return done
        except torch.cuda.OutOfMemoryError:
            if at_once <= 1:
                raise
            at_once = max(1, at_once // 2)
            logger.debug("a chunk of %d frames did not fit, halving", size)
            del done
            _release()


def _release() -> None:
    """Hand back whatever the caching allocator is holding but not using."""
    try:
        import comfy.model_management as model_management
    except ImportError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
    model_management.soft_empty_cache(force=True)


def empty_prompt() -> torch.Tensor:
    """The prompt embedding the checkpoints condition on, as it ships with the pack.

    Returns:
        A ``(1, 2, 1024)`` float32 tensor.

    Raises:
        FileNotFoundError: The file is not beside the pack's other data.
        KeyError: The file holds no embedding.
    """
    from safetensors.torch import load_file

    path = paths.data_directory() / "models" / EMPTY_PROMPT
    if not path.is_file():
        raise FileNotFoundError(
            f"The Marigold prompt embedding is missing from the pack at {path}. Reinstall "
            f"the pack: the file ships with it and is not downloaded."
        )
    held = load_file(str(path))
    if PROMPT_KEY not in held:
        raise KeyError(
            f"{path} holds {', '.join(sorted(held))}, not {PROMPT_KEY}. Reinstall the pack "
            f"to replace the file."
        )
    return held[PROMPT_KEY]


def load(name: str = "Marigold IID Appearance", device: str | None = None):
    """Load an intrinsic decomposition checkpoint.

    Args:
        name: One of the keys of :data:`MODELS`.
        device: Device name for inference, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``model`` is a :class:`Marigold` resting on
        the offload device.

    Raises:
        ValueError: ``name`` is not a key of :data:`MODELS`.
        ModelUnavailable: No local weights, and ``features.network`` is off.
    """
    if name not in MODELS:
        raise ValueError(
            f"Intrinsics model must be one of {', '.join(MODELS)}, not {name!r}"
        )
    repo_id = MODELS[name]
    return managed_module(
        ("marigold", repo_id), lambda: _build(name, repo_id, device), device=device
    )


def _build(name: str, repo_id: str, device: str | None) -> Marigold:
    """Read one checkpoint off disk into a :class:`Marigold`.

    Args:
        name: The widget option being loaded.
        repo_id: The repository publishing it.
        device: Device name inference will run on, or ``None`` for ComfyUI's.

    Returns:
        The assembled model, in the precision that device runs in.
    """
    from safetensors.torch import load_file

    from . import compute_device

    # The autoencoder is looked for under every intrinsics repository, not only this one.
    files = published_files(
        FOLDER,
        repo_id,
        (UNET_FILE, VAE_FILE),
        also_search=[other for other in MODELS.values() if other != repo_id],
        feature=FEATURE,
        what="The intrinsic decomposition network",
    )
    dtype = _dtype(compute_device(device))

    weights = load_file(str(files[UNET_FILE]))
    # Built without storage, then handed the mapped tensors themselves.
    with torch.device("meta"):
        unet = UNet2DCondition(
            weights["conv_in.weight"].shape[1], weights["conv_out.weight"].shape[0]
        )
    unet.load_state_dict(weights, assign=True)
    del weights

    with torch.device("meta"):
        vae = AutoencoderKL()
    vae.load_state_dict(load_file(str(files[VAE_FILE])), assign=True)

    model = Marigold(unet, vae, empty_prompt(), TARGETS[name], name)
    logger.debug(
        "built %s from %s and %s in %s", name, files[UNET_FILE], files[VAE_FILE], dtype
    )
    return model.to(dtype).eval()


def _dtype(device) -> torch.dtype:
    """The precision a Marigold checkpoint runs in on ``device``."""
    try:
        import comfy.model_management as model_management
    except ImportError:
        logger.debug("comfy.model_management is unavailable, so this model runs in float32")
        return torch.float32
    return torch.float16 if model_management.should_use_fp16(device) else torch.float32
