"""Latent interpolation and reverse sampling shared by the sequence samplers.

The previous loop's latent is mixed into the new one by one of three interpolation modes,
and :func:`unsample` drives a latent back up the noise schedule.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

#: The ways one loop's latent is mixed into the next, in the order the widget offers them.
LATENT_INTERPOLATION_MODES = ["Blend", "Slerp", "Cosine Interp"]

#: How the seed moves from one loop to the next, in the order the widget offers them.
SEED_MODES = ["increment", "decrement", "random", "fixed"]

#: Largest value a ComfyUI seed widget accepts.
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def alternate_seed(seed: int, previous: int, current: int) -> tuple[int, int]:
    """Swap between two seeds that drift apart, for the alternating loops.

    Args:
        seed: The base seed the run was started with.
        previous: The seed held back from the last swap.
        current: The seed the last alternating loop ran on.

    Returns:
        The ``(previous, current)`` pair after this loop.
    """
    if seed % 3 == 0:
        previous, current = current, (previous + 1 if seed // 2 % 2 == 0 else previous - 1)
    return previous, current


def advance_seed(seed, mode: str):
    """The seed the next loop runs on.

    Args:
        seed: The seed the current loop ran on.
        mode: One of :data:`SEED_MODES`.

    Returns:
        The next seed, or ``None`` for a mode name that is not one of the four.
    """
    if mode == "increment":
        return seed + 1
    if mode == "decrement":
        return seed - 1
    if mode == "random":
        return random.randint(0, MAX_SEED)
    if mode == "fixed":
        return seed
    return None


def slerp(strength, tensor_from, tensor_to, epsilon: float = 1e-6):
    """Spherical linear interpolation between two tensors, along their last axis.

    Args:
        strength: Interpolation factor. 0.0 is ``tensor_from``, 1.0 is ``tensor_to``.
        tensor_from: Tensor interpolated away from.
        tensor_to: Tensor interpolated towards.
        epsilon: Smallest sine treated as non-zero. Where two vectors are parallel the
            arc has no direction, and those positions fall back to whichever end
            ``strength`` is nearer.

    Returns:
        A tensor shaped like the inputs.
    """
    low_norm = F.normalize(tensor_from, p=2, dim=-1, eps=epsilon)
    high_norm = F.normalize(tensor_to, p=2, dim=-1, eps=epsilon)

    dot_product = torch.clamp((low_norm * high_norm).sum(dim=-1), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    zero_mask = torch.isclose(
        sin_omega, torch.tensor([0.0], device=sin_omega.device), atol=epsilon
    )
    sin_omega = torch.where(zero_mask, torch.tensor([1.0], device=sin_omega.device), sin_omega)
    from_scale = torch.sin((1.0 - strength) * omega) / sin_omega
    to_scale = torch.sin(strength * omega) / sin_omega

    blended = from_scale.unsqueeze(-1) * tensor_from + to_scale.unsqueeze(-1) * tensor_to
    return torch.where(
        zero_mask.unsqueeze(-1),
        tensor_from if strength < 0.5 else tensor_to,
        blended,
    )


def blend_conditioning(strength, last_pair, next_pair):
    """Interpolate between two conditioning pairs along the arc between them.

    Args:
        strength: Interpolation factor. 0.0 is ``last_pair``, 1.0 is ``next_pair``.
        last_pair: The ``[tensor, dict]`` pair the previous loop used.
        next_pair: The ``[tensor, dict]`` pair this loop would otherwise use.

    Returns:
        A new ``[tensor, dict]`` pair.
    """
    blended = slerp(strength, last_pair[0].clone(), next_pair[0].clone())
    pooled = slerp(
        strength,
        last_pair[1]["pooled_output"].clone(),
        next_pair[1]["pooled_output"].clone(),
    )
    return [blended, {"pooled_output": pooled}]


def slerp_latents(val, low, high):
    """Spherical linear interpolation between two latents, one batch item at a time.

    Args:
        val: Interpolation factor. 0.0 is ``low``, 1.0 is ``high``.
        low: Latent samples interpolated away from.
        high: Latent samples interpolated towards.

    Returns:
        Latent samples shaped like the inputs.
    """
    dims = low.shape

    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)

    # A zero-length row normalises to NaN, which would poison the whole batch item.
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm * high_norm).sum(1))
    sin_omega = torch.sin(omega)
    blended = (torch.sin((1.0 - val) * omega) / sin_omega).unsqueeze(1) * low + (
        torch.sin(val * omega) / sin_omega
    ).unsqueeze(1) * high
    return blended.reshape(dims)


def blend_latents(alpha, latent_1, latent_2):
    """Straight linear blend of two latents.

    Args:
        alpha: Blend factor. 0.0 is ``latent_1``, 1.0 is ``latent_2``.
        latent_1: Latent samples blended away from.
        latent_2: Latent samples blended towards.

    Returns:
        Latent samples shaped like the inputs.
    """
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor([alpha], dtype=latent_1.dtype, device=latent_1.device)
    return (1 - alpha) * latent_1 + alpha * latent_2


def cosine_interp_latents(val, low, high):
    """Linear blend of two latents with the factor eased by a cosine.

    Args:
        val: Interpolation factor. 0.0 is ``low``, 1.0 is ``high``.
        low: Latent samples interpolated away from.
        high: Latent samples interpolated towards.

    Returns:
        Latent samples shaped like the inputs.
    """
    if not isinstance(val, torch.Tensor):
        val = torch.tensor([val], dtype=low.dtype, device=low.device)
    eased = (1 - torch.cos(val * math.pi)) / 2
    return (1 - eased) * low + eased * high


def interpolate_latents(mode: str, strength, previous, current):
    """Mix the previous loop's latent into this loop's by one of the named modes.

    Args:
        mode: One of :data:`LATENT_INTERPOLATION_MODES`. Any other name returns
            ``current`` untouched.
        strength: Interpolation factor. 0.0 keeps ``previous``, 1.0 keeps ``current``.
        previous: Latent samples from the loop before.
        current: Latent samples the sampler just produced.

    Returns:
        The mixed latent samples.
    """
    if mode == "Blend":
        return blend_latents(strength, previous, current)
    if mode == "Slerp":
        return slerp_latents(strength, previous, current)
    if mode == "Cosine Interp":
        return cosine_interp_latents(strength, previous, current)
    return current


def unsample(
    model,
    seed,
    cfg,
    sampler_name,
    steps,
    end_at_step,
    scheduler,
    normalize,
    positive,
    negative,
    latent_image,
):
    """Run the sampler with its noise schedule reversed, adding noise back to a latent.

    Args:
        model: The diffusion model to run.
        seed: Noise seed handed to the sampler.
        cfg: Classifier-free guidance scale.
        sampler_name: A ``comfy.samplers.KSampler.SAMPLERS`` entry.
        steps: Total steps in the schedule.
        end_at_step: How far back up the schedule to walk, counted from the end.
        scheduler: A ``comfy.samplers.KSampler.SCHEDULERS`` entry.
        normalize: ``"enable"`` to rescale the result to zero mean and unit variance.
            Any other value leaves it as the sampler produced it.
        positive: Positive CONDITIONING.
        negative: Negative CONDITIONING.
        latent_image: LATENT dict to walk back.

    Returns:
        A LATENT dict holding the re-noised samples on the CPU.
    """
    import comfy.model_management
    import comfy.sampler_helpers
    import comfy.samplers
    import comfy.utils

    device = comfy.model_management.get_torch_device()
    end_at_step = steps - min(end_at_step, steps - 1)

    latent = latent_image
    samples_in = latent["samples"].to(device)

    noise = torch.zeros(
        samples_in.size(), dtype=samples_in.dtype, layout=samples_in.layout, device=device
    )
    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = comfy.sampler_helpers.prepare_mask(latent["noise_mask"], noise.shape, device)

    positive_copy = comfy.sampler_helpers.convert_cond(positive)
    negative_copy = comfy.sampler_helpers.convert_cond(negative)

    models, inference_memory = comfy.sampler_helpers.get_additional_models(
        {"positive": positive, "negative": negative}, model.model_dtype()
    )
    comfy.model_management.load_models_gpu(
        [model] + models, model.memory_required(noise.shape) + inference_memory
    )

    sampler = comfy.samplers.KSampler(
        model.model,
        steps=steps,
        device=device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=1.0,
        model_options=model.model_options,
    )
    # Flipping the sigmas walks the schedule the other way; the offset keeps the last
    # sigma off zero, which the samplers divide by.
    sigmas = sampler.sigmas.flip(0) + 0.0001

    progress = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        progress.update_absolute(step + 1, total_steps)

    samples = sampler.sample(
        noise,
        positive_copy,
        negative_copy,
        cfg=cfg,
        latent_image=samples_in,
        force_full_denoise=False,
        denoise_mask=noise_mask,
        sigmas=sigmas,
        start_step=0,
        last_step=end_at_step,
        callback=callback,
        seed=seed,
    )

    if normalize == "enable":
        samples = (samples - samples.mean()) / samples.std()

    comfy.sampler_helpers.cleanup_additional_models(models)

    out = latent.copy()
    out["samples"] = samples.cpu()
    return out
