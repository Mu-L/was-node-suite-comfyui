"""Turning a keyframed camera move into one sampling grid per frame.

A grid holds one source coordinate per output pixel in normalized device coordinates,
where -1 and 1 are the picture edges on both axes, and feeds
``torch.nn.functional.grid_sample``.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Optional

import torch

__all__ = [
    "DOLLY_MODES",
    "EASINGS",
    "apply_easing",
    "build_base_grid",
    "build_camera_shake",
    "clamp_value",
    "create_frame_grid",
    "interpolate_camera_paths",
    "load_trajectory_config",
    "normalize_keyframes",
]

#: Shapes the dolly falloff can take.
DOLLY_MODES = ("radial", "aspect", "box")

#: Interpolation curves a keyframe may name. Anything else is treated as ``linear``.
EASINGS = (
    "linear",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "smoothstep",
    "smootherstep",
)

#: Scalar properties, as ``name -> value when the spec does not give one``.
SCALAR_PROPERTIES = {
    "zoom": 1.0,
    "angle": 0.0,
    "dolly_strength": 0.0,
    "dolly_feather": 0.5,
    "sphereize_strength": 0.0,
    "sphereize_feather": 0.5,
    "depth_strength": 0.0,
}

#: Paired properties, as ``name -> (default x, default y)``.
VECTOR_PROPERTIES = {
    "center": (0.5, 0.5),
    "pan": (0.0, 0.0),
    "tilt": (0.0, 0.0),
    "dolly_radius": (0.3, 0.3),
    "sphereize_radius": (0.4, 0.4),
}


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    """Hold a number inside a range."""
    return max(min_value, min(max_value, value))


def apply_easing(t: float, ease_type: str) -> float:
    """Shape a 0-to-1 progress value into an easing curve.

    Args:
        t: Progress through a segment, clamped to ``[0, 1]``.
        ease_type: One of :data:`EASINGS`. An unknown name is treated as ``linear``.

    Returns:
        The eased progress, still in ``[0, 1]``.
    """
    t = clamp_value(t, 0.0, 1.0)
    eased = (ease_type or "linear").lower()

    if eased == "ease_in":
        return t * t
    if eased == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if eased == "ease_in_out":
        return 2.0 * t * t if t < 0.5 else 1.0 - 2.0 * (1.0 - t) * (1.0 - t)
    if eased == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if eased == "smootherstep":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return t


def load_trajectory_config(text: str) -> dict[str, Any]:
    """Parse the trajectory spec.

    Args:
        text: JSON document. Empty text gives a single keyframe that holds the picture
            still, and does not raise.

    Returns:
        The parsed document.

    Raises:
        ValueError: The text is not valid JSON, with the parser's own position in the
            message.
    """
    stripped = text.strip()
    if not stripped:
        return {"loop": False, "default_ease": "linear", "keyframes": [{"frame": 0}]}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"the trajectory spec is not valid JSON: {error}\n"
            f"JSON allows no comments and no trailing commas, and every key needs double "
            f"quotes."
        ) from error


def extract_vec2(
    data: dict[str, Any],
    combined_key: str,
    x_key: str,
    y_key: str,
    default_x: float,
    default_y: float,
) -> tuple[float, float, bool]:
    """Read a paired value written either as one list or as two separate keys.

    Args:
        data: One keyframe.
        combined_key: Key holding ``[x, y]``, such as ``"center"``.
        x_key: Key holding x on its own, such as ``"center_x"``.
        y_key: Key holding y on its own.
        default_x: Value used when neither spelling is present.
        default_y: Same for y.

    Returns:
        ``(x, y, stated)``, where ``stated`` reports whether the keyframe mentioned the
        property at all, which is what decides between interpolating to it and
        extrapolating past it from a speed.
    """
    combined = data.get(combined_key)
    if isinstance(combined, (list, tuple)) and len(combined) == 2:
        return float(combined[0]), float(combined[1]), True
    x = float(data[x_key]) if x_key in data else default_x
    y = float(data[y_key]) if y_key in data else default_y
    return x, y, (x_key in data or y_key in data)


def extract_vec2_speed(
    data: dict[str, Any], combined_key: str, x_key: str, y_key: str
) -> tuple[Optional[float], Optional[float]]:
    """Read a paired rate of change, written either as one list or as two keys.

    Args:
        data: One keyframe.
        combined_key: Key holding ``[dx, dy]`` per frame, such as ``"pan_speed"``.
        x_key: Key holding the x rate on its own.
        y_key: Key holding the y rate on its own.

    Returns:
        ``(x rate, y rate)``, each ``None`` where the keyframe gives no rate.
    """
    combined = data.get(combined_key)
    if isinstance(combined, (list, tuple)) and len(combined) == 2:
        return float(combined[0]), float(combined[1])
    x = float(data[x_key]) if x_key in data else None
    y = float(data[y_key]) if y_key in data else None
    return x, y


#: Range each property is held inside, as ``name -> (low, high)``. Anything not listed is
#: taken as written.
LIMITS = {
    "center": (0.0, 1.0),
    "pan": (-1.0, 1.0),
    "tilt": (-1.0, 1.0),
    "dolly_radius": (0.0, 1.0),
    "sphereize_radius": (0.0, 1.0),
    "dolly_feather": (0.0, 1.0),
    "sphereize_feather": (0.0, 1.0),
}

#: Properties whose rate of change may be given. A later keyframe leaving one out
#: extrapolates rather than interpolates.
SPEED_PROPERTIES = ("zoom", "angle", "center", "pan", "tilt")


def _default_keyframe(default_ease: str, default_dolly_mode: str, default_depth: float) -> dict:
    """The keyframe used when the spec names none: everything at rest on frame 0."""
    keyframe: dict[str, Any] = {"frame": 0, "ease": default_ease, "dolly_mode": default_dolly_mode}
    for name, value in SCALAR_PROPERTIES.items():
        keyframe[name] = float(default_depth) if name == "depth_strength" else value
        keyframe[f"{name}_explicit"] = True
        keyframe[f"{name}_speed"] = None
    for name, (x, y) in VECTOR_PROPERTIES.items():
        keyframe[f"{name}_x"] = x
        keyframe[f"{name}_y"] = y
        keyframe[f"{name}_explicit"] = True
        keyframe[f"{name}_speed_x"] = None
        keyframe[f"{name}_speed_y"] = None
    return keyframe


def normalize_keyframes(
    config: dict[str, Any],
    num_frames: int,
    default_dolly_mode: str,
    default_depth_strength: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Expand the spec's keyframes into one fully populated record each.

    Args:
        config: The parsed spec.
        num_frames: Frames the move covers; keyframe numbers are held inside it.
        default_dolly_mode: Dolly falloff used where a keyframe names none.
        default_depth_strength: Depth parallax used where a keyframe names none.

    Returns:
        ``(keyframes, loop)`` with the keyframes sorted by frame number and never empty,
        and ``loop`` reporting whether the move wraps from the last keyframe back to the
        first.
    """
    loop = bool(config.get("loop", False))
    default_ease = str(config.get("default_ease", "linear"))
    raw = config.get("keyframes", [])

    if not isinstance(raw, list) or not raw:
        return [_default_keyframe(default_ease, default_dolly_mode, default_depth_strength)], loop

    normalized: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue

        keyframe: dict[str, Any] = {
            "frame": int(clamp_value(int(entry.get("frame", 0)), 0, num_frames - 1)),
            "ease": str(entry.get("ease", default_ease)),
        }

        for name, fallback in SCALAR_PROPERTIES.items():
            default = float(default_depth_strength) if name == "depth_strength" else fallback
            value = float(entry.get(name, default))
            if name == "zoom" and value <= 0.0:
                value = 1.0
            low_high = LIMITS.get(name)
            if low_high is not None:
                value = clamp_value(value, *low_high)
            keyframe[name] = value
            keyframe[f"{name}_explicit"] = name in entry
            speed_key = f"{name}_speed"
            keyframe[speed_key] = (
                float(entry[speed_key])
                if name in SPEED_PROPERTIES and speed_key in entry
                else None
            )

        for name, (default_x, default_y) in VECTOR_PROPERTIES.items():
            x, y, stated = extract_vec2(
                entry, name, f"{name}_x", f"{name}_y", default_x, default_y
            )
            low_high = LIMITS.get(name)
            if low_high is not None:
                x = clamp_value(x, *low_high)
                y = clamp_value(y, *low_high)
            keyframe[f"{name}_x"] = x
            keyframe[f"{name}_y"] = y
            keyframe[f"{name}_explicit"] = stated
            if name in SPEED_PROPERTIES:
                speed_x, speed_y = extract_vec2_speed(
                    entry, f"{name}_speed", f"{name}_speed_x", f"{name}_speed_y"
                )
            else:
                speed_x = speed_y = None
            keyframe[f"{name}_speed_x"] = speed_x
            keyframe[f"{name}_speed_y"] = speed_y

        mode = str(entry.get("dolly_mode", default_dolly_mode)).lower()
        keyframe["dolly_mode"] = mode if mode in DOLLY_MODES else default_dolly_mode

        normalized.append(keyframe)

    if not normalized:
        normalized.append(
            _default_keyframe(default_ease, default_dolly_mode, default_depth_strength)
        )

    normalized.sort(key=lambda entry: entry["frame"])
    return normalized, loop


def _segment_scalar(
    name: str, before: dict[str, Any], after: dict[str, Any], frames_delta: int
) -> tuple[float, float]:
    """The start and end value of one scalar property across a segment."""
    start = float(before[name])
    end = float(after[name])
    speed = before.get(f"{name}_speed")
    if not bool(after.get(f"{name}_explicit", True)) and speed is not None and frames_delta > 0:
        end = start + float(speed) * float(frames_delta)
    return start, end


def _segment_vector(
    name: str, before: dict[str, Any], after: dict[str, Any], frames_delta: int
) -> tuple[float, float, float, float]:
    """The start and end values of one paired property across a segment."""
    start_x = float(before[f"{name}_x"])
    start_y = float(before[f"{name}_y"])
    end_x = float(after[f"{name}_x"])
    end_y = float(after[f"{name}_y"])
    speed_x = before.get(f"{name}_speed_x")
    speed_y = before.get(f"{name}_speed_y")
    stated = bool(after.get(f"{name}_explicit", True))
    if not stated and frames_delta > 0 and (speed_x is not None or speed_y is not None):
        if speed_x is not None:
            end_x = start_x + float(speed_x) * float(frames_delta)
        if speed_y is not None:
            end_y = start_y + float(speed_y) * float(frames_delta)
    return start_x, start_y, end_x, end_y


def interpolate_camera_paths(
    keyframes: list[dict[str, Any]], num_frames: int, loop: bool
) -> dict[str, list]:
    """Evaluate every property at every frame.

    Args:
        keyframes: Normalized keyframes from :func:`normalize_keyframes`.
        num_frames: Frames to produce.
        loop: Whether the move wraps from the last keyframe back to the first.

    Returns:
        One list per property, each ``num_frames`` long, keyed by the property's name.
        Paired properties appear as ``"<name>_x"`` and ``"<name>_y"``, and the dolly
        falloff shape as ``"dolly_mode"``.
    """
    tracks: dict[str, list] = {name: [] for name in SCALAR_PROPERTIES}
    for name in VECTOR_PROPERTIES:
        tracks[f"{name}_x"] = []
        tracks[f"{name}_y"] = []
    tracks["dolly_mode"] = []

    first, last = keyframes[0], keyframes[-1]

    for frame_index in range(num_frames):
        before = after = None
        for keyframe in keyframes:
            if keyframe["frame"] <= frame_index:
                before = keyframe
            if keyframe["frame"] >= frame_index and after is None:
                after = keyframe

        if before is None:
            before = first
        wrapped = False
        if after is None:
            if loop and len(keyframes) > 1 and frame_index > last["frame"]:
                before, after, wrapped = last, first, True
            else:
                after = last

        if wrapped:
            before_frame = last["frame"]
            after_frame = first["frame"] + num_frames
        else:
            before_frame = before["frame"]
            after_frame = after["frame"]

        if before_frame == after_frame:
            progress = 0.0
            frames_delta = 1
        else:
            frames_delta = max(after_frame - before_frame, 1)
            progress = clamp_value((frame_index - before_frame) / float(frames_delta), 0.0, 1.0)

        eased = apply_easing(progress, before.get("ease", "linear"))

        for name in SCALAR_PROPERTIES:
            start, end = _segment_scalar(name, before, after, frames_delta)
            tracks[name].append(start + (end - start) * eased)

        for name in VECTOR_PROPERTIES:
            start_x, start_y, end_x, end_y = _segment_vector(name, before, after, frames_delta)
            tracks[f"{name}_x"].append(start_x + (end_x - start_x) * eased)
            tracks[f"{name}_y"].append(start_y + (end_y - start_y) * eased)

        tracks["dolly_mode"].append(before.get("dolly_mode", "radial"))

    return tracks


def build_camera_shake(
    num_frames: int,
    enable: bool,
    position_amplitude: float,
    rotation_amplitude: float,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Generate a handheld wobble as a bounded random walk.

    Args:
        num_frames: Frames to generate.
        enable: Whether to generate anything at all.
        position_amplitude: Largest offset, in normalized device coordinates where 2.0 is
            the full width of the picture.
        rotation_amplitude: Largest rotation, in degrees.
        seed: Seed for the walk. The same seed always gives the same wobble.

    Returns:
        ``(x offsets, y offsets, angles)``, each ``num_frames`` long and all zero when
        ``enable`` is false or both amplitudes are zero.
    """
    shake_x = [0.0] * num_frames
    shake_y = [0.0] * num_frames
    shake_angle = [0.0] * num_frames

    if not enable or (position_amplitude <= 0.0 and rotation_amplitude <= 0.0):
        return shake_x, shake_y, shake_angle

    rng = random.Random(int(seed))
    current_x = current_y = current_angle = 0.0

    for index in range(1, num_frames):
        if position_amplitude > 0.0:
            current_x = clamp_value(
                current_x + rng.uniform(-position_amplitude / 3.0, position_amplitude / 3.0),
                -position_amplitude, position_amplitude,
            )
            current_y = clamp_value(
                current_y + rng.uniform(-position_amplitude / 3.0, position_amplitude / 3.0),
                -position_amplitude, position_amplitude,
            )
        if rotation_amplitude > 0.0:
            current_angle = clamp_value(
                current_angle + rng.uniform(-rotation_amplitude / 3.0, rotation_amplitude / 3.0),
                -rotation_amplitude, rotation_amplitude,
            )
        shake_x[index] = current_x
        shake_y[index] = current_y
        shake_angle[index] = current_angle

    return shake_x, shake_y, shake_angle


def build_base_grid(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """The untransformed sampling grid, one coordinate per output pixel.

    Args:
        height: Output height in pixels.
        width: Output width in pixels.
        device: Device the grid is built on.
        dtype: Dtype the grid is built in.

    Returns:
        A ``(1, height, width, 2)`` tensor holding ``(x, y)`` in ``[-1, 1]``.
    """
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def _falloff(distance: torch.Tensor, feather: float) -> torch.Tensor:
    """Shape a normalized distance into a smooth 0 to 1 weight."""
    weight = distance * distance * (3.0 - 2.0 * distance)
    return weight ** (1.0 + 3.0 * (1.0 - clamp_value(feather, 0.0, 1.0)))


def create_frame_grid(
    base_grid: torch.Tensor,
    center_x_norm: float,
    center_y_norm: float,
    zoom: float,
    angle_deg: float,
    pan_x_norm: float,
    pan_y_norm: float,
    tilt_x: float,
    tilt_y: float,
    dolly_strength: float,
    dolly_radius_x: float,
    dolly_radius_y: float,
    dolly_feather: float,
    sphereize_strength: float,
    sphereize_radius_x: float,
    sphereize_radius_y: float,
    sphereize_feather: float,
    dolly_mode: str,
    depth_grid: Optional[torch.Tensor],
    depth_strength: float,
    shake_x_norm: float,
    shake_y_norm: float,
) -> torch.Tensor:
    """Build the sampling grid for one frame.

    Args:
        base_grid: The untransformed grid from :func:`build_base_grid`.
        center_x_norm: Point the move rotates and zooms about, in ``[0, 1]`` across the
            width.
        center_y_norm: Same, down the height.
        zoom: Magnification. Above 1.0 pushes in, below 1.0 pulls out.
        angle_deg: Rotation in degrees.
        pan_x_norm: Sideways shift, where 0.5 moves half the picture's width.
        pan_y_norm: Same, vertically.
        tilt_x: Vertical keystone, shearing the frame as though the camera looked up or
            down.
        tilt_y: Horizontal keystone.
        dolly_strength: Barrel or pincushion push inside the dolly radius. Positive pushes
            the middle outwards, negative pulls it in.
        dolly_radius_x: Half-width of the dolly region, in ``[0, 1]``.
        dolly_radius_y: Half-height of the dolly region.
        dolly_feather: How gradually the dolly falls off, in ``[0, 1]``.
        sphereize_strength: Fisheye bulge applied over the whole frame.
        sphereize_radius_x: Half-width of the bulge.
        sphereize_radius_y: Half-height of the bulge.
        sphereize_feather: How gradually the bulge falls off.
        dolly_mode: One of :data:`DOLLY_MODES`; ``box`` gives a rectangular region,
            ``aspect`` compensates for a non-square frame.
        depth_grid: Optional single-channel depth map at the output size. Where present,
            the warp is applied in full to the near parts of the picture and held back on
            the far parts, which is what makes a move read as parallax.
        depth_strength: How strongly the depth map holds the warp back, in ``[-1, 1]``.
            Negative reverses which end of the depth range moves.
        shake_x_norm: Handheld offset for this frame, added after the rotation.
        shake_y_norm: Same, vertically.

    Returns:
        A ``(1, height, width, 2)`` grid ready for ``grid_sample``.
    """
    center_x = center_x_norm * 2.0 - 1.0
    center_y = center_y_norm * 2.0 - 1.0
    pan_x = pan_x_norm * 2.0
    pan_y = pan_y_norm * 2.0

    x = base_grid[..., 0] - center_x
    y = base_grid[..., 1] - center_y

    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    scale = 1.0 / (zoom if zoom > 0.0 else 1.0)

    # Rotation and zoom come first: rotating after a warp would drag the warp around with
    # it.
    rotated_x = (cos_t * x - sin_t * y) * scale
    rotated_y = (sin_t * x + cos_t * y) * scale

    moved_x = rotated_x + center_x + pan_x + shake_x_norm
    moved_y = rotated_y + center_y + pan_y + shake_y_norm

    camera_x, camera_y = moved_x, moved_y

    tilted_x = moved_x * (1.0 + tilt_y * ((moved_y + 1.0) * 0.5)) if tilt_y != 0.0 else moved_x
    tilted_y = moved_y * (1.0 + tilt_x * ((moved_x + 1.0) * 0.5)) if tilt_x != 0.0 else moved_y

    dollied_x, dollied_y = tilted_x, tilted_y
    if dolly_strength != 0.0 and dolly_radius_x > 0.0 and dolly_radius_y > 0.0:
        radius_x = max(dolly_radius_x * 2.0, 1e-6)
        radius_y = max(dolly_radius_y * 2.0, 1e-6)
        offset_x = tilted_x - center_x
        offset_y = tilted_y - center_y

        mode = (dolly_mode or "radial").lower()
        if mode == "box":
            distance = torch.max(torch.abs(offset_x) / radius_x, torch.abs(offset_y) / radius_y)
        else:
            distance = torch.sqrt((offset_x / radius_x) ** 2 + (offset_y / radius_y) ** 2 + 1e-8)
            if mode == "aspect":
                height, width = tilted_x.shape[1], tilted_x.shape[2]
                distance = distance * (float(width) / float(height) if height > 0 else 1.0)

        inside = distance <= 1.0
        if inside.any():
            weight = _falloff(torch.clamp(distance[inside], 0.0, 1.0), dolly_feather)
            strength = float(dolly_strength)
            factor = 1.0 / (1.0 + strength * weight) if strength > 0.0 else 1.0 + strength * weight
            dollied_x = offset_x.clone()
            dollied_y = offset_y.clone()
            dollied_x[inside] = offset_x[inside] * factor
            dollied_y[inside] = offset_y[inside] * factor
            dollied_x = center_x + dollied_x
            dollied_y = center_y + dollied_y

    warped_x, warped_y = dollied_x, dollied_y
    if sphereize_strength != 0.0 and sphereize_radius_x > 0.0 and sphereize_radius_y > 0.0:
        radius_x = max(sphereize_radius_x * 2.0, 1e-6)
        radius_y = max(sphereize_radius_y * 2.0, 1e-6)
        offset_x = dollied_x - center_x
        offset_y = dollied_y - center_y
        distance = torch.sqrt((offset_x / radius_x) ** 2 + (offset_y / radius_y) ** 2 + 1e-8)
        factor = 1.0 + float(sphereize_strength) * _falloff(
            torch.clamp(distance, 0.0, 1.0), sphereize_feather
        )
        warped_x = center_x + offset_x * factor
        warped_y = center_y + offset_y * factor

    if depth_grid is not None and depth_strength != 0.0:
        depth = depth_grid
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        low = depth.amin()
        high = depth.amax()
        if float(high - low) > 1e-6:
            normalized = (depth - low) / (high - low)
        else:
            normalized = torch.zeros_like(depth)

        alpha = clamp_value(abs(depth_strength), 0.0, 1.0)
        held = normalized if depth_strength >= 0.0 else 1.0 - normalized
        keep = torch.clamp(1.0 - alpha * held, 0.0, 1.0)
        final_x = camera_x + (warped_x - camera_x) * keep
        final_y = camera_y + (warped_y - camera_y) * keep
    else:
        final_x, final_y = warped_x, warped_y

    return torch.stack((final_x, final_y), dim=-1)
