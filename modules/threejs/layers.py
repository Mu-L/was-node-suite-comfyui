"""Mixing two Three.js materials through a mask, channel by channel.

A channel both materials agree on is left alone; any other becomes a new texture.
"""

from __future__ import annotations

__all__ = ["CHANNELS", "MAX_EDGE", "mixed"]

from ..log import get_logger
from ..interface import three_asset
from .spec import create_spec
from .textures import as_png

logger = get_logger("threejs.layers")

#: Largest side, in pixels, a mixed channel is written at.
MAX_EDGE = 4096

#: Smallest side, so a mask alone still gives something usable.
MIN_EDGE = 8

#: Each channel a mix covers, as the descriptor's dependency name, the material setting that
#: stands in where no texture is wired, and how many channels the picture carries.
CHANNELS = (
    ("map", "color", 3),
    ("normalMap", None, 3),
    ("roughnessMap", "roughness", 1),
    ("metalnessMap", "metalness", 1),
    ("emissiveMap", "emissive", 3),
    ("aoMap", None, 1),
    ("bumpMap", None, 1),
    ("displacementMap", None, 1),
    ("alphaMap", None, 1),
)

#: The channel names a mix owns, so a dependency outside them is carried across untouched.
OWNED = frozenset(name for name, _, _ in CHANNELS)

#: Material settings mixed as plain numbers when neither side carries a texture for them.
SCALARS = (
    "roughness",
    "metalness",
    "opacity",
    "emissiveIntensity",
    "bumpScale",
    "displacementScale",
    "aoMapIntensity",
    "normalScale",
)


def _hex_to_rgb(value, fallback=(1.0, 1.0, 1.0)):
    """One ``#rrggbb`` string as three floats in ``[0, 1]``.

    Args:
        value: The colour as written on the material.
        fallback: What to answer where the string is unreadable.

    Returns:
        ``(red, green, blue)``.
    """
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(part * 2 for part in text)
    if len(text) != 6:
        return fallback
    try:
        return tuple(int(text[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    except ValueError:
        return fallback


def _texture_image(spec):
    """One texture descriptor's picture, where its bytes are still held.

    Args:
        spec: A texture descriptor, or None.

    Returns:
        A PIL image, or None where nothing is wired or the bytes have been dropped.
    """
    if not isinstance(spec, dict):
        return None
    address = str((spec.get("params") or {}).get("url", ""))
    if not address.startswith(three_asset.ROUTE):
        logger.warning(
            "a texture at %s is not held by this server, so the channel it fills is mixed "
            "from the material's plain setting instead",
            address[:60] or "an unknown address",
        )
        return None
    body = three_asset.read(address.partition("key=")[2])
    if body is None:
        return None

    import io as _io

    from PIL import Image

    return Image.open(_io.BytesIO(body))


def _size(pictures, mask_size):
    """The size a mixed channel is written at.

    Args:
        pictures: The pictures taking part, any of which may be None.
        mask_size: The mask's own size, used where no picture is wired.

    Returns:
        ``(width, height)``, bounded by :data:`MIN_EDGE` and :data:`MAX_EDGE`.
    """
    widths = [p.width for p in pictures if p is not None] + [mask_size[0]]
    heights = [p.height for p in pictures if p is not None] + [mask_size[1]]
    wide = min(MAX_EDGE, max(MIN_EDGE, max(widths)))
    tall = min(MAX_EDGE, max(MIN_EDGE, max(heights)))
    return wide, tall


def _as_array(picture, standin, channels, size):
    """One side of a channel as an array at the mixing size.

    Args:
        picture: The wired texture's picture, or None.
        standin: The material setting to fall back on, already in ``[0, 1]``.
        channels: How many channels the array carries.
        size: ``(width, height)`` to work at.

    Returns:
        A float array shaped ``(height, width, channels)``.
    """
    import numpy as np
    from PIL import Image

    mode = "RGB" if channels == 3 else "L"
    if picture is not None:
        resized = picture.convert(mode).resize(size, Image.BILINEAR)
        return np.asarray(resized).astype(np.float32) / 255.0

    flat = np.zeros((size[1], size[0], channels), dtype=np.float32)
    if channels == 3:
        flat[:, :] = np.asarray(standin, dtype=np.float32)
    else:
        flat[:, :, 0] = float(standin)
    return flat


def mixed(base: dict, top: dict, mask, mask_size) -> dict:
    """One material blended into another through a mask.

    Args:
        base: The material descriptor showing where the mask is black.
        top: The material descriptor showing where the mask is white.
        mask: The mask as a float array shaped ``(height, width)`` in ``[0, 1]``.
        mask_size: The mask's ``(width, height)``.

    Returns:
        A material descriptor carrying the mixed channels and settings.
    """
    import numpy as np
    from PIL import Image

    base_params = dict(base.get("params") or {})
    top_params = dict(top.get("params") or {})
    base_deps = dict(base.get("deps") or {})
    top_deps = dict(top.get("deps") or {})

    params = dict(base_params)
    deps = {}
    mixed_names = []

    for name, standin, channels in CHANNELS:
        under = _texture_image(base_deps.get(name))
        over = _texture_image(top_deps.get(name))
        setting = standin is not None and base_params.get(standin) != top_params.get(standin)
        if under is None and over is None and not setting:
            # Neither side textures this channel and both agree on the setting behind it.
            if name in base_deps:
                deps[name] = base_deps[name]
            continue

        size = _size((under, over), mask_size)
        first = _as_array(
            under,
            _hex_to_rgb(base_params.get(standin)) if channels == 3 else base_params.get(standin, 0.0),
            channels,
            size,
        )
        second = _as_array(
            over,
            _hex_to_rgb(top_params.get(standin)) if channels == 3 else top_params.get(standin, 0.0),
            channels,
            size,
        )

        weight = np.asarray(
            Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
            .resize(size, Image.BILINEAR),
            dtype=np.float32,
        )[..., None] / 255.0

        blended = first * (1.0 - weight) + second * weight
        picture = np.clip(blended, 0.0, 1.0)
        if channels == 1:
            picture = np.repeat(picture, 3, axis=-1)

        body = as_png((picture * 255.0).astype(np.uint8))
        deps[name] = create_spec(
            "texture",
            "TextureURL",
            params={
                "url": "%s?key=%s" % (three_asset.ROUTE, three_asset.keep(body)),
                "colorSpace": "srgb" if name in ("map", "emissiveMap") else "linear-srgb",
                "wrapS": "repeat",
                "wrapT": "repeat",
                "repeat": [1.0, 1.0],
                "offset": [0.0, 0.0],
                "rotation": 0.0,
                "flipY": True,
                "anisotropy": 1,
            },
            meta={"source": "mix"},
        )
        mixed_names.append(name)
        # The picture now carries the whole channel, so the plain setting must not tint it.
        if standin is not None:
            params[standin] = "#ffffff" if channels == 3 else 1.0

    # A setting no channel picture covers is a single number for the whole surface, so it is
    # taken at the mask's average: an all-white mask lands exactly on the top material.
    strength = float(np.clip(mask, 0.0, 1.0).mean())
    for name in SCALARS:
        first = base_params.get(name)
        second = top_params.get(name)
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            params[name] = first * (1.0 - strength) + second * strength

    # A dependency outside the mixed channels, such as a shader texture, is carried across.
    for name, value in top_deps.items():
        if name not in deps and name not in OWNED:
            deps[name] = value

    logger.info(
        "Three Material Mix blended %d channel(s): %s",
        len(mixed_names), ", ".join(mixed_names) or "none",
    )
    return create_spec(
        "material",
        base.get("type") or "MeshStandardMaterial",
        params=params,
        deps=deps,
    )
