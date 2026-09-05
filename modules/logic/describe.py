"""Naming the kind of value on a wire."""

from __future__ import annotations

__all__ = ["describe_value"]

#: Keys that identify a value the sockets carry as a dictionary rather than a tensor.
DICT_SHAPES = {
    "samples": "LATENT",
    "waveform": "AUDIO",
    "sample_rate": "AUDIO",
}


def _tensor_type(value) -> str:
    """The socket type a bare tensor most likely came from, read from its shape."""
    shape = tuple(getattr(value, "shape", ()) or ())
    if len(shape) == 4:
        return "LATENT" if shape[1] in (4, 16) and shape[1] != shape[-1] else "IMAGE"
    if len(shape) == 3:
        return "IMAGE" if shape[-1] in (1, 3, 4) else "MASK"
    if len(shape) == 2:
        return "MASK"
    return "TENSOR"


def describe_value(value) -> dict:
    """What a value is, for a graph that branches on the kind.

    Args:
        value: Whatever the socket carried.

    Returns:
        ``{"type_name", "python_type", "shape", "batch_size", "is_empty"}``. ``type_name``
        is the socket type in capitals, and ``TENSOR`` or ``UNKNOWN`` where it cannot be
        told apart.
    """
    empty = {
        "type_name": "NONE", "python_type": "NoneType", "shape": "",
        "batch_size": 0, "is_empty": True,
    }
    if value is None:
        return empty

    python_type = type(value).__name__
    shape_attr = getattr(value, "shape", None)

    if shape_attr is not None and hasattr(value, "dtype"):
        sizes = tuple(int(size) for size in shape_attr)
        return {
            "type_name": _tensor_type(value),
            "python_type": python_type,
            "shape": "x".join(str(size) for size in sizes),
            "batch_size": sizes[0] if sizes else 0,
            "is_empty": not sizes or any(size == 0 for size in sizes),
        }

    if isinstance(value, dict):
        for key, named in DICT_SHAPES.items():
            if key in value:
                inner = describe_value(value[key])
                return {
                    "type_name": named, "python_type": python_type,
                    "shape": inner["shape"], "batch_size": inner["batch_size"],
                    "is_empty": inner["is_empty"],
                }
        return {
            "type_name": "DICT", "python_type": python_type,
            "shape": f"{len(value)} entries", "batch_size": 0, "is_empty": not value,
        }

    if isinstance(value, bool):
        return {"type_name": "BOOLEAN", "python_type": python_type, "shape": "",
                "batch_size": 0, "is_empty": False}
    if isinstance(value, int):
        return {"type_name": "INT", "python_type": python_type, "shape": "",
                "batch_size": 0, "is_empty": False}
    if isinstance(value, float):
        return {"type_name": "FLOAT", "python_type": python_type, "shape": "",
                "batch_size": 0, "is_empty": False}
    if isinstance(value, str):
        return {"type_name": "STRING", "python_type": python_type, "shape": f"{len(value)} chars",
                "batch_size": 0, "is_empty": not value.strip()}
    if isinstance(value, (list, tuple)):
        return {"type_name": "LIST", "python_type": python_type,
                "shape": f"{len(value)} entries", "batch_size": len(value),
                "is_empty": not value}

    # A loaded model, a VAE, a ControlNet: named by its class, which is what tells them apart.
    return {"type_name": python_type.upper(), "python_type": python_type, "shape": "",
            "batch_size": 0, "is_empty": False}
