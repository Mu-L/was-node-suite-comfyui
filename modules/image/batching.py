"""Joining image batches end to end: the slot series, the arity and the size agreement.

Images are float tensors shaped ``(batch, height, width, channels)``. The batch axis is
ignored when sizes are compared.
"""

from __future__ import annotations

from comfy_api.latest import io

__all__ = [
    "IMAGE_SLOTS",
    "MAX_SLOTS",
    "RESIZE_ADVICE",
    "as_batch",
    "check_image_dimensions",
    "frames",
    "image_slot_template",
]

#: Slot ids of a lettered image series, in the order the slots are drawn and joined.
IMAGE_SLOTS = (
    "images_a", "images_b", "images_c", "images_d", "images_e", "images_f", "images_g",
    "images_h", "images_i", "images_j", "images_k", "images_l", "images_m", "images_n",
    "images_o", "images_p", "images_q", "images_r", "images_s", "images_t", "images_u",
    "images_v", "images_w", "images_x", "images_y", "images_z",
)

#: How many slots :data:`IMAGE_SLOTS` names.
MAX_SLOTS = 26

#: Closing sentence of the refusal a size mismatch raises.
RESIZE_ADVICE = (
    "Resize the odd one, or use Image Batch Advanced, which brings every slot to the size of "
    "the first."
)


def _described(shape) -> str:
    """One frame shape, as a size and a channel count.

    Args:
        shape: The axes after the batch axis, ``(height, width, channels)``.

    Returns:
        A phrase such as ``1856x2254 with 3 channel(s)``.
    """
    if len(shape) == 3:
        return f"{shape[1]}x{shape[0]} with {shape[2]} channel(s)"
    return "x".join(str(axis) for axis in shape)


def check_image_dimensions(tensors, names, node="Image Batch", advice=RESIZE_ADVICE) -> None:
    """Reject image batches whose frames do not share a size or a channel count.

    Args:
        tensors: Image tensors, in slot order.
        names: Socket id of each entry, used to name the offenders.
        node: Display name of the calling node, for the refusal.
        advice: Closing sentence of the refusal, naming what to do about it.

    Raises:
        ValueError: Two entries differ in height, width or channel count.
    """
    shapes: dict[tuple[int, ...], list[str]] = {}
    for name, tensor in zip(names, tensors):
        shapes.setdefault(tuple(int(axis) for axis in tensor.shape[1:]), []).append(name)
    if len(shapes) < 2:
        return
    listed = ", ".join(
        f"{'/'.join(sorted(set(slots)))} is {_described(shape)}"
        for shape, slots in shapes.items()
    )
    raise ValueError(
        f"{node} joins its images into one batch, so every connected image must be the same "
        f"size and hold the same number of channels. These do not match: {listed}. {advice}"
    )


def image_slot_template() -> io.Autogrow.TemplateNames:
    """The growing image slot series a batching node declares.

    Returns:
        A template over :data:`IMAGE_SLOTS`, one ``IMAGE`` socket per name, the first always
        drawn and the rest appearing as the one before it is filled.
    """
    return io.Autogrow.TemplateNames(
        input=io.Image.Input(
            "images",
            tooltip=(
                "One image or batch to add. A new empty slot appears as soon as this "
                "one is connected, and the slots are joined in the order they are "
                "listed."
            ),
        ),
        names=list(IMAGE_SLOTS),
        min=1,
    )


def as_batch(tensor, slot: str = "image"):
    """One image input as a batch, whatever arity it arrived with.

    Args:
        tensor: An ``IMAGE`` tensor, ``(batch, height, width, channels)`` or a single
            ``(height, width, channels)`` frame.
        slot: Socket id the tensor arrived on, named in the refusal.

    Returns:
        The tensor with a batch axis, a single frame answered as a batch of one.

    Raises:
        ValueError: The tensor carries neither three nor four axes.
    """
    shape = tuple(int(axis) for axis in getattr(tensor, "shape", ()))
    if len(shape) == 4:
        return tensor
    if len(shape) == 3:
        return tensor.unsqueeze(0)
    described = f"shaped {shape}" if shape else "not a tensor at all"
    raise ValueError(
        f"{slot} did not arrive as an image. An IMAGE is (batch, height, width, channels), or "
        f"(height, width, channels) for a single frame, and this one is {described}. Feed the "
        f"slot from an image node, or reshape the tensor before batching it."
    )


def frames(tensor, slot: str = "image") -> list:
    """The individual images in one image input, each as a batch of one.

    Args:
        tensor: An ``IMAGE`` tensor, ``(batch, height, width, channels)`` or a single
            ``(height, width, channels)`` frame.
        slot: Socket id the tensor arrived on, named in the refusal.

    Returns:
        One ``(1, height, width, channels)`` tensor per image, in batch order. Empty where
        the batch holds no frames.

    Raises:
        ValueError: The tensor carries neither three nor four axes.
    """
    batch = as_batch(tensor, slot)
    return [batch[index : index + 1] for index in range(int(batch.shape[0]))]
