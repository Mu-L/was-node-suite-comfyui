"""What a batching node made, as something its node can draw.

Everything publishes under the node's own id. Sizes are the tensor's own memory,
``elements x itemsize``.
"""

from __future__ import annotations

from . import run_result

#: Channel counts spelled the way a picture is described, so a readout names a mode rather than
#: a number. Anything else is reported as its own count.
IMAGE_MODES = {1: "L", 3: "RGB", 4: "RGBA"}


def readable_bytes(count) -> str:
    """A byte count in the largest unit that keeps it readable.

    Args:
        count: The number of bytes.

    Returns:
        The count with its unit, such as ``191.6 MB``.
    """
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def memory_of(tensor) -> int:
    """How many bytes a tensor occupies, or 0 when it cannot be measured.

    Args:
        tensor: A torch tensor, or anything else.

    Returns:
        ``element count x element size`` in bytes.
    """
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def describe_images(tensor) -> tuple[str, str]:
    """An image batch's frame size and channel mode.

    Args:
        tensor: An ``IMAGE`` tensor, ``(batch, height, width, channels)``.

    Returns:
        ``(size, mode)``, such as ``("1856x2254", "RGB")``.
    """
    height, width, channels = int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3])
    return f"{width}x{height}", IMAGE_MODES.get(channels, f"{channels} channels")


def describe_masks(tensor) -> tuple[str, str]:
    """A mask batch's frame size and mode.

    Args:
        tensor: A ``MASK`` tensor, whose last two axes are the height and the width.

    Returns:
        ``(size, mode)``.
    """
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    return f"{width}x{height}", "MASK"


def describe_latents(samples) -> tuple[str, str]:
    """A latent batch's frame size and channel count.

    Args:
        samples: A latent's ``samples`` tensor, ``(batch, channels, height, width)``.

    Returns:
        ``(size, mode)``, the size being the latent's own grid rather than pixels.
    """
    channels, height, width = int(samples.shape[1]), int(samples.shape[-2]), int(samples.shape[-1])
    return f"{width}x{height} latent", f"{channels} channels"


def _sentence(frames: int, slots: int) -> str:
    """What the node built, in words, with the plurals right.

    Args:
        frames: How many frames the finished batch holds.
        slots: How many inputs were connected.

    Returns:
        A sentence such as ``2 frames from 2 slots``.
    """
    frame_word = "frame" if frames == 1 else "frames"
    slot_word = "slot" if slots == 1 else "slots"
    return f"{frames} {frame_word} from {slots} {slot_word}"


def publish(frames, slots, size, mode, memory, refused=None, fitted=None, node_id=None) -> bool:
    """Store what a batch node built, or why it refused to build one.

    Args:
        frames: How many frames the finished batch holds.
        slots: How many inputs were connected.
        size: The frame size, as :func:`describe_images` and friends spell it.
        mode: The channel mode.
        memory: The batch's own memory in bytes.
        refused: What stopped the build, naming the slots and sizes that disagreed.
            Given, the report is a warning and the counts describe what was offered.
        fitted: How many slots had to be brought to size, for a node that does that. Left
            out, the report says nothing about fitting.
        node_id: The publishing node's graph id. Left out, the id of the node ComfyUI is
            executing, so a node needs no hidden input to report itself.

    Returns:
        Whatever :func:`run_result.publish` answered, which is False when no browser is
        connected and the readout is not worth building.
    """
    counts = {"frames": int(frames), "slots": int(slots)}
    facts = {"size": size, "mode": mode, "memory": readable_bytes(memory)}
    if fitted:
        facts["fitted"] = f"{int(fitted)} slot(s) brought to size"

    if refused:
        return run_result.publish(
            status=run_result.WARNING,
            summary=refused,
            counts=counts,
            facts=facts,
            node_id=node_id,
        )
    return run_result.publish(
        status=run_result.OK,
        summary=_sentence(int(frames), int(slots)),
        counts=counts,
        facts=facts,
        node_id=node_id,
    )


def publish_sample(kept, frames, strategy, size, detail=None, facts=None, warn=False,
                   node_id=None) -> bool:
    """Store what a frame sampler kept, for the strip on its node.

    Args:
        kept: How many frames the sampler answered.
        frames: How many there were to choose from.
        strategy: Which strategy chose them, for the row that names it.
        size: The frame size, as :func:`describe_images` spells it.
        detail: How that strategy reads in a sentence, for the summary line. Left out, the
            strategy's own name stands in, which repeats the row below it.
        facts: Anything further worth a row, as a mapping of name to value.
        warn: True where the pick is worth drawing in the warning colour, such as an index
            held to the last frame. A sample that kept every frame is a warning whether or
            not this is set.
        node_id: The publishing node's graph id. Left out, the id of the node ComfyUI is
            executing.

    Returns:
        Whatever :func:`run_result.publish` answered, which is False when no browser is
        connected and the readout is not worth building.
    """
    kept, frames = int(kept), int(frames)
    word = "frame" if kept == 1 else "frames"
    return run_result.publish(
        # Keeping everything is worth saying rather than reporting as a sample: it means the
        # clip was already shorter than what was asked for and the node changed nothing.
        status=run_result.WARNING if warn or kept >= frames else run_result.OK,
        summary=(
            f"kept all {frames} {word}" if kept >= frames
            else f"kept {kept} of {frames} {word}, {detail or strategy}"
        ),
        counts={"kept": kept, "frames": frames},
        facts={"size": size, "strategy": strategy, **(facts or {})},
        node_id=node_id,
    )
