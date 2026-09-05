"""Append images to an animated GIF that grows across prompts."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ....modules.io import rooted
from ....modules import log
from ....modules.convert.tensors import tensor2pil
from ....modules.interface import file_report
from ....modules.util import sandbox

logger = log.get_logger("nodes.animation.writer")


def output_directory_default() -> str:
    """The ``filename`` default, resolved when the schema is built.

    Returns:
        The absolute path of ComfyUI's output directory.
    """
    import folder_paths

    # The v2 widget froze this default to ComfyUI's output directory, so it is resolved
    # here rather than written as a literal.
    return folder_paths.get_output_directory()


class WriteToGIF(io.ComfyNode):
    """Add every image in the batch to one animated GIF, creating it on the first run."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Write to GIF",
            display_name="Write to GIF",
            search_aliases=["Write to GIF", "append gif", "animated gif", "gif writer"],
            category="WAS Suite/Animation",
            description=(
                "Append the images to an animated GIF that keeps growing across prompts, "
                "fading in from the frame already at the end of the file."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The images to append. A batch is appended one image at a time, in "
                        "batch order, each faded in from the one before it."
                    ),
                ),
                io.Int.Input(
                    "transition_frames",
                    default=30,
                    min=2,
                    max=60,
                    step=1,
                    tooltip=(
                        "How many blended frames are drawn between the last frame in the file "
                        "and the incoming image. 2 is almost a hard cut, 30 is a smooth fade. "
                        "Ignored the first time, when there is nothing to fade from."
                    ),
                ),
                io.Float.Input(
                    "image_delay_ms",
                    default=2500.0,
                    min=0.1,
                    max=60000.0,
                    step=0.1,
                    tooltip=(
                        "How long each appended image is held once the fade into it has "
                        "finished, in milliseconds. 2500 holds it for two and a half seconds."
                    ),
                ),
                io.Float.Input(
                    "duration_ms",
                    default=0.1,
                    min=0.1,
                    max=60000.0,
                    step=0.1,
                    tooltip=(
                        "How long each blended frame is shown, in milliseconds. The default "
                        "0.1 asks for the shortest frame the format allows, which most "
                        "players round up to about 10ms; raise it to slow the fade down."
                    ),
                ),
                io.Int.Input(
                    "loops",
                    default=0,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "How many times the animation plays before it stops. 0 = forever, "
                        "1 = once through, 3 = three times. Written on every append, so the "
                        "finished file carries the count set on the last run."
                    ),
                ),
                io.Int.Input(
                    "max_size",
                    default=512,
                    min=128,
                    max=1280,
                    step=1,
                    tooltip=(
                        "Longest side any frame is written at, in pixels. 512 keeps a big "
                        "render down to a shareable file; an image already smaller is left "
                        "alone. The first frame still sets the canvas, and a later one is "
                        "centred on it."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. filename names the part below it, so "
                        "'[time(%Y-%m-%d)]/clip' files each day's under a dated folder."
                    ),
                ),
                io.String.Input(
                    "filename",
                    default="morph_writer",
                    multiline=False,
                    tooltip=(
                        "Name of the GIF, without an extension, '.gif' is added. Every prompt "
                        "using the same name appends to the same file, so change the name, or "
                        "put a token such as [time(%Y-%m-%d)] in it, to start a new animation."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image_pass",
                    tooltip="The images exactly as they arrived, so this node can sit mid-chain.",
                ),
                io.String.Output(
                    display_name="filepath_text",
                    tooltip="Full path of the GIF that was written, '.gif' included.",
                ),
                io.String.Output(
                    display_name="filename_text",
                    tooltip=(
                        "The filename widget as it was typed, with no folder and no extension, "
                        "for feeding a caption or a log."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """NaN never equals itself, so every prompt appends another image."""
        return float("NaN")

    @classmethod
    def execute(
        cls,
        image,
        transition_frames=30,
        image_delay_ms=2500.0,
        duration_ms=0.1,
        loops=0,
        max_size=512,
        root=rooted.DEFAULT,
        filename="morph_writer",
    ) -> io.NodeOutput:
        import comfy.utils

        from ....modules.media import gif


        transition_frames = min(max(transition_frames, 2), 60)
        duration_ms = min(max(duration_ms, 0.1), 60000.0)
        below, _, leaf = (filename or "").replace("\\", "/").rpartition("/")
        target = sandbox.resolve_write_file(
            rooted.destination(root, below), leaf + ".gif"
        )
        os.makedirs(target.parent, exist_ok=True)

        writer = gif.GifMorphWriter(
            int(transition_frames), int(duration_ms), int(image_delay_ms),
            loop=int(loops), max_size=int(max_size),
        )
        progress = comfy.utils.ProgressBar(len(image))
        for frame in image:
            writer.write(tensor2pil(frame), str(target))
            progress.update(1)

        logger.info("appended %s image(s) to %s", len(image), target)
        file_report.publish(
            [str(target)], kind="gif", facts={
                "appended": f"{len(image)} frame(s)",
                "loops": "forever" if not loops else str(loops),
                "max size": f"{max_size}px",
            }
        )
        return io.NodeOutput(image, str(target), filename)
