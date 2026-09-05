"""Animated GIF, APNG and WebP writing.

Two writers, both taking PIL images. :data:`FORMATS` maps a ``filetype`` to its Pillow
format, suffix and save options.
"""

from __future__ import annotations

import os

from PIL import Image

from .. import log
from ..util import sandbox

__all__ = ["FORMATS", "GifMorphWriter", "canvas_size", "morph_images"]

logger = log.get_logger("media.gif")

#: ``filetype`` -> ``(Pillow format, suffix, extra save options)``.
#:
#: WebP carries 24-bit colour and an alpha channel, where GIF is limited to 256 indexed
#: colours and one fully transparent entry. ``method=4`` is Pillow's default encoder effort.
FORMATS = {
    "GIF": ("GIF", ".gif", {}),
    "APNG": ("PNG", ".png", {}),
    "WEBP": ("WEBP", ".webp", {"quality": 90, "method": 4}),
    "WEBP_LOSSLESS": ("WEBP", ".webp", {"lossless": True, "method": 4}),
}

#: The format a ``filetype`` outside :data:`FORMATS` is written as, which is what every
#: caller wrote before the others were offered.
DEFAULT_FORMAT = "GIF"

#: Formats limited to an indexed palette, which are quantised whatever the caller asks for.
INDEXED_FORMATS = ("GIF",)

#: How a palette is chosen. ``per_frame`` fits one to each frame, which is the most faithful
#: single frame and lets a static area shift colour as the palette changes under it.
#: ``global`` fits one palette to the whole animation, which holds a static area steady.
PALETTE_MODES = ("per_frame", "global")

#: Colours a palette may hold. GIF cannot exceed the upper bound.
MIN_COLORS = 2
MAX_COLORS = 256

#: Longest side each frame is reduced to before a global palette is measured from it. The
#: palette needs the colour distribution, not the resolution, and a full-size montage of a
#: long sequence would be the largest allocation in the writer.
PALETTE_SAMPLE = 128


def canvas_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Scale a canvas down so neither edge exceeds ``max_size``, keeping its aspect ratio.

    Args:
        width: Width in pixels of the largest source.
        height: Height in pixels of the largest source.
        max_size: Longest edge the canvas may have. At or below 0 nothing is scaled.

    Returns:
        ``(width, height)``, unchanged when both edges already fit, and never below 1.
    """
    longest = max(width, height)
    if max_size <= 0 or longest <= max_size:
        return width, height
    scale = max_size / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def quantize_frames(
    frames: list[Image.Image],
    palette_mode: str = "per_frame",
    max_colors: int = MAX_COLORS,
    dither: bool = True,
) -> list[Image.Image]:
    """Reduce frames to an indexed palette.

    Args:
        frames: Source images, all the same size.
        palette_mode: A value from :data:`PALETTE_MODES`. An unknown value fits a palette
            per frame.
        max_colors: Palette entries, clamped to :data:`MIN_COLORS`-:data:`MAX_COLORS`.
        dither: Spread the quantisation error between neighbouring pixels, which trades
            visible speckle for the flat bands a hard mapping leaves in a gradient.

    Returns:
        One ``P`` mode image per input frame, in the same order. Under ``global`` every
        image carries the same palette. ``frames`` is left empty.
    """
    colors = max(MIN_COLORS, min(MAX_COLORS, int(max_colors)))
    spread = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE

    shared = _shared_palette(frames, colors) if palette_mode == "global" else None

    # Each source frame is released as its reduced frame is built, so the two lists never
    # both hold the whole animation.
    reduced = []
    for index, frame in enumerate(frames):
        if shared is not None:
            reduced.append(frame.quantize(palette=shared, dither=spread))
        else:
            reduced.append(
                frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors, dither=spread)
            )
        frames[index] = None
    frames.clear()
    return reduced


def _shared_palette(frames: list[Image.Image], colors: int) -> Image.Image:
    """One palette measured across every frame.

    Args:
        frames: Source images.
        colors: Palette entries.

    Returns:
        A ``P`` mode image whose palette covers the whole sequence, for
        :meth:`PIL.Image.Image.quantize` to map against.
    """
    width, height = frames[0].size
    scale = min(1.0, PALETTE_SAMPLE / max(width, height))
    size = (max(1, int(width * scale)), max(1, int(height * scale)))

    montage = Image.new("RGB", (size[0], size[1] * len(frames)))
    for index, frame in enumerate(frames):
        montage.paste(frame.resize(size, Image.Resampling.BILINEAR), (0, size[1] * index))
    return montage.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)


def morph_images(
    images: list[Image.Image],
    steps: int = 10,
    max_size: int = 512,
    loop: int | None = None,
    still_duration: int = 30,
    duration: float = 0.1,
    output_path: str = "output",
    filename: str = "morph",
    filetype: str = "GIF",
    palette_mode: str = "per_frame",
    max_colors: int = MAX_COLORS,
    dither: bool = True,
) -> str | None:
    """Cross-fade images into one animation and write it to disk.

    Args:
        images: Source images, in order. At least one is required.
        steps: Blended frames rendered between each consecutive pair.
        max_size: Longest edge of the canvas, in pixels. The canvas is otherwise sized
            from the largest source, and is only ever scaled down to this.
        loop: How many times the animation plays, where 0 plays forever. ``None`` writes no
            loop count, which most viewers read as playing once.
        still_duration: Frame duration for each source image.
        duration: Frame duration for each blended frame.
        output_path: Destination directory, ``/``-separated.
        filename: Destination file name, without an extension.
        filetype: A key of :data:`FORMATS`. An unrecognised value writes a
            :data:`DEFAULT_FORMAT`.
        palette_mode: A value from :data:`PALETTE_MODES`, used when the frames are
            quantised.
        max_colors: Palette entries. Applied to an :data:`INDEXED_FORMATS` target always,
            and to a full-colour target only below :data:`MAX_COLORS`.
        dither: Spread the quantisation error between neighbouring pixels.

    Returns:
        The absolute path written, or ``None`` when the file could not be opened.
    """
    pil_format, suffix, save_options = FORMATS.get(filetype, FORMATS[DEFAULT_FORMAT])
    output_file = str(sandbox.resolve_write_file(output_path, filename + suffix))

    max_width = max(im.size[0] for im in images)
    max_height = max(im.size[1] for im in images)
    max_width, max_height = canvas_size(max_width, max_height, max_size)
    max_aspect_ratio = max_width / max_height

    def padded_images():
        for im in images:
            aspect_ratio = im.size[0] / im.size[1]
            if aspect_ratio > max_aspect_ratio:
                new_height = max(1, int(max_width / aspect_ratio))
                padding = (max_height - new_height) // 2
                padded_im = Image.new("RGB", (max_width, max_height), color=(0, 0, 0))
                padded_im.paste(im.resize((max_width, new_height)), (0, padding))
            else:
                new_width = max(1, int(max_height * aspect_ratio))
                padding = (max_width - new_width) // 2
                padded_im = Image.new("RGB", (max_width, max_height), color=(0, 0, 0))
                padded_im.paste(im.resize((new_width, max_height)), (padding, 0))
            yield padded_im

    canvas = list(padded_images())
    canvas.append(canvas[0].copy())

    frames = []
    durations = []
    for i in range(len(canvas) - 1):
        frames.append(canvas[i])
        durations.append(still_duration)

        for step in range(steps):
            frames.append(Image.blend(canvas[i], canvas[i + 1], step / float(steps)))
            durations.append(duration)

    # The closing frame is the first image again, held like any other source image. Its
    # duration is appended, not inserted at the front: one duration per frame, in the order
    # the frames are written.
    frames.append(canvas[-1])
    durations.append(still_duration)

    # A full-colour target is left alone at the default, so asking for no reduction costs
    # nothing and writes exactly what the blend produced.
    if pil_format in INDEXED_FORMATS or max_colors < MAX_COLORS:
        frames = quantize_frames(frames, palette_mode, max_colors, dither)

    options = dict(save_options)
    if loop is not None:
        options["loop"] = loop

    try:
        frames[0].save(
            output_file,
            format=pil_format,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            **options,
        )
    except OSError as error:
        logger.error("unable to save output to %s due to the following error: %s", output_file, error)
        return None
    except Exception as error:
        logger.error("unable to write the animation due to the following error: %s", error)
        return None

    logger.info("morphing completed, output saved as %s", output_file)

    return output_file


class GifMorphWriter:
    """Appends one image at a time to a growing GIF.

    Args:
        transition_frames: Blended frames rendered between the last frame already in the
            file and the incoming image.
        duration_ms: Frame duration for each blended frame.
        still_image_delay_ms: Frame duration for each appended image.
        loop: Animation loop count written on every rewrite, where 0 loops forever.
        max_size: Longest side any frame is written at, in pixels. 0 writes each image at
            its own size.
    """

    def __init__(
        self,
        transition_frames: int = 30,
        duration_ms: int = 100,
        still_image_delay_ms: int = 2500,
        loop: int = 0,
        max_size: int = 0,
    ):
        self.transition_frames = transition_frames
        self.duration_ms = duration_ms
        self.still_image_delay_ms = still_image_delay_ms
        self.loop = loop
        self.max_size = max_size

    def fit(self, image: Image.Image) -> Image.Image:
        """One image brought within ``max_size`` on its longest side, keeping its shape."""
        longest = max(image.size)
        if self.max_size <= 0 or longest <= self.max_size:
            return image
        scale = self.max_size / longest
        return image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )

    def write(self, image: Image.Image, gif_path: str) -> None:
        """Append ``image`` to the GIF at ``gif_path``, creating it if it is not there.

        Args:
            image: The image to append.
            gif_path: Destination file, which is both read and written.
        """
        image = self.fit(image)
        if not os.path.isfile(gif_path):
            with Image.new("RGBA", image.size) as new_gif:
                new_gif.paste(image.convert("RGBA"))
                new_gif.info["duration"] = self.still_image_delay_ms
                new_gif.save(
                    gif_path,
                    format="GIF",
                    save_all=True,
                    append_images=[],
                    duration=self.still_image_delay_ms,
                    loop=self.loop,
                )
            logger.info("created new GIF animation at: %s", gif_path)
        else:
            with Image.open(gif_path) as gif:
                n_frames = gif.n_frames
                if n_frames > 0:
                    gif.seek(n_frames - 1)
                    last_frame = gif.copy()
                else:
                    last_frame = None

                end_image = image
                steps = self.transition_frames - 1 if last_frame is not None else self.transition_frames

                if last_frame is not None:
                    image = self.pad_to_size(image, last_frame.size)

                frames = self.generate_transition_frames(last_frame, image, steps)

                still_frame = end_image.copy()

                gif_frames = []
                for i in range(n_frames):
                    gif.seek(i)
                    gif_frame = gif.copy()
                    gif_frames.append(gif_frame)

                for frame in frames:
                    frame.info["duration"] = self.duration_ms
                    gif_frames.append(frame)

                still_frame.info["duration"] = self.still_image_delay_ms
                gif_frames.append(still_frame)

                gif_frames[0].save(
                    gif_path,
                    format="GIF",
                    save_all=True,
                    append_images=gif_frames[1:],
                    optimize=True,
                    loop=self.loop,
                )

                logger.info("edited existing GIF animation at: %s", gif_path)

    def pad_to_size(self, image: Image.Image, size: tuple[int, int]) -> Image.Image:
        """Centre an image on a transparent canvas of ``size``.

        Args:
            image: Source image.
            size: ``(width, height)`` of the canvas.

        Returns:
            An ``RGBA`` image of exactly ``size``. An image larger than the canvas is
            pasted at a negative offset and cropped by it rather than being scaled.
        """
        new_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
        x_offset = (size[0] - image.width) // 2
        y_offset = (size[1] - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        return new_image

    def generate_transition_frames(
        self,
        start_frame: Image.Image | None,
        end_image: Image.Image,
        num_frames: int,
    ) -> list[Image.Image]:
        """Blend from one image to another.

        Args:
            start_frame: Image the transition starts from. ``None`` returns no frames.
            end_image: Image the transition runs towards.
            num_frames: Frames to render.

        Returns:
            ``num_frames`` ``RGBA`` images at evenly spaced weights, excluding both ends:
            the first is one step away from ``start_frame`` and the last one step short of
            ``end_image``.
        """
        if start_frame is None:
            return []

        start_frame = start_frame.convert("RGBA")
        end_image = end_image.convert("RGBA")

        frames = []
        for i in range(1, num_frames + 1):
            weight = i / (num_frames + 1)
            frame = Image.blend(start_frame, end_image, weight)
            frames.append(frame)
        return frames
