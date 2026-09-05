"""Load a video from ComfyUI's input folder or from a web address."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.compat.types import WAS_VIDEO_METADATA
from ...modules.image import sizing
from ...modules.media import reader, sampling
from ...modules.util import sandbox
from .load_video import load

logger = log.get_logger("nodes.io")

#: Config key of the group that permits this node to reach the network. The node itself is
#: default tier and normally given a file from the input folder.
FEATURE = "features.network"

#: Text before the digest in the name a downloaded video is cached under.
NAME_PREFIX = "was_video_"

#: Container extensions a downloaded name may keep. Anything else is saved as ``.mp4``,
#: since libavformat reads a file by its content rather than by its name.
EXTENSIONS = frozenset(reader.VIDEO_EXTENSIONS)


def cache_name(url: str) -> str:
    """The temp file one address is kept in.

    Args:
        url: The address, as the widget holds it.

    Returns:
        A file name built from the address's digest, so the same address always names the
        same file and is fetched once.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    extension = os.path.splitext(urlsplit(url).path)[1].lower()
    return f"{NAME_PREFIX}{digest}{extension if extension in EXTENSIONS else '.mp4'}"


def fetch(url: str) -> str:
    """Download a video into ComfyUI's temp folder, when the network group permits it.

    Args:
        url: An ``http`` or ``https`` address naming a video file.

    Returns:
        The path the video was written to. An address already fetched is read from the temp
        folder instead of being fetched again.

    Raises:
        DependencyError: ``requests`` or ``tqdm`` is not installed.
        PathNotAllowed: ComfyUI's temp folder is outside every permitted write root.
        ValueError: ``features.network`` is off, the address is not an ``http`` one, the
            download did not complete, or what arrived holds no video.
    """
    import folder_paths

    from ...modules.config import group_enabled
    from ...modules.util import net

    address = (url or "").strip()
    if not address.lower().startswith(("http://", "https://")):
        raise ValueError(
            f"`{address}` is not a web address. url takes an http or https address naming a "
            f"video file. Leave it empty to read the file chosen above instead"
        )
    if not group_enabled(FEATURE):
        raise ValueError(
            f"not fetching {address}: {FEATURE} is off, so this pack makes no network "
            f"request of its own. Turn that group on in config.yaml to let this node "
            f"download, or leave url empty and pick a file from the list above"
        )

    target = sandbox.resolve_write_file(folder_paths.get_temp_directory(), cache_name(address))
    if target.is_file() and target.stat().st_size:
        logger.info("reading %s from %s, which an earlier run downloaded", address, target)
        return str(sandbox.resolve_read(target))

    os.makedirs(target.parent, exist_ok=True)
    # Written under a part name and moved into place, so an interrupted download is not
    # read back as a complete one on the next run.
    partial = sandbox.resolve_write_file(target.parent, target.name + ".part")
    try:
        arrived = net.download_file(address, partial.name, str(partial.parent))
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"{address} could not be reached: {error}. Check that the address opens in a "
            f"browser, and that this machine is allowed out to it"
        ) from error
    if not arrived:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"{address} did not download; the log names the status the server answered "
            f"with. An address behind a sign-in page answers this way"
        )
    _reject_unreadable(address, partial)
    os.replace(partial, target)
    logger.info("downloaded %s to %s", address, target)
    return str(sandbox.resolve_read(target))


def _reject_unreadable(url: str, path: Path) -> None:
    """Delete a download that holds no video, before it is cached under its address.

    Args:
        url: The address it came from, named in the message.
        path: The file the response body was written to.

    Raises:
        DependencyError: PyAV is not installed.
        ValueError: The file holds nothing that reads as a video.
    """
    from ...modules.deps import DependencyError

    try:
        reader.probe(str(path))
    except DependencyError:
        raise
    except Exception as error:
        size = path.stat().st_size if path.is_file() else 0
        path.unlink(missing_ok=True)
        raise ValueError(
            f"{url} answered {size} byte(s) that hold no video ({error}). An address behind "
            f"a sign-in page, a consent screen or a player page answers with a web page, "
            f"which downloads like a file. Open the address in a browser and use the one "
            f"the video itself is served from, ending in .mp4 or another container"
        ) from error


class LoadVideoUpload(io.ComfyNode):
    """Load a video chosen in ComfyUI's input folder, or downloaded from a web address."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadVideoUpload",
            display_name="Load Video (Upload)",
            search_aliases=[
                "WASLoadVideoUpload",
                "Load Video (Upload)",
                "video from url",
                "download video",
                "upload video",
                "video link",
            ],
            category="WAS Suite/IO",
            description=(
                "Load a video and hand on everything in it at once: the video itself, its "
                "frames as an image batch, its sound, and how long it is. Upload a file "
                "with the button on the node and play it back there, or paste a web address "
                "into url and the file is downloaded to ComfyUI's temp folder first. "
                "Downloading needs features.network on in config.yaml. Frames are chosen "
                "and sized exactly as Load Video beside it does them, 16 of them unless "
                "told otherwise."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=reader.video_labels(),
                    upload=io.UploadType.video,
                    tooltip=(
                        "Which video to read, from ComfyUI's input folder. The button below "
                        "uploads one and selects it, and the player shows what is selected. "
                        "Ignored while url holds an address."
                    ),
                ),
                io.String.Input(
                    "url",
                    default="",
                    multiline=False,
                    tooltip=(
                        "A web address to download the video from instead, such as "
                        "https://example.com/clip.mp4. It lands in ComfyUI's temp folder and "
                        "is fetched once, then read from there. Needs features.network on in "
                        "config.yaml. Empty reads the file chosen above."
                    ),
                ),
                io.Int.Input(
                    "num_frames",
                    default=16,
                    min=0,
                    max=reader.MAX_FRAMES,
                    tooltip=(
                        "How many frames to keep, chosen by the strategy below. 16 by "
                        "default; a clip can hold thousands and a batch is one "
                        f"tensor in memory. 0 takes every frame in the range, up to the "
                        f"{reader.MAX_FRAMES} ceiling."
                    ),
                ),
                io.Combo.Input(
                    "strategy",
                    options=list(sampling.STRATEGIES),
                    default="uniform",
                    tooltip=(
                        "How num_frames are chosen. uniform = evenly spaced; head = first; "
                        "center = middle; tail = last; random = a seeded pick; every_nth = "
                        "every nth. uniform gives a contact sheet of a whole clip, head "
                        "gives a run that plays."
                    ),
                ),
                io.Int.Input(
                    "nth",
                    default=1,
                    min=1,
                    max=limits.max_resolution(),
                    tooltip=(
                        "Step between the frames the strategy may choose from. 1 uses every "
                        "frame; 2 thins to every other one first, so `head` takes the opening "
                        "of the clip on alternate frames. It applies to every strategy."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Seed for random, so a re-run keeps the same frames. Ignored by the "
                        "other strategies. Any whole number; `0` is as good a seed as any."
                    ),
                ),
                io.Float.Input(
                    "target_fps",
                    default=0.0,
                    min=0.0,
                    max=reader.MAX_RATE,
                    step=0.01,
                    tooltip=(
                        "Rate the frames come out at. 0 keeps the file's own. A lower rate "
                        "drops frames and a higher one repeats them, so the clip runs for "
                        "the same time either way. Set it to match a model that wants 8 or "
                        "16 fps."
                    ),
                ),
                io.Int.Input(
                    "start",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    optional=True,
                    tooltip=(
                        "First frame to consider, counting from 0 through the file's own "
                        "frames. Negative counts back from the end, so -60 starts sixty "
                        "frames before it."
                    ),
                ),
                io.Int.Input(
                    "end",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    optional=True,
                    tooltip=(
                        "Last frame to consider, inclusive. -1 is the final frame, which is "
                        "the whole clip together with a start of 0."
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=list(sizing.MODES),
                    default=sizing.FIT_AND_PAD,
                    tooltip=(
                        "How each frame meets the size below. `fit and pad` keeps the whole "
                        "frame and pads the rest, `fill and crop` fills the size and trims "
                        "the overhang, `stretch` distorts to fit, `crop or pad` never "
                        "resamples."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Width every frame is brought to. 0 takes the width the file was "
                        "encoded at, which is what loads a clip at its own size."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Height every frame is brought to. 0 takes the height the file was "
                        "encoded at."
                    ),
                ),
                io.Int.Input(
                    "max_size",
                    default=1024,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    optional=True,
                    tooltip=(
                        "Longest edge the derived size is held to, keeping the aspect. Only "
                        "read when width and height are 0, which is where a 4K clip would "
                        "otherwise fill memory. 0 lifts the cap."
                    ),
                ),
                io.Combo.Input(
                    "interpolation",
                    options=list(sizing.FILTER_NAMES),
                    default=sizing.DEFAULT_FILTER,
                    optional=True,
                    tooltip="Resampling filter. `lanczos` is the sharpest for a downscale.",
                ),
                io.Combo.Input(
                    "align",
                    options=list(sizing.ALIGNMENT_NAMES),
                    default=sizing.DEFAULT_ALIGNMENT,
                    optional=True,
                    tooltip=(
                        "Which part of a frame survives a crop, and which side carries the "
                        "wider bar of a pad."
                    ),
                ),
                io.String.Input(
                    "pad_color",
                    default="#000000",
                    optional=True,
                    tooltip="Fill for space a frame does not cover. Any Pillow colour.",
                ),
                io.Combo.Input(
                    "channels",
                    options=list(sizing.CHANNELS),
                    default="RGB",
                    optional=True,
                    tooltip=(
                        "Channels the image batch carries. `RGBA` keeps the pad transparent. "
                        "The video output is always colour, since a video carries no "
                        "transparency."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The frames that were kept, with their sound, as a video at the rate "
                        "below. Wire it into Save Video, or into any node taking a VIDEO."
                    ),
                ),
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The same frames as one image batch, in playback order, every one at "
                        "the same size."
                    ),
                ),
                io.Audio.Output(
                    display_name="audio",
                    tooltip=(
                        "The sound playing under the frames that were kept, from where they "
                        "start and for as long as they run. Empty when the file is silent, "
                        "so read has_audio before wiring this into a save node."
                    ),
                ),
                WAS_VIDEO_METADATA.Output(
                    display_name="metadata",
                    tooltip=(
                        "What this read measured: the rate, the frame count, the size, the "
                        "duration, the bit depth and whether there is sound, beside the same "
                        "figures for the file itself. Wire it into Video Metadata to read any "
                        "of them as a number."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls, file, url="", num_frames=16, strategy="uniform", nth=1, seed=0, target_fps=0.0,
        resize_mode=sizing.FIT_AND_PAD, width=0, height=0, start=0, end=-1, max_size=1024,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB",
    ):
        """The address, or when the chosen file was last written, so an edit is read again."""
        import folder_paths

        address = (url or "").strip()
        if address:
            # The address itself, so a downloaded video is fetched once and read from the
            # temp folder on every run after it.
            return address
        # An empty name resolves to the input folder itself, which exists, so it is refused
        # before the folder is asked about it.
        chosen = (file or "").strip()
        if not chosen or not folder_paths.exists_annotated_filepath(chosen):
            return float("NaN")
        return os.path.getmtime(reader.input_path(file))

    @classmethod
    def validate_inputs(cls, file, url=""):
        """Whether there is something to read: an address, or a file still in the folder."""
        import folder_paths

        if (url or "").strip():
            return True
        if not (file or "").strip():
            return "nothing to load. Pick a video from the list, upload one, or paste an address into url"
        if not folder_paths.exists_annotated_filepath(file):
            return (
                f"`{file}` is not in ComfyUI's input, output or temp folder. Pick "
                f"another, or upload it again"
            )
        return True

    @classmethod
    def execute(
        cls, file, url="", num_frames=16, strategy="uniform", nth=1, seed=0, target_fps=0.0,
        resize_mode=sizing.FIT_AND_PAD, width=0, height=0, start=0, end=-1, max_size=1024,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB",
    ) -> io.NodeOutput:
        """Read the chosen or downloaded video and hand on its frames, sound and measurements.

        Raises:
            DependencyError: PyAV is not installed, or requests is not installed for a
                download.
            PathNotAllowed: The file resolved outside every permitted read root.
            ValueError: Nothing was chosen, ``features.network`` is off for an address, the
                download failed, or no frame could be decoded.
        """
        address = (url or "").strip()
        path = fetch(address) if address else reader.input_path(file)
        return load(
            path, num_frames, strategy, nth, seed, target_fps, resize_mode, width, height,
            start, end, max_size, interpolation, align, pad_color, channels,
        )
