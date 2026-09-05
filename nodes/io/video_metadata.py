"""Read what a video load measured as separate numbers."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import WAS_VIDEO_METADATA


def figure(metadata, key: str, fallback):
    """One figure out of a metadata mapping.

    Args:
        metadata: What a loader answered, or anything that is not a mapping.
        key: Which figure to read.
        fallback: Answer for a missing key and for an unreadable value.

    Returns:
        The value as the same type as ``fallback``.
    """
    if not isinstance(metadata, dict):
        return fallback
    value = metadata.get(key, fallback)
    try:
        return type(fallback)(value)
    except (TypeError, ValueError):
        return fallback


class VideoMetadata(io.ComfyNode):
    """Split a video load's measurements into the numbers a graph wires."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoMetadata",
            display_name="Video Metadata",
            search_aliases=[
                "WASVideoMetadata",
                "Video Metadata",
                "video fps",
                "frame count",
                "video duration",
                "video size",
            ],
            category="WAS Suite/IO",
            description=(
                "Read what a video load measured: the rate, the frame count, the duration "
                "and the size of the frames that came out, and the same figures for the "
                "file they came from. The loaders answer one metadata socket rather than a "
                "column of numbers, and this opens it where a number is actually wanted."
            ),
            inputs=[
                WAS_VIDEO_METADATA.Input(
                    "metadata",
                    tooltip="The metadata output of Load Video or Load Video (Upload).",
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="fps",
                    tooltip=(
                        "Frames per second the batch plays at: target_fps when one was "
                        "given, otherwise the rate the file was encoded at."
                    ),
                ),
                io.Int.Output(
                    display_name="frame_count",
                    tooltip=(
                        "How many frames came out, after the range, the strategy and "
                        "target_fps have all been applied. Feed it to a sampler that needs "
                        "its length told to it."
                    ),
                ),
                io.Float.Output(
                    display_name="duration",
                    tooltip=(
                        "Seconds the frames that came out play for, which is frame_count "
                        "divided by fps."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip="Frame width in pixels, after resizing.",
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip="Frame height in pixels, after resizing.",
                ),
                io.Boolean.Output(
                    display_name="has_audio",
                    tooltip=(
                        "True when the load's audio output carries sound. False for a silent "
                        "file, and for one whose sound track PyAV could not decode."
                    ),
                ),
                io.Int.Output(
                    display_name="bit_depth",
                    tooltip=(
                        "Bits per colour component the file is encoded at, 8 for most "
                        "footage and 10 for HDR and higher end capture. The video output "
                        "carries the same depth, so a save writes it back as it came in."
                    ),
                ),
                io.Float.Output(
                    display_name="source_fps",
                    tooltip="The rate the file itself was encoded at, before target_fps.",
                ),
                io.Int.Output(
                    display_name="source_frame_count",
                    tooltip=(
                        "Frames the whole file holds, before the range and the strategy cut "
                        "it down. Compare it with frame_count to see how much was kept."
                    ),
                ),
                io.Float.Output(
                    display_name="source_duration",
                    tooltip="Seconds the whole file runs for, whatever was kept from it.",
                ),
                io.Int.Output(
                    display_name="source_width",
                    tooltip="Frame width the file holds, before resizing.",
                ),
                io.Int.Output(
                    display_name="source_height",
                    tooltip="Frame height the file holds, before resizing.",
                ),
                io.String.Output(
                    display_name="filename",
                    tooltip=(
                        "The file the frames were read from, without its folder. Feed it to "
                        "a filename prefix so a render is named after its source."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, metadata) -> io.NodeOutput:
        """Answer each figure the load recorded.

        Args:
            metadata: What Load Video or Load Video (Upload) answered.

        Returns:
            The thirteen figures, each falling back to zero, false or empty where the load
            recorded nothing for it.
        """
        return io.NodeOutput(
            figure(metadata, "fps", 0.0),
            figure(metadata, "frame_count", 0),
            figure(metadata, "duration", 0.0),
            figure(metadata, "width", 0),
            figure(metadata, "height", 0),
            bool(figure(metadata, "has_audio", False)),
            figure(metadata, "bit_depth", 8),
            figure(metadata, "source_fps", 0.0),
            figure(metadata, "source_frame_count", 0),
            figure(metadata, "source_duration", 0.0),
            figure(metadata, "source_width", 0),
            figure(metadata, "source_height", 0),
            figure(metadata, "filename", ""),
        )
