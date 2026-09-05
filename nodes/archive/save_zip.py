"""Write an archive held on a wire into a folder."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules.io import rooted
from ...modules import log
from ...modules.archive import container
from ...modules.compat.types import ZIP
from ...modules.interface import run_result
from ...modules.util import filenames, sandbox

logger = log.get_logger("nodes.archive")

#: Folder the archive is written into when the widget is left alone.
DEFAULT_PATH = "./ComfyUI/output/[time(%Y-%m-%d)]"


class SaveZip(io.ComfyNode):
    """Write a ZIP to a folder under a name, and answer where it landed."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASSaveZip",
            display_name="Save ZIP",
            search_aliases=[
                'WASSaveZip',
                "Save ZIP",
                "save zip",
                "write zip",
                "export archive",
                "zip export",
            ],
            category="WAS Suite/Archive",
            description=(
                "Write an archive to a folder under ComfyUI's output, and answer "
                "where it landed. filename_number_padding of 0 writes the exact name "
                "and replaces a file already there."
            ),
            inputs=[
                ZIP.Input(
                    "zip",
                    tooltip=(
                        "The archive to write, from ZIP Add or Open ZIP. It is written "
                        "exactly as it stands."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. filename_prefix names the part below it, so "
                        "'[time(%Y-%m-%d)]/notes' files each day's under a dated folder."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="archive",
                    multiline=False,
                    tooltip=(
                        "Start of the file name, before the delimiter and the number. "
                        "'archive' gives archive_0001.zip. Eg: renders"
                    ),
                ),
                io.String.Input(
                    "filename_delimiter",
                    default="_",
                    multiline=False,
                    tooltip=(
                        "What sits between the prefix and the number. '_' gives "
                        "archive_0001.zip and '-' gives archive-0001.zip. Eg: _"
                    ),
                ),
                io.Int.Input(
                    "filename_number_padding",
                    default=4,
                    min=0,
                    max=9,
                    step=1,
                    tooltip=(
                        "Digits the number is padded to, so 4 gives 'archive_0001.zip'. 0 "
                        "writes 'archive.zip' with no number, which replaces a file of that "
                        "name already in the folder."
                    ),
                ),
                io.String.Input(
                    "filename_suffix",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "Added after the number and before '.zip'. Empty adds nothing. Eg: "
                        "_final"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="zip_path",
                    tooltip="The full path of the archive that was written.",
                ),
                io.Int.Output(
                    display_name="file_count",
                    tooltip="How many files the written archive holds.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        zip=None,
        root=rooted.DEFAULT,
        filename_prefix="archive",
        filename_delimiter="_",
        filename_number_padding=4,
        filename_suffix="",
    ) -> io.NodeOutput:
        """Write the archive and answer where it landed.

        Raises:
            NotAnArchive: ``zip`` carries no archive.
            PathNotAllowed: The folder resolved outside every permitted write root.
            OSError: The folder could not be made, or the file could not be written.
        """
        archive = container.require_archive(zip, "zip")

        below, _, leaf = (filename_prefix or "").replace("\\", "/").rpartition("/")
        directory = rooted.destination(root, below)
        filename_prefix = leaf
        if not directory.exists():
            logger.warning("the path `%s` doesn't exist! Creating it...", directory)
            os.makedirs(directory, exist_ok=True)

        filename = filenames.generate_filename(
            directory,
            filename_prefix,
            filename_delimiter,
            int(filename_number_padding),
            container.SUFFIX,
            filename_suffix,
        )
        target = sandbox.resolve_write_file(directory, filename)
        replaced = target.exists()

        data = cls.payload(archive)
        target.write_bytes(data)

        count = len(archive.files)
        logger.info(
            "Save ZIP wrote %d file(s) to %s (%d bytes)%s",
            count,
            target,
            len(data),
            ", replacing what was there" if replaced else "",
        )
        listing = chr(10).join(entry.name for entry in archive.files)
        run_result.publish(
            summary=(
                f"{'Overwrote' if replaced else 'Wrote'} {target.name}, {count} entr"
                f"{'y' if count == 1 else 'ies'}"
            ),
            counts={"entries": count, "bytes": len(data)},
            facts={
                "folder": str(directory),
                "file": target.name,
                "replaced": "yes" if replaced else "no",
            },
            bodies=run_result.body("contents", listing),
        )
        return io.NodeOutput(str(target), count)

    @classmethod
    def payload(cls, archive) -> bytes:
        """The archive's own bytes, whether it was built in memory or opened from disk.

        Args:
            archive: The archive to write.

        Returns:
            The whole file.
        """
        held = getattr(archive, "_data", None)
        if held:
            return bytes(held)
        source = getattr(archive, "source", None)
        if source:
            return sandbox.resolve_read(source).read_bytes()
        raise ValueError(
            "the archive carries neither its bytes nor the file it came from, so there is "
            "nothing to write."
        )
