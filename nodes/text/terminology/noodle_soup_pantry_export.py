"""Write the stored Noodle Soup Prompts pantry out as a file."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.interface import library_report, run_result
from ....modules.io import rooted
from ....modules.prompt import nsp
from ....modules.util import sandbox

logger = log.get_logger("nodes.text.terminology")

#: What each scope writes, in the order the widget offers them.
WHOLE = "the whole pantry"
LOCAL = "only what you added"

SCOPES = (WHOLE, LOCAL)


class NoodleSoupPantryExport(io.ComfyNode):
    """Write the stored terminology out as a pantry JSON file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNoodleSoupPantryExport",
            display_name="Noodle Soup Pantry Export",
            search_aliases=[
                "WASNoodleSoupPantryExport",
                "Noodle Soup Pantry Export",
                "nsp",
                "noodle soup",
                "pantry",
                "save terminology",
            ],
            category="WAS Suite/Text/Terminology",
            description=(
                "Write the stored Noodle Soup Prompts terminology out as a JSON file, to "
                "share it, back it up or edit it by hand. Noodle Soup Pantry Import reads "
                "the same file back. A file already at that name is replaced."
            ),
            inputs=[
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, "
                        "listed by its own name. filename names the part below it."
                    ),
                ),
                io.String.Input(
                    "filename",
                    default="nsp_pantry.json",
                    multiline=False,
                    tooltip=(
                        "Name of the file, and any folder below root to put it in. Eg: "
                        "nsp_pantry.json, or terminology/[time(%Y-%m-%d)].json to file "
                        "each day's under a dated name."
                    ),
                ),
                io.Combo.Input(
                    "scope",
                    options=list(SCOPES),
                    tooltip=(
                        "`the whole pantry` writes every terminology, published words "
                        "included. `only what you added` writes just the words added from a "
                        "node or brought in from a file, which is the portable copy of your "
                        "own additions."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="path",
                    tooltip="The full path of the file that was written.",
                ),
                io.Int.Output(
                    display_name="terms",
                    tooltip="How many terminologies were written.",
                ),
                io.Int.Output(
                    display_name="entries",
                    tooltip="How many words were written, counting every terminology.",
                ),
            ],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, root=rooted.DEFAULT, filename="nsp_pantry.json", scope=WHOLE) -> io.NodeOutput:
        """Write the pantry and answer where it landed.

        Raises:
            ValueError: ``filename`` is empty.
            PathNotAllowed: The folder resolved outside every permitted write root.
            OSError: The folder could not be made.
        """
        wanted = str(filename or "").strip()
        if not wanted:
            raise ValueError(
                "no file name was given, so there is nowhere to write. Type a name such as "
                "nsp_pantry.json"
            )
        below, _, leaf = wanted.replace("\\", "/").rpartition("/")
        if not leaf:
            raise ValueError(
                f"`{wanted}` names a folder rather than a file. End it with a file name, "
                f"such as {below}/nsp_pantry.json"
            )
        directory = rooted.destination(root, below)
        directory.mkdir(parents=True, exist_ok=True)
        target = sandbox.resolve_write_file(directory, leaf)

        report = nsp.export_pantry(target, local_only=str(scope) == LOCAL)
        logger.info(
            "wrote %s term(s) and %s entry(s) to %s", report["terms"], report["entries"], target
        )
        cls.report(target, scope, report)
        return io.NodeOutput(str(target), report["terms"], report["entries"])

    @classmethod
    def report(cls, target, scope, report) -> None:
        """Draw what was written and where on the node."""
        if not report["saved"]:
            status = run_result.ERROR
            summary = f"{target.name} could not be written, so no file was left behind"
        elif report["entries"]:
            status = run_result.OK
            summary = (
                f"wrote {report['entries']} entry(s) across {report['terms']} term(s) to "
                f"{target.name}"
            )
        else:
            status = run_result.WARNING
            summary = f"{target.name} was written holding nothing, since there was nothing to write"
        library_report.publish(
            summary=summary,
            counts={"terms": report["terms"], "entries": report["entries"]},
            facts={"file": str(target), "scope": str(scope)},
            status=status,
        )
