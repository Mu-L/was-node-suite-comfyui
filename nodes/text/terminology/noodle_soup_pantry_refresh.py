"""Merge the published Noodle Soup Prompts pantry into the stored one."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.interface import library_report, run_result
from ....modules.prompt import nsp

logger = log.get_logger("nodes.text.terminology")

#: What each mode does, in the order the widget offers them.
CHECK = "check what is new"
MERGE = "merge it in"

MODES = (CHECK, MERGE)


class NoodleSoupPantryRefresh(io.ComfyNode):
    """Fetch the published pantry and merge it into the stored one."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNoodleSoupPantryRefresh",
            display_name="Noodle Soup Pantry Refresh",
            search_aliases=[
                "WASNoodleSoupPantryRefresh",
                "Noodle Soup Pantry Refresh",
                "nsp",
                "noodle soup",
                "pantry",
                "update terminology",
            ],
            category="WAS Suite/Text/Terminology",
            description=(
                "Fetch the published Noodle Soup Prompts pantry and merge it into the "
                "stored one. Words you added are kept, words you removed are not put back, "
                "and a terminology of your own is untouched. The download is checked in "
                "full before anything is stored, so a failed fetch changes nothing. Needs "
                "features.network on in config.yaml."
            ),
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=list(MODES),
                    tooltip=(
                        "`check what is new` downloads the published pantry and reports "
                        "what a merge would change, storing nothing. `merge it in` stores "
                        "the merged result."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "What the merge did, or would do, as one line per figure. Eg: "
                        "'terms added 2'."
                    ),
                ),
                io.Int.Output(
                    display_name="terms_added",
                    tooltip="Terminologies the published pantry has that the stored one did not.",
                ),
                io.Int.Output(
                    display_name="entries_added",
                    tooltip="Words the published pantry has that the stored one did not.",
                ),
                io.Int.Output(
                    display_name="yours_kept",
                    tooltip=(
                        "Words kept that the published pantry no longer has, which are the "
                        "ones added here."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, mode=CHECK) -> io.NodeOutput:
        """Fetch the pantry and merge or report it.

        Raises:
            ValueError: ``features.network`` is off, the download failed, or what came
                back is not a pantry.
        """
        preview = str(mode) != MERGE
        try:
            report = nsp.refresh_pantry(preview=preview)
        except OSError as error:
            raise ValueError(
                f"the Noodle Soup Prompts pantry could not be downloaded from "
                f"{nsp.PANTRY_URL} ({error}). Check that this machine can reach the "
                f"internet, then run the node again. Nothing stored was changed"
            ) from error

        lines = "\n".join(
            f"{name.replace('_', ' ')} {value}"
            for name, value in report.items()
            if name != "saved"
        )
        logger.info("pantry refresh (%s):\n%s", mode, lines)
        cls.report(preview, report, lines)
        return io.NodeOutput(
            lines,
            report["terms_added"],
            report["entries_added"],
            report["entries_kept"],
        )

    @classmethod
    def report(cls, preview, report, lines) -> None:
        """Draw what the merge changed, or would change, on the node."""
        changing = report["terms_added"] + report["entries_added"] + report["entries_retired"]
        if not preview and not report["saved"]:
            status = run_result.ERROR
            summary = "the merged pantry could not be stored, so nothing was changed"
        elif preview:
            status = run_result.WARNING if changing else run_result.OK
            summary = (
                f"a merge would change {changing} entry(s); nothing was stored"
                if changing
                else "the stored pantry already holds everything published"
            )
        else:
            status = run_result.OK
            summary = (
                f"{report['terms_added']} term(s) added, {report['entries_added']} "
                f"entry(s) added, {report['entries_kept']} of yours kept"
            )
        library_report.publish(
            summary=summary,
            counts={
                "terms +": report["terms_added"],
                "terms ~": report["terms_updated"],
                "entries +": report["entries_added"],
                "entries -": report["entries_retired"],
                "yours kept": report["entries_kept"],
                "declined": report["entries_declined"],
            },
            facts={
                "mode": CHECK if preview else MERGE,
                "stored": "no" if preview else ("yes" if report["saved"] else "no"),
                "source": nsp.PANTRY_URL,
            },
            lines=lines.splitlines(),
            listing="figures",
            status=status,
        )
