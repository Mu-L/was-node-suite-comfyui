"""Save a positive and negative prompt into the style library under a name."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.interface import library_report, run_result
from ....modules.prompt import styles

logger = log.get_logger("text.styles")

#: How many characters of a prompt an auto-generated name quotes.
PREVIEW_LENGTH = 32


class PromptStyleSave(io.ComfyNode):
    """Store a prompt pair in the style library."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPromptStyleSave",
            display_name="Prompt Style Save",
            search_aliases=[
                "WASPromptStyleSave",
                "Prompt Style Save",
                "style",
                "save style",
                "a1111 styles",
            ],
            category="WAS Suite/Text/Styles",
            description=(
                "Save a positive and negative prompt into the style library under a name, "
                "so Prompt Styles Selector can call it back. Saving over a name replaces "
                "what it held, and a pair already in the library under another name is "
                "left where it is rather than stored twice."
            ),
            inputs=[
                io.String.Input(
                    "name",
                    default="",
                    multiline=False,
                    placeholder="Eg: cinematic film still",
                    tooltip=(
                        "What to call the style in the selector menu. Eg: cinematic film "
                        "still. Left empty, the name is made from the date and the first "
                        "32 characters of the prompt."
                    ),
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    placeholder="Eg: cinematic film still, shallow depth of field",
                    tooltip=(
                        "The positive prompt to store; STRING. Eg: `cinematic film still, "
                        "shallow depth of field, highly detailed`"
                    ),
                ),
                io.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: anime, cartoon, low contrast",
                    tooltip=(
                        "The negative prompt to store; STRING. Empty stores a style with "
                        "no negative half. Eg: `anime, cartoon, low contrast`"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="name",
                    tooltip=(
                        "The name the style is stored under, which is the generated one "
                        "when name was left empty."
                    ),
                ),
                io.String.Output(
                    display_name="positive_string",
                    tooltip="The positive prompt as stored, for a positive CLIP Text Encode.",
                ),
                io.String.Output(
                    display_name="negative_string",
                    tooltip="The negative prompt as stored, for a negative CLIP Text Encode.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, name="", prompt="", negative_prompt="") -> io.NodeOutput:
        """Store the pair and answer the name it went under.

        Raises:
            ValueError: Both prompts are empty, or the style could not be stored.
        """
        wanted = str(name or "").strip()
        positive = str(prompt or "")
        negative = str(negative_prompt or "")
        if not positive.strip() and not negative.strip():
            raise ValueError(
                "both prompt boxes are empty, so there is no style to save. Type at least "
                "a positive or a negative prompt"
            )

        library = styles.PromptStyles(preview_length=PREVIEW_LENGTH)
        held = dict(library.get_prompts())
        stored = library.add_style(
            prompt=positive,
            negative_prompt=negative,
            auto=not wanted,
            name=wanted or None,
        )
        if stored is None:
            raise ValueError(
                "the style could not be named, so nothing was saved. Type a name, or leave "
                "the name empty and type a prompt for it to be named after"
            )

        logger.info("the style `%s` is in the library", stored)
        cls.report(stored, wanted, held, len(library.get_prompts()), positive, negative)
        return io.NodeOutput(stored, positive, negative)

    @classmethod
    def report(cls, stored, wanted, held, total, positive, negative) -> None:
        """Draw what was saved and what the library holds on the node."""
        pair = {"prompt": positive, "negative_prompt": negative}
        status = run_result.OK
        if wanted and stored != wanted:
            status = run_result.WARNING
            summary = f"these prompts were already saved as `{stored}`, so `{wanted}` was not added"
        elif held.get(stored) == pair:
            status = run_result.WARNING
            summary = f"`{stored}` already held these prompts, so nothing changed"
        elif stored in held:
            summary = f"replaced the style `{stored}`"
        else:
            summary = f"saved the style `{stored}`"
        library_report.publish(
            summary=summary,
            counts={"styles": total, "prompt": len(positive), "negative": len(negative)},
            facts={"name": stored, "entry": "replaced" if stored in held else "new"},
            lines=[line for line in (positive, negative) if line],
            listing="prompts",
            status=status,
        )
