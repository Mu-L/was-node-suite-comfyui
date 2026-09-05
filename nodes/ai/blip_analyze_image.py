"""Image captioning and visual question answering with BLIP."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.sockets import require_input
from ...modules.compat.types import BLIP_MODEL
from ...modules.convert.tensors import tensor2pil

REQUIRES = "blip"

logger = log.get_logger("nodes.ai.blip")

#: ``mode`` widget option -> the key of the model in the BLIP_MODEL socket that answers it.
MODE_TASKS = {"caption": "caption", "interrogate": "question"}


class BlipAnalyzeImage(io.ComfyNode):
    """Describe an image in words, or answer a question about it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BLIP Analyze Image",
            display_name="BLIP Analyze Image",
            search_aliases=[
                "BLIP Analyze Image",
                "caption",
                "interrogate",
                "image to text",
                "vqa",
            ],
            category="WAS Suite/Image/AI",
            description=(
                "Turn an image into text: either a caption describing it, or an answer to a "
                "question about it. Enable features.blip to load this node."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to describe. Every image in the batch gets its own "
                        "caption or answer."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["caption", "interrogate"],
                    tooltip=(
                        "`caption` writes a description of the image and ignores the question "
                        "widget. `interrogate` answers the question instead, so 'What colour "
                        "is the car?' gives back a colour."
                    ),
                ),
                io.String.Input(
                    "question",
                    default="What does the background consist of?",
                    multiline=True,
                    dynamic_prompts=False,
                    tooltip=(
                        "The question to answer in `interrogate` mode. Plain language works "
                        "best and short answers are the norm: 'How many people are there?' "
                        "answers with a number. Ignored in `caption` mode."
                    ),
                ),
                BLIP_MODEL.Input(
                    "blip_model",
                    tooltip="Both models from BLIP Model Loader.",
                ),
                io.Int.Input(
                    "min_length",
                    min=1,
                    max=1024,
                    default=24,
                    optional=True,
                    tooltip=(
                        "Shortest answer the model may stop at, in tokens, which are roughly "
                        "words. Raise it to force a wordier caption; a question answer is "
                        "usually padded out rather than improved by it."
                    ),
                ),
                io.Int.Input(
                    "max_length",
                    min=2,
                    max=1024,
                    default=64,
                    optional=True,
                    tooltip=(
                        "Longest answer the model may produce, in tokens. The answer is cut "
                        "off here, so raise it if captions end mid-sentence."
                    ),
                ),
                io.Int.Input(
                    "num_beams",
                    min=1,
                    max=12,
                    default=5,
                    optional=True,
                    tooltip=(
                        "How many candidate wordings are explored before the best is picked. "
                        "1 is fastest and takes the first thing that comes; 5 is the usual "
                        "trade; 12 is slower and a little more considered."
                    ),
                ),
                io.Int.Input(
                    "no_repeat_ngram_size",
                    min=1,
                    max=12,
                    default=3,
                    optional=True,
                    tooltip=(
                        "Blocks any run of this many words from appearing twice, which stops "
                        "'a man on a man on a man'. 3 is a good default; 1 forbids repeating "
                        "even single words, including 'the'."
                    ),
                ),
                io.Boolean.Input(
                    "early_stopping",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Stop searching as soon as enough finished candidates exist rather "
                        "than exploring to the end. Faster, and it tends to give shorter "
                        "answers."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="FULL_CAPTIONS",
                    tooltip=(
                        "Every caption in one string, separated by blank lines, for saving to "
                        "a text file or feeding a prompt box."
                    ),
                ),
                io.String.Output(
                    display_name="CAPTIONS",
                    is_output_list=True,
                    tooltip=(
                        "One caption per image, as a list, so downstream nodes run once per "
                        "image."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        mode,
        question,
        blip_model,
        min_length=24,
        max_length=64,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=False,
    ) -> io.NodeOutput:
        """Caption the images, or answer the question about them.

        Raises:
            ValueError: Nothing is connected to the blip_model input.
        """
        import comfy.utils

        require_input(
            blip_model,
            "BLIP Analyze Image",
            "blip_model",
            "model",
            "BLIP Model Loader",
            "BLIP_MODEL",
        )

        backend = blip_model[MODE_TASKS[mode]]
        device = backend.load()
        processor = backend.processor
        model = backend.model
        progress = comfy.utils.ProgressBar(len(images))

        captions = []
        for image in images:
            pil_image = tensor2pil(image).convert("RGB")
            if mode == "caption":
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
            else:
                inputs = processor(images=pil_image, text=question, return_tensors="pt").to(device)

            generated = model.generate(
                **inputs,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )
            caption = processor.decode(generated[0], skip_special_tokens=True)
            captions.append(caption)
            logger.info("BLIP %s: %s", "caption" if mode == "caption" else "answer", caption)
            progress.update(1)

        # The separator follows every caption, the last one included, so the joined string
        # ends with a blank line.
        full_captions = "".join(caption + "\n\n" for caption in captions)

        return io.NodeOutput(full_captions, captions)
