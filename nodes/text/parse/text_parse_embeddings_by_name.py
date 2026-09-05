"""Rewrite bare embedding names in a prompt into ComfyUI's ``embedding:`` syntax."""

from __future__ import annotations

import os
import re

from comfy_api.latest import io

from ....modules import log

logger = log.get_logger("nodes.text.parse")


class TextParseA1111Embeddings(io.ComfyNode):
    """Prefix every installed embedding's name in ``text`` with ``embedding:``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Parse A1111 Embeddings",
            display_name="Text Parse A1111 Embeddings",
            search_aliases=[
                "Text Parse A1111 Embeddings",
                "embeddings",
                "textual inversion",
                "a1111",
            ],
            category="WAS Suite/Text/Parse",
            description=(
                "Convert A1111-style embedding names in a prompt to ComfyUI's "
                "embedding:name syntax, using the embeddings installed on this machine."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: photo, badhands",
                    tooltip=(
                        "Prompt naming embeddings by file name; STRING. Installed names "
                        "get the `embedding:` prefix. Eg: `photo, badhands`"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The prompt with every installed embedding's name prefixed, so "
                        "'photo, badhands' becomes 'photo, embedding:badhands'. An embedding "
                        "that is not installed on this machine is left as written."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        for basename in cls._embedding_names():
            pattern = re.compile(r"\b(?<!embedding:){}\b".format(re.escape(basename)))
            replacement = "embedding:{}".format(basename)
            text = re.sub(pattern, replacement, text)

        return io.NodeOutput(text)

    @classmethod
    def fingerprint_inputs(cls, text):
        """Re-run when the set of installed embeddings changes, not only the text."""
        return (text, tuple(cls._embedding_names()))

    @staticmethod
    def _embedding_names() -> list[str]:
        """Extension-stripped file names of every embedding, in folder order."""
        import folder_paths

        names = []
        for embeddings_path in folder_paths.get_folder_paths("embeddings"):
            try:
                filenames = os.listdir(embeddings_path)
            except OSError as error:
                logger.debug("embeddings folder %s is unreadable (%s)", embeddings_path, error)
                continue
            for filename in filenames:
                names.append(os.path.splitext(filename)[0])
        return names
