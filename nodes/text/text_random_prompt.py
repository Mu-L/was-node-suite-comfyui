"""Fetch a random prompt from the Lexica search API."""

from __future__ import annotations

import random

from comfy_api.latest import io

from ...modules import deps, log

REQUIRES = "network"

logger = log.get_logger("text.random_prompt")

#: Queried when the search field is left empty, so the node returns something varied
#: rather than nothing.
DEFAULT_QUERIES = ["portrait", "landscape", "anime", "superhero", "animal", "nature", "scenery"]

SEARCH_URL = "https://lexica.art/api/v1/search"

#: Emitted when the API is unreachable or returns no images. Kept as the literal string
#: v2 returned, which workflows downstream of this node test for.
NOT_FOUND = "404 not found error"

#: Seconds to wait for the API. v2 passed no timeout at all, which leaves the prompt
#: worker blocked on a socket that cannot be cancelled from the frontend.
TIMEOUT = 10


class TextRandomPrompt(io.ComfyNode):
    """Return the prompt of a random image matching a Lexica search."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Random Prompt",
            display_name="Text Random Prompt",
            search_aliases=["Text Random Prompt", "lexica", "random prompt"],
            category="WAS Suite/Text",
            description=(
                "Search lexica.art and return the prompt of one random result. An empty "
                "search term picks a subject at random. Needs an internet connection."
            ),
            not_idempotent=True,
            inputs=[
                io.String.Input(
                    "search_seed",
                    multiline=False,
                    tooltip=(
                        "Subject to search lexica.art for, for example 'cyberpunk city'. "
                        "Left empty, one of portrait, landscape, anime, superhero, animal, "
                        "nature or scenery is searched for instead."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The prompt of one random image the search matched. Reads "
                        "'404 not found error' when the search returned nothing or the site "
                        "could not be reached."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, search_seed=None) -> io.NodeOutput:
        requests = deps.require("requests", feature="features.network")

        query = search_seed
        if query in ["", " "]:
            query = None
        if not query:
            query = random.choice(DEFAULT_QUERIES)

        try:
            response = requests.get(SEARCH_URL, params={"q": query}, timeout=TIMEOUT)
            images = response.json().get("images", [])
            if not images:
                return io.NodeOutput(NOT_FOUND)
            prompt = random.choice(images).get("prompt")
            if prompt is None:
                return io.NodeOutput(NOT_FOUND)
        except Exception as error:
            logger.error("Unable to establish connection to the Lexica API (%s).", error)
            prompt = NOT_FOUND

        return io.NodeOutput(prompt)
