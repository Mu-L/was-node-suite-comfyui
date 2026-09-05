"""True random numbers from the RANDOM.ORG service."""

from __future__ import annotations

import hashlib

from comfy_api.latest import io

from ...modules import deps, log
from ...modules.compat.types import NUMBER

REQUIRES = "network"

logger = log.get_logger("nodes.number")

#: Config key that enables this module. Named in dependency errors.
FEATURE = "features.network"

#: RANDOM.ORG's JSON-RPC endpoint. Documented at https://api.random.org/json-rpc/2.
API_URL = "https://api.random.org/json-rpc/2/invoke"

#: The widget default, and the two other spellings of "no key was entered".
UNSET_KEYS = (None, "00000000-0000-0000-0000-000000000000", "")

#: Seconds to wait for the service. Without one, an unanswered request holds the prompt
#: worker open indefinitely.
TIMEOUT = 30


def draw_integers(api_key: str | None, amount: int, minimum: int, maximum: int) -> list[int]:
    """Fetch true random integers from RANDOM.ORG.

    Args:
        api_key: RANDOM.ORG API key. A missing or placeholder key is reported and no
            request is made.
        amount: How many integers to draw.
        minimum: Inclusive lower bound.
        maximum: Inclusive upper bound.

    Returns:
        The drawn integers, or ``[0]`` when no key was given or the service did not
        answer with a result.

    Raises:
        DependencyError: ``requests`` is not installed.
    """
    if api_key in UNSET_KEYS:
        logger.error(
            "No API key provided! A valid RANDOM.ORG API key is required to use "
            "`True Random.org Number Generator`"
        )
        return [0]

    requests = deps.require("requests", feature=FEATURE)
    payload = {
        "jsonrpc": "2.0",
        "method": "generateIntegers",
        "params": {
            "apiKey": api_key,
            "n": amount,
            "min": minimum,
            "max": maximum,
            "replacement": True,
            "base": 10,
        },
        "id": 1,
    }

    response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    if response.status_code != 200:
        logger.error("RANDOM.ORG answered with status %s", response.status_code)
        return [0]

    data = response.json()
    if "result" not in data:
        logger.error("RANDOM.ORG returned no result: %s", data.get("error", data))
        return [0]

    return data["result"]["random"]["data"]


class TrueRandomNumberGenerator(io.ComfyNode):
    """Draw one true random integer from RANDOM.ORG.

    In 'fixed' mode the same key keeps returning the number already drawn.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="True Random.org Number Generator",
            display_name="True Random.org Number Generator",
            search_aliases=[
                "True Random.org Number Generator",
                "random.org",
                "true random",
                "entropy",
            ],
            category="WAS Suite/Number",
            description=(
                "Draw a true random integer from RANDOM.ORG, which derives it from "
                "atmospheric noise rather than a pseudo-random generator. Requires a free "
                "API key from https://api.random.org and an internet connection; without "
                "one the node emits 0."
            ),
            inputs=[
                io.String.Input(
                    "api_key",
                    default="00000000-0000-0000-0000-000000000000",
                    multiline=False,
                    tooltip=(
                        "A RANDOM.ORG API key, free from https://api.random.org, in the form "
                        "of a UUID. The all-zero default is a placeholder: while it is there, "
                        "no request is sent and the node reports the missing key and emits 0."
                    ),
                ),
                io.Float.Input(
                    "minimum",
                    default=0,
                    min=-18446744073709551615,
                    max=18446744073709551615,
                    tooltip=(
                        "The lowest number that can be drawn, itself included. Any fraction "
                        "is cut off first, since the service only draws whole numbers."
                    ),
                ),
                io.Float.Input(
                    "maximum",
                    default=10000000,
                    min=-18446744073709551615,
                    max=18446744073709551615,
                    tooltip=(
                        "The highest number that can be drawn, itself included. Any fraction "
                        "is cut off first. A range the service will not accept comes back as "
                        "0, with the refusal in the log."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["random", "fixed"],
                    tooltip=(
                        "How often a new number is fetched. `random` asks the service for a "
                        "fresh one on every prompt. `fixed` keeps handing out the number "
                        "already drawn until the key or one of the bounds changes, which "
                        "saves requests while the rest of a workflow is being tuned."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The drawn whole number, or 0 when no key was given or the service "
                        "did not answer."
                    ),
                ),
                io.Float.Output(
                    tooltip="The same number as a decimal, so 42 leaves here as 42.0.",
                ),
                io.Int.Output(
                    tooltip="The same number on an INT socket, for a seed widget.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, api_key, minimum, maximum, mode):
        """The key's digest in 'fixed' mode, NaN in 'random' mode so a new draw is made."""
        if mode == "fixed":
            return hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return float("NaN")

    @classmethod
    def execute(cls, api_key, minimum, maximum, mode) -> io.NodeOutput:
        number = draw_integers(api_key, 1, int(minimum), int(maximum))[0]
        return io.NodeOutput(number, float(number), int(number))
