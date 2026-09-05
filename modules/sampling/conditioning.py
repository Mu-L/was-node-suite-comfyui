"""Prompt encoding for the sequence text encoders.

A CONDITIONING value is a list of ``[tensor, dict]`` pairs.
"""

from __future__ import annotations

#: Token weight normalisation strategies offered by the sequence text encoders.
TOKEN_NORMALIZATIONS = ["none", "mean", "length", "length+mean"]

#: Prompt weighting dialects offered by the sequence text encoders.
WEIGHT_INTERPRETATIONS = ["comfy", "A1111", "compel", "comfy++"]

#: Node id of BlenderNeko's advanced text encoder, which reads the two settings above.
ADVANCED_ENCODER = "BNK_CLIPTextEncodeAdvanced"


def advanced_encoder():
    """The registered advanced text encoder class, or ``None``.

    Returns:
        The class registered under :data:`ADVANCED_ENCODER` by whichever pack provides
        it, or ``None`` when no pack does.
    """
    import nodes

    return nodes.NODE_CLASS_MAPPINGS.get(ADVANCED_ENCODER)


def encode_prompt(clip, text: str, token_normalization: str, weight_interpretation: str):
    """Encode one prompt into a ``[tensor, dict]`` conditioning pair.

    Args:
        clip: The CLIP model to encode with.
        text: One prompt.
        token_normalization: One of :data:`TOKEN_NORMALIZATIONS`.
        weight_interpretation: One of :data:`WEIGHT_INTERPRETATIONS`.

    Returns:
        A ``[tensor, dict]`` pair, one entry of a CONDITIONING list.
    """
    import nodes

    advanced = advanced_encoder()
    if advanced is not None:
        encoded = advanced().encode(
            clip=clip,
            text=text,
            token_normalization=token_normalization,
            weight_interpretation=weight_interpretation,
        )
    else:
        encoded = nodes.CLIPTextEncode().encode(clip=clip, text=text)
    return [encoded[0][0][0], encoded[0][0][1]]
