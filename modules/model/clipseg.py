"""CLIPSeg text-prompted segmentation, on transformers.

One cached processor and model serve every CLIPSeg node, on ComfyUI's compute device.
"""

from __future__ import annotations

from .. import deps
from . import managed, resolve

__all__ = ["load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.clipseg"

#: ``folder_paths`` model folder searched for CLIPSeg checkpoints.
FOLDER = "clipseg"

#: The repository v2's widgets default to.
DEFAULT_REPO = "CIDAS/clipseg-rd64-refined"


def load(repo_id: str = "", device: str | None = None):
    """Load a CLIPSeg processor and segmentation model.

    Args:
        repo_id: Hugging Face repository id. Empty or blank selects :data:`DEFAULT_REPO`.
        device: Device name, or None for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``processor`` and ``model`` are the CLIPSeg
        pair.

    Raises:
        DependencyError: transformers is not importable.
        ModelUnavailable: No local checkpoint, and ``features.network`` is off.
    """
    repo_id = repo_id.strip() or DEFAULT_REPO

    transformers = deps.require("transformers", feature=FEATURE)
    pretrained, cache_dir = resolve(FOLDER, repo_id, feature=FEATURE)

    def build():
        processor = transformers.CLIPSegProcessor.from_pretrained(pretrained, cache_dir=cache_dir)
        model = transformers.CLIPSegForImageSegmentation.from_pretrained(
            pretrained, cache_dir=cache_dir
        )
        return processor, model

    return managed(("clipseg", pretrained), build, device=device)
