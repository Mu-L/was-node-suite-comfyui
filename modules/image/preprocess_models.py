"""Running the models a preprocessor answers with.

Every function takes the loaded model and an ``IMAGE`` batch in ``[0, 1]``, then whatever
that answer reads, and answers a ``(batch, channels, height, width)`` tensor on a 0 to 255
scale.
"""

from __future__ import annotations

import torch

from . import lines as drawing
from . import pose, preprocess, segmentation

#: Levels the TEED answer is quantised onto, which is what pushes its floor to black.
SAFE_STEPS = 2

__all__ = [
    "anyline",
    "depth",
    "intrinsics",
    "line_segments",
    "lines",
    "restore",
    "segments",
    "skeleton",
    "soft_edges",
]


def depth(loaded, image, resolution: int) -> torch.Tensor:
    """Run the depth model over a batch and answer its map at the image's own size.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the model reads at.

    Returns:
        A ``(batch, 1, height, width)`` map on a 0 to 255 scale, bright for near.
    """
    import numpy as np
    import torch.nn.functional as functional

    import comfy.utils

    backend = loaded.backend
    device = backend.load()
    height, width = int(image.shape[1]), int(image.shape[2])
    longest = max(height, width)
    edge = min(int(resolution), longest)
    scale = edge / float(longest)
    read = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))

    progress = comfy.utils.ProgressBar(len(image))
    estimates = []
    for frame in image:
        source = (frame[..., :3].float().cpu().numpy() * 255.0).astype(np.uint8)
        inputs = backend.processor(images=source, return_tensors="pt").to(device)
        with torch.no_grad():
            predicted = backend.model(**inputs).predicted_depth
        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)
        estimates.append(
            functional.interpolate(
                predicted.float(), size=read, mode="bicubic", align_corners=False
            )
        )
        progress.update(1)

    # The floor and span are taken across the batch rather than per frame.
    floor = torch.stack([found.amin() for found in estimates]).amin()
    span = (torch.stack([found.amax() for found in estimates]).amax() - floor).clamp_min(1e-6)

    maps = [
        functional.interpolate(
            (found - floor) / span, size=(height, width), mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        for found in estimates
    ]
    return torch.cat(maps, dim=0) * 255.0


#: The classes an animal pose model is pointed at, as the detector names them.
ANIMALS = (
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
)

#: Layout name -> the function that draws it.
DRAWN = {
    "body": "draw",
    "wholebody": "draw_wholebody",
    "animal": "draw_animal",
}


def skeleton(loaded, image, threshold: float) -> torch.Tensor:
    """Find the subjects in every frame and draw their skeletons.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived, holding a detector and a poser.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        threshold: How sure the detector has to be to pose a subject.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    import numpy as np

    import comfy.utils

    pair = loaded.backend
    detector, poser = pair.detector, pair.poser
    device = detector.load()
    poser.load()
    height, width = int(image.shape[1]), int(image.shape[2])
    wanted = _subject_labels(detector.model.config, pair.layout)
    render = getattr(pose, DRAWN.get(pair.layout, "draw"))

    progress = comfy.utils.ProgressBar(len(image))
    drawn = []
    for frame in image:
        picture = (frame[..., :3].float().cpu().numpy() * 255.0).astype(np.uint8)
        boxes = _subjects(detector, picture, height, width, threshold, wanted, device)
        if boxes is None:
            drawn.append(torch.zeros((1, 3, height, width), dtype=torch.float32))
        else:
            points, scores = _joints(poser, picture, boxes, pair.experts, device)
            drawn.append(render(points, scores, height, width).cpu())
        progress.update(1)
    return torch.cat(drawn, dim=0)


def _subject_labels(config, layout: str) -> set:
    """Which class indices the detector should keep for one keypoint layout."""
    labels = getattr(config, "id2label", None) or {}
    wanted = ANIMALS if layout == "animal" else ("person",)
    found = {int(index) for index, name in labels.items() if str(name).lower() in wanted}
    return found or {0}


def _subjects(detector, picture, height, width, threshold, wanted, device):
    """Every subject's box as ``(count, 4)`` in ``x, y, width, height``, or None."""
    inputs = detector.processor(images=picture, return_tensors="pt").to(device)
    with torch.no_grad():
        found = detector.model(**inputs)
    sizes = torch.tensor([(height, width)], device=device)
    results = detector.processor.post_process_object_detection(
        found, target_sizes=sizes, threshold=float(threshold)
    )[0]
    keep = torch.isin(results["labels"], torch.tensor(sorted(wanted), device=device))
    boxes = results["boxes"][keep]
    if boxes.numel() == 0:
        return None
    corners = boxes.cpu()
    return torch.stack(
        [corners[:, 0], corners[:, 1], corners[:, 2] - corners[:, 0],
         corners[:, 3] - corners[:, 1]],
        dim=1,
    )


def _joints(poser, picture, boxes, experts: int, device):
    """Each subject's joints in pixels, and the confidence beside them."""
    inputs = poser.processor(picture, boxes=[boxes], return_tensors="pt").to(device)
    extra = {}
    if experts > 1:
        # A mixture of experts is told which dataset it is answering for, and 0 is COCO.
        extra["dataset_index"] = torch.zeros(len(boxes), dtype=torch.long, device=device)
    with torch.no_grad():
        estimated = poser.model(**inputs, **extra)
    people = poser.processor.post_process_pose_estimation(estimated, boxes=[boxes])[0]
    points = torch.stack([person["keypoints"].cpu() for person in people])
    scores = torch.stack([person["scores"].cpu() for person in people])
    return points, scores


def segments(loaded, image, resolution: int) -> torch.Tensor:
    """Label every pixel and paint the labels in the ADE20K palette.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the model reads at.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    import numpy as np
    import torch.nn.functional as functional

    import comfy.utils

    backend = loaded.backend
    device = backend.load()
    height, width = int(image.shape[1]), int(image.shape[2])
    longest = max(height, width)
    edge = min(int(resolution), longest)
    scale = edge / float(longest)
    read = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))

    progress = comfy.utils.ProgressBar(len(image))
    painted = []
    for frame in image:
        source = (frame[..., :3].float().cpu().numpy() * 255.0).astype(np.uint8)
        inputs = backend.processor(images=source, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = backend.model(**inputs).logits
        # Classes are chosen at the working size and then grown, so a class never appears
        # from the interpolation of two others.
        at_read = functional.interpolate(
            logits.float(), size=read, mode="bilinear", align_corners=False
        )
        classes = at_read.argmax(dim=1)
        grown = functional.interpolate(
            classes.unsqueeze(1).float(), size=(height, width), mode="nearest"
        ).squeeze(1)
        painted.append(segmentation.colourise(grown).cpu())
        progress.update(1)
    return torch.cat(painted, dim=0)


def soft_edges(loaded, image, resolution: int) -> torch.Tensor:
    """Trace soft edges at every scale the network answers and average them into one map.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived, holding the edge network.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the network reads at.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    import torch.nn.functional as functional

    import comfy.utils

    network, device, (height, width), read = _network(loaded, image, resolution)
    # TEED joins three branches at one size, so both sides have to divide by eight.
    read = _rounded(_at_least(read, SMALLEST_SIDE.get(loaded.name, 1)),
                    8 if loaded.name == "TEED Soft Edge" else 1)
    stepped = loaded.name == "TEED Soft Edge"
    fused = loaded.name == "PiDiNet Soft Edge"
    scale = 1.0 / 255.0 if fused else 1.0

    progress = comfy.utils.ProgressBar(len(image))
    traced = []
    for frame in image:
        planes = _planes(frame, device) * 255.0
        if fused:
            planes = planes.flip(1) * scale
        if read != tuple(planes.shape[-2:]):
            planes = functional.interpolate(planes, size=read, mode="area")
        with torch.no_grad():
            stages = network(planes)
        if fused:
            # PiDiNet answers a fused map last and has already taken its own sigmoid.
            edges = functional.interpolate(
                stages[-1].float(), size=(height, width), mode="bilinear", align_corners=False
            )
        else:
            grown = [
                functional.interpolate(
                    s.float(), size=(height, width), mode="bilinear", align_corners=False
                )
                for s in stages
            ]
            # The scales are averaged before the sigmoid, which is how these are read.
            edges = torch.sigmoid(torch.cat(grown, dim=1).mean(dim=1, keepdim=True))
            if stepped:
                edges = _stepped(edges, SAFE_STEPS)
        traced.append((edges.clamp(0.0, 1.0) * 255.0).repeat(1, 3, 1, 1).cpu())
        progress.update(1)
    return torch.cat(traced, dim=0)


def _network(loaded, image, resolution: int):
    """The network, the device it runs on, and the size the frame is read at.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the network reads at.

    Returns:
        ``(network, device, (height, width), (read_height, read_width))``.
    """
    backend = loaded.backend
    device = backend.load()
    network = backend.model
    height, width = int(image.shape[1]), int(image.shape[2])
    longest = max(height, width)
    edge = min(int(resolution), longest)
    scale = edge / float(longest)
    read = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    return network, device, (height, width), read


#: Shortest side each network will accept, by the menu name that reaches it.
SMALLEST_SIDE = {
    "HED Soft Edge": 16,
    "PiDiNet Soft Edge": 8,
    "TEED Soft Edge": 8,
}


def _planes(frame, device) -> torch.Tensor:
    """One frame as ``(1, 3, height, width)`` float32 on the device.

    Args:
        frame: ``(height, width, channels)`` in ``[0, 1]``, of any float dtype.
        device: Where the network runs.

    Returns:
        A float32 tensor, since a network built in float32 refuses any other.
    """
    return frame[..., :3].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)


def _at_least(size, smallest: int):
    """A size with neither side below ``smallest``."""
    return max(smallest, size[0]), max(smallest, size[1])


def _rounded(size, multiple: int):
    """A size taken up to the next whole multiple on both sides."""
    return (
        max(multiple, -(-size[0] // multiple) * multiple),
        max(multiple, -(-size[1] // multiple) * multiple),
    )


def lines(loaded, image, resolution: int) -> torch.Tensor:
    """Trace drawn lines with whichever lineart network was wired in.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the network reads at.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    import torch.nn.functional as functional

    import comfy.utils

    network, device, (height, width), read = _network(loaded, image, resolution)
    name = loaded.name
    multiple = 256 if name == "Lineart Anime" else 16 if name == "Manga Line" else 4
    read = _rounded(read, multiple)

    progress = comfy.utils.ProgressBar(len(image))
    traced = []
    for frame in image:
        planes = _planes(frame, device)
        planes = functional.interpolate(planes, size=read, mode="area")
        if name == "Lineart Anime":
            fed = planes * 2.0 - 1.0
        elif name == "Manga Line":
            weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
            fed = (planes * weights).sum(dim=1, keepdim=True) * 255.0
        else:
            fed = planes
        with torch.no_grad():
            answer = network(fed)
        answer = answer[0] if isinstance(answer, (list, tuple)) else answer
        if name == "Manga Line":
            answer = answer / 255.0
        elif name == "Lineart Anime":
            # This one ends in a hyperbolic tangent, so it answers -1 to 1 rather than 0 to 1.
            answer = (answer + 1.0) * 0.5
        grown = functional.interpolate(
            answer.float(), size=(height, width), mode="bilinear", align_corners=False
        )
        traced.append(((1.0 - grown.clamp(0.0, 1.0)) * 255.0).repeat(1, 3, 1, 1).cpu())
        progress.update(1)
    return torch.cat(traced, dim=0)


def line_segments(loaded, image, resolution: int, score: float, shortest: float):
    """Find the straight runs in each frame and draw them.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the network reads at.
        score: Confidence a centre must reach.
        shortest: Shortest segment kept, in centre-map samples.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    import torch.nn.functional as functional

    import comfy.utils

    from ..model import mlsd as mlsd_model

    network, device, (height, width), _read = _network(loaded, image, resolution)
    side = mlsd_model.INPUT_SIZE

    progress = comfy.utils.ProgressBar(len(image))
    drawn = []
    for frame in image:
        planes = _planes(frame, device)
        square = functional.interpolate(planes, size=(side, side), mode="area")
        # The network reads a fourth channel of ones beside the three colours.
        opaque = torch.ones_like(square[:, :1])
        fed = torch.cat([square, opaque], dim=1) * 2.0 - 1.0
        with torch.no_grad():
            answer = network(fed)
        found = drawing.decode(answer, score, shortest)
        drawn.append(
            drawing.draw(found, height, width, width / float(side), height / float(side)).cpu()
        )
        progress.update(1)
    return torch.cat(drawn, dim=0)


def _stepped(x: torch.Tensor, steps: int) -> torch.Tensor:
    """Quantise a 0 to 1 answer onto whole steps.

    Args:
        x: The answer, on a 0 to 1 scale.
        steps: How many steps the range is cut into.

    Returns:
        A tensor of the same shape, holding only the step values.
    """
    return torch.floor(x * float(steps + 1)) / float(steps)


def anyline(loaded, image, resolution: int, speck: float) -> torch.Tensor:
    """Merge a fine edge pass with a lineart pass, dropping the specks between them.

    Args:
        loaded: The built edge network and the name it came from.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge both passes read at.
        speck: Smallest run of connected samples kept, in samples.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale, dark lines on white.
    """
    import torch.nn.functional as functional

    import comfy.utils

    backend = loaded.backend
    device = backend.load()
    network = backend.model
    height, width = int(image.shape[1]), int(image.shape[2])
    longest = max(height, width)
    edge = min(int(resolution), longest)
    scale = edge / float(longest)
    read = _rounded(
        (max(1, int(round(height * scale))), max(1, int(round(width * scale)))), 8
    )

    progress = comfy.utils.ProgressBar(len(image))
    merged = []
    for frame in image:
        planes = _planes(frame, device) * 255.0
        small = functional.interpolate(planes, size=read, mode="area")
        with torch.no_grad():
            stages = network(small)
        grown = [
            functional.interpolate(s.float(), size=(height, width), mode="bilinear",
                                   align_corners=False)
            for s in stages
        ]
        fine = torch.sigmoid(torch.cat(grown, dim=1).mean(dim=1, keepdim=True))

        # The lineart half is traced at the settings AnyLine asks it for, not the node's.
        colour = _planes(frame, device) * 255.0
        strokes = preprocess.lineart_simple(colour, 2.0, 3)[:, :1] / 255.0
        strokes = 1.0 - strokes

        kept = _without_specks(strokes, float(speck))
        # Screened together, so whichever pass found a line keeps it.
        both = 1.0 - (1.0 - fine) * (1.0 - kept)
        merged.append(((1.0 - both.clamp(0.0, 1.0)) * 255.0).repeat(1, 3, 1, 1).cpu())
        progress.update(1)
    return torch.cat(merged, dim=0)


def _without_specks(x: torch.Tensor, smallest: float) -> torch.Tensor:
    """Drop every run of connected samples shorter than ``smallest``.

    Args:
        x: ``(1, 1, height, width)`` on a 0 to 1 scale.
        smallest: Fewest connected samples a run may hold and be kept.

    Returns:
        A tensor of the same shape with the short runs cleared.
    """
    from .. import deps

    if deps.optional("scipy") is None or smallest <= 1:
        return x
    from scipy.ndimage import label

    marked = (x[0, 0] > 0.1).cpu().numpy()
    labelled, found = label(marked)
    if not found:
        return x
    counts = torch.from_numpy(labelled.reshape(-1)).bincount()
    tiny = (counts < int(smallest)).numpy()
    tiny[0] = True
    keep = torch.from_numpy(~tiny[labelled]).to(x.device, dtype=x.dtype)
    return x * keep.unsqueeze(0).unsqueeze(0)


def intrinsics(loaded, image, resolution: int, steps: int, seed: int, channel: str):
    """Decompose a batch and answer one of the maps the model reads out of it.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived, holding the checkpoint.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the networks read at.
        steps: Denoising steps per frame.
        seed: Chooses between equally good answers.
        channel: One of the names in the model's entry of
            :data:`~modules.model.marigold.MAPS`.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.

    Raises:
        ValueError: The model does not answer ``channel``.
    """
    from . import intrinsic_maps

    return intrinsic_maps.answer(loaded, image, resolution, steps, seed, channel)


#: Side multiple each restoration network needs, by the module answering it. SCUNet pads and
#: crops on its own, so it takes any size.
RESTORE_MULTIPLE = {
    "nafnet": 8, "cidnet": 8, "retinexformer": 8, "darkir": 8, "scunet": 1,
}

#: Smallest tile worth cutting. Below this a tile carries too little of the picture for a
#: network to read, and anything under it is taken as tiling turned off.
SMALLEST_TILE = 128

#: Share of a tile that neighbouring tiles share.
TILE_OVERLAP = 4


def _held_to(network, multiple: int):
    """Wrap a network so any frame size reaches it, answering the size it was given.

    Args:
        network: The network, which needs both sides on a multiple.
        multiple: Side multiple the network reads.

    Returns:
        A callable taking and answering ``(1, 3, height, width)``.
    """
    import torch.nn.functional as functional

    def run(planes: torch.Tensor) -> torch.Tensor:
        height, width = int(planes.shape[-2]), int(planes.shape[-1])
        down, right = (-height) % multiple, (-width) % multiple
        if down or right:
            planes = functional.pad(planes, (0, right, 0, down), mode="replicate")
        return network(planes)[:, :, :height, :width]

    return run


def restore(loaded, image, family: str, tile: int = 0) -> torch.Tensor:
    """Run a restoration network over a batch at the frame's own size.

    Args:
        loaded: The ``PREPROCESSOR_MODEL`` that arrived, holding the network.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        family: Module the network came from, a key of :data:`RESTORE_MULTIPLE`.
        tile: Square edge each pass reads, or under :data:`SMALLEST_TILE` for whole frames.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    import comfy.utils

    from .tiled_upscale import tiled_upscale

    backend = loaded.backend
    device = backend.load()
    network = backend.model
    height, width = int(image.shape[1]), int(image.shape[2])
    multiple = RESTORE_MULTIPLE.get(family, 8)
    run = _held_to(network, multiple)
    planes = image[..., :3].permute(0, 3, 1, 2).to(dtype=torch.float32)

    edge = int(tile)
    if edge >= SMALLEST_TILE:
        overlap = max(multiple, (edge // TILE_OVERLAP) // multiple * multiple)
        across = len(range(0, max(1, width - edge), max(1, edge - overlap))) + 1
        down = len(range(0, max(1, height - edge), max(1, edge - overlap))) + 1
        answered = tiled_upscale(
            planes,
            run,
            tile_size=edge,
            overlap=overlap,
            output_device="cpu",
            pbar=comfy.utils.ProgressBar(len(image) * across * down),
            target_height=height,
            target_width=width,
            device=device,
        )
        return answered.clamp(0.0, 1.0) * 255.0

    progress = comfy.utils.ProgressBar(len(image))
    answered = []
    for frame in planes:
        with torch.no_grad():
            out = run(frame.unsqueeze(0).to(device))
        answered.append(out.clamp(0.0, 1.0).float().cpu())
        progress.update(1)
    return torch.cat(answered, dim=0) * 255.0
