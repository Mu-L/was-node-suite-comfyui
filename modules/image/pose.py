"""Drawing a body, whole-body or animal pose as an OpenPose ControlNet reads it.

Keypoints are ``(subject, joint, 2)`` in pixels with a ``(subject, joint)`` score beside
them, in COCO, COCO-WholeBody or AP-10K order.
"""

from __future__ import annotations

import torch

__all__ = [
    "ANIMAL_COLOURS", "ANIMAL_EDGES", "ANIMAL_JOINTS", "COCO_TO_OPENPOSE", "COLOURS",
    "FACE", "FEET", "HANDS", "HAND_EDGES", "JOINTS", "LIMBS", "draw", "draw_animal",
    "draw_wholebody", "to_openpose",
]

#: The 18 joints in OpenPose order, which is the order the limbs and colours below index.
JOINTS = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
)

#: OpenPose joint index -> the COCO joint that fills it, or None where it is derived. COCO
#: has no neck, so it is the midpoint of the two shoulders.
COCO_TO_OPENPOSE = (
    0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3,
)

#: Joint pairs each limb runs between, as indices into :data:`JOINTS`.
LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17),
)

#: One colour per joint, and the first 17 double as the limb colours.
COLOURS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
)

#: Half-width of a limb in pixels before it is scaled for the frame.
STICK_WIDTH = 4

#: Radius of a joint in pixels before it is scaled for the frame.
JOINT_RADIUS = 4

#: Share of a limb's colour it is filled at, so a joint drawn over it still reads.
LIMB_SHADE = 0.6

#: Score a joint has to reach to be drawn at all.
MIN_SCORE = 0.3


def to_openpose(keypoints: torch.Tensor, scores: torch.Tensor):
    """Rearrange COCO joints into OpenPose order, deriving the neck.

    Args:
        keypoints: ``(person, 17, 2)`` in pixels.
        scores: ``(person, 17)``.

    Returns:
        ``(points, confidence)`` shaped ``(person, 18, 2)`` and ``(person, 18)``.
    """
    people = keypoints.shape[0]
    points = keypoints.new_zeros((people, len(JOINTS), 2))
    confidence = scores.new_zeros((people, len(JOINTS)))
    for target, source in enumerate(COCO_TO_OPENPOSE):
        if source is None:
            continue
        points[:, target] = keypoints[:, source]
        confidence[:, target] = scores[:, source]
    # The neck sits between the shoulders, and is only as certain as the weaker of them.
    left, right = 5, 6
    points[:, 1] = (keypoints[:, left] + keypoints[:, right]) * 0.5
    confidence[:, 1] = torch.minimum(scores[:, left], scores[:, right])
    return points, confidence


def draw(keypoints: torch.Tensor, scores: torch.Tensor, height: int, width: int):
    """Draw every person's skeleton on black, in OpenPose's own colours.

    Args:
        keypoints: ``(person, 17, 2)`` COCO joints in pixels.
        scores: ``(person, 17)`` confidence per joint.
        height: Canvas height in pixels.
        width: Canvas width in pixels.

    Returns:
        A ``(1, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    device = keypoints.device
    canvas = torch.zeros((3, height, width), dtype=torch.float32, device=device)
    if keypoints.numel() == 0:
        return canvas.unsqueeze(0)

    points, confidence = to_openpose(keypoints.float(), scores.float())
    scale = _scale(height, width)
    rows = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
    columns = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)

    for person in range(points.shape[0]):
        for limb, (first, second) in enumerate(LIMBS):
            if min(confidence[person, first], confidence[person, second]) < MIN_SCORE:
                continue
            shade = [c * LIMB_SHADE for c in COLOURS[limb]]
            _fill(canvas, _limb_mask(
                points[person, first], points[person, second],
                STICK_WIDTH * scale, rows, columns,
            ), shade)
        for joint in range(len(JOINTS)):
            if confidence[person, joint] < MIN_SCORE:
                continue
            _fill(canvas, _disc(
                points[person, joint], JOINT_RADIUS * scale, rows, columns,
            ), COLOURS[joint])
    return canvas.unsqueeze(0)


def _scale(height: int, width: int) -> float:
    """How much wider a limb is drawn on a large frame than on a small one."""
    longest = max(height, width)
    return 1.0 if longest < 500 else float(min(2 + longest // 1000, 7))


def _disc(centre: torch.Tensor, radius: float, rows, columns) -> torch.Tensor:
    """A filled circle as a boolean mask the size of the canvas."""
    dx = columns - centre[0]
    dy = rows - centre[1]
    return (dx * dx + dy * dy) <= radius * radius


def _limb_mask(start, end, half_width: float, rows, columns) -> torch.Tensor:
    """A filled ellipse spanning two joints, as a boolean mask the size of the canvas."""
    middle_x = (start[0] + end[0]) * 0.5
    middle_y = (start[1] + end[1]) * 0.5
    span_x = end[0] - start[0]
    span_y = end[1] - start[1]
    length = torch.sqrt(span_x * span_x + span_y * span_y).clamp_min(1e-6)
    along_x, along_y = span_x / length, span_y / length
    dx = columns - middle_x
    dy = rows - middle_y
    # Distance along the limb and across it, which is the ellipse in its own frame.
    along = dx * along_x + dy * along_y
    across = -dx * along_y + dy * along_x
    semi = (length * 0.5).clamp_min(1e-6)
    return (along / semi) ** 2 + (across / max(half_width, 1e-6)) ** 2 <= 1.0


def _fill(canvas: torch.Tensor, mask: torch.Tensor, colour) -> None:
    """Paint one colour over every sample a mask selects."""
    for channel, level in enumerate(colour):
        canvas[channel] = torch.where(
            mask, torch.tensor(float(level), device=canvas.device), canvas[channel]
        )


#: Where each part of a COCO-WholeBody answer starts, as ``(first, count)`` pairs.
FEET = (17, 6)
FACE = (23, 68)
HANDS = ((91, 21), (112, 21))

#: Ankle and the three foot points it carries, in COCO-WholeBody indices.
FOOT_EDGES = ((15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22))

#: Colour every foot edge is drawn in.
FOOT_COLOUR = (255, 255, 255)

#: Colour every face point is drawn in.
FACE_COLOUR = (255, 255, 255)

#: Radius of a face point in pixels before it is scaled for the frame.
FACE_RADIUS = 1.5

#: Point pairs each finger bone runs between, as indices into one hand's 21 points.
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
)

#: Half-width of a finger bone in pixels before it is scaled for the frame.
HAND_WIDTH = 1.0

#: Radius of a hand point in pixels before it is scaled for the frame.
HAND_RADIUS = 2.0

#: Colour every hand point is drawn in.
HAND_COLOUR = (0, 0, 255)

#: The 17 AP-10K joints in order, which the edges below index.
ANIMAL_JOINTS = (
    "left_eye", "right_eye", "nose", "neck", "tail_root",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_back_paw",
    "right_hip", "right_knee", "right_back_paw",
)

#: Joint pairs each AP-10K link runs between, with the colour it is drawn in.
ANIMAL_EDGES = (
    ((0, 1), (0, 0, 255)), ((0, 2), (0, 0, 255)), ((1, 2), (0, 0, 255)),
    ((2, 3), (0, 255, 0)), ((3, 4), (0, 255, 0)),
    ((3, 5), (0, 255, 255)), ((5, 6), (0, 255, 255)), ((6, 7), (0, 255, 255)),
    ((3, 8), (6, 156, 250)), ((8, 9), (6, 156, 250)), ((9, 10), (6, 156, 250)),
    ((4, 11), (0, 255, 255)), ((11, 12), (0, 255, 255)), ((12, 13), (0, 255, 255)),
    ((4, 14), (6, 156, 250)), ((14, 15), (6, 156, 250)), ((15, 16), (6, 156, 250)),
)

#: One colour per AP-10K joint, grouped as head, spine, left limbs and right limbs.
ANIMAL_COLOURS = (
    (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (0, 255, 0),
    (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (6, 156, 250), (6, 156, 250), (6, 156, 250),
    (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (6, 156, 250), (6, 156, 250), (6, 156, 250),
)


def draw_wholebody(keypoints: torch.Tensor, scores: torch.Tensor, height: int, width: int):
    """Draw every person's body, feet, face and hands on black.

    Args:
        keypoints: ``(person, 133, 2)`` COCO-WholeBody points in pixels.
        scores: ``(person, 133)`` confidence per point.
        height: Canvas height in pixels.
        width: Canvas width in pixels.

    Returns:
        A ``(1, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    drawn = draw(keypoints[:, :17], scores[:, :17], height, width)
    canvas = drawn[0]
    if keypoints.numel() == 0:
        return canvas.unsqueeze(0)

    device = keypoints.device
    points, confidence = keypoints.float(), scores.float()
    scale = _scale(height, width)
    rows = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
    columns = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)
    ring = _hue_ring(len(HAND_EDGES))

    for person in range(points.shape[0]):
        for first, second in FOOT_EDGES:
            if min(confidence[person, first], confidence[person, second]) < MIN_SCORE:
                continue
            _fill(canvas, _limb_mask(
                points[person, first], points[person, second],
                STICK_WIDTH * scale * 0.5, rows, columns,
            ), [level * LIMB_SHADE for level in FOOT_COLOUR])

        start, count = FACE
        for point in range(start, start + count):
            if confidence[person, point] < MIN_SCORE:
                continue
            _fill(canvas, _disc(
                points[person, point], FACE_RADIUS * scale, rows, columns,
            ), FACE_COLOUR)

        for base, size in HANDS:
            for edge, (first, second) in enumerate(HAND_EDGES):
                one, two = base + first, base + second
                if min(confidence[person, one], confidence[person, two]) < MIN_SCORE:
                    continue
                _fill(canvas, _limb_mask(
                    points[person, one], points[person, two],
                    HAND_WIDTH * scale, rows, columns,
                ), ring[edge])
            for point in range(base, base + size):
                if confidence[person, point] < MIN_SCORE:
                    continue
                _fill(canvas, _disc(
                    points[person, point], HAND_RADIUS * scale, rows, columns,
                ), HAND_COLOUR)
    return canvas.unsqueeze(0)


def draw_animal(keypoints: torch.Tensor, scores: torch.Tensor, height: int, width: int):
    """Draw every animal's skeleton on black, in the AP-10K colours.

    Args:
        keypoints: ``(animal, 17, 2)`` AP-10K joints in pixels.
        scores: ``(animal, 17)`` confidence per joint.
        height: Canvas height in pixels.
        width: Canvas width in pixels.

    Returns:
        A ``(1, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    device = keypoints.device
    canvas = torch.zeros((3, height, width), dtype=torch.float32, device=device)
    if keypoints.numel() == 0:
        return canvas.unsqueeze(0)

    points, confidence = keypoints.float(), scores.float()
    scale = _scale(height, width)
    rows = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
    columns = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)

    for animal in range(points.shape[0]):
        for (first, second), colour in ANIMAL_EDGES:
            if min(confidence[animal, first], confidence[animal, second]) < MIN_SCORE:
                continue
            _fill(canvas, _limb_mask(
                points[animal, first], points[animal, second],
                STICK_WIDTH * scale, rows, columns,
            ), [level * LIMB_SHADE for level in colour])
        for joint in range(len(ANIMAL_JOINTS)):
            if confidence[animal, joint] < MIN_SCORE:
                continue
            _fill(canvas, _disc(
                points[animal, joint], JOINT_RADIUS * scale, rows, columns,
            ), ANIMAL_COLOURS[joint])
    return canvas.unsqueeze(0)


def _hue_ring(count: int) -> tuple:
    """``count`` colours spaced evenly round the hue circle, fully saturated."""
    colours = []
    for index in range(count):
        sixth, offset = divmod(index * 6.0 / count, 1.0)
        rising, falling = offset, 1.0 - offset
        wheel = (
            (1.0, rising, 0.0), (falling, 1.0, 0.0), (0.0, 1.0, rising),
            (0.0, falling, 1.0), (rising, 0.0, 1.0), (1.0, 0.0, falling),
        )[int(sixth) % 6]
        colours.append(tuple(level * 255.0 for level in wheel))
    return tuple(colours)
