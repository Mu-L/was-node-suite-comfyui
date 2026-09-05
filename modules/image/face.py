"""Turning a face detection into the square crop a node answers with.

Frames are ``(height, width, 3)`` arrays of 8-bit RGB pixels. A window is ``(left, top,
right, bottom)`` and is always square and inside the frame.
"""

from __future__ import annotations

__all__ = [
    "CASCADES",
    "EMPTY_SIZE",
    "EYE_CASCADE",
    "MIN_FACE_SIZE",
    "crop_window",
    "largest",
    "try_order",
    "window",
]

import numpy as np

#: Classifier cascades a face node offers, in the order they are tried. The chosen one is
#: moved to the front and the rest stay as fallbacks, so a picture the selected classifier
#: misses can still be answered by another.
#:
#: :data:`EYE_CASCADE` is the exception: it finds an eye rather than a face, so it is
#: searched only when it is the selection.
CASCADES = (
    "lbpcascade_animeface.xml",
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt_tree.xml",
    "haarcascade_profileface.xml",
    "haarcascade_upperbody.xml",
    "haarcascade_eye.xml",
)

#: The one cascade that is not a face detector.
EYE_CASCADE = "haarcascade_eye.xml"

#: Shortest side, in pixels, the cropped face is resized up to.
MIN_FACE_SIZE = 64

#: Side in pixels of the black frame answered when no face is found.
EMPTY_SIZE = 512


def try_order(cascade_name: str | None) -> list[str]:
    """The cascades to try, in order, for one selection.

    Args:
        cascade_name: File name of the classifier to try first, one of :data:`CASCADES`.
            Anything else leaves the default order in place.

    Returns:
        File names, the chosen one first. The eye cascade appears only when it is the
        choice.
    """
    order = [name for name in CASCADES if name != EYE_CASCADE or name == cascade_name]
    if cascade_name in order:
        order.remove(cascade_name)
        order.insert(0, cascade_name)
    return order


def largest(faces):
    """The biggest of several detections.

    Args:
        faces: ``(x, y, width, height)`` rows.

    Returns:
        The row covering the most pixels. Detectors list their hits in no dependable order,
        so this is what keeps two runs on one picture cropping the same thing.
    """
    return max(faces, key=lambda face: int(face[2]) * int(face[3]))


def window(img, face, padding: float = 0.25):
    """The padded square to cut around one detection.

    Args:
        img: Source pixels as an ``RGB`` array shaped ``(height, width, 3)``.
        face: ``(x, y, width, height)`` of the detection.
        padding: Margin around the detection as a fraction of the face size.

    Returns:
        ``(left, top, right, bottom)``, square and inside the image.
    """
    x, y, w, h = face

    # Trim a detection that overhangs an edge down to the part inside the image.
    left_adjust = max(0, -x)
    right_adjust = max(0, x + w - img.shape[1])
    top_adjust = max(0, -y)
    bottom_adjust = max(0, y + h - img.shape[0])

    x += left_adjust
    y += top_adjust
    w -= left_adjust + right_adjust
    h -= top_adjust + bottom_adjust

    face_size = min(h, w)
    center_x = x + w // 2
    center_y = y + h // 2
    half_size = (face_size + int(face_size * padding)) // 2
    top = max(0, center_y - half_size)
    bottom = min(img.shape[0], center_y + half_size)
    left = max(0, center_x - half_size)
    right = min(img.shape[1], center_x + half_size)

    # Whichever axis was clipped by the image edge decides the side length, so the window
    # stays square.
    half_crop = min(right - left, bottom - top) // 2

    # The centre moves off the face rather than the window leaving the image.
    center_x = min(max(center_x, half_crop), img.shape[1] - half_crop)
    center_y = min(max(center_y, half_crop), img.shape[0] - half_crop)

    return (
        center_x - half_crop,
        center_y - half_crop,
        center_x + half_crop,
        center_y + half_crop,
    )


def crop_window(img, box):
    """Cut one square out of one image and square it up.

    Args:
        img: Source pixels as an ``RGB`` array shaped ``(height, width, 3)``.
        box: ``(left, top, right, bottom)`` as :func:`window` answers it.

    Returns:
        An ``RGB`` image of the squared crop, at least :data:`MIN_FACE_SIZE` on a side.
    """
    from PIL import Image

    left, top, right, bottom = box
    face_img = img[top:bottom, left:right, :]

    size = max(face_img.shape[:2])
    pad_h = (size - face_img.shape[0]) // 2
    pad_w = (size - face_img.shape[1]) // 2
    face_img = np.pad(face_img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

    square = Image.fromarray(face_img).convert("RGB")
    return square.resize((max(size, MIN_FACE_SIZE),) * 2, Image.BILINEAR)
