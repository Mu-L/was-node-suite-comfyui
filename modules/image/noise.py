"""Procedural noise generators.

A classic Perlin sum-of-octaves field, a power-fractal variant of it, and a Worley
(cellular) field. The two Perlin generators evaluate one octave at a time with torch, in
float64.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

from .. import deps

__all__ = ["WorleyNoise", "perlin_noise", "perlin_power_fractal"]


def perlin_noise(
    width: int,
    height: int,
    octaves: int,
    persistence: float,
    scale: float,
    seed: int | None = None,
) -> Image.Image:
    """Render fractal Perlin noise as a greyscale image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        octaves: Number of frequencies summed. Zero sums nothing and renders black.
        persistence: Amplitude ratio between one octave and the next.
        scale: Pixels per unit of noise space at octave 0. Larger is smoother.
        seed: Seeds the ``random`` module before the permutation table is shuffled, 0
            included. ``None`` leaves the shared state alone.

    Returns:
        An ``RGB`` image of ``(width, height)`` holding a greyscale field.
    """
    device = deps.float64_device()
    p = _permutation(seed, device)
    plane = torch.zeros((), dtype=torch.float64, device=device)

    noise_map = torch.zeros((height, width), dtype=torch.float64, device=device)
    amplitude = 1.0

    for octave in range(octaves):
        frequency = 2 ** octave
        x, y = _coordinates(width, height, scale, frequency, device)
        noise_map = noise_map + _noise(x, y, plane, p) * amplitude
        amplitude *= persistence

    return _render(noise_map)


def perlin_power_fractal(
    width: int,
    height: int,
    octaves: int,
    persistence: float,
    lacunarity: float,
    exponent: float,
    scale: float,
    seed: int | None = None,
) -> Image.Image:
    """Render a Perlin power fractal as a greyscale image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        octaves: Number of frequencies summed. Zero sums nothing and renders black.
        persistence: Amplitude ratio between one octave and the next.
        lacunarity: Frequency ratio between one octave and the next.
        exponent: Power applied to each octave's amplitude before it is summed.
        scale: Pixels per unit of noise space at octave 0. Larger is smoother.
        seed: Seeds the ``random`` module before the permutation table is shuffled, 0
            included. ``None`` leaves the shared state alone.

    Returns:
        An ``RGB`` image of ``(width, height)`` holding a greyscale field.
    """
    device = deps.float64_device()
    p = _permutation(seed, device)
    plane = torch.zeros((), dtype=torch.float64, device=device)

    noise_map = torch.zeros((height, width), dtype=torch.float64, device=device)
    amplitude = 1.0

    for octave in range(octaves):
        frequency = lacunarity ** octave
        amplitude *= persistence
        x, y = _coordinates(width, height, scale, frequency, device)
        noise_map = noise_map + _noise(x, y, plane, p) * amplitude ** exponent

    return _render(noise_map)


class WorleyNoise:
    """Worley (cellular) noise over a random point set, drawn and rendered during construction.

    Attributes:
        points: ``(density, 2)`` array of point coordinates, both axes drawn from
            ``[0, width)``. On a non-square canvas the y coordinates are therefore drawn
            from the width, not the height.
        colors: ``(density, 3)`` array of RGB colours, one per point, used by flat mode.
        data: ``(height, width)`` array of distances to the ``option``-th nearest point.
        image: The rendered image.
    """

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        density: int = 50,
        option: int = 0,
        use_broadcast_ops: bool = True,
        flat: bool = False,
        seed: int | None = None,
    ):
        """Draw the points and render the field.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.
            density: Number of feature points.
            option: Which nearest neighbour supplies the distance. 0 is the nearest
                point, 1 the second nearest, and so on; it indexes the sorted distance
                array, so it must be less than ``density``.
            use_broadcast_ops: Stored on the instance and not read.
            flat: Render flat Voronoi cells rather than a distance field.
            seed: Seed for the point and colour draw. ``None`` draws from entropy.
        """
        self.height = height
        self.width = width
        self.density = density
        self.use_broadcast_ops = use_broadcast_ops
        self.seed = seed
        self.generate_points_and_colors()
        self.calculate_noise(option)
        self.image = self.generateImage(option, flat_mode=flat)

    def generate_points_and_colors(self) -> None:
        """Draw the feature points and their colours, setting ``points`` and ``colors``."""
        rng = np.random.default_rng(self.seed)
        self.points = rng.integers(0, self.width, (self.density, 2))
        self.colors = rng.integers(0, 256, (self.density, 3))

    def calculate_noise(self, option: int) -> None:
        """Fill ``data`` with the distance from each pixel to its ``option``-th neighbour.

        Args:
            option: Index into the per-pixel sorted distance array.
        """
        self.data = np.zeros((self.height, self.width))
        for h in range(self.height):
            for w in range(self.width):
                distances = np.sqrt(np.sum((self.points - np.array([w, h])) ** 2, axis=1))
                self.data[h, w] = np.sort(distances)[option]

    def generateImage(self, option: int, flat_mode: bool = False) -> Image.Image:
        """Render the noise.

        Args:
            option: Accepted and unused; flat mode reads the nearest point directly and
                distance mode reads the field :meth:`calculate_noise` already built.
            flat_mode: Fill each Voronoi cell with its point's colour instead of
                rendering the distance field.

        Returns:
            An ``RGB`` image in flat mode, otherwise an ``L`` image holding the distance
            field stretched to fill 0-255.
        """
        if flat_mode:
            flat_color_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for h in range(self.height):
                for w in range(self.width):
                    closest_point_idx = np.argmin(np.sum((self.points - np.array([w, h])) ** 2, axis=1))
                    flat_color_data[h, w, :] = self.colors[closest_point_idx]
            return Image.fromarray(flat_color_data, 'RGB')
        else:
            min_val, max_val = np.min(self.data), np.max(self.data)
            data_scaled = (self.data - min_val) / (max_val - min_val) * 255
            data_scaled = data_scaled.astype(np.uint8)
            return Image.fromarray(data_scaled, 'L')


def _permutation(seed: int | None, device) -> torch.Tensor:
    """Build the gradient lookup table the Perlin generators index.

    Args:
        seed: Seeds the ``random`` module before the shuffle, 0 included, so one seed
            always builds one table. ``None`` leaves the shared state alone, which is what
            a caller with no seed of its own asks for.
        device: Device the table is built on.

    Returns:
        A 512-entry int64 tensor: a shuffled permutation of 0-255 followed by itself, so
        the sum of two table entries indexes it without wrapping.
    """
    if seed is not None:
        random.seed(seed)

    p = np.arange(256, dtype=np.int32)
    random.shuffle(p)
    return torch.as_tensor(np.concatenate((p, p)), dtype=torch.int64, device=device)


def _coordinates(width: int, height: int, scale: float, frequency: float, device):
    """Noise-space coordinates of one octave's grid.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Pixels per unit of noise space at frequency 1.
        frequency: Frequency multiplier for this octave.
        device: Device the vectors are built on.

    Returns:
        ``(x, y)`` as float64 tensors shaped ``(1, width)`` and ``(height, 1)``, each
        value being ``pixel / scale * frequency``. An expression over both broadcasts to
        the whole ``(height, width)`` grid.
    """
    x = torch.arange(width, dtype=torch.float64, device=device) / scale * frequency
    y = torch.arange(height, dtype=torch.float64, device=device) / scale * frequency
    return x.unsqueeze(0), y.unsqueeze(1)


def _fade(t: torch.Tensor) -> torch.Tensor:
    """Perlin's quintic ease curve, ``6t^5 - 15t^4 + 10t^3``.

    Args:
        t: Position within a cell, in ``[0, 1)``.

    Returns:
        The eased position, flat at both ends of the interval.
    """
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _lerp(t: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Interpolate from ``a`` to ``b`` by ``t``.

    Args:
        t: Interpolation weight, 0 selecting ``a`` and 1 selecting ``b``.
        a: Value at ``t`` of 0.
        b: Value at ``t`` of 1.

    Returns:
        ``a + t * (b - a)``.
    """
    return a + t * (b - a)


def _grad(hashed: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Dot the offset from a cell corner with the gradient that corner's hash selects.

    Args:
        hashed: Permutation table entry for the corner.
        x: Offset from the corner along x.
        y: Offset from the corner along y.
        z: Offset from the corner along z.

    Returns:
        The dot product, broadcast over the shapes of the arguments.
    """
    h = hashed & 15
    u = torch.where(h < 8, x, y)
    v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
    return torch.where((h & 1) == 0, u, -u) + torch.where((h & 2) == 0, v, -v)


def _noise(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Sample Perlin's improved gradient noise over a whole grid at once.

    Args:
        x: Noise-space x coordinates, broadcastable against ``y`` and ``z``.
        y: Noise-space y coordinates.
        z: Noise-space z coordinates.
        p: 512-entry permutation table from :func:`_permutation`.

    Returns:
        A float64 tensor of the broadcast shape, holding values in roughly ``[-1, 1]``.
    """
    X = torch.floor(x).to(torch.int64) & 255
    Y = torch.floor(y).to(torch.int64) & 255
    Z = torch.floor(z).to(torch.int64) & 255

    x = x - torch.floor(x)
    y = y - torch.floor(y)
    z = z - torch.floor(z)

    u = _fade(x)
    v = _fade(y)
    w = _fade(z)

    A = p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B = p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z

    # float64 and this nesting order are the per-pixel evaluation order the field is
    # defined by; reassociating the interpolations or narrowing the dtype changes it.
    return _lerp(w, _lerp(v, _lerp(u, _grad(p[AA], x, y, z), _grad(p[BA], x - 1, y, z)),
                             _lerp(u, _grad(p[AB], x, y - 1, z), _grad(p[BB], x - 1, y - 1, z))),
                 _lerp(v, _lerp(u, _grad(p[AA + 1], x, y, z - 1), _grad(p[BA + 1], x - 1, y, z - 1)),
                          _lerp(u, _grad(p[AB + 1], x, y - 1, z - 1), _grad(p[BB + 1], x - 1, y - 1, z - 1))))


def _render(noise_map: torch.Tensor) -> Image.Image:
    """Stretch a noise field to fill 0-255 and render it as greyscale.

    Args:
        noise_map: Float64 field of any range, shaped ``(height, width)``.

    Returns:
        An ``RGB`` image of the field. A field with no range to stretch, which is what
        zero octaves produces, renders black.
    """
    noise_map = noise_map.cpu().numpy()
    min_value = np.min(noise_map)
    max_value = np.max(noise_map)
    noise_map = np.interp(noise_map, (min_value, max_value), (0, 255)).astype(np.uint8)
    return Image.fromarray(noise_map, mode='L').convert("RGB")
