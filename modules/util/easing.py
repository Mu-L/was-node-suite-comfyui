"""Easing curves mapping ``[0, 1]`` onto ``[0, 1]``.

Every curve satisfies ``f(0) == 0`` and ``f(1) == 1``. The ``back`` and ``elastic``
families overshoot in between and can return values outside the unit range.
"""

from __future__ import annotations

import math

__all__ = ["EASINGS", "EASING_NAMES", "ease", "ease_series"]

#: Overshoot constant for the ``back`` family, from the original Penner equations. 1.70158
#: is the value that overshoots by roughly 10 percent.
_BACK = 1.70158

#: The same constant scaled for the symmetric ``ease_in_out_back``, which runs each half of
#: the curve over half the distance and needs the overshoot scaled to match.
_BACK_INOUT = _BACK * 1.525

#: Period of the ``elastic`` oscillation, as a fraction of the curve.
_ELASTIC_PERIOD = 0.3

#: Angular frequency the elastic curves oscillate at, derived from :data:`_ELASTIC_PERIOD`.
_ELASTIC_OMEGA = (2 * math.pi) / _ELASTIC_PERIOD


def _linear(t: float) -> float:
    return t


def _in_sine(t: float) -> float:
    return 1 - math.cos((t * math.pi) / 2)


def _out_sine(t: float) -> float:
    return math.sin((t * math.pi) / 2)


def _in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2


def _in_quad(t: float) -> float:
    return t * t


def _out_quad(t: float) -> float:
    return 1 - (1 - t) ** 2


def _in_out_quad(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 2) / 2


def _in_cubic(t: float) -> float:
    return t**3


def _out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def _in_out_cubic(t: float) -> float:
    return 4 * t**3 if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def _in_quart(t: float) -> float:
    return t**4


def _out_quart(t: float) -> float:
    return 1 - (1 - t) ** 4


def _in_out_quart(t: float) -> float:
    return 8 * t**4 if t < 0.5 else 1 - ((-2 * t + 2) ** 4) / 2


def _in_quint(t: float) -> float:
    return t**5


def _out_quint(t: float) -> float:
    return 1 - (1 - t) ** 5


def _in_out_quint(t: float) -> float:
    return 16 * t**5 if t < 0.5 else 1 - ((-2 * t + 2) ** 5) / 2


def _in_expo(t: float) -> float:
    return 0.0 if t == 0 else 2 ** (10 * t - 10)


def _out_expo(t: float) -> float:
    return 1.0 if t == 1 else 1 - 2 ** (-10 * t)


def _in_out_expo(t: float) -> float:
    if t == 0 or t == 1:
        return float(t)
    return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2


def _in_circ(t: float) -> float:
    return 1 - math.sqrt(max(0.0, 1 - t * t))


def _out_circ(t: float) -> float:
    return math.sqrt(max(0.0, 1 - (t - 1) ** 2))


def _in_out_circ(t: float) -> float:
    if t < 0.5:
        return (1 - math.sqrt(max(0.0, 1 - (2 * t) ** 2))) / 2
    return (math.sqrt(max(0.0, 1 - (-2 * t + 2) ** 2)) + 1) / 2


def _in_back(t: float) -> float:
    return (_BACK + 1) * t**3 - _BACK * t**2


def _out_back(t: float) -> float:
    return 1 + (_BACK + 1) * (t - 1) ** 3 + _BACK * (t - 1) ** 2


def _in_out_back(t: float) -> float:
    if t < 0.5:
        return ((2 * t) ** 2 * ((_BACK_INOUT + 1) * 2 * t - _BACK_INOUT)) / 2
    return ((2 * t - 2) ** 2 * ((_BACK_INOUT + 1) * (t * 2 - 2) + _BACK_INOUT) + 2) / 2


def _in_elastic(t: float) -> float:
    if t == 0 or t == 1:
        return float(t)
    return -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * _ELASTIC_OMEGA)


def _out_elastic(t: float) -> float:
    if t == 0 or t == 1:
        return float(t)
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * _ELASTIC_OMEGA) + 1


def _in_out_elastic(t: float) -> float:
    if t == 0 or t == 1:
        return float(t)
    omega = (2 * math.pi) / 4.5
    if t < 0.5:
        return -(2 ** (20 * t - 10) * math.sin((20 * t - 11.125) * omega)) / 2
    return (2 ** (-20 * t + 10) * math.sin((20 * t - 11.125) * omega)) / 2 + 1


def _out_bounce(t: float) -> float:
    """The bounce curve, and the definition the other two bounce curves are built from.

    Args:
        t: Progress in ``[0, 1]``.

    Returns:
        The eased value, in ``[0, 1]``.
    """
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def _in_bounce(t: float) -> float:
    return 1 - _out_bounce(1 - t)


def _in_out_bounce(t: float) -> float:
    if t < 0.5:
        return (1 - _out_bounce(1 - 2 * t)) / 2
    return (1 + _out_bounce(2 * t - 1)) / 2


#: Curve name -> the function computing it. Ordered as it is presented to the user: the
#: identity first, then the four polynomial families from gentlest to steepest, then the
#: three that do something other than accelerate.
EASINGS = {
    "linear": _linear,
    "ease_in_sine": _in_sine,
    "ease_out_sine": _out_sine,
    "ease_in_out_sine": _in_out_sine,
    "ease_in_quad": _in_quad,
    "ease_out_quad": _out_quad,
    "ease_in_out_quad": _in_out_quad,
    "ease_in_cubic": _in_cubic,
    "ease_out_cubic": _out_cubic,
    "ease_in_out_cubic": _in_out_cubic,
    "ease_in_quart": _in_quart,
    "ease_out_quart": _out_quart,
    "ease_in_out_quart": _in_out_quart,
    "ease_in_quint": _in_quint,
    "ease_out_quint": _out_quint,
    "ease_in_out_quint": _in_out_quint,
    "ease_in_expo": _in_expo,
    "ease_out_expo": _out_expo,
    "ease_in_out_expo": _in_out_expo,
    "ease_in_circ": _in_circ,
    "ease_out_circ": _out_circ,
    "ease_in_out_circ": _in_out_circ,
    "ease_in_back": _in_back,
    "ease_out_back": _out_back,
    "ease_in_out_back": _in_out_back,
    "ease_in_elastic": _in_elastic,
    "ease_out_elastic": _out_elastic,
    "ease_in_out_elastic": _in_out_elastic,
    "ease_in_bounce": _in_bounce,
    "ease_out_bounce": _out_bounce,
    "ease_in_out_bounce": _in_out_bounce,
}

#: The curve names in presentation order, for a combo's ``options``.
EASING_NAMES = tuple(EASINGS)


def ease(name: str, t: float) -> float:
    """Apply one curve to a progress value.

    Args:
        name: A key of :data:`EASINGS`.
        t: Progress. Values outside ``[0, 1]`` are clamped before the curve is applied.

    Returns:
        The eased value. Within ``[0, 1]`` for every curve except the ``back`` and
        ``elastic`` families, which overshoot both ends by design.

    Raises:
        KeyError: ``name`` is not a known curve.
    """
    curve = EASINGS[name]
    return float(curve(min(1.0, max(0.0, float(t)))))


def ease_series(name: str, count: int, endpoint: bool = True) -> list[float]:
    """The curve sampled at evenly spaced points.

    Args:
        name: A key of :data:`EASINGS`.
        count: How many samples to take. A ``count`` of 1 yields ``[0.0]``, since one
            sample cannot describe a span.
        endpoint: Whether the last sample sits at ``t = 1``. ``False`` stops one step
            short, which is what a seamless loop needs: the final frame of the loop is
            the first frame of the next pass, so emitting both repeats it.

    Returns:
        ``count`` eased values.

    Raises:
        KeyError: ``name`` is not a known curve.
        ValueError: ``count`` is less than 1.
    """
    if count < 1:
        raise ValueError(f"count must be 1 or more, not {count}")
    if count == 1:
        return [0.0]
    divisor = (count - 1) if endpoint else count
    return [ease(name, step / divisor) for step in range(count)]
