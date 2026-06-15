from __future__ import annotations

import math


def sign(value: float) -> float:
    """Return the scalar sign used by reversible flow components."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def sqrt_or_nan(value: float) -> float:
    """Fast scalar square root with NumPy-like negative-input behavior."""
    return math.sqrt(value) if value >= 0.0 else math.nan


def isclose_numpy_default(a: float, b: float) -> bool:
    """Scalar equivalent of ``numpy.isclose`` with default tolerances."""
    return abs(a - b) <= (1e-8 + 1e-5 * abs(b))


def pressure_drop_flow_rate(
    pressure_drop: float,
    density: float,
    discharge_coefficient: float,
    area: float,
) -> float:
    """Return signed incompressible restriction flow for a pressure drop."""
    return (
        sign(pressure_drop)
        * discharge_coefficient
        * area
        * sqrt_or_nan(2.0 * density * abs(pressure_drop))
    )


def divide_or_nan(numerator: float, denominator: float) -> float:
    """Scalar divide with NumPy-like zero-division behavior."""
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return math.nan
    return math.copysign(math.inf, numerator)
