"""
Steady-state Lookup callable example: concrete column sizing.

Physical layout
---------------

        Building load
             |
             v
        +----------+
        |          |
        | Concrete |
        |  column  |
        |          |
        +----------+
             |
             v
        Foundation

This example demonstrates that FullFlow's Lookup component is not limited to
ThermoProp objects. Any callable can be wrapped by Lookup.

Here, a plain Python function estimates concrete properties from curing
conditions:

    concrete_properties(curing_days, curing_temperature, water_cement_ratio)

The steady-state solver then sizes the column cross-sectional area so that:

    applied_stress = allowable_stress

No custom FullFlow component is required.
"""

from types import SimpleNamespace

import math

from fullflow import *


# -----------------------------------------------------------------------------
# User-defined callable lookup
# -----------------------------------------------------------------------------

def concrete_properties(curing_days, curing_temperature, water_cement_ratio):
    """
    Estimate simple concrete design properties.

    This is intentionally an arbitrary engineering-style correlation, not a
    code-approved structural design model. It is just a compact example of a
    user-defined callable that maps inputs to outputs.
    """

    reference_strength = 40.0e6

    age_factor = 1.0 - math.exp(-curing_days / 10.0)
    temperature_factor = 1.0 + 0.015 * (curing_temperature - 293.15)
    water_cement_factor = 0.45 / water_cement_ratio

    compressive_strength = reference_strength * age_factor * temperature_factor * water_cement_factor
    compressive_strength = max(compressive_strength, 1.0e6)

    compressive_strength_mpa = compressive_strength / 1.0e6
    elastic_modulus = 4700.0 * math.sqrt(compressive_strength_mpa) * 1.0e6

    allowable_stress = 0.35 * compressive_strength

    return SimpleNamespace(
        compressive_strength=compressive_strength,
        allowable_stress=allowable_stress,
        elastic_modulus=elastic_modulus,
    )


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

ColumnDesign = Network("Concrete Column Design")


# -----------------------------------------------------------------------------
# Inputs and solve variable
# -----------------------------------------------------------------------------

AppliedLoad = State(2.0e6)             # N
CuringDays = State(14.0)               # days
CuringTemperature = State(293.15)      # K
WaterCementRatio = State(0.45)         # -

ColumnArea = State(0.20, bounds=(0.01, 5.0))


# -----------------------------------------------------------------------------
# Callable Lookup
# -----------------------------------------------------------------------------

Concrete = Lookup(
    "Concrete Properties",
    ColumnDesign,
    concrete_properties,
    curing_days=CuringDays,
    curing_temperature=CuringTemperature,
    water_cement_ratio=WaterCementRatio,
)


# -----------------------------------------------------------------------------
# Design equations
# -----------------------------------------------------------------------------

AppliedStress = AppliedLoad / ColumnArea
ColumnDiameter = (4.0 * ColumnArea / math.pi) ** 0.5

Balance(
    "Column Area Balance",
    ColumnDesign,
    variable=ColumnArea,
    function=AppliedStress - Concrete.allowable_stress,
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

ColumnDesign.track("Applied Load [N]", AppliedLoad)
ColumnDesign.track("Curing Days [day]", CuringDays)
ColumnDesign.track("Curing Temperature [K]", CuringTemperature)
ColumnDesign.track("Water Cement Ratio [-]", WaterCementRatio)
ColumnDesign.track("Concrete Compressive Strength [Pa]", Concrete.compressive_strength)
ColumnDesign.track("Concrete Allowable Stress [Pa]", Concrete.allowable_stress)
ColumnDesign.track("Concrete Elastic Modulus [Pa]", Concrete.elastic_modulus)
ColumnDesign.track("Required Column Area [m2]", ColumnArea)
ColumnDesign.track("Equivalent Round Column Diameter [m]", ColumnDiameter)
ColumnDesign.track("Applied Stress [Pa]", AppliedStress)


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------

filename = "concrete_column"

SteadyState(ColumnDesign).solve(
    verbose=True,
    filename=filename,
)