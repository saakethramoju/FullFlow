"""
COPV blowdown into a high-pressure downstream receiver.

Physical layout
---------------

    High-pressure GN2 COPV
    6000 psia, 300 K
    20 L
          |
          |  0.075 in valve/orifice
          v
    Fixed downstream receiver
    3000 psia


Model purpose
-------------
This example demonstrates a simple high-pressure COPV blowdown using an
ideal-gas pressurant model and a compressible orifice.

The downstream boundary is intentionally set to 3000 psia instead of
atmospheric pressure. This keeps the ideal-gas nitrogen state inside the
property-valid temperature range while still showing the valve transition
from choked flow to unchoked flow.

For GN2, the critical pressure ratio is roughly 0.53. Therefore, with a
3000 psia downstream boundary, the valve starts choked at 6000 psia and
becomes unchoked once the COPV pressure falls below roughly:

    3000 psia / 0.53 ≈ 5650 psia

Assumptions
-----------
- Nitrogen is modeled as an ideal gas.
- The COPV is a lumped, well-mixed volume.
- The tank blowdown is adiabatic.
- The downstream receiver pressure is fixed.
- The valve is modeled as a compressible orifice.
- Wall heat transfer, real-gas effects, and two-phase behavior are neglected.
"""

import numpy as np

from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

COPVBlowdown = Network("COPV Blowdown")


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

psi_to_pa = 6894.76


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

initial_copv_pressure = 6000 * psi_to_pa
initial_copv_temperature = 300

downstream_pressure = 3000 * psi_to_pa


# ---------------------------------------------------------------------------
# Pressurant properties
# ---------------------------------------------------------------------------
# The Lookup component creates an IdealGas object and exposes its properties as
# FullFlow states. As the COPV pressure and temperature change, this lookup
# updates density, internal energy, enthalpy, gas constant, and gamma.

Pressurant = Lookup(
    "Pressurant Gas",
    COPVBlowdown,
    IdealGas,
    "gn2",
    pressure=initial_copv_pressure,
    temperature=initial_copv_temperature,
)


# ---------------------------------------------------------------------------
# COPV control volume
# ---------------------------------------------------------------------------
# This is the storage element. It owns the transient mass and energy balances.
#
# pressure, temperature, density, internal_energy, and enthalpy are connected
# to the Pressurant lookup so the gas properties update with the tank state.
#
# energy_variable="T" means the transient energy balance solves directly for
# tank temperature.

COPV = Volume(
    "COPV",
    COPVBlowdown,
    volume=20 / 1000,
    pressure=Pressurant.pressure,
    temperature=Pressurant.temperature,
    density=Pressurant.density,
    internal_energy=Pressurant.internal_energy,
    enthalpy=Pressurant.enthalpy,
    energy_variable="T",
)


# ---------------------------------------------------------------------------
# Valve geometry
# ---------------------------------------------------------------------------
# The valve is represented as a circular compressible orifice.

valve_diameter = 0.075 / 39.37
valve_area = (np.pi / 4) * valve_diameter**2


# ---------------------------------------------------------------------------
# Compressible valve/orifice
# ---------------------------------------------------------------------------
# Positive mass flow is from the COPV to the downstream receiver.
#
# The upstream total pressure and temperature are taken from the COPV. Since the
# COPV is a lumped tank with negligible bulk velocity, the tank static and total
# temperatures are effectively the same for this simple model.
#
# The valve mass flow is connected to COPV.mass_flow_out, so the volume loses
# mass according to the orifice equation.

Valve = CompressibleOrifice(
    "Valve",
    COPVBlowdown,
    upstream_total_pressure=COPV.pressure,
    upstream_total_temperature=COPV.temperature,
    downstream_pressure=downstream_pressure,
    discharge_coefficient=1,
    cross_sectional_area=valve_area,
    gas_constant=Pressurant.gas_constant,
    specific_heat_ratio=Pressurant.specific_heat_ratio,
    upstream_static_enthalpy=Pressurant.enthalpy,
    upstream_static_temperature=COPV.temperature,
    mass_flow=COPV.mass_flow_out,
)


# ---------------------------------------------------------------------------
# Transient solve
# ---------------------------------------------------------------------------
# This runs a fixed-step transient simulation and exports the tracked solution
# history to COPVBlowdown.h5.

Transient(COPVBlowdown).solve(
    dt=0.01,
    t_final=30.0,
    filename="COPVBlowdown",
    verbose=True,
)