"""
Transient N2/O2/argon mixer example with inertial flow tubes.

Physical layout
---------------

        N2/O2 source                       Argon source
        fixed pressure                     fixed pressure
        N2/O2 mixture                      pure Ar
              |                                |
              |                                |
              v                                v
      +----------------+              +----------------+
      | N2/O2 flowtube |              | Argon flowtube |
      | with inertia   |              | with inertia   |
      +----------------+              | area opens     |
              |                       | during transient
              |                       +----------------+
              |                                |
              +---------------+----------------+
                              |
                              v
                    +-------------------+
                    | Mixer volume      |
                    |                   |
                    | Pressure storage: |
                    |     Volume        |
                    |                   |
                    | Species storage:  |
                    |     Composition   |
                    |                   |
                    | Gas properties:   |
                    |     IdealGas      |
                    |     through Lookup|
                    |                   |
                    | Initial mixture:  |
                    |     N2 = 0.79     |
                    |     O2 = 0.20     |
                    |     Ar = 0.01     |
                    +-------------------+
                              |
                              v
                      +---------------+
                      | Outlet        |
                      | flowtube      |
                      | with inertia  |
                      +---------------+
                              |
                              v
                         Downstream
                         fixed pressure


Model notes
-----------

This example separates the flow problem, gas-property problem, and species
problem.

The FlowTube components calculate the transient inlet and outlet mass flows.
Each FlowTube has a real mass-flow dynamic equation:

    d(mdot)/dt = momentum_error / length

The IdealGas objects are wrapped in FullFlow Lookup components. As pressure,
temperature, and composition change, the lookups update density, gas constant,
specific heat ratio, and other gas properties.

The Volume component represents the mixer pressure node. It stores total gas
mass and solves the mixer pressure from total mass conservation:

    mass_mixer = density_mixer * volume_mixer

    d(mass_mixer)/dt = mdot_N2O2 + mdot_Ar - mdot_out

The Composition component represents the species inventory inside the mixer.
It is given MixerVolume.mass, so it uses the same variable mixer mass as the
pressure Volume:

    amount_i = mass_mixer * mixer_mass_fraction_i

    d(amount_i)/dt = mdot_N2O2 * x_i,N2O2
                   + mdot_Ar   * x_i,Ar
                   - mdot_out  * x_i,mixer

The outlet composition is passed as None, so the outlet uses the current mixer
composition states. For this well-mixed volume:

    mixer composition = outlet composition

The pressure/flow network is initialized first with the argon inlet nearly
closed. The Composition component is then added, and the initialized full model
is exported with static=True so the requested initial composition remains:

    N2 = 0.79, O2 = 0.20, Ar = 0.01
"""

import numpy as np

from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

MixerNetwork = Network("Transient Composition Mixer")


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------

filename = "transient_composition_mixer"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

temperature = State(300.0)

friction_factor = 0.03


# ---------------------------------------------------------------------------
# Mixer geometry
# ---------------------------------------------------------------------------

mixer_volume = 0.50


# ---------------------------------------------------------------------------
# Boundary pressures
# ---------------------------------------------------------------------------

n2_o2_source_pressure = State(220_000.0)
argon_source_pressure = State(190_000.0)
downstream_pressure = State(101_325.0)


# ---------------------------------------------------------------------------
# Mixer composition states
# ---------------------------------------------------------------------------

# These are the internal mixer composition states. They also represent the
# outlet composition because the mixer is assumed to be perfectly mixed.
mixer_n2_mass_fraction = State(0.79)
mixer_o2_mass_fraction = State(0.20)
mixer_ar_mass_fraction = State(0.01)

mixer_composition = {
    "N2": mixer_n2_mass_fraction,
    "O2": mixer_o2_mass_fraction,
    "Ar": mixer_ar_mass_fraction,
}


# ---------------------------------------------------------------------------
# Source stream compositions
# ---------------------------------------------------------------------------

# The N2/O2 inlet contains no argon. Since the initial mixer composition has
# 79% N2, 20% O2, and 1% argon, the N2/O2 source is normalized over only the
# N2 and O2 portions.
n2_o2_composition = {
    "N2": 0.79 / (0.79 + 0.20),
    "O2": 0.20 / (0.79 + 0.20),
    "Ar": 0.0,
}

argon_composition = {
    "N2": 0.0,
    "O2": 0.0,
    "Ar": 1.0,
}


# ---------------------------------------------------------------------------
# Pressure states
# ---------------------------------------------------------------------------

# Mixer pressure is solved by the Volume component.
mixer_pressure = State(150_000.0)


# ---------------------------------------------------------------------------
# Gas property lookups
# ---------------------------------------------------------------------------

# The source gas lookups are fixed-pressure, fixed-temperature property models.
# They provide the source densities used by the inlet FlowTube components.
N2O2Gas = Lookup(
    "N2/O2 Source Gas",
    MixerNetwork,
    IdealGas,
    n2_o2_composition,
    basis="mass",
    pressure=n2_o2_source_pressure,
    temperature=temperature,
)

ArgonGas = Lookup(
    "Argon Source Gas",
    MixerNetwork,
    IdealGas,
    argon_composition,
    basis="mass",
    pressure=argon_source_pressure,
    temperature=temperature,
)

# The mixer gas lookup is the important one. Its composition inputs are the
# live mixer composition states, and its pressure input is the mixer pressure
# solved by MixerVolume.
MixerGas = Lookup(
    "Mixer Gas",
    MixerNetwork,
    IdealGas,
    mixer_composition,
    basis="mass",
    pressure=mixer_pressure,
    temperature=temperature,
)

# The downstream gas lookup is used only as a downstream density estimate for
# the outlet FlowTube. It uses the same composition as the well-mixed outlet.
DownstreamGas = Lookup(
    "Downstream Gas",
    MixerNetwork,
    IdealGas,
    mixer_composition,
    basis="mass",
    pressure=downstream_pressure,
    temperature=temperature,
)


# ---------------------------------------------------------------------------
# Source and outlet stream containers
# ---------------------------------------------------------------------------

N2O2Source = Component("N2/O2 Source", MixerNetwork)
N2O2Source.pressure = n2_o2_source_pressure
N2O2Source.density = N2O2Gas.density
N2O2Source.mass_flow = State(0.20)
N2O2Source.composition = n2_o2_composition


ArgonSource = Component("Argon Source", MixerNetwork)
ArgonSource.pressure = argon_source_pressure
ArgonSource.density = ArgonGas.density
ArgonSource.mass_flow = State(0.001)
ArgonSource.composition = argon_composition


MixedOutlet = Component("Mixed Outlet", MixerNetwork)
MixedOutlet.pressure = downstream_pressure
MixedOutlet.density = DownstreamGas.density
MixedOutlet.mass_flow = State(0.20)
MixedOutlet.composition = mixer_composition


# ---------------------------------------------------------------------------
# Mixer pressure volume
# ---------------------------------------------------------------------------

# The Volume component stores total gas mass and solves mixer pressure.
#
# The density comes from the MixerGas Lookup, not from a hand-written ideal-gas
# equation in this script.
MixerVolume = Volume(
    "Mixer Volume",
    MixerNetwork,
    pressure=MixerGas.pressure,
    volume=mixer_volume,
    density=MixerGas.density,
    mass_flow_in=N2O2Source.mass_flow + ArgonSource.mass_flow,
    mass_flow_out=MixedOutlet.mass_flow,
)


# ---------------------------------------------------------------------------
# N2/O2 inlet flowtube
# ---------------------------------------------------------------------------

# This inertial FlowTube determines the N2/O2 inlet mass flow from source
# pressure, mixer pressure, friction, and flow inertia.
N2O2Line = FlowTube(
    "N2/O2 Inlet FlowTube",
    MixerNetwork,
    mass_flow=N2O2Source.mass_flow,
    upstream_static_pressure=N2O2Source.pressure,
    downstream_static_pressure=MixerVolume.pressure,
    length=2.0,
    hydraulic_diameter=0.030,
    cross_sectional_area=(np.pi / 4) * 0.030**2,
    upstream_density=N2O2Source.density,
    downstream_density=MixerVolume.density,
    friction_factor=friction_factor,
)


# ---------------------------------------------------------------------------
# Argon inlet area schedule
# ---------------------------------------------------------------------------

# The argon inlet is represented as a FlowTube with a scheduled effective area.
# The area starts very small, then opens smoothly. Keeping it nonzero avoids a
# true singular zero-area line.
argon_area_closed = 1.0e-6
argon_area_open = 2.0e-4

argon_effective_area = State(argon_area_closed)

argon_valve_open_delay = 1.0
argon_valve_open_time = 3.0


def argon_area_schedule(t):
    """
    Open the argon inlet smoothly.

    The area starts nearly closed so the mixer initially flows mostly N2/O2.
    After one second, the argon inlet opens. Argon begins entering the mixer,
    so the argon mass fraction rises and the N2/O2 mass fractions fall toward
    their new steady-state values.
    """

    if t <= argon_valve_open_delay:
        return argon_area_closed

    ramp_fraction = (t - argon_valve_open_delay) / argon_valve_open_time

    if ramp_fraction >= 1.0:
        return argon_area_open

    ramp_fraction = 0.5 - 0.5 * np.cos(np.pi * ramp_fraction)

    return argon_area_closed + (argon_area_open - argon_area_closed) * ramp_fraction


ArgonAreaSequence = Sequence(
    "Argon Effective Area",
    MixerNetwork,
    target=argon_effective_area,
    function=argon_area_schedule,
)


# ---------------------------------------------------------------------------
# Argon inlet flowtube
# ---------------------------------------------------------------------------

# This inertial FlowTube determines the argon inlet mass flow. The scheduled
# effective area acts like a simple opening valve.
ArgonLine = FlowTube(
    "Argon Inlet FlowTube",
    MixerNetwork,
    mass_flow=ArgonSource.mass_flow,
    upstream_static_pressure=ArgonSource.pressure,
    downstream_static_pressure=MixerVolume.pressure,
    length=0.75,
    hydraulic_diameter=0.020,
    cross_sectional_area=argon_effective_area,
    upstream_density=ArgonSource.density,
    downstream_density=MixerVolume.density,
    friction_factor=friction_factor,
)


# ---------------------------------------------------------------------------
# Outlet flowtube
# ---------------------------------------------------------------------------

# This inertial FlowTube determines the outlet mass flow from mixer pressure to
# the downstream boundary.
OutletLine = FlowTube(
    "Outlet FlowTube",
    MixerNetwork,
    mass_flow=MixedOutlet.mass_flow,
    upstream_static_pressure=MixerVolume.pressure,
    downstream_static_pressure=MixedOutlet.pressure,
    length=2.5,
    hydraulic_diameter=0.040,
    cross_sectional_area=(np.pi / 4) * 0.040**2,
    upstream_density=MixerVolume.density,
    downstream_density=MixedOutlet.density,
    friction_factor=friction_factor,
)


# ---------------------------------------------------------------------------
# Flow initialization
# ---------------------------------------------------------------------------

# First initialize only the pressure/flow/property network with the argon inlet
# nearly closed. The Composition component is not created yet, so the requested
# initial mixer composition cannot be driven to steady state.
SteadyState(MixerNetwork).solve(
    verbose=True,
)


# ---------------------------------------------------------------------------
# Transient composition storage
# ---------------------------------------------------------------------------

# With mass assigned, Composition becomes a transient species inventory.
#
# The mass passed here is MixerVolume.mass, not a fixed value. Therefore the
# species amounts are:
#
#     amount_i = MixerVolume.mass * mixer_mass_fraction_i
#
# and the mixer composition evolves with the variable total gas inventory.
MixerComposition = Composition(
    "Mixer Composition",
    MixerNetwork,
    inlets=[
        (N2O2Source.mass_flow, N2O2Source.composition),
        (ArgonSource.mass_flow, ArgonSource.composition),
    ],
    outlets=[
        (MixedOutlet.mass_flow, None),
    ],
    solve={
        "N2": mixer_n2_mass_fraction,
        "O2": mixer_o2_mass_fraction,
        "Ar": mixer_ar_mass_fraction,
    },
    mass=MixerVolume.mass,
)


# ---------------------------------------------------------------------------
# Tracked outputs
# ---------------------------------------------------------------------------

composition_sum = (
    mixer_n2_mass_fraction
    + mixer_o2_mass_fraction
    + mixer_ar_mass_fraction
)

# Mixer composition states. For this well-mixed volume, these are also the
# outlet composition fractions.
MixerNetwork.track("Mixer N2 Mass Fraction [-]", mixer_n2_mass_fraction)
MixerNetwork.track("Mixer O2 Mass Fraction [-]", mixer_o2_mass_fraction)
MixerNetwork.track("Mixer Ar Mass Fraction [-]", mixer_ar_mass_fraction)
MixerNetwork.track("Mixer Composition Sum [-]", composition_sum)

# Flow quantities.
MixerNetwork.track("N2/O2 Inlet Mass Flow [kg/s]", N2O2Source.mass_flow)
MixerNetwork.track("Argon Inlet Mass Flow [kg/s]", ArgonSource.mass_flow)
MixerNetwork.track("Outlet Mass Flow [kg/s]", MixedOutlet.mass_flow)

# Pressure, density, mass, area, and gas-property quantities.
MixerNetwork.track("Argon Effective Area [m^2]", argon_effective_area)
MixerNetwork.track("Mixer Pressure [Pa]", MixerVolume.pressure)
MixerNetwork.track("Mixer Density [kg/m^3]", MixerVolume.density)
MixerNetwork.track("Mixer Stored Mass [kg]", MixerVolume.mass)
MixerNetwork.track("Mixer Gas Constant [J/kg/K]", MixerGas.gas_constant)
MixerNetwork.track("Mixer Specific Heat Ratio [-]", MixerGas.specific_heat_ratio)


# ---------------------------------------------------------------------------
# Initial-condition export
# ---------------------------------------------------------------------------

# This writes the initialized full mixer network to the steady_state section of
# the HDF5 file, including the Composition component and the initial composition.
#
# static=True is important here. A normal steady-state solve would drive the
# composition derivatives to zero and would overwrite the requested initial
# 79% N2, 20% O2, 1% Ar mixer composition.
SteadyState(MixerNetwork).solve(
    verbose=True,
    static=True,
    filename=filename,
)


# ---------------------------------------------------------------------------
# Transient solve
# ---------------------------------------------------------------------------

# Run long enough for the argon inlet to open and for the mixer composition to
# move toward its new steady-state value.
Transient(MixerNetwork).solve(
    dt=0.01,
    t_final=20.0,
    filename=filename,
    statistics=True,
)