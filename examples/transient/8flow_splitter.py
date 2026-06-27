"""
Transient particulate separator / flow splitter example.

Physical layout
---------------

              Contaminated water source
              fixed pressure
              water + particulate slurry
              concentration changes during transient
                         |
                         v
                +----------------+
                | Feed flowtube  |
                | with inertia   |
                +----------------+
                         |
                         v
                +----------------+
                | Separator node |
                |                |
                | Volume used as |
                | algebraic node |
                +----------------+
                    |        |
                    |        |
                    v        v
        +---------------+  +----------------------+
        | Clean outlet  |  | Concentrate outlet   |
        | flowtube      |  | flowtube             |
        | nearly pure   |  | adjustable area      |
        | water         |  | particulate-rich     |
        +---------------+  +----------------------+
                    |        |
                    v        v
             clean water   reject / concentrate


Model notes
-----------

This example models a particulate separator as a flow splitter.

The inlet is contaminated water:

    water + particulate

The clean outlet is prescribed to contain only a very small particulate
carryover. The concentrate outlet composition is solved by the Composition
component.

The separator node is a Volume without volume/density storage, so it acts as an
algebraic flow node:

    mdot_feed - mdot_clean - mdot_concentrate = 0

The three FlowTube components provide transient mass-flow inertia:

    d(mdot)/dt = momentum_error / length

The Composition component conserves particulate mass:

    mdot_feed*x_feed
        - mdot_clean*x_clean
        - mdot_concentrate*x_concentrate = 0

A Balance adjusts the concentrate outlet area so that the concentrate stream
stays at the desired particulate mass fraction. This is equivalent to an ideal
controller moving a reject valve or concentrate outlet opening.

The water + particulate properties are calculated with callable Lookup
functions. The slurry lookup returns density, viscosity, and particulate volume
fraction from the current particulate mass fraction.
"""

from types import SimpleNamespace

import numpy as np

from fullflow import *


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

SplitterNetwork = Network("Transient Particulate Splitter")


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------

filename = "transient_particulate_splitter"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

temperature = State(300.0)

water_density_value = 997.0
particle_density_value = 2650.0
water_dynamic_viscosity_value = 8.55e-4

roughness = 2.0e-5


# ---------------------------------------------------------------------------
# Callable slurry property model
# ---------------------------------------------------------------------------

def slurry_properties(particulate_mass_fraction, temperature=300.0):
    """
    Return simple water + particulate slurry properties.

    Parameters
    ----------
    particulate_mass_fraction : float
        Particulate mass fraction in the slurry.
    temperature : float
        Temperature in K. This simple example keeps water properties fixed, but
        temperature is accepted so the lookup has a realistic callable shape.

    Returns
    -------
    SimpleNamespace
        density, dynamic_viscosity, and particulate_volume_fraction.

    Notes
    -----
    The density is calculated from additive specific volumes:

        1/rho_mix = Y_water/rho_water + Y_particle/rho_particle

    The viscosity uses a simple Krieger-Dougherty style correction based on
    particulate volume fraction. This is not a design-grade slurry model; it is
    only meant to make the example physically interesting.
    """

    Yp = float(particulate_mass_fraction)

    # Numerical guard for nonlinear iterations.
    Yp = min(max(Yp, 0.0), 0.60)

    Yw = 1.0 - Yp

    density = 1.0 / (Yw / water_density_value + Yp / particle_density_value)

    particulate_volume_fraction = (Yp / particle_density_value) / (
        Yw / water_density_value + Yp / particle_density_value
    )

    maximum_packing_fraction = 0.62
    relative_viscosity = (1.0 - particulate_volume_fraction / maximum_packing_fraction) ** (-2.5 * maximum_packing_fraction)

    dynamic_viscosity = water_dynamic_viscosity_value * relative_viscosity

    return SimpleNamespace(
        density=density,
        dynamic_viscosity=dynamic_viscosity,
        particulate_volume_fraction=particulate_volume_fraction,
    )


def tube_friction_factor(
    mass_flow,
    density,
    dynamic_viscosity,
    hydraulic_diameter,
    cross_sectional_area,
    roughness,
):
    """
    Return a Darcy friction factor from mass flow and slurry properties.

    The function uses laminar friction for Re < 2300 and the Haaland equation
    for turbulent flow.
    """

    mdot = abs(float(mass_flow))
    rho = float(density)
    mu = float(dynamic_viscosity)
    D = float(hydraulic_diameter)
    A = abs(float(cross_sectional_area))

    A = max(A, 1.0e-12)
    mu = max(mu, 1.0e-12)

    velocity = mdot / (rho * A)
    reynolds_number = rho * velocity * D / mu

    if reynolds_number < 1.0e-12:
        friction_factor = 0.02
    elif reynolds_number < 2300.0:
        friction_factor = 64.0 / reynolds_number
    else:
        relative_roughness = roughness / D
        friction_factor = 1.0 / (
            -1.8 * np.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number)
        ) ** 2

    return SimpleNamespace(
        friction_factor=friction_factor,
        reynolds_number=reynolds_number,
    )


# ---------------------------------------------------------------------------
# Boundary pressures
# ---------------------------------------------------------------------------

feed_source_pressure = State(450_000.0)
clean_outlet_pressure = State(101_325.0)
concentrate_outlet_pressure = State(130_000.0)


# ---------------------------------------------------------------------------
# Separator pressure node
# ---------------------------------------------------------------------------

separator_pressure = State(260_000.0)


# ---------------------------------------------------------------------------
# Stream mass-flow states
# ---------------------------------------------------------------------------

feed_mass_flow = State(0.40)
clean_mass_flow = State(0.36)
concentrate_mass_flow = State(0.04)


# ---------------------------------------------------------------------------
# Composition states
# ---------------------------------------------------------------------------

# Feed contamination changes during the transient.
feed_particulate_mass_fraction = State(0.01)
feed_water_mass_fraction = 1.0 - feed_particulate_mass_fraction

# Clean outlet is nearly pure water. This is the residual particulate carryover
# through the separator.
clean_particulate_mass_fraction = State(5.0e-5)
clean_water_mass_fraction = 1.0 - clean_particulate_mass_fraction

# The concentrate stream is solved from particulate conservation.
concentrate_particulate_mass_fraction = State(0.20)
concentrate_water_mass_fraction = 1.0 - concentrate_particulate_mass_fraction

feed_composition = {
    "water": feed_water_mass_fraction,
    "particulate": feed_particulate_mass_fraction,
}

clean_composition = {
    "water": clean_water_mass_fraction,
    "particulate": clean_particulate_mass_fraction,
}

concentrate_composition = {
    "water": concentrate_water_mass_fraction,
    "particulate": concentrate_particulate_mass_fraction,
}


# ---------------------------------------------------------------------------
# Source and outlet stream containers
# ---------------------------------------------------------------------------

FeedSource = Component("Contaminated Water Source", SplitterNetwork)
FeedSource.pressure = feed_source_pressure
FeedSource.mass_flow = feed_mass_flow
FeedSource.composition = feed_composition


CleanOutlet = Component("Clean Water Outlet", SplitterNetwork)
CleanOutlet.pressure = clean_outlet_pressure
CleanOutlet.mass_flow = clean_mass_flow
CleanOutlet.composition = clean_composition


ConcentrateOutlet = Component("Concentrate Outlet", SplitterNetwork)
ConcentrateOutlet.pressure = concentrate_outlet_pressure
ConcentrateOutlet.mass_flow = concentrate_mass_flow
ConcentrateOutlet.composition = concentrate_composition


# ---------------------------------------------------------------------------
# Slurry property lookups
# ---------------------------------------------------------------------------

FeedProperties = Lookup(
    "Feed Slurry Properties",
    SplitterNetwork,
    slurry_properties,
    particulate_mass_fraction=feed_particulate_mass_fraction,
    temperature=temperature,
)

CleanProperties = Lookup(
    "Clean Water Properties",
    SplitterNetwork,
    slurry_properties,
    particulate_mass_fraction=clean_particulate_mass_fraction,
    temperature=temperature,
)

ConcentrateProperties = Lookup(
    "Concentrate Slurry Properties",
    SplitterNetwork,
    slurry_properties,
    particulate_mass_fraction=concentrate_particulate_mass_fraction,
    temperature=temperature,
)


# ---------------------------------------------------------------------------
# Tube geometry
# ---------------------------------------------------------------------------

feed_tube_length = 2.0
feed_tube_diameter = 0.030
feed_tube_area = (np.pi / 4.0) * feed_tube_diameter**2

clean_tube_length = 2.0
clean_tube_diameter = 0.035
clean_tube_area = (np.pi / 4.0) * clean_tube_diameter**2

concentrate_tube_length = 1.5
concentrate_tube_diameter = 0.020

# The concentrate outlet behaves like a controlled reject valve.
#
# The Balance below changes this area so the concentrate stream stays at the
# desired particulate mass fraction.
concentrate_effective_area = State(2.0e-5)


# ---------------------------------------------------------------------------
# Friction factor lookups
# ---------------------------------------------------------------------------

FeedFriction = Lookup(
    "Feed Tube Friction",
    SplitterNetwork,
    tube_friction_factor,
    mass_flow=FeedSource.mass_flow,
    density=FeedProperties.density,
    dynamic_viscosity=FeedProperties.dynamic_viscosity,
    hydraulic_diameter=feed_tube_diameter,
    cross_sectional_area=feed_tube_area,
    roughness=roughness,
)

CleanFriction = Lookup(
    "Clean Tube Friction",
    SplitterNetwork,
    tube_friction_factor,
    mass_flow=CleanOutlet.mass_flow,
    density=CleanProperties.density,
    dynamic_viscosity=CleanProperties.dynamic_viscosity,
    hydraulic_diameter=clean_tube_diameter,
    cross_sectional_area=clean_tube_area,
    roughness=roughness,
)

ConcentrateFriction = Lookup(
    "Concentrate Tube Friction",
    SplitterNetwork,
    tube_friction_factor,
    mass_flow=ConcentrateOutlet.mass_flow,
    density=ConcentrateProperties.density,
    dynamic_viscosity=ConcentrateProperties.dynamic_viscosity,
    hydraulic_diameter=concentrate_tube_diameter,
    cross_sectional_area=concentrate_effective_area,
    roughness=roughness,
)


# ---------------------------------------------------------------------------
# Separator node
# ---------------------------------------------------------------------------

# This Volume is used as an algebraic junction. Since volume and density are
# omitted, it enforces:
#
#     feed_mass_flow = clean_mass_flow + concentrate_mass_flow
SeparatorNode = Volume(
    "Separator Node",
    SplitterNetwork,
    pressure=separator_pressure,
    mass_flow_in=FeedSource.mass_flow,
    mass_flow_out=CleanOutlet.mass_flow + ConcentrateOutlet.mass_flow,
)


# ---------------------------------------------------------------------------
# Feed flowtube
# ---------------------------------------------------------------------------

FeedTube = FlowTube(
    "Feed FlowTube",
    SplitterNetwork,
    mass_flow=FeedSource.mass_flow,
    upstream_static_pressure=FeedSource.pressure,
    downstream_static_pressure=SeparatorNode.pressure,
    length=feed_tube_length,
    hydraulic_diameter=feed_tube_diameter,
    cross_sectional_area=feed_tube_area,
    upstream_density=FeedProperties.density,
    downstream_density=FeedProperties.density,
    friction_factor=FeedFriction.friction_factor,
)


# ---------------------------------------------------------------------------
# Clean water outlet flowtube
# ---------------------------------------------------------------------------

CleanTube = FlowTube(
    "Clean Water FlowTube",
    SplitterNetwork,
    mass_flow=CleanOutlet.mass_flow,
    upstream_static_pressure=SeparatorNode.pressure,
    downstream_static_pressure=CleanOutlet.pressure,
    length=clean_tube_length,
    hydraulic_diameter=clean_tube_diameter,
    cross_sectional_area=clean_tube_area,
    upstream_density=CleanProperties.density,
    downstream_density=CleanProperties.density,
    friction_factor=CleanFriction.friction_factor,
)


# ---------------------------------------------------------------------------
# Concentrate outlet flowtube
# ---------------------------------------------------------------------------

ConcentrateTube = FlowTube(
    "Concentrate FlowTube",
    SplitterNetwork,
    mass_flow=ConcentrateOutlet.mass_flow,
    upstream_static_pressure=SeparatorNode.pressure,
    downstream_static_pressure=ConcentrateOutlet.pressure,
    length=concentrate_tube_length,
    hydraulic_diameter=concentrate_tube_diameter,
    cross_sectional_area=concentrate_effective_area,
    upstream_density=ConcentrateProperties.density,
    downstream_density=ConcentrateProperties.density,
    friction_factor=ConcentrateFriction.friction_factor,
)


# ---------------------------------------------------------------------------
# Composition splitter
# ---------------------------------------------------------------------------

# Particulate conservation solves the concentrate particulate fraction for the
# current flow split.
#
# Water conservation is redundant because the Volume component already enforces
# total mass conservation and each stream's water fraction is defined as:
#
#     water = 1 - particulate
SplitterComposition = Composition(
    "Particulate Splitter Composition",
    SplitterNetwork,
    inlets=[
        (FeedSource.mass_flow, FeedSource.composition),
    ],
    outlets=[
        (CleanOutlet.mass_flow, CleanOutlet.composition),
        (ConcentrateOutlet.mass_flow, ConcentrateOutlet.composition),
    ],
    solve={
        "particulate": concentrate_particulate_mass_fraction,
    },
)


# ---------------------------------------------------------------------------
# Concentrate concentration control
# ---------------------------------------------------------------------------

desired_concentrate_particulate_mass_fraction = 0.20

# This balance adjusts the concentrate outlet area so the concentrate stream
# stays at the desired particulate loading. In a physical system, this would be
# analogous to adjusting a reject valve.
ConcentrateControl = Balance(
    "Concentrate Concentration Control",
    SplitterNetwork,
    variable=concentrate_effective_area,
    function=concentrate_particulate_mass_fraction - desired_concentrate_particulate_mass_fraction,
)


# ---------------------------------------------------------------------------
# Feed contamination transient
# ---------------------------------------------------------------------------

feed_particulate_initial = 0.01
feed_particulate_final = 0.04

feed_contamination_delay = 2.0
feed_contamination_ramp_time = 5.0


def feed_contamination_schedule(t):
    """
    Increase the incoming particulate contamination during the transient.

    The separator control balance responds by changing the concentrate outlet
    area so the concentrate outlet remains at the requested particulate mass
    fraction.
    """

    if t <= feed_contamination_delay:
        return feed_particulate_initial

    ramp_fraction = (t - feed_contamination_delay) / feed_contamination_ramp_time

    if ramp_fraction >= 1.0:
        return feed_particulate_final

    ramp_fraction = 0.5 - 0.5 * np.cos(np.pi * ramp_fraction)

    return feed_particulate_initial + (feed_particulate_final - feed_particulate_initial) * ramp_fraction


FeedContaminationSequence = Sequence(
    "Feed Particulate Fraction",
    SplitterNetwork,
    target=feed_particulate_mass_fraction,
    function=feed_contamination_schedule,
)


# ---------------------------------------------------------------------------
# Tracked outputs
# ---------------------------------------------------------------------------

SplitterNetwork.track("Feed Particulate Mass Fraction [-]", feed_particulate_mass_fraction)
SplitterNetwork.track("Clean Particulate Mass Fraction [-]", clean_particulate_mass_fraction)
SplitterNetwork.track("Concentrate Particulate Mass Fraction [-]", concentrate_particulate_mass_fraction)

SplitterNetwork.track("Feed Mass Flow [kg/s]", FeedSource.mass_flow)
SplitterNetwork.track("Clean Mass Flow [kg/s]", CleanOutlet.mass_flow)
SplitterNetwork.track("Concentrate Mass Flow [kg/s]", ConcentrateOutlet.mass_flow)

SplitterNetwork.track("Separator Pressure [Pa]", SeparatorNode.pressure)
SplitterNetwork.track("Concentrate Effective Area [m^2]", concentrate_effective_area)

SplitterNetwork.track("Feed Density [kg/m^3]", FeedProperties.density)
SplitterNetwork.track("Clean Density [kg/m^3]", CleanProperties.density)
SplitterNetwork.track("Concentrate Density [kg/m^3]", ConcentrateProperties.density)

SplitterNetwork.track("Feed Dynamic Viscosity [Pa-s]", FeedProperties.dynamic_viscosity)
SplitterNetwork.track("Clean Dynamic Viscosity [Pa-s]", CleanProperties.dynamic_viscosity)
SplitterNetwork.track("Concentrate Dynamic Viscosity [Pa-s]", ConcentrateProperties.dynamic_viscosity)

SplitterNetwork.track("Feed Reynolds Number [-]", FeedFriction.reynolds_number)
SplitterNetwork.track("Clean Reynolds Number [-]", CleanFriction.reynolds_number)
SplitterNetwork.track("Concentrate Reynolds Number [-]", ConcentrateFriction.reynolds_number)

SplitterNetwork.track("Feed Friction Factor [-]", FeedFriction.friction_factor)
SplitterNetwork.track("Clean Friction Factor [-]", CleanFriction.friction_factor)
SplitterNetwork.track("Concentrate Friction Factor [-]", ConcentrateFriction.friction_factor)


# ---------------------------------------------------------------------------
# Steady-state initialization
# ---------------------------------------------------------------------------

# This solves the initial flow split at the initial contamination level. The
# concentrate outlet area is adjusted so the concentrate starts at the target
# particulate mass fraction.
SteadyState(SplitterNetwork).solve(
    verbose=True,
    filename=filename,
)


# ---------------------------------------------------------------------------
# Transient solve
# ---------------------------------------------------------------------------

# During the transient, the feed contamination rises from 1% to 4% by mass.
# The control Balance adjusts the concentrate outlet area to keep the
# concentrate stream at 20% particulate by mass.
Transient(SplitterNetwork).solve(
    dt=0.01,
    t_final=15.0,
    filename=filename,
    statistics=True,
)