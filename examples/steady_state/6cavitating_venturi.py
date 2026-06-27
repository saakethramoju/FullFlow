"""
Cavitating Venturi Injector Network
===================================

This example demonstrates a steady-state RP-1/LOX injector, chamber, and nozzle
network with switchable inlet models on the fuel and oxidizer sides.

The model is intentionally organized like a small engine feed system. The fuel
and oxidizer begin at source conditions, flow through selectable inlet-line
models, enter injector manifolds, pass through injector orifices, mix in a
combustion chamber, and discharge through an isentropic nozzle.

Physical Layout
---------------

Fuel side:

    RP-1 Source
        |
        v
    Fuel Inlet Model
        |\
        | +-- Venturi option
        | +-- Darcy + Churchill option
        v
    Fuel Injector Manifold
        |
        v
    Fuel Injector Orifices
        |
        v

                         Combustion Chamber  --->  Isentropic Nozzle
        ^
        |
    Ox Injector Orifices
        ^
        |
    Ox Injector Manifold
        ^
        | +-- Darcy + Churchill option
        | +-- Venturi option
        |/
    Ox Inlet Model
        ^
        |
    LOX Source

Modeling Approach
-----------------

The inlet models are wrapped in FullFlow Model objects. This allows the same
network to solve with different physical assumptions without rebuilding the rest
of the system. In this example, each inlet can be represented either by:

    1. A cavitating venturi, or
    2. A Darcy-Weisbach line with a Churchill friction-factor update.

The injector manifolds are Volume components. They enforce mass balance and make
manifold pressures iteration variables.

The injector orifices use discharge-coefficient relations to compute fuel and
oxidizer mass flow from manifold pressure, chamber pressure, liquid density, Cd,
and total injector area.

Combustion gas properties are read from a precomputed RP-1/LOX map instead of
calling equilibrium chemistry during every solver evaluation. The map relates:

    chamber_pressure, mixture_ratio -> chamber_temperature, gamma, gas_constant

The mixture-ratio State is created before the orifices so the map can be wired
early. Later, the same State is made derived using:

    mixture_ratio <<= OxOrf.mass_flow / FuelOrf.mass_flow

This is useful because ChamberMap already holds a reference to mixture_ratio. By
mutating that State instead of replacing it, the map sees the solved live mixture
ratio while still having a startup guess.

Demonstrates
------------
- Switchable model options with Model
- Cavitating and noncavitating venturi behavior
- Darcy-Weisbach pressure loss with Churchill friction updates
- Injector manifold mass balances
- Discharge-coefficient injector orifices
- HDF5 combustion-products maps
- Derived States using the <<= shorthand
- Coupling a chamber mass balance to a nozzle mass-flow relation
"""

import numpy as np

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Map file settings
# -----------------------------------------------------------------------------
# The chamber gas property map is expected to live in RP1_LOX.h5. The map is
# generated separately from the solve because equilibrium calculations are much
# more expensive than interpolation during each network evaluation.
filename = "RP1_LOX"


# -----------------------------------------------------------------------------
# Optional combustion-map generation
# -----------------------------------------------------------------------------
# Uncomment this block when the RP1_LOX.h5 map file needs to be generated or
# regenerated. The Axis names must match the rp1_lox_products_map function
# argument names.
#
# Generated layout:
#
#     RP1_LOX.h5
#       /products                         top-level map object
#       /products/axes/chamber_pressure
#       /products/axes/mixture_ratio
#       /products/outputs/chamber_temperature
#       /products/outputs/gamma
#       /products/outputs/gas_constant
#
# The map is intentionally low-resolution here because it is only a user example.
# Increase the axis counts for production-quality interpolation.

fuel = Propellant("rp-1", temperature=298.15)
ox = Propellant("LOX", temperature=90.17)


def rp1_lox_products_map(chamber_pressure, mixture_ratio):
    """Return RP-1/LOX equilibrium gas properties for one grid point."""
    r = Reactants(
        fuels=fuel,
        oxidizers=ox,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=r,
        pressure=chamber_pressure,
    )

    return {
        "chamber_temperature": eq.temperature,
        "gamma": eq.gamma,
        "gas_constant": eq.gas_constant,
    }


generate_map(
    filename,
    group="products",
    axes=[
        Axis.linear("chamber_pressure", start=100 * 6894.76, stop=1000 * 6894.76, count=10, units="Pa"),
        Axis.linear("mixture_ratio", start=1.0, stop=4.0, count=10),
    ],
    evaluate=rp1_lox_products_map,
    overwrite=True,
    raise_errors=True,
)



# -----------------------------------------------------------------------------
# Network and source states
# -----------------------------------------------------------------------------
# The Network owns every component and State connection in this example.
CavNetwork = Network("Cavitating Venturi Network")

# Feed-system source fluids. These are upstream reservoir/source conditions. The
# inlet models use these pressures and fluid properties as boundary conditions.
FuelSource = Propellant("RP-1", temperature=300, pressure=400 * 6894.76)
LOXSource = Propellant("LOX", temperature=90.17, pressure=400 * 6894.76)

# Primary solve states shared between components. They are created early so all
# downstream components, maps, and tracked outputs reference the same objects.
fuel_manifold_pressure = State(350 * 6894.76)
ox_manifold_pressure = State(350 * 6894.76)
chamber_pressure = State(300 * 6894.76)

# Startup guess for mixture ratio. This State is connected to ChamberMap below,
# then later converted into a derived State from oxidizer/fuel mass flow.
mixture_ratio = State(2)


# -----------------------------------------------------------------------------
# Manifold fluid property lookups
# -----------------------------------------------------------------------------
# RP-1 manifold properties are evaluated from composition, fixed temperature, and
# the current fuel manifold pressure.
FuelManifoldFluid = Lookup(
    "Fuel Injector Manifold Fluid",
    CavNetwork,
    Propellant,
    FuelSource.composition,
    temperature=298.15,
    pressure=fuel_manifold_pressure,
)

# LOX manifold properties are evaluated as a liquid using quality=0.0 and the
# current oxidizer manifold pressure.
OxManifoldFluid = Lookup(
    "Ox Injector Manifold Fluid",
    CavNetwork,
    Propellant,
    LOXSource.composition,
    quality=0.0,
    pressure=ox_manifold_pressure,
)


# -----------------------------------------------------------------------------
# Combustion-products map
# -----------------------------------------------------------------------------
# ChamberMap reads gas properties from RP1_LOX.h5. The input dictionary keys must
# match the map axis names. The State variable names themselves can be anything.
ChamberMap = Map.from_hdf5(
    "ProductsMap",
    CavNetwork,
    filename,
    group="products",
    inputs={
        "chamber_pressure": chamber_pressure,
        "mixture_ratio": mixture_ratio,
    },
)


# -----------------------------------------------------------------------------
# Shared inlet interface states
# -----------------------------------------------------------------------------
# These states connect whichever inlet model option is active to the rest of the
# network. For example, the fuel Darcy option and fuel venturi option both write
# to fuel_inlet_mass_flow.
fuel_inlet_mass_flow = State(1.5)
fuel_inlet_friction_factor = State(0.02)

ox_inlet_mass_flow = State(3.0)
ox_inlet_friction_factor = State(0.02)


# -----------------------------------------------------------------------------
# Inlet line geometry
# -----------------------------------------------------------------------------
# FullFlow uses SI units internally. The hydraulic diameters are specified in
# inches and converted to meters before calculating cross-sectional area.
fuel_inlet_hydraulic_diameter = 0.75 / 39.37
ox_inlet_hydraulic_diameter = 0.75 / 39.37

fuel_inlet_cross_sectional_area = (np.pi / 4) * fuel_inlet_hydraulic_diameter**2
ox_inlet_cross_sectional_area = (np.pi / 4) * ox_inlet_hydraulic_diameter**2

inlet_line_length = 1.0


# -----------------------------------------------------------------------------
# Fuel inlet model
# -----------------------------------------------------------------------------
# The fuel inlet can be solved with either a cavitating venturi or a Darcy line.
# The order list sets the preferred option order when model options are swept.
FuelInletModel = Model(
    "Fuel Inlet",
    CavNetwork,
    order=[
        "Venturi",
        "Darcy",
    ],
)

# Darcy option. Churchill calculates friction factor from Reynolds number, and
# DarcyWeisbach uses that friction factor to calculate line pressure drop.
FuelInletModel.option(
    "Darcy",
    Churchill.template(
        "Fuel Inlet Line Friction",
        mass_flow=fuel_inlet_mass_flow,
        friction_factor=fuel_inlet_friction_factor,
        hydraulic_diameter=fuel_inlet_hydraulic_diameter,
        dynamic_viscosity=FuelSource.dynamic_viscosity,
        cross_sectional_area=fuel_inlet_cross_sectional_area,
    ),
    DarcyWeisbach.template(
        "Fuel Inlet Line",
        mass_flow=fuel_inlet_mass_flow,
        upstream_pressure=FuelSource.pressure,
        downstream_pressure=FuelManifoldFluid.pressure,
        length=inlet_line_length,
        cross_sectional_area=fuel_inlet_cross_sectional_area,
        hydraulic_diameter=fuel_inlet_hydraulic_diameter,
        density=FuelSource.density,
        friction_factor=fuel_inlet_friction_factor,
    ),
)

# Venturi option. CavitatingVenturi switches between cavitating and
# noncavitating behavior based on throat pressure relative to vapor pressure.
FuelInletModel.option(
    "Venturi",
    CavitatingVenturi.template(
        upstream_pressure=FuelSource.pressure,
        downstream_pressure=FuelManifoldFluid.pressure,
        density=FuelSource.density,
        throat_area=(np.pi / 4) * (1.0 / 39.37)**2,
        vapor_pressure=FuelSource.saturation_pressure,
        pressure_recovery_factor=0.85,
        cavitating_discharge_coefficient=0.94,
        noncavitating_discharge_coefficient=0.6,
        mass_flow=fuel_inlet_mass_flow,
    ),
)


# -----------------------------------------------------------------------------
# Oxidizer inlet model
# -----------------------------------------------------------------------------
# The oxidizer side has the same two-option structure as the fuel side. The
# default order is different here to demonstrate that each Model can have its own
# priority order.
OxInletModel = Model(
    "Ox Inlet",
    CavNetwork,
    order=[
        "Darcy",
        "Venturi",
    ],
)

OxInletModel.option(
    "Darcy",
    Churchill.template(
        "Ox Inlet Line Friction",
        mass_flow=ox_inlet_mass_flow,
        friction_factor=ox_inlet_friction_factor,
        hydraulic_diameter=ox_inlet_hydraulic_diameter,
        dynamic_viscosity=LOXSource.dynamic_viscosity,
        cross_sectional_area=ox_inlet_cross_sectional_area,
    ),
    DarcyWeisbach.template(
        "Ox Inlet Line",
        mass_flow=ox_inlet_mass_flow,
        upstream_pressure=LOXSource.pressure,
        downstream_pressure=OxManifoldFluid.pressure,
        length=inlet_line_length,
        cross_sectional_area=ox_inlet_cross_sectional_area,
        hydraulic_diameter=ox_inlet_hydraulic_diameter,
        density=LOXSource.density,
        friction_factor=ox_inlet_friction_factor,
    ),
)

OxInletModel.option(
    "Venturi",
    CavitatingVenturi.template(
        upstream_pressure=LOXSource.pressure,
        downstream_pressure=OxManifoldFluid.pressure,
        density=LOXSource.density,
        throat_area=(np.pi / 4) * (1.0 / 39.37)**2,
        vapor_pressure=LOXSource.saturation_pressure,
        pressure_recovery_factor=0.85,
        cavitating_discharge_coefficient=0.94,
        noncavitating_discharge_coefficient=0.6,
        mass_flow=ox_inlet_mass_flow,
    ),
)


# -----------------------------------------------------------------------------
# Injector manifolds
# -----------------------------------------------------------------------------
# The manifold Volume components are real lumped storage volumes.  In steady
# state, FullFlow drives their mass derivatives to zero:
#
#     mass_dot = mass_flow_in - mass_flow_out = 0
#
# Their pressures are dynamic solve variables and their outlet mass flows are
# used by the downstream injector orifice components.
FuelManifold = Volume(
    "Fuel Injector Manifold",
    CavNetwork,
    volume=1,
    pressure=fuel_manifold_pressure,
    density=FuelManifoldFluid.density,
    mass_flow_in=fuel_inlet_mass_flow,
)

OxManifold = Volume(
    "Ox Injector Manifold",
    CavNetwork,
    volume=1,
    pressure=ox_manifold_pressure,
    density=OxManifoldFluid.density,
    mass_flow_in=ox_inlet_mass_flow,
)


# -----------------------------------------------------------------------------
# Injector orifices
# -----------------------------------------------------------------------------
# These components calculate injector mass flow from manifold pressure, chamber
# pressure, liquid density, discharge coefficient, and total injector area.
FuelOrf = DischargeCoefficient(
    "Fuel Injector Orifices",
    CavNetwork,
    upstream_pressure=FuelManifold.pressure,
    downstream_pressure=chamber_pressure,
    density=FuelManifoldFluid.density,
    discharge_coefficient=1,
    cross_sectional_area=0.555e-4,
    mass_flow=FuelManifold.mass_flow_out,
)

OxOrf = DischargeCoefficient(
    "Ox Injector Orifices",
    CavNetwork,
    upstream_pressure=OxManifold.pressure,
    downstream_pressure=chamber_pressure,
    density=OxManifoldFluid.density,
    discharge_coefficient=1,
    cross_sectional_area=1.25e-4,
    mass_flow=OxManifold.mass_flow_out,
)


# -----------------------------------------------------------------------------
# Derived mixture ratio
# -----------------------------------------------------------------------------
# This mutates the existing mixture_ratio State object rather than replacing it.
# ChamberMap already holds a reference to mixture_ratio, so after this line the
# map sees the live solved value:
#
#     mixture_ratio = ox_mass_flow / fuel_mass_flow
#
# The original State(2) value remains useful as a startup fallback before both
# mass-flow values are available.
mixture_ratio <<= OxOrf.mass_flow / FuelOrf.mass_flow

# Equivalent verbose form:
# mixture_ratio.derive_from(OxOrf.mass_flow / FuelOrf.mass_flow)


# Ideal-gas density estimate for the chamber products map.
# The combustion map supplies chamber temperature and gas constant; pressure is
# the chamber pressure solve variable.
chamber_density = chamber_pressure / (ChamberMap.gas_constant * ChamberMap.chamber_temperature)


# -----------------------------------------------------------------------------
# Combustion chamber
# -----------------------------------------------------------------------------
# The chamber is a lumped storage volume.  In steady state, FullFlow drives:
#
#     mass_dot = fuel_mass_flow + ox_mass_flow - nozzle_mass_flow = 0
Chamber = Volume(
    "Combustion Chamber",
    CavNetwork,
    volume=1,
    pressure=chamber_pressure,
    density=chamber_density,
    mass_flow_in=OxOrf.mass_flow + FuelOrf.mass_flow,
)


# -----------------------------------------------------------------------------
# Nozzle
# -----------------------------------------------------------------------------
# The nozzle receives chamber total pressure and mapped chamber gas properties.
# Its mass flow is connected to Chamber.mass_flow_out so the chamber mass balance
# and nozzle flow solution are solved together.
Nozzle = IsentropicNozzle(
    "Nozzle",
    CavNetwork,
    upstream_total_pressure=Chamber.pressure,
    upstream_total_temperature=ChamberMap.chamber_temperature,
    ambient_pressure=101325,
    specific_heat_ratio=ChamberMap.gamma,
    gas_constant=ChamberMap.gas_constant,
    throat_area=(6.05 / 1550),
    expansion_ratio=4.5,
    mass_flow=Chamber.mass_flow_out,
)


# -----------------------------------------------------------------------------
# Tracked outputs
# -----------------------------------------------------------------------------
# Tracks are displayed by the solver and exported as friendly aliases when a filename is
# supplied to solve(...). Pressures are converted to psia for readability.
CavNetwork.track("Fuel Injector Manifold Pressure [psia]", fuel_manifold_pressure / 6894.76)
CavNetwork.track("Ox Injector Manifold Pressure [psia]", ox_manifold_pressure / 6894.76)
CavNetwork.track("Chamber Pressure [psia]", chamber_pressure / 6894.76)
CavNetwork.track("Mixture Ratio", mixture_ratio)


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------
# This solve call sweeps all options for OxInletModel only. FuelInletModel still
# follows its configured active/default option behavior.
#
# To sweep both fuel and oxidizer combinations, run a multi-model sweep or solve
# the individual model combinations manually.
SteadyState(CavNetwork).solve(
    verbose=True,
    jacobian_method="2-point",
    statistics=True,
    model=OxInletModel,
    evaluate_all_model_options=True,
    filename=filename,
)
