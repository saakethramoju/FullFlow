"""
Cavitating venturi injector network example.

This script demonstrates a steady-state injector + chamber + nozzle network with
switchable inlet model options for the fuel and oxidizer sides.

## Network structure

FuelSource  -> Fuel inlet model -> Fuel manifold -> Fuel injector orifices
LOXSource   -> Ox inlet model   -> Ox manifold   -> Ox injector orifices

Fuel + ox then enter a combustion chamber volume, and the chamber discharges
through an isentropic nozzle.

A precomputed RP-1/LOX combustion-products map is used to estimate chamber gas
properties from chamber pressure and mixture ratio:

```
chamber_pressure, mixture_ratio -> chamber_temperature, gamma, gas_constant
```

The mixture ratio State is created before the injector orifices so the chamber
map can be defined early. Later, the same State is made derived using:

```
mixture_ratio <<= OxOrf.mass_flow / FuelOrf.mass_flow
```

That keeps the original State object connected to the map while allowing its
value to update from the solved fuel and oxidizer mass flows.
"""

import numpy as np

from fullflow import *
from thermoprop import *

# =============================================================================

# Map file settings

# =============================================================================

filename = "RP1_LOX"

# =============================================================================

# Optional combustion-map generation

# =============================================================================

#

# Uncomment this block when the RP1_LOX.h5 map file needs to be generated or

# regenerated. The Axis names must match the rp1_lox_products_map function

# argument names.

#

# The generated group layout is:

#

# RP1_LOX.h5

# /products

# /axes/chamber_pressure

# /axes/mixture_ratio

# /outputs/chamber_temperature

# /outputs/gamma

# /outputs/gas_constant

#

# The map is intentionally generated separately from the solve because

# ThermoProp equilibrium calls are much more expensive than HDF5 interpolation.

'''
fuel = Propellant("rp-1", temperature=298.15)
ox = Propellant("LOX", temperature=90.17)

def rp1_lox_products_map(chamber_pressure, mixture_ratio):
"""Return RP-1/LOX equilibrium gas properties for one map grid point."""
r = Reactants(
fuels=fuel,
oxidizers=ox,
mixture_ratio=mixture_ratio,
)

```
eq = Equilibrium(
    reactants=r,
    pressure=chamber_pressure,
)

return {
    "chamber_temperature": eq.temperature,
    "gamma": eq.gamma,
    "gas_constant": eq.gas_constant,
}
```

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
'''

# =============================================================================

# Network and source states

# =============================================================================

CavNetwork = Network("Cavitating Venturi Network")

# Feed-system source fluids.

#

# These are upstream reservoir/source conditions. The injector inlet models use

# these pressures and fluid properties as their upstream boundary conditions.

FuelSource = Propellant("RP-1", temperature=300, pressure=400 * 6894.76)
LOXSource = Propellant("LOX", temperature=90.17, pressure=400 * 6894.76)

# Primary solve states.

#

# These are shared between components. They are created early so downstream

# components, maps, and tracked outputs can all reference the same State objects.

fuel_manifold_pressure = State(350 * 6894.76)
ox_manifold_pressure = State(350 * 6894.76)
chamber_pressure = State(300 * 6894.76)

# Startup guess for mixture ratio.

#

# This State is passed into ChamberMap now, then later converted into a derived

# State from OxOrf.mass_flow / FuelOrf.mass_flow using <<=. This allows the map

# to be defined before the injector orifices are created.

mixture_ratio = State(2)

# =============================================================================

# Manifold fluid property lookups

# =============================================================================

# RP-1 manifold properties are evaluated from composition, fixed temperature,

# and current manifold pressure.

FuelManifoldFluid = Lookup(
"Fuel Injector Manifold Fluid",
CavNetwork,
Propellant,
FuelSource.composition,
temperature=298.15,
pressure=fuel_manifold_pressure,
)

# LOX manifold properties are evaluated as saturated/subcooled liquid using

# quality=0.0 and the current oxidizer manifold pressure.

OxManifoldFluid = Lookup(
"Ox Injector Manifold Fluid",
CavNetwork,
Propellant,
LOXSource.composition,
quality=0.0,
pressure=ox_manifold_pressure,
)

# =============================================================================

# Combustion-products map

# =============================================================================

# ChamberMap reads gas properties from RP1_LOX.h5.

#

# The input dictionary keys must match the map axis names:

#

# "chamber_pressure"

# "mixture_ratio"

#

# The State variable names can be anything. Only the dictionary keys must match

# the HDF5 map axis names.

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

# =============================================================================

# Shared inlet interface states

# =============================================================================

# These states connect whichever inlet model option is active to the rest of the

# network. For example, the Darcy and Venturi inlet options both write to the

# same fuel_inlet_mass_flow State.

fuel_inlet_mass_flow = State(1.5)
fuel_inlet_friction_factor = State(0.02)

ox_inlet_mass_flow = State(3.0)
ox_inlet_friction_factor = State(0.02)

# =============================================================================

# Inlet line geometry

# =============================================================================

fuel_inlet_hydraulic_diameter = 0.75 / 39.37
ox_inlet_hydraulic_diameter = 0.75 / 39.37

fuel_inlet_cross_sectional_area = (np.pi / 4) * fuel_inlet_hydraulic_diameter**2
ox_inlet_cross_sectional_area = (np.pi / 4) * ox_inlet_hydraulic_diameter**2

inlet_line_length = 1.0

# =============================================================================

# Fuel inlet model

# =============================================================================

# The fuel inlet can be modeled either as a cavitating venturi or as a Darcy line.

#

# The model order sets the default/priority order when model options are chosen.

FuelInletModel = Model(
"Fuel Inlet",
CavNetwork,
order=[
"Venturi",
"Darcy",
],
)

# Darcy option:

#

# Churchill calculates the friction factor from Reynolds number.

# DarcyWeisbach then uses that friction factor to calculate the line pressure

# drop for the same mass flow.

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

# Venturi option:

#

# CavitatingVenturi switches between cavitating and noncavitating behavior based

# on throat pressure relative to vapor pressure.

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

# =============================================================================

# Oxidizer inlet model

# =============================================================================

# The oxidizer inlet has the same two-option structure as the fuel inlet.

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

# =============================================================================

# Injector manifolds

# =============================================================================

# The manifold Volume components enforce mass balance:

#

# mass_flow_in - mass_flow_out = 0

#

# The pressure States are iteration variables. The outlet mass flows are used by

# the downstream injector orifice components.

FuelManifold = Volume(
"Fuel Injector Manifold",
CavNetwork,
volume=1,
pressure=fuel_manifold_pressure,
mass_flow_in=fuel_inlet_mass_flow,
)

OxManifold = Volume(
"Ox Injector Manifold",
CavNetwork,
volume=1,
pressure=ox_manifold_pressure,
mass_flow_in=ox_inlet_mass_flow,
)

# =============================================================================

# Injector orifices

# =============================================================================

# These components calculate injector mass flow from manifold pressure, chamber

# pressure, liquid density, Cd, and total injector area.

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

# =============================================================================

# Derived mixture ratio

# =============================================================================

# This mutates the existing mixture_ratio State object rather than replacing it.

#

# ChamberMap already holds a reference to mixture_ratio, so after this line the

# map sees the live solved value:

#

# mixture_ratio = ox_mass_flow / fuel_mass_flow

#

# The initial value State(2.2) remains useful as a startup fallback before both

# mass-flow values are available.

mixture_ratio <<= OxOrf.mass_flow / FuelOrf.mass_flow

# Equivalent verbose form:

# mixture_ratio.derive_from(OxOrf.mass_flow / FuelOrf.mass_flow)

# =============================================================================

# Combustion chamber

# =============================================================================

# The chamber volume enforces:

#

# fuel_mass_flow + ox_mass_flow - nozzle_mass_flow = 0

Chamber = Volume(
"Combustion Chamber",
CavNetwork,
volume=1,
pressure=chamber_pressure,
mass_flow_in=OxOrf.mass_flow + FuelOrf.mass_flow,
)

# =============================================================================

# Nozzle

# =============================================================================

# The nozzle receives total chamber conditions from:

#

# pressure:      Chamber.pressure

# temperature:   ChamberMap.chamber_temperature

# gamma:         ChamberMap.gamma

# gas constant:  ChamberMap.gas_constant

#

# The nozzle mass flow is connected to Chamber.mass_flow_out so the chamber mass

# balance and nozzle choking/expansion solution are solved together.

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

# =============================================================================

# Tracked outputs

# =============================================================================

CavNetwork.track("Fuel Injector Manifold Pressure [psia]", fuel_manifold_pressure / 6894.76)
CavNetwork.track("Ox Injector Manifold Pressure [psia]", ox_manifold_pressure / 6894.76)
CavNetwork.track("Chamber Pressure [psia]", chamber_pressure / 6894.76)
CavNetwork.track("Mixture Ratio", mixture_ratio)

# =============================================================================

# Solve

# =============================================================================

# This solve call sweeps all options for OxInletModel only.

#

# FuelInletModel still follows its configured active/default option behavior.

# To sweep both fuel and oxidizer model combinations, run a multi-model sweep or

# solve separate combinations manually.

SteadyState(CavNetwork).solve(
verbose=True,
jacobian_method="2-point",
statistics=True,
model=OxInletModel,
evaluate_all_model_options=True,
# filename="cavitating_venturi",
)