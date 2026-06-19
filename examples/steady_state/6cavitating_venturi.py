# This example demonstrates a cavitating-venturi injector network with
# switchable model options for the fuel and oxidizer inlets.

import numpy as np

from fullflow import *
from thermoprop import *


CavNetwork = Network("Cavitating Venturi Network")

FuelSource = Propellant("RP-1", temperature=300, pressure=400 * 6894.76)
LOXSource = Propellant("LOX", temperature=90.17, pressure=400 * 6894.76)


fuel_manifold_pressure = State(350 * 6896.76)
ox_manifold_pressure = State(350 * 6896.76)
chamber_pressure = State(300 * 6894.76)


FuelManifoldFluid = Lookup(
    "Fuel Injector Manifold Fluid",
    CavNetwork,
    Propellant,
    FuelSource.composition,
    temperature=300,
    pressure=fuel_manifold_pressure,
)

OxManifoldFluid = Lookup(
    "Ox Injector Manifold Fluid",
    CavNetwork,
    Propellant,
    LOXSource.composition,
    quality=0.0,
    pressure=ox_manifold_pressure,
)


Props = Lookup(
    "Reacting Propellants",
    CavNetwork,
    Reactants,
    fuels=FuelManifoldFluid,
    oxidizers=OxManifoldFluid,
    mixture_ratio=2,
)

ChamberGas = Lookup(
    "Combustion Chamber Gas",
    CavNetwork,
    Equilibrium,
    reactants=Props,
    pressure=chamber_pressure,
)


# Shared inlet interface states.
# These connect the active model option to the rest of the network.

fuel_inlet_mass_flow = State(1.5)
fuel_inlet_friction_factor = State(0.02)

ox_inlet_mass_flow = State(3.0)
ox_inlet_friction_factor = State(0.02)


# Reasonable line geometry guesses.

fuel_inlet_hydraulic_diameter = 0.75 / 39.37
ox_inlet_hydraulic_diameter = 0.75 / 39.37

fuel_inlet_cross_sectional_area = (np.pi / 4) * fuel_inlet_hydraulic_diameter**2
ox_inlet_cross_sectional_area = (np.pi / 4) * ox_inlet_hydraulic_diameter**2

inlet_line_length = 1.0


FuelInletModel = Model(
    "Fuel Inlet",
    CavNetwork,
    order=[
        "Darcy",
        "Venturi",
    ],
)

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


Props.mixture_ratio = OxOrf.mass_flow / FuelOrf.mass_flow


Chamber = Volume(
    "Combustion Chamber",
    CavNetwork,
    volume=1,
    pressure=chamber_pressure,
    mass_flow_in=OxOrf.mass_flow + FuelOrf.mass_flow,
)


CavNetwork.track(
    "Combustion Chamber Gas",
    ChamberGas.CombustionGas,
    attributes={
        "Pressure [Pa]": "pressure",
        "Temperature [K]": "temperature",
        "Density [kg/m3]": "density",
        "Enthalpy [J/kg]": "enthalpy",
        "Entropy [J/kg-K]": "entropy",
        "Cp [J/kg-K]": "specific_heat",
        "Cv [J/kg-K]": "specific_heat_cv",
        "Gamma": "gamma",
        "Gas Constant [J/kg-K]": "gas_constant",
        "Molecular Weight": "molar_mass",
        "Speed of Sound [m/s]": "speed_of_sound",
        "Viscosity [Pa-s]": "dynamic_viscosity",
        "Conductivity [W/m-K]": "conductivity",
        "Prandtl": "prandtl",
    },
    items={
        "Mole Fractions": "mole_fractions",
        "Mass Fractions": "mass_fractions",
    },
    minimum=1e-4,
    max_items=12,
)


Nozzle = IsentropicNozzle(
    "Nozzle",
    CavNetwork,
    upstream_total_pressure=Chamber.pressure,
    upstream_total_temperature=ChamberGas.temperature,
    ambient_pressure=101325,
    specific_heat_ratio=ChamberGas.gamma,
    gas_constant=ChamberGas.gas_constant,
    throat_area=(6.05 / 1550),
    expansion_ratio=4.5,
    mass_flow=Chamber.mass_flow_out,
)


CavNetwork.track("Fuel Injector Manifold Pressure [psia]", fuel_manifold_pressure / 6894.76)
CavNetwork.track("Ox Injector Manifold Pressure [psia]", ox_manifold_pressure / 6894.76)
CavNetwork.track("Chamber Pressure [psia]", chamber_pressure / 6894.76)
CavNetwork.track("Mixture Ratio", Props.mixture_ratio)


SteadyState(CavNetwork).solve(
    verbose=True,
    jacobian_method="2-point",
    statistics=True,
    model=OxInletModel,
    evaluate_all_model_options=True,
)