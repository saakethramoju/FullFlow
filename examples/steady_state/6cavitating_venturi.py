# This example demon


import numpy as np

from fullflow import *
from thermoprop import *

CavNetwork = Network("Cavitating Venturi Network")

FuelSource = Propellant("RP-1", temperature=300, pressure=400*6894.76)
LOXSource = Propellant("LOX", temperature=90.17, pressure=400*6894.76)


fuel_manifold_pressure = State(350 * 6896.76, bounds=(0, 400*6894.76), keep_feasible=True)
ox_manifold_pressure = State(350 * 6896.76, bounds=(0, 400*6894.76), keep_feasible=True)
chamber_pressure = State(300 * 6894.76, bounds=(0, None))

FuelManifoldFluid = Lookup(
    "Fuel Injector Manifold Fluid",
    CavNetwork,
    Propellant,
    FuelSource.composition,
    temperature = 300,
    pressure = fuel_manifold_pressure,
)

OxManifoldFluid = Lookup(
    "Ox Injector Manifold Fluid",
    CavNetwork,
    Propellant,
    LOXSource.composition,
    quality = 0.0,
    pressure = ox_manifold_pressure
)


Props = Lookup(
    "Reacting Propellants",
    CavNetwork,
    Reactants,
    fuels = FuelManifoldFluid,
    oxidizers = OxManifoldFluid,
    mixture_ratio = 2
)

ChamberGas = Lookup(
    "Combustion Chamber Gas",
    CavNetwork,
    Equilibrium,
    reactants = Props,
    pressure = chamber_pressure
)


FuelVenturi = CavitatingVenturi(
    "Fuel Cavitating Venturi",
    CavNetwork,
    upstream_pressure=FuelSource.pressure,
    downstream_pressure=FuelManifoldFluid.pressure,
    density=FuelSource.density,
    throat_area=(np.pi/4) * (1.0 / 39.37)**2,
    vapor_pressure=FuelSource.saturation_pressure,
    pressure_recovery_factor=0.85,
    cavitating_discharge_coefficient=0.94,
    noncavitating_discharge_coefficient=0.6
)



FuelVenturi = ModelOption(
    "Fuel Venturi",
    
)




FuelInletModel = Model(
    "Fuel Inlet",
    CavNetwork,
)


OxVenturi = CavitatingVenturi(
    "Ox Cavitating Venturi",
    CavNetwork,
    upstream_pressure=LOXSource.pressure,
    downstream_pressure=OxManifoldFluid.pressure,
    density=LOXSource.density,
    throat_area=(np.pi/4) * (1.0 / 39.37)**2,
    vapor_pressure=LOXSource.saturation_pressure,
    pressure_recovery_factor=0.85,
    cavitating_discharge_coefficient=0.94,
    noncavitating_discharge_coefficient=0.6
)


FuelManifold = Volume(
    "Fuel Injector Manifold",
    CavNetwork,
    volume=1,
    pressure=fuel_manifold_pressure,
    mass_flow_in=FuelVenturi.mass_flow
)


OxManifold = Volume(
    "Ox Injector Manifold",
    CavNetwork,
    volume=1,
    pressure=ox_manifold_pressure,
    mass_flow_in=OxVenturi.mass_flow
)

FuelOrf = DischargeCoefficient(
    "Fuel Injector Orifices",
    CavNetwork,
    upstream_pressure=FuelManifold.pressure,
    downstream_pressure=chamber_pressure,
    density=FuelManifoldFluid.density,
    discharge_coefficient=1,
    cross_sectional_area=0.555e-4,
    mass_flow=FuelManifold.mass_flow_out
)

OxOrf = DischargeCoefficient(
    "Ox Injector Orifices",
    CavNetwork,
    upstream_pressure=OxManifold.pressure,
    downstream_pressure=chamber_pressure,
    density=OxManifoldFluid.density,
    discharge_coefficient=1,
    cross_sectional_area=1.25e-4,
    mass_flow=OxManifold.mass_flow_out
)

Props.mixture_ratio = OxOrf.mass_flow / FuelOrf.mass_flow


Chamber = Volume(
    "Combustion Chamber",
    CavNetwork,
    volume=1,
    pressure=chamber_pressure,
    mass_flow_in=OxOrf.mass_flow + FuelOrf.mass_flow
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
    mass_flow=Chamber.mass_flow_out
)

CavNetwork.track("Fuel Injector Manifold Pressure [psia]", fuel_manifold_pressure / 6894.76)
CavNetwork.track("Ox Injector Manifold Pressure [psia]", ox_manifold_pressure / 6894.76)
CavNetwork.track("Chamber Pressure [psia]", chamber_pressure / 6894.76)
CavNetwork.track("Mixture Ratio", Props.mixture_ratio)

SteadyState(CavNetwork).solve(
    verbose=True,
    jacobian_method='2-point',
    statistics=True,
)