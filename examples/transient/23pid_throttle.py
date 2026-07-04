from fullflow import *
import fullplot as fplt
from thermoprop import *
import math

psia_to_pa = 6894.76
in_to_m = 1 / 39.37
in2_to_m2 = 1 / 1550
in3_to_m3 = in2_to_m2 * in2_to_m2
lbf_to_n = 4.44822


Engine = Network("Engine")


fuel = Fluid("RP1", pressure=450*psia_to_pa, temperature=300)
ox = Fluid("LOX", pressure=400*psia_to_pa, quality=0.0)


InjFuel = Lookup(
    "Injector Manifold Fuel",
    Engine,
    Propellant,
    "rp-1",
    pressure = 350 * psia_to_pa,
    temperature = 300
)

InjOx = Lookup(
    "Injector Manifold Oxidizer",
    Engine,
    Propellant,
    "lox",
    pressure = 350 * psia_to_pa,
    temperature = 90
)

InjReactants = Lookup(
    "Injector Reactants",
    Engine,
    Reactants,
    fuels = InjFuel,
    oxidizers = InjOx,
    mixture_ratio = 2.3
)

ChamberGas = Lookup(
    "Chamber Gas",
    Engine,
    Equilibrium,
    reactants = InjReactants,
    pressure = 300 * psia_to_pa
)

FuelMain = DischargeCoefficient(
    "Fuel Main Line",
    Engine,
    upstream_pressure=fuel.pressure,
    downstream_pressure=InjFuel.pressure,
    density=fuel.density,
    discharge_coefficient=0.4,
    cross_sectional_area=(math.pi/4) * (1 * in_to_m)**2,
    #length=4
)


OxMain = DischargeCoefficient(
    "Oxidizer Main Line",
    Engine,
    upstream_pressure=ox.pressure,
    downstream_pressure=InjOx.pressure,
    density=ox.density,
    discharge_coefficient=0.5,
    cross_sectional_area=(math.pi/4) * (1 * in_to_m)**2,
    #length=4
)


FuelManifold = Volume(
    "Injector Fuel Manifold",
    Engine,
    volume= 25 * in3_to_m3,
    pressure=InjFuel.pressure,
    density=InjFuel.density,
    mass_flow_in=FuelMain.mass_flow
)

OxManifold = Volume(
    "Injector Oxidizer Manifold",
    Engine,
    volume= 25 * in3_to_m3,
    pressure=InjOx.pressure,
    density=InjOx.density,
    mass_flow_in=OxMain.mass_flow
)


FuelOrf = DischargeCoefficient(
    "Injector Fuel Orifices",
    Engine,
    upstream_pressure=InjFuel.pressure,
    downstream_pressure=ChamberGas.pressure,
    density=InjFuel.density,
    discharge_coefficient=1,
    cross_sectional_area=0.555 * 1e-4,
    mass_flow=FuelManifold.mass_flow_out
)

OxOrf = DischargeCoefficient(
    "Injector Ox Orifices",
    Engine,
    upstream_pressure=InjOx.pressure,
    downstream_pressure=ChamberGas.pressure,
    density=InjOx.density,
    discharge_coefficient=1,
    cross_sectional_area=1.25 * 1e-4,
    mass_flow=OxManifold.mass_flow_out
)


InjReactants.mixture_ratio = OxOrf.mass_flow / FuelOrf.mass_flow


Chamber = Volume(
    "Combustion Chamber",
    Engine,
    volume = (25 * in2_to_m2) * (8 * in_to_m),
    pressure = ChamberGas.pressure,
    density=ChamberGas.density,
    mass_flow_in=FuelOrf.mass_flow + OxOrf.mass_flow
)

Nozzle = IsentropicNozzle(
    "Nozzle",
    Engine,
    upstream_total_pressure=Chamber.pressure,
    upstream_total_temperature=ChamberGas.temperature,
    ambient_pressure=14.67 * psia_to_pa,
    specific_heat_ratio=ChamberGas.gamma,
    gas_constant=ChamberGas.gas_constant,
    throat_area=6 * in2_to_m2,
    expansion_ratio=6,
    mass_flow=Chamber.mass_flow_out,
)

MRBalance = Balance(
    "Mixture Ratio Balance",
    Engine,
    variable=OxOrf.discharge_coefficient,
    function=InjReactants.mixture_ratio - 2.0,
    #bounds=(0.01, 2.0),
    #keep_feasible=True,
)

F = Nozzle.mass_flow * Nozzle.exit_velocity + (Nozzle.exit_static_pressure - Nozzle.ambient_pressure) * Nozzle.expansion_ratio * Nozzle.throat_area


Engine.track("Fuel Injector Pressure [psia]", InjFuel.pressure / psia_to_pa)
Engine.track("Ox Injector Pressure [psia]", InjOx.pressure / psia_to_pa)
Engine.track("Chamber Pressure [psia]", Chamber.pressure / psia_to_pa)

Engine.track("Mixture Ratio", InjReactants.mixture_ratio)
Engine.track("Fuel Mass Flow [kg/s]", FuelOrf.mass_flow)
Engine.track("Ox Mass Flow [kg/s]", OxOrf.mass_flow)

Engine.track("Thrust [lbf]", F / lbf_to_n)


Transient(Engine).solve(
    dt = 0.1,
    t_final=0.2,
    verbose=True,
    statistics=True
)

SteadyState(Engine).solve(
    verbose=True,
    statistics=True,
)