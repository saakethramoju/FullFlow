from fullflow import *
import fullplot as fplt
from thermoprop import *
import math

psia_to_pa = 6894.76
in_to_m = 1 / 39.37
in2_to_m2 = 1 / 1550
in3_to_m3 = in_to_m**3
lbf_to_n = 4.44822



filename = "23pid_throttle"
generate_combustion_map = False



if generate_combustion_map:
    map_fuel = Propellant("rp-1", temperature=300)
    map_ox = Propellant("lox", temperature=90)

    def rp1_lox_products(chamber_pressure, mixture_ratio):
        reactants = Reactants(
            fuels=map_fuel,
            oxidizers=map_ox,
            mixture_ratio=mixture_ratio,
        )

        gas = Equilibrium(
            reactants=reactants,
            pressure=chamber_pressure,
        )

        return {
            "temperature": gas.temperature,
            "gamma": gas.gamma,
            "gas_constant": gas.gas_constant,
            "density": gas.density,
        }

    fplt.generate_map(
        filename,
        group="products",
        axes=[
            fplt.Axis.linear(
                "chamber_pressure",
                start=250 * psia_to_pa,
                stop=420 * psia_to_pa,
                count=30,
                units="Pa",
            ),
            fplt.Axis.linear(
                "mixture_ratio",
                start=1.5,
                stop=3.5,
                count=30,
            ),
        ],
        evaluate=rp1_lox_products,
        overwrite=True,
        raise_errors=True,
    )







Engine = Network("Engine")


fuel = Fluid("RP1", pressure=400*psia_to_pa, temperature=300)
ox = Fluid("LOX", pressure=400*psia_to_pa, quality=0.0)


InjFuel = Lookup(
    "Injector Manifold Fuel",
    Engine,
    Propellant,
    "rp-1",
    pressure = 380 * psia_to_pa,
    temperature = 300
)

InjOx = Lookup(
    "Injector Manifold Oxidizer",
    Engine,
    Propellant,
    "lox",
    pressure = 380 * psia_to_pa,
    temperature = 90
)

chamber_pressure = State(350 * psia_to_pa)
mixture_ratio = State(2.0)

ChamberGas = Map.from_hdf5(
    "Chamber Gas Map",
    Engine,
    filename,
    group="products",
    inputs={
        "chamber_pressure": chamber_pressure,
        "mixture_ratio": mixture_ratio,
    },
)

'''
FuelMainValve = Sequence(
    "Fuel Main Valve Cd",
    Engine,
    times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
    values= [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

OxMainValve = Sequence(
    "Oxidizer Main Valve Cd",
    Engine,
    times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
    values= [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
'''

ox_cd = State(0.5)


FuelMain = DischargeCoefficient(
    "Fuel Main Line",
    Engine,
    upstream_pressure=fuel.pressure,
    downstream_pressure=InjFuel.pressure,
    density=fuel.density,
    discharge_coefficient=0.75,
    cross_sectional_area=(math.pi/4) * (1.0 * in_to_m)**2,
    #length=4,
    #mass_flow=0
)


OxMain = DischargeCoefficient(
    "Oxidizer Main Line",
    Engine,
    upstream_pressure=ox.pressure,
    downstream_pressure=InjOx.pressure,
    density=ox.density,
    discharge_coefficient=ox_cd,
    cross_sectional_area=(math.pi/4) * (1.0 * in_to_m)**2,
    #length=4,
    #mass_flow=0
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
    downstream_pressure=chamber_pressure,
    density=InjFuel.density,
    discharge_coefficient=0.6,
    cross_sectional_area=0.555 * 1e-4,
    mass_flow=FuelManifold.mass_flow_out
)

OxOrf = DischargeCoefficient(
    "Injector Ox Orifices",
    Engine,
    upstream_pressure=InjOx.pressure,
    downstream_pressure=chamber_pressure,
    density=InjOx.density,
    discharge_coefficient=0.6,
    cross_sectional_area=1.25 * 1e-4,
    mass_flow=OxManifold.mass_flow_out
)

mixture_ratio <<= OxOrf.mass_flow / FuelOrf.mass_flow


Chamber = Volume(
    "Combustion Chamber",
    Engine,
    volume=(25 * in2_to_m2) * (8 * in_to_m),
    pressure=chamber_pressure,
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



F = Nozzle.mass_flow * Nozzle.exit_velocity + (Nozzle.exit_static_pressure - Nozzle.ambient_pressure) * Nozzle.expansion_ratio * Nozzle.throat_area


Engine.track("Fuel Injector Pressure [psia]", InjFuel.pressure / psia_to_pa)
Engine.track("Ox Injector Pressure [psia]", InjOx.pressure / psia_to_pa)
Engine.track("Chamber Pressure [psia]", Chamber.pressure / psia_to_pa)

Engine.track("Mixture Ratio", mixture_ratio)
Engine.track("Fuel Mass Flow [kg/s]", FuelOrf.mass_flow)
Engine.track("Ox Mass Flow [kg/s]", OxOrf.mass_flow)

Engine.track("Thrust [lbf]", F / lbf_to_n)


SteadyState(Engine).solve(
    verbose=True,
    statistics=True,
    filename=filename
)



MRController = PID(
    "MR Controller",
    Engine,
    feedback=mixture_ratio,
    setpoint=2.3,
    command=ox_cd,
    proportional_gain=0.8,
    integral_gain=1.0,
    derivative_gain=0.0,
    minimum=1e-3,
)



Transient(Engine).solve(
    dt=0.01, 
    t_final=1, 
    verbose=True, 
    statistics=True,
    filename=filename
)



