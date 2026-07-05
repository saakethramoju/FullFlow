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



runline_length = 5.0
runline_area = 0.5e-4

fuel_throttle_cda = 0.5e-4
ox_throttle_cda = 1.0e-4

fuel_main_cd = fuel_throttle_cda / runline_area
ox_main_cd = ox_throttle_cda / runline_area



fuel = Fluid("RP1", pressure=450 * psia_to_pa, temperature=300)
ox = Fluid("LOX", pressure=400 * psia_to_pa, quality=0.0)


InjFuel = Lookup(
    "Injector Manifold Fuel",
    Engine,
    Propellant,
    "rp-1",
    pressure=350 * psia_to_pa,
    temperature=300,
)

InjOx = Lookup(
    "Injector Manifold Oxidizer",
    Engine,
    Propellant,
    "lox",
    pressure=350 * psia_to_pa,
    temperature=90,
)

chamber_pressure = State(300 * psia_to_pa)
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




ox_cmd = State(ox_main_cd)


PcController = PID(
    "Pc Controller",
    Engine,
    feedback=chamber_pressure / psia_to_pa,
    setpoint=285,
    command=ox_cmd,
    trim=ox_main_cd,
    proportional_gain=0.01,
    integral_gain=0.05,
    derivative_gain=0.0,
    minimum=0.5,
    maximum=3.0,
)

'''
ox_cmd = State(ox_main_cd)

def ox_cd_step(t):
    if t < 1.0:
        return 2.0      # CdA = 2.0 * 0.5e-4 = 1.0e-4

    return 2.3          # CdA = 2.3 * 0.5e-4 = 1.15e-4

OxCdSequence = Sequence(
    "Ox Cd Step",
    Engine,
    target=ox_cmd,
    function=ox_cd_step,
)

'''


fuel_main_mdot = State(1.0)

FuelMain = DischargeCoefficient(
    "Fuel Main Line",
    Engine,
    upstream_pressure=fuel.pressure,
    downstream_pressure=InjFuel.pressure,
    density=fuel.density,
    discharge_coefficient=fuel_main_cd,
    cross_sectional_area=runline_area,
    length=runline_length,
    mass_flow=fuel_main_mdot,
)


ox_main_mdot = State(2.0)

OxMain = DischargeCoefficient(
    "Oxidizer Main Line",
    Engine,
    upstream_pressure=ox.pressure,
    downstream_pressure=InjOx.pressure,
    density=ox.density,
    discharge_coefficient=ox_cmd,
    cross_sectional_area=runline_area,
    length=runline_length,
    mass_flow=ox_main_mdot,
)

FuelManifold = Volume(
    "Injector Fuel Manifold",
    Engine,
    volume= 0.1287,
    pressure=InjFuel.pressure,
    density=InjFuel.density,
    mass_flow_in=FuelMain.mass_flow
)

OxManifold = Volume(
    "Injector Oxidizer Manifold",
    Engine,
    volume= 0.1287,
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
    discharge_coefficient=1.0,
    cross_sectional_area=0.5e-4,
    mass_flow=FuelManifold.mass_flow_out,
)

OxOrf = DischargeCoefficient(
    "Injector Ox Orifices",
    Engine,
    upstream_pressure=InjOx.pressure,
    downstream_pressure=chamber_pressure,
    density=InjOx.density,
    discharge_coefficient=1.0,
    cross_sectional_area=1.0e-4,
    mass_flow=OxManifold.mass_flow_out,
)

mixture_ratio <<= OxOrf.mass_flow / FuelOrf.mass_flow


Chamber = Volume(
    "Combustion Chamber",
    Engine,
    volume=6.0e-2,
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
    throat_area=6.05 * in2_to_m2,
    expansion_ratio=4.7,
    mass_flow=Chamber.mass_flow_out,
)



F = Nozzle.mass_flow * Nozzle.exit_velocity + (Nozzle.exit_static_pressure - Nozzle.ambient_pressure) * Nozzle.expansion_ratio * Nozzle.throat_area


Engine.track("Fuel Injector Pressure [psia]", InjFuel.pressure / psia_to_pa)
Engine.track("Ox Injector Pressure [psia]", InjOx.pressure / psia_to_pa)
Engine.track("Chamber Pressure [psia]", Chamber.pressure / psia_to_pa)

Engine.track("Ox Cd Command", ox_cmd)

Engine.track("Mixture Ratio", mixture_ratio)
Engine.track("Fuel Mass Flow [kg/s]", FuelOrf.mass_flow)
Engine.track("Ox Mass Flow [kg/s]", OxOrf.mass_flow)

Engine.track("Thrust [lbf]", F / lbf_to_n)


SteadyState(Engine).solve(
    verbose=True,
    statistics=True,
    filename=filename
)


Transient(Engine).solve(
    dt=0.01,
    t_final=2.0,
    verbose=True,
    statistics=True,
    filename=filename,
)



result = fplt.open(filename).at("Engine/transient/runs/base/tracks")
#result.tree()

mr = result.trace(y="Mixture_Ratio", x="time", name="Mixture Ratio")
ox_cmd = result.trace(y="Ox Cd Command", x="time", name="Ox Cd", role="command")
pc = result.trace(y="Chamber Pressure [psia]", x="time", name="Chamber Pressure")
fipt = result.trace(y="Fuel Injector Pressure [psia]", x="time", name="Fuel Inj Pressure")
oipt = result.trace(y="Ox Injector Pressure [psia]", x="time", name="Ox Inj Pressure")

result.plot(
    y=ox_cmd,
    y2=[pc, fipt, oipt],
    xlabel="Time [s]",
    ylabel="Oxidizer Discharge Coefficient",
    y2label="Chamber Pressure [psia]",
    title="Pc PID Control",
)
result.plot(
    y=[mr],
    y2=ox_cmd,
    xlabel="Time [s]",
    ylabel="Mixture Ratio",
    y2label="Oxidizer Discharge Coefficient",
    title="Pc PID Control",
)


fplt.show()
