import numpy as np

from fullflow import *
from thermoprop import *


BangBangSim = Network("Bang Bang Simulation")

psi_to_pa = 6894.76

initial_copv_pressure = 5500 * psi_to_pa
initial_copv_temperature = 300

tank_set_pressure = 450 * psi_to_pa

bang_bang_cda = State(0.0)
ullage_pressure = State(101325)

liquid_volume = State(110 / 1000)
ullage_volume = 125 / 1000 - liquid_volume

Pressurant = Lookup(
    "Pressurant Gas",
    BangBangSim,
    IdealGas,
    "gn2",
    pressure=initial_copv_pressure,
    temperature=initial_copv_temperature,
)

UllageGas = Lookup(
    "Ullage Gas",
    BangBangSim,
    IdealGas,
    "gn2",
    temperature = 300,
    pressure = ullage_pressure,
)

Fuel = Lookup(
    "Fuel",
    BangBangSim,
    Fluid,
    "ch4",
    pressure = UllageGas.pressure,
    temperature = 100
)


NodeFluid = Lookup(
    "Node Fluid",
    BangBangSim,
    Fluid,
    Fuel.fluid,
    pressure = 101325,
    temperature = 100
)



COPV = Volume(
    "COPV",
    BangBangSim,
    volume=20 / 1000,
    pressure=Pressurant.pressure,
    temperature=Pressurant.temperature,
    density=Pressurant.density,
    internal_energy=Pressurant.internal_energy,
    enthalpy=Pressurant.enthalpy,
    energy_variable="T",
)


def bang_bang_condition(t, pressure):
    if pressure > tank_set_pressure + 5 * psi_to_pa:
        return 0.0
    if pressure < tank_set_pressure - 5 * psi_to_pa:
        return 1.0
    return bang_bang_cda.value


BangBangSequence = Sequence(
    "Bang Bang Cd",
    BangBangSim,
    target=bang_bang_cda,
    function=bang_bang_condition,
    inputs=[ullage_pressure],
)



BangBangValve = CompressibleOrifice(
    "Bang Bang Valve",
    BangBangSim,
    upstream_total_pressure=COPV.pressure,
    upstream_total_temperature=Pressurant.temperature,
    downstream_pressure=UllageGas.pressure,
    discharge_coefficient=bang_bang_cda,
    cross_sectional_area=0.01 / 1550,
    gas_constant=Pressurant.gas_constant,
    specific_heat_ratio=Pressurant.gamma,
    upstream_static_enthalpy=Pressurant.enthalpy,
    upstream_static_temperature=Pressurant.temperature,
    mass_flow=COPV.mass_flow_out
)


Ullage = Volume(
    "Ullage",
    BangBangSim,
    volume = ullage_volume,
    pressure=UllageGas.pressure,
    temperature=UllageGas.temperature,
    density=UllageGas.density,
    internal_energy=UllageGas.internal_energy,
    #enthalpy=UllageGas.enthalpy,
    energy_variable="T",
    mass_flow_in=BangBangValve.mass_flow,
    total_enthalpy_in=BangBangValve.total_enthalpy
)


TankLiquid = Volume(
    "Tank Liquid",
    BangBangSim,
    volume=liquid_volume,
    pressure=Fuel.pressure,
    density=Fuel.density,
    mass_flow_out=0,
    solve_volume=True
)


MainLine = DarcyWeisbach(
    "Main Line",
    BangBangSim,
    mass_flow=TankLiquid.mass_flow_out,
    upstream_pressure=TankLiquid.pressure,
    downstream_pressure=NodeFluid.pressure,
    length=3,
    hydraulic_diameter=0.5 / 39.37,
    cross_sectional_area=(np.pi / 4) * (0.5 / 39.37)**2,
    density=TankLiquid.density,
    friction_factor=0.02,
    #height_change=-3
)


Node = Volume(
    "Liquid Manifold",
    BangBangSim,
    volume=(np.pi / 4) * (0.075 / 39.37)**2,
    pressure=NodeFluid.pressure,
    density=NodeFluid.density,
    mass_flow_in=MainLine.mass_flow,
)


main_valve_cd = State(0.0)


main_valve_ramp_time = 0.5


def make_main_valve_sequence():
    opened = False
    open_time = None

    def main_valve_sequence(t, tank_pressure):
        nonlocal opened, open_time

        if not opened and tank_pressure >= tank_set_pressure:
            opened = True
            open_time = t

        if not opened:
            return 0.0

        ramp_fraction = (t - open_time) / main_valve_ramp_time

        if ramp_fraction >= 1.0:
            return 0.6

        return 0.6 * max(0.0, ramp_fraction)

    return main_valve_sequence


MainValveSequence = Sequence(
    "Main Valve Cd",
    BangBangSim,
    target=main_valve_cd,
    function=make_main_valve_sequence(),
    inputs=[ullage_pressure],
)



MainValve = DischargeCoefficient(
    "Main Valve",
    BangBangSim,
    upstream_pressure=Node.pressure,
    downstream_pressure=101325,
    density=Node.density,
    discharge_coefficient=main_valve_cd,
    cross_sectional_area=MainLine.cross_sectional_area,
    mass_flow=Node.mass_flow_out
)

SteadyState(BangBangSim).solve(verbose=True)


Transient(BangBangSim).solve(
    dt = 0.01,
    t_final=25.0,
    filename="BangBang",
    rtol=1e-5
)