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
    "rp-1",
    pressure = UllageGas.pressure,
    temperature = 300
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


BangBangSchedule = Schedule(
    "Bang Bang Cd",
    BangBangSim,
    target=bang_bang_cda,
    function=bang_bang_condition,
    inputs=[ullage_pressure],
)


BangBangValve = DischargeCoefficient(
    "Bang Bang Valve",
    BangBangSim,
    upstream_pressure=COPV.pressure,
    downstream_pressure=UllageGas.pressure,
    density=Pressurant.density,
    discharge_coefficient=bang_bang_cda,
    cross_sectional_area=0.0015 / 1550,
    mass_flow=COPV.mass_flow_out
)


Ullage = Volume(
    "Ullage",
    BangBangSim,
    volume = 7.5 / 1000,
    pressure=UllageGas.pressure,
    density=UllageGas.density,
    mass_flow_in=BangBangValve.mass_flow 
)


TankLiquid = Volume(
    "Tank Liquid",
    BangBangSim,
    volume=67.5 / 1000,
    pressure=Fuel.pressure,
    density=Fuel.density,
    mass_flow_out=0
)


MainLine = DarcyWeisbach(
    "Main Line",
    BangBangSim,
    mass_flow=0,
    upstream_pressure=Ullage.pressure,
    downstream_pressure=101325,
    length=3,
    hydraulic_diameter=0.75 / 39.37,
    cross_sectional_area=(np.pi / 4) * (0.75 / 39.37)**2,
    density=Fuel.density,
    friction_factor=2e-4,
    height_change=-3
)


SteadyState(BangBangSim).solve(verbose=True)


Transient(BangBangSim).solve(
    dt = 0.01,
    t_final=50.0,
    filename="BangBang"
)