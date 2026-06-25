from fullflow import *
from thermoprop import *

COPVBlowdown = Network("COPV Blowdown")

psi_to_pa = 6894.76


Pressurant = Lookup(
    "Pressurant Gas",
    COPVBlowdown,
    IdealGas,
    "gn2",
    pressure=1000 * psi_to_pa,
    temperature=300,
)

COPV = Volume(
    "COPV",
    COPVBlowdown,
    volume=20 / 1000,
    pressure=Pressurant.pressure,
    temperature=Pressurant.temperature,
    density=Pressurant.density,
    internal_energy=Pressurant.internal_energy,
    enthalpy=Pressurant.enthalpy,
    energy_variable="T",
)

valve_diameter = 0.075 / 39.37
valve_area = (np.pi / 4) * valve_diameter**2

Valve = CompressibleOrifice(
    "Valve",
    COPVBlowdown,
    upstream_total_pressure=COPV.pressure,
    upstream_total_temperature=COPV.temperature,
    downstream_pressure=101325,
    discharge_coefficient=1,
    cross_sectional_area=valve_area,
    gas_constant=Pressurant.gas_constant,
    specific_heat_ratio=Pressurant.specific_heat_ratio,
    upstream_static_enthalpy=Pressurant.enthalpy,
    upstream_static_temperature=COPV.temperature,
    mass_flow=COPV.mass_flow_out,
)

Transient(COPVBlowdown).solve(
    dt=0.01,
    t_final=60.0,
    filename="COPVBlowdown",
    verbose=True,
)