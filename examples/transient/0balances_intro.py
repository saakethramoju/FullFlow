from fullflow import *
from thermoprop import *


Test = Network("Test")


source_pressure = State(3e5)
node_pressure = State(101325.0)
downstream_pressure = State(101325.0)

valve_cda = State(1.0e-12)
outlet_cda = State(0.0)

def valve_ramp(t):
    if t < 0.0:
        return 1.0e-12

    if t < 1.0:
        return 1.0e-12 + t

    return 1.0


def outlet_bang_bang(t, pressure):
    if pressure > 2.2e5:
        return 1.0

    if pressure < 1.8e5:
        return 0.0

    return outlet_cda.value


ValveCdASchedule = Schedule(
    "Valve CdA",
    Test,
    target=valve_cda,
    function=valve_ramp,
)


OutletCdASchedule = Schedule(
    "Outlet CdA",
    Test,
    target=outlet_cda,
    function=outlet_bang_bang,
    inputs=[node_pressure],
)


SourceFluid = Lookup(
    "Source Water",
    Test,
    Fluid,
    "water",
    pressure=source_pressure,
    temperature=300,
)


NodeFluid = Lookup(
    "Node Water",
    Test,
    Fluid,
    "water",
    pressure=node_pressure,
    temperature=300,
)


Inlet = DischargeCoefficient(
    "Inlet",
    Test,
    upstream_pressure=source_pressure,
    downstream_pressure=node_pressure,
    density=SourceFluid.density,
    discharge_coefficient=valve_cda,
    cross_sectional_area=1.0e-5,
    length=1,
    mass_flow=0,
)


Outlet = DischargeCoefficient(
    "Outlet",
    Test,
    upstream_pressure=node_pressure,
    downstream_pressure=downstream_pressure,
    density=NodeFluid.density,
    discharge_coefficient=outlet_cda,
    cross_sectional_area=1.0e-5,
)


Node = Volume(
    "Node",
    Test,
    pressure=node_pressure,
    volume=1.0e-1,
    density=NodeFluid.density,
    mass_flow_in=Inlet.mass_flow,
    mass_flow_out=Outlet.mass_flow,
)


'''
def node_pressure_error():
    return node_pressure.value / desired_node_pressure.value - 1.0


OutletAreaBalance = Balance(
    "Outlet Area Balance",
    Test,
    variable=outlet_area,
    function=node_pressure_error,
)
'''

Transient(Test).solve(
    dt=0.001,
    t_final=3.0,
    filename="test",
    verbose=True,
    statistics=True
)