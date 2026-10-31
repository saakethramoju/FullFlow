"""Transient gas feed system with two volumes and an adjustable outlet valve."""

from types import SimpleNamespace

import fullplot as fplt
from fullflow import *


filename = "2transient"
network = Network("Transient Gas Feed")

source_pressure = State(1.6e5)
inlet_area = State(1.5e-4)
valve_area = State(1.2e-4)
effective_area = State()
transfer_flow = State(0.02)

Sequence(
    "Source Pressure Ramp", network,
    target=source_pressure,
    times=[0.0, 1.0],
    values=[1.6e5, 1.85e5],
)


def gas(pressure):
    return SimpleNamespace(density=pressure / (287 * 300))


Source = Lookup("Source Gas", network, gas, pressure=source_pressure)
Node1Gas = Lookup("Node 1 Gas", network, gas, pressure=State(1.5e5))
Node2Gas = Lookup("Node 2 Gas", network, gas, pressure=State(1.35e5))

Inlet = DischargeCoefficient(
    "Inlet Pipe", network,
    upstream_pressure=Source.pressure,
    downstream_pressure=Node1Gas.pressure,
    density=Source.density,
    discharge_coefficient=0.85,
    cross_sectional_area=inlet_area,
    length=5,
    mass_flow=0.02,
)

Volume(
    "Node 1", network,
    pressure=Node1Gas.pressure,
    volume=0.06,
    density=Node1Gas.density,
    mass_flow_in=Inlet.mass_flow,
    mass_flow_out=transfer_flow,
)

Transfer = DarcyWeisbach(
    "Transfer Pipe", network,
    mass_flow=transfer_flow,
    upstream_pressure=Node1Gas.pressure,
    downstream_pressure=Node2Gas.pressure,
    length=4,
    hydraulic_diameter=0.015,
    density=Node1Gas.density,
    friction_factor=0.02,
)

SeriesCdA(
    "Outlet Restriction", network,
    effective_areas=[valve_area, 1.6e-4],
    effective_area=effective_area,
)

Outlet = DischargeCoefficient(
    "Outlet Pipe", network,
    upstream_pressure=Node2Gas.pressure,
    downstream_pressure=1.1e5,
    density=Node2Gas.density,
    discharge_coefficient=0.85,
    cross_sectional_area=effective_area,
)

Volume(
    "Node 2", network,
    pressure=Node2Gas.pressure,
    volume=0.08,
    density=Node2Gas.density,
    mass_flow_in=Transfer.mass_flow,
    mass_flow_out=Outlet.mass_flow,
)

SteadyState(network).solve()
transient = Transient(network)
transient.solve(dt=0.05, t_final=1.0, filename=filename)

times = [0.2, 0.5, 0.8]
transient.linearize(
    inputs=[inlet_area, valve_area],
    outputs=[Node1Gas.pressure, Node2Gas.pressure, Outlet.mass_flow],
    times=times,
    filename=filename,
)
transient.linearize(
    inputs=[inlet_area, valve_area],
    outputs=[Node1Gas.pressure, Node2Gas.pressure, Outlet.mass_flow],
    filename=filename,
)

file = fplt.open(filename + ".h5")
points = file.linearization(solver="transient", times=times)
final = file.linearization(solver="transient")

print("Final states:", final.states)
print("Final A:\n", final.A)
print("Final B:\n", final.B)

sensitivity = fplt.Trace.from_arrays(
    "Outlet Flow / Valve Area",
    times,
    [point.D[2, 1] for point in points.values()],
)

fplt.plot(
    [sensitivity],
    xlabel="Time [s]",
    ylabel="d(Mass Flow)/d(Area) [kg/(s m^2)]",
    title="Transient Valve Sensitivity",
    save=filename + ".png",
)

fplt.show()