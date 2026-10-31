"""Linearize a two-node feed system during a quasi-steady pressure sweep."""

import fullplot as fplt
from fullflow import *


filename = "3steady_state_sweep"
network = Network("Steady Pressure Sweep")

source_pressure = State(4e5)
node1_pressure = State(3.1e5)
node2_pressure = State(2.1e5)
valve_area = State(1.2e-4)
effective_area = State()

Sequence(
    "Source Pressure Ramp", network,
    target=source_pressure,
    times=[0.0, 1.0],
    values=[4e5, 6e5],
)

Inlet = DischargeCoefficient(
    "Inlet Pipe", network,
    upstream_pressure=source_pressure,
    downstream_pressure=node1_pressure,
    density=1000,
    discharge_coefficient=0.85,
    cross_sectional_area=1.5e-4,
)

Transfer = DischargeCoefficient(
    "Transfer Pipe", network,
    upstream_pressure=node1_pressure,
    downstream_pressure=node2_pressure,
    density=1000,
    discharge_coefficient=0.85,
    cross_sectional_area=1.3e-4,
)

SeriesCdA(
    "Outlet Restriction", network,
    effective_areas=[valve_area, 1.6e-4],
    effective_area=effective_area,
)

Outlet = DischargeCoefficient(
    "Outlet Pipe", network,
    upstream_pressure=node2_pressure,
    downstream_pressure=101325,
    density=1000,
    discharge_coefficient=0.85,
    cross_sectional_area=effective_area,
)

Volume(
    "Node 1", network,
    pressure=node1_pressure,
    mass_flow_in=Inlet.mass_flow,
    mass_flow_out=Transfer.mass_flow,
)
Volume(
    "Node 2", network,
    pressure=node2_pressure,
    mass_flow_in=Transfer.mass_flow,
    mass_flow_out=Outlet.mass_flow,
)

steady = SteadyState(network)
steady.solve(dt=0.1, t_final=1.0, rtol=1e-6, filename=filename)

times = [0.2, 0.5, 0.8, 1.0]
steady.linearize(
    inputs=[source_pressure, valve_area],
    outputs=[node1_pressure, node2_pressure, Outlet.mass_flow],
    times=times,
    filename=filename,
)

points = fplt.open(filename + ".h5").linearization(
    solver="steady_state", times=times,
)

for time, point in points.items():
    print(f"t = {time:g} s, mass flow = {point.y0[2]:.4f} kg/s")
    print("D:\n", point.D)

node1_gain = fplt.Trace.from_arrays(
    "Node 1 Pressure / Valve Area",
    times,
    [point.D[0, 1] for point in points.values()],
)
node2_gain = fplt.Trace.from_arrays(
    "Node 2 Pressure / Valve Area",
    times,
    [point.D[1, 1] for point in points.values()],
)
fplt.plot(
    [node1_gain, node2_gain],
    xlabel="Time [s]",
    ylabel="d(Pressure)/d(Area) [Pa/m²]",
    title="Quasi-Steady Valve Sensitivity",
    save=filename + ".png",
)

fplt.show()