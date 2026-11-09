"""Steady-state sensitivity of a manifold and two restrictions."""

from fullflow import *


filename = "1steady_state"
network = Network("Steady Manifold")

source_pressure = State(4e5)
manifold_pressure = State(2.5e5)
inlet_area = State(1.5e-4)
valve_area = State(1.2e-4)
effective_area = State()

Inlet = DischargeCoefficient(
    "Inlet Pipe", network,
    upstream_pressure=source_pressure,
    downstream_pressure=manifold_pressure,
    density=1000,
    discharge_coefficient=0.85,
    cross_sectional_area=inlet_area,
)

SeriesCdA(
    "Outlet Restriction", network,
    effective_areas=[valve_area, 1.6e-4],
    effective_area=effective_area,
)

Outlet = DischargeCoefficient(
    "Outlet Pipe", network,
    upstream_pressure=manifold_pressure,
    downstream_pressure=101325,
    density=1000,
    discharge_coefficient=0.85,
    cross_sectional_area=effective_area,
)

Volume(
    "Manifold", network,
    pressure=manifold_pressure,
    mass_flow_in=Inlet.mass_flow,
    mass_flow_out=Outlet.mass_flow,
)


def pressure_drop():
    return source_pressure.value - manifold_pressure.value


steady = SteadyState(network)
steady.solve(filename=filename)

linear = steady.linearize(
    inputs=[source_pressure, inlet_area, valve_area],
    outputs=[manifold_pressure, Outlet.mass_flow, pressure_drop],
    filename=filename,
)

linear.print()
print("Direct input/output sensitivities:\n", linear.D)
