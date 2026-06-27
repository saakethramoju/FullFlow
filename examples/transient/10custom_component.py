"""
Simple Custom Transient Components: Draining Tank and Opening Valve
==================================================================

This example shows the smallest useful transient custom-component pattern.
There are only two custom components:

1. ``SimpleTank``
   - owns the dynamic equation for tank level

2. ``SimpleValve``
   - owns the algebraic balance for outlet flow

Physical layout
---------------

      Tank level h(t)

      ┌───────────────┐
      │               │
      │     Tank      │
      │               │
      └───────┬───────┘
              │
              │ Q_out
              ▼
        Opening Valve
              │
              ▼
            Drain

Model notes
-----------

The tank has one dynamic equation:

    dh/dt = -Q_out / A

The valve has one algebraic balance:

    Q_out = opening(t) * K_max * (h_tank - h_drain)

The valve opening is driven by a built-in ``Sequence``. It ramps from closed to
open over the first 5 seconds, then stays open. As the valve opens, the tank
monotonically drains.
"""

from fullflow import *


# ---------------------------------------------------------------------------
# Custom component 1: tank dynamics
# ---------------------------------------------------------------------------
class SimpleTank(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        level: State,
        cross_sectional_area: State | float,
        volume_flow_out: State,
    ):
        self.level_dot = 0.0
        self.setup()

    def evaluate_states(self):
        # Positive outlet flow drains the tank.
        self.level_dot = -self.volume_flow_out.value / self.cross_sectional_area.value

    @property
    def dynamics(self):
        # FullFlow integrates this with backward Euler during the transient.
        return [(self.level, self.level_dot)]


# ---------------------------------------------------------------------------
# Custom component 2: valve algebraic balance
# ---------------------------------------------------------------------------
class SimpleValve(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        upstream_level: State,
        downstream_level: State | float,
        opening: State,
        max_conductance: State | float,
        volume_flow: State,
    ):
        self.flow_error = 0.0
        self.setup()

    def evaluate_states(self):
        # Linear valve law. The solver changes volume_flow until this residual
        # becomes zero.
        target_flow = self.opening.value * self.max_conductance.value * (self.upstream_level.value - self.downstream_level.value)
        self.flow_error = self.volume_flow.value - target_flow

    @property
    def balances(self):
        # FullFlow solves this algebraic equation at every transient timestep.
        return [(self.volume_flow, self.flow_error)]


# ---------------------------------------------------------------------------
# Network and shared states
# ---------------------------------------------------------------------------
TankNetwork = Network("Simple Tank Valve Transient")

level = State(1.0)          # initial tank level
outlet_flow = State(0.0)    # algebraic initial guess
valve_opening = State(0.0)  # driven by the Sequence below


# ---------------------------------------------------------------------------
# Valve opening command
# ---------------------------------------------------------------------------
# The valve starts closed, ramps open over 5 seconds, then stays open.
ValveOpeningSchedule = Sequence(
    "Valve Opening Schedule",
    TankNetwork,
    target=valve_opening,
    times=[0.0, 5.0, 20.0],
    values=[0.0, 1.0, 1.0],
)


# ---------------------------------------------------------------------------
# Custom component instances
# ---------------------------------------------------------------------------
Tank = SimpleTank(
    "Tank",
    TankNetwork,
    level=level,
    cross_sectional_area=1.0,
    volume_flow_out=outlet_flow,
)

OutletValve = SimpleValve(
    "Outlet Valve",
    TankNetwork,
    upstream_level=level,
    downstream_level=0.0,
    opening=valve_opening,
    max_conductance=0.20,
    volume_flow=outlet_flow,
)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
TankNetwork.track("Tank Level", level)
TankNetwork.track("Valve Opening", valve_opening)
TankNetwork.track("Outlet Flow", outlet_flow)


# ---------------------------------------------------------------------------
# Transient solve
# ---------------------------------------------------------------------------
Transient(TankNetwork).solve(
    dt=0.05,
    t_final=20.0,
    filename="simple_tank_valve_transient",
    verbose=True,
)
