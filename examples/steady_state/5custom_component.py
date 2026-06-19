"""
Custom Component Example: Liquid Oxygen Pump Feed System
========================================================

This example demonstrates how to create and use custom FullFlow components in
a steady-state liquid oxygen pump system.

Network Layout
--------------
    LOX Source
        |
        v
    CustomRestriction
        |
        v
    Pump Inlet Volume
        |
        v
    CustomPump
        |
        v
    Discharge Pressure Target

The model represents liquid oxygen flowing from a source condition through a
fitting or restriction into a pump inlet node. A custom pump component then
raises the pressure to a specified discharge pressure.

This example is intended to teach the basic structure of custom FullFlow
components.

Two custom components are defined:

    CustomRestriction
        A simple square-law hydraulic restriction. It computes mass flow
        directly from pressure drop, density, area, and loss coefficient.

    CustomPump
        A simple pump curve component. It uses a quadratic head curve as a
        function of volumetric flow rate and solves for the mass flow required
        to hit a requested discharge pressure.

Together, the system forms a coupled nonlinear problem:

    Restriction:
        mdot = sign(ΔP) A sqrt(2 ρ |ΔP| / K)

    Pump:
        Q = mdot / ρ

        H = A Q² + B Q + C

        P2 = P1 + ρ g H

    Pump residual:
        P2_predicted - discharge_pressure = 0

The steady-state solver adjusts the pump inlet pressure and pump mass flow until
the pump inlet volume mass balance and the pump pressure-rise equation are both
satisfied.

Demonstrates
------------
- Creating custom FullFlow components
- Using self.setup() to automatically register inputs and outputs
- Direct state evaluation inside evaluate_states()
- Defining custom iteration variables
- Defining custom residual equations
- Coupling custom components to built-in FullFlow components
- Chaining ThermoProp lookups using .composition
- Tracking scalar values and object attributes
"""

import numpy as np

from fullflow import *
from thermoprop import *


# =============================================================================
# Custom Components
# =============================================================================

class CustomRestriction(Component):
    """
    Simple incompressible square-law restriction.

    This component models pressure loss through a fitting, valve, bend, or other
    hydraulic restriction using the standard loss-coefficient relation:

        ΔP = K ρ V² / 2

    where:

        ΔP  = upstream pressure - downstream pressure
        K   = loss coefficient
        ρ   = fluid density
        V   = flow velocity

    Since:

        V = mdot / (ρ A)

    the mass flow can be written as:

        mdot = sign(ΔP) A sqrt(2 ρ |ΔP| / K)

    This component does not define a residual equation. Instead, it directly
    calculates the mass flow from the current pressure drop. This makes it a good
    first example of a custom component that evaluates an output state.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        FullFlow network that owns this component.
    upstream_pressure : State
        Pressure upstream of the restriction [Pa].
    downstream_pressure : State
        Pressure downstream of the restriction [Pa].
    density : State
        Fluid density used in the restriction calculation [kg/m^3].
    cross_sectional_area : float
        Restriction flow area [m^2].
    loss_coefficient : float
        Dimensionless loss coefficient K.
    mass_flow : State, optional
        Mass flow through the restriction [kg/s]. If not supplied, FullFlow
        creates it automatically.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_pressure: State,
        downstream_pressure: State,
        density: State,
        cross_sectional_area: float,
        loss_coefficient: float,
        mass_flow: State | None = None,
    ):
        # self.setup() automatically:
        #
        # - stores all __init__ arguments as component attributes
        # - wraps plain numeric values into State objects when appropriate
        # - creates optional output states such as mass_flow when None is passed
        # - registers this component with the network
        self.setup()

    def evaluate_states(self):
        """
        Calculate restriction mass flow from the current pressure drop.

        This method is called repeatedly by the solver while the nonlinear
        system is being evaluated.
        """

        # Current pressure drop across the restriction.
        #
        # Positive dP means flow from upstream to downstream.
        # Negative dP means reverse flow.
        dP = self.upstream_pressure.value - self.downstream_pressure.value

        # Resolve connected FullFlow states to numeric values.
        rho = self.density.value
        A = self.cross_sectional_area.value
        K = self.loss_coefficient.value

        # Avoid sqrt(0) sign ambiguity when the pressure drop is exactly zero.
        if dP == 0:
            mass_flow = 0.0
        else:
            # Square-law restriction relation.
            #
            # np.sign(dP) preserves the direction of flow.
            # abs(dP) keeps the square root positive.
            mass_flow = np.sign(dP) * A * np.sqrt(2.0 * rho * abs(dP) / K)

        # Write the calculated value into the component's mass_flow state.
        self.mass_flow.value = mass_flow


class CustomPump(Component):
    """
    Simple liquid pump with a quadratic head-flow curve.

    This component models a pump using a simple curve for pump head as a
    function of volumetric flow rate:

        H = A Q² + B Q + C

    where:

        H = pump head rise [m]
        Q = volumetric flow rate [m^3/s]

    The pressure rise is then:

        ΔP = ρ g H

    and the predicted discharge pressure is:

        P2_predicted = P1 + ρ g H

    Unlike CustomRestriction, this component defines a residual equation. The
    solver varies mass_flow until the predicted discharge pressure matches the
    requested discharge pressure:

        P2_predicted - discharge_pressure = 0

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        FullFlow network that owns this component.
    mass_flow : State
        Pump mass flow rate [kg/s]. This is the iteration variable.
    density : State
        Pump inlet fluid density [kg/m^3].
    upstream_pressure : State
        Pump inlet pressure [Pa].
    discharge_pressure : State
        Required pump discharge pressure [Pa].
    gravitational_acceleration : float, optional
        Gravitational acceleration [m/s^2].
    """

    def __init__(
        self, 
        name: str,
        network: Network,
        mass_flow: State,
        density: State,
        upstream_pressure: State,
        discharge_pressure: State,
        gravitational_acceleration: float = 9.80665
    ):
        # self.setup() stores all inputs, wraps numeric values, and registers
        # the component with the network.
        self.setup()

    def evaluate_states(self):
        """
        Evaluate the pump curve and calculate predicted discharge pressure.
        """

        # Resolve connected states.
        mdot = self.mass_flow.value
        rho = self.density.value
        P1 = self.upstream_pressure.value
        g = self.gravitational_acceleration.value

        # Pump curve coefficients.
        #
        # These are tutorial/example coefficients chosen to produce a reasonable
        # LOX pump pressure rise near 5-10 kg/s.
        #
        # H = A Q² + B Q + C
        #
        # A < 0 gives decreasing head as flow increases, which is typical for a
        # simple centrifugal-pump-style head curve.
        A = -2.6e5
        B = 0.0
        C = 40.0

        # Convert mass flow to volumetric flow.
        Q = mdot / rho

        # Pump head rise from the curve [m].
        H = A*Q**2 + B*Q + C

        # Convert head rise to pressure rise and add it to inlet pressure.
        self.P2_predicted = P1 + rho*g*H

    @property
    def iteration_variables(self):
        """
        States that the nonlinear solver is allowed to vary.

        Here the solver changes pump mass flow until the discharge pressure
        residual is satisfied.
        """
        return [self.mass_flow]

    @property
    def residuals(self):
        """
        Residual equations enforced by the nonlinear solver.

        The pump is solved when:

            P2_predicted = discharge_pressure
        """
        return [self.P2_predicted - self.discharge_pressure.value]


# =============================================================================
# Build Network
# =============================================================================

# Create the FullFlow network that owns all components, states, balances, and
# tracked outputs in this example.
LOXPumpSystem = Network("Liquid Oxygen Pump System")


# -----------------------------------------------------------------------------
# Source LOX property lookup
#
# ThermoProp's Propellant wrapper calculates liquid oxygen properties at the
# source condition.
#
# The source is specified by:
#
#     T = 90.17 K
#     P = 50 psia
#
# The pressure conversion uses:
#
#     1 psi = 6894.76 Pa
#
# Outputs such as:
#
#     LOX.pressure
#     LOX.temperature
#     LOX.density
#     LOX.composition
#
# behave like FullFlow state-like attributes and can be connected directly into
# components.
# -----------------------------------------------------------------------------

LOX = Lookup(
    "Liquid Oxygen",
    LOXPumpSystem,
    Propellant,
    "lox",
    temperature = 90.17,
    pressure = 50 * 6894.76
)


# -----------------------------------------------------------------------------
# Pump inlet / node LOX property lookup
#
# This lookup represents the fluid state at the pump inlet node.
#
# Instead of hardcoding "lox" again, it uses:
#
#     LOX.composition
#
# This demonstrates lookup chaining. The node fluid uses the same fluid identity
# as the source fluid, but evaluates it at a different pressure.
#
# The node pressure starts at 45 psia, but because NodeFluid.pressure is also
# connected to the Volume component, the steady-state solver can change it.
# -----------------------------------------------------------------------------

NodeFluid = Lookup(
    "Node LOX",
    LOXPumpSystem,
    Propellant,
    LOX.composition,
    temperature = 90.17,
    pressure = 45 * 6894.76
)


# Track the source density so it appears in the final solution summary.
LOXPumpSystem.track("Source LOX Density", LOX.density)


# -----------------------------------------------------------------------------
# Inlet fitting / restriction
#
# This custom component calculates the mass flow from the LOX source to the pump
# inlet node.
#
# It uses the source pressure, node pressure, source density, fitting area, and
# loss coefficient.
#
# Since mass_flow is not provided, FullFlow creates:
#
#     Fitting.mass_flow
#
# automatically.
# -----------------------------------------------------------------------------

Fitting = CustomRestriction(
    "Fitting",
    LOXPumpSystem,
    upstream_pressure=LOX.pressure,
    downstream_pressure=NodeFluid.pressure,
    density=LOX.density,
    cross_sectional_area=(np.pi/4) * (1.5 / 39.37)**2,
    loss_coefficient=1.5652,
)


# -----------------------------------------------------------------------------
# Pump inlet volume
#
# The Volume component enforces mass conservation at the pump inlet node:
#
#     mass_flow_in - mass_flow_out = 0
#
# The inlet flow comes from the custom restriction:
#
#     mass_flow_in = Fitting.mass_flow
#
# The outlet flow goes into the pump:
#
#     mass_flow_out = Node.mass_flow_out
#
# An initial guess of 5 kg/s is supplied for mass_flow_out.
#
# The pressure of this volume is the same state as NodeFluid.pressure, so the
# solver can adjust the pump inlet pressure while keeping the ThermoProp lookup
# connected to the same pressure state.
# -----------------------------------------------------------------------------

Node = Volume(
    "Pump Inlet",
    LOXPumpSystem,
    pressure=NodeFluid.pressure,
    volume=1,
    mass_flow_in=Fitting.mass_flow,
    mass_flow_out=5
)


# -----------------------------------------------------------------------------
# Custom LOX pump
#
# The pump receives mass flow from the pump inlet volume and uses the node fluid
# density to calculate pressure rise.
#
# The discharge pressure is specified as:
#
#     80 psia
#
# The solver varies:
#
#     LOXPump.mass_flow
#
# which is the same state as:
#
#     Node.mass_flow_out
#
# until:
#
#     LOXPump.P2_predicted = LOXPump.discharge_pressure
# -----------------------------------------------------------------------------

LOXPump = CustomPump(
    "LOX Pump",
    LOXPumpSystem,
    mass_flow=Node.mass_flow_out,
    density=NodeFluid.density,
    upstream_pressure=Node.pressure,
    discharge_pressure=80 * 6894.76,
)


# -----------------------------------------------------------------------------
# Track node fluid properties
#
# This tracks several attributes from the NodeFluid lookup output object.
#
# The labels on the left are user-facing names in the solution table.
# The strings on the right are attribute names on the Propellant object.
# -----------------------------------------------------------------------------

LOXPumpSystem.track(
    "Node Fluid Characteristics",
    NodeFluid,
    attributes={
        "Pressure [Pa]": "pressure",
        "Temperature [K]": "temperature",
        "Density [kg/m^3]": "density",
        "Dynamic Viscosity [Pa-s]": "dynamic_viscosity",
        "Saturation Pressure [Pa]": "saturation_pressure"
    }
)


# -----------------------------------------------------------------------------
# Derived pump performance quantities
#
# FullFlow state-like objects support algebraic operations. These expressions
# create derived states that update automatically during and after the solve.
#
# Pressure rise:
#
#     dP = P_discharge - P_inlet
#
# Head rise:
#
#     H = dP / (ρ g)
#
# The tracked values are converted to common engineering units:
#
#     H  [m]  -> [ft]
#     dP [Pa] -> [psid]
# -----------------------------------------------------------------------------

dP = LOXPump.discharge_pressure - LOXPump.upstream_pressure
H = dP / (LOXPump.density * LOXPump.gravitational_acceleration)

LOXPumpSystem.track("Head Rise [ft]", H*3.28084)
LOXPumpSystem.track("Pressure Rise [psid]", dP/6894.76)


# -----------------------------------------------------------------------------
# Solve network
#
# The steady-state solver iterates on the pump inlet pressure and pump mass flow
# until:
#
#     1. Pump inlet mass conservation is satisfied.
#     2. Pump discharge pressure matches the target value.
#
# verbose=True prints the solver summary and final network solution.
# -----------------------------------------------------------------------------

SteadyState(LOXPumpSystem).solve(verbose=True)