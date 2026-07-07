from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class Rotor(Component):
    """Single-shaft rotor speed dynamic from net torque and polar moment of inertia.

        ``Rotor`` integrates shaft speed in revolutions per minute.  The component
        expects net torque in SI units and moment of inertia in kg*m^2; it converts
        angular acceleration from rad/s^2 to rpm/s.  Steady-state solving drives
        ``net_torque`` to zero through the speed derivative, while transient solving
        integrates ``rotor_speed``."""
    def __init__(
        self,
        name: str,
        network: Network,
        rotor_speed: State, # rpm
        polar_moment_of_inertia: float | None = None,
        net_torque: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        self.rotor_speed_dot = self.net_torque.value / self.polar_moment_of_inertia.value * 30.0 / math.pi

    @property
    def dynamics(self):
        # Rotor speed is a real dynamic state.  SteadyState drives the torque
        # balance to zero through rotor_speed_dot; Transient integrates speed.
        """Return dynamic equations contributed by this component.
        
                A two-item tuple ``(state, derivative)`` means the solver integrates that
                state directly.  A three-item tuple ``(iteration_state, stored_state,
                derivative)`` means the nonlinear solver iterates a convenient state but
                conserves/integrates a different stored quantity.  Steady-state solves
                drive the derivative to zero."""
        return [(self.rotor_speed, self.rotor_speed_dot)]




class GasTurbine(Component):
    """Simple gas turbine power extraction component.

        The component uses a turbine flow parameter to compute mass flow from
        upstream total pressure and temperature, gas constant, and rotor speed.  It
        computes shaft power from torque and angular speed, then derives efficiency
        either from a supplied ideal enthalpy drop or from an ideal-gas pressure
        ratio relation.

        Optional enthalpy states let the turbine update discharge total enthalpy for
        coupled pump-turbine or gas-generator examples."""
    def __init__(self, 
                 name: str,
                 network: Network,
                 rotor_speed: State,
                 torque: State,
                 flow_parameter: State,
                 upstream_total_pressure: State,
                 upstream_total_temperature: State,
                 downstream_pressure: State,
                 gas_constant: float,
                 specific_heat_ratio: float,
                 upstream_total_enthalpy: State | None = None,
                 ideal_total_enthalpy_change: float | None = None,
                 
                 efficiency: State | None = None,
                 discharge_total_enthalpy: State | None = None,
                 shaft_power: State | None = None,
                 mass_flow: State | None = None):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()


    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        N = self.rotor_speed.value
        T = self.torque.value
        FP = self.flow_parameter.value
        Po = self.upstream_total_pressure.value
        To = self.upstream_total_temperature.value
        Pout = self.downstream_pressure.value
        R = self.gas_constant.value
        g = self.specific_heat_ratio.value

        omega = (math.pi / 30.0) * N

        mdot = FP * Po / math.sqrt(R * To)
        shaft_power = T * omega

        self.mass_flow.value = mdot
        self.shaft_power.value = shaft_power

        actual_total_enthalpy_change = shaft_power / mdot

        if self.upstream_total_enthalpy.is_assigned:
            ho_in = self.upstream_total_enthalpy.value
            self.discharge_total_enthalpy.value = ho_in - actual_total_enthalpy_change


        if self.ideal_total_enthalpy_change.is_assigned:
            ideal_total_enthalpy_change = self.ideal_total_enthalpy_change.value
            self.efficiency.value = actual_total_enthalpy_change / ideal_total_enthalpy_change
        else:
            cp = g * R / (g - 1.0)
            ideal_total_enthalpy_change = cp * To * (1.0 - (Pout / Po) ** ((g - 1.0) / g))
            self.efficiency.value = actual_total_enthalpy_change / ideal_total_enthalpy_change





class ConstantDensityPump(Component):
    """Constant-density pump pressure-rise component with optional shaft-power bookkeeping.

        The pump predicts discharge pressure from upstream pressure, density,
        gravitational acceleration, and map head rise.  It exposes an algebraic
        balance that varies ``mass_flow`` until predicted discharge pressure matches
        the connected discharge pressure.  Optional torque and rotor speed inputs
        allow hydraulic power, shaft power, efficiency, volumetric flow, and
        discharge total enthalpy to be calculated."""
    def __init__(self,
                 name: str, 
                 network: Network,
                 mass_flow: State,
                 rotor_speed: State,
                 head_rise: State,
                 density: State,
                 upstream_pressure: State,
                 discharge_pressure: State,
                 torque: State | None = None,
                 upstream_total_enthalpy: State | None = None,
                 discharge_total_enthalpy: State | None = None,
                 gravitational_acceleration: float = 9.80665,
                 efficiency: State | None = None,
                 shaft_power: State | None = None,
                 volumetric_flow: State | None = None,):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        mdot = self.mass_flow.value
        H = self.head_rise.value 
        g = self.gravitational_acceleration.value
        rho = self.density.value
        po_in = self.upstream_pressure.value

        Q = mdot / rho

        self.po_out = po_in + rho * g * H
        self.discharge_pressure_error = self.po_out - self.discharge_pressure.value

        self.volumetric_flow.value = Q

        if self.torque.is_assigned:
            T = self.torque.value
            N = self.rotor_speed.value

            omega = (math.pi / 30.0) * N

            shaft_power = T * omega
            hydraulic_power = rho * g * H * Q

            if abs(shaft_power) > 1e-12:
                eta = hydraulic_power / shaft_power
            else:
                eta = 0.0

            self.efficiency.value = eta
            self.shaft_power.value = shaft_power

            if self.upstream_total_enthalpy.is_assigned and abs(mdot) > 1e-12:
                ho_in = self.upstream_total_enthalpy.value
                dho = shaft_power / mdot
                ho_out = ho_in + dho
                self.discharge_total_enthalpy.value = ho_out

    @property
    def balances(self):
        # This pump has no storage of its own. The solver varies mass_flow until
        # the pump curve predicts the connected discharge pressure.
        """Return algebraic equations contributed by this component.
        
                Each tuple is ``(iteration_variable, residual)``.  Steady-state and
                transient solvers vary the iteration variable until the residual is zero.
                Components without algebraic closure equations return an empty list or do
                not define this property."""
        return [(self.mass_flow, self.discharge_pressure_error)]







class PolytropicPump(Component):

    """Compressible/polytropic pump pressure-rise component.

        This component is intended for pump-map workflows where head rise, upstream
        density, discharge density, and pressure ratio are all relevant.  It computes
        a density-pressure slope, converts head to specific work, predicts discharge
        pressure, and exposes a balance on ``mass_flow``.  Optional shaft-power
        inputs update efficiency and discharge total enthalpy."""
    def __init__(self,
                 name: str, 
                 network: Network,
                 mass_flow: State,
                 rotor_speed: State,
                 head_rise: State, 
                 upstream_pressure: State,
                 discharge_pressure: State,
                 upstream_density: State,
                 discharge_density: State,
                 torque: State | None = None,
                 upstream_total_enthalpy: State | None = None,
                 gravitational_acceleration: float = 9.80665,
                 discharge_total_enthalpy: State | None = None,
                 efficiency: State | None = None,
                 shaft_power: State | None = None):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        H = self.head_rise.value
        mdot = self.mass_flow.value
        g = self.gravitational_acceleration.value
        rho1 = self.upstream_density.value
        rho2 = self.discharge_density.value
        p_in = self.upstream_pressure.value
        p_out = self.discharge_pressure.value

        # Pump maps usually report head in distance units.
        # ROCETS polytropic headrise uses specific work units.
        H_specific = g * H

        pressure_ratio = p_out / p_in
        density_ratio = rho2 / rho1

        log_pressure_ratio = math.log(pressure_ratio)
        log_density_ratio = math.log(density_ratio)

        if log_pressure_ratio != 0.0:
            density_pressure_slope = log_density_ratio / log_pressure_ratio
        elif log_density_ratio == 0.0:
            density_pressure_slope = math.nan
        else:
            density_pressure_slope = math.copysign(math.inf, log_density_ratio)

        beta = 1.0 / (1.0 - density_pressure_slope)

        self.predicted_discharge_pressure = rho2 * (H_specific / beta + p_in / rho1)
        self.discharge_pressure_error = self.predicted_discharge_pressure - self.discharge_pressure.value

        if self.torque.is_assigned:
            N = self.rotor_speed.value
            T = self.torque.value

            omega = (math.pi / 30.0) * N
            shaft_power = T * omega

            hydraulic_power = mdot * H_specific

            if abs(shaft_power) > 1e-12:
                eta = hydraulic_power / shaft_power
            else:
                eta = 0.0

            self.efficiency.value = eta
            self.shaft_power.value = shaft_power

            if self.upstream_total_enthalpy.is_assigned and abs(eta) > 1e-12:
                ho_in = self.upstream_total_enthalpy.value
                dho = H_specific / eta
                ho_out = ho_in + dho
                self.discharge_total_enthalpy.value = ho_out

    @property
    def balances(self):
        # Algebraic pump equation: vary mass_flow until predicted discharge
        # pressure equals the connected discharge pressure.
        """Return algebraic equations contributed by this component.
        
                Each tuple is ``(iteration_variable, residual)``.  Steady-state and
                transient solvers vary the iteration variable until the residual is zero.
                Components without algebraic closure equations return an empty list or do
                not define this property."""
        return [(self.mass_flow, self.discharge_pressure_error)]
