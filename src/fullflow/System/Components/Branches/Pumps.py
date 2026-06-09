from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class ConstantDensityPump(Component):
    """
    Constant-density pump pressure-rise component.

    `ConstantDensityPump` models a pump for an approximately incompressible
    fluid with constant density. The component solves volumetric flow as an
    iteration variable by matching the predicted discharge total pressure to the
    assigned discharge total pressure.

    If upstream total enthalpy is assigned, the component also computes
    discharge total enthalpy from shaft power and mass flow.

    Residuals
    ---------
    pressure_balance : float
        Enforces consistency between predicted and assigned discharge pressure

        ``predicted_discharge_total_pressure - discharge_total_pressure = 0``

    Iteration Variables
    -------------------
    volumetric_flow : State
        Pump volumetric flow rate

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    rotor_speed : State
        Pump rotor speed
    head_rise : State
        Pump head rise
    volumetric_flow : State
        Pump volumetric flow rate
    density : State
        Fluid density
    torque : State
        Shaft torque
    upstream_total_pressure : State
        Upstream total pressure
    discharge_total_pressure : State
        Discharge total pressure
    upstream_total_enthalpy : State, optional
        Upstream total enthalpy
    gravitational_acceleration : float, optional
        Gravitational acceleration

    Outputs
    -------
    discharge_total_enthalpy : State, optional
        Discharge total enthalpy
    efficiency : State, optional
        Pump efficiency
    shaft_power : State, optional
        Shaft power
    mass_flow : State, optional
        Pump mass flow rate

    Notes
    -----
    Angular speed is evaluated from:

        ``omega = pi * rotor_speed / 30``

    Shaft power is evaluated from:

        ``shaft_power = torque * omega``

    Hydraulic power is evaluated from:

        ``hydraulic_power = density * gravity * head_rise * volumetric_flow``

    Discharge total pressure is evaluated from:

        ``discharge_total_pressure = upstream_total_pressure
        + density * gravity * head_rise``

    Efficiency is evaluated from:

        ``efficiency = hydraulic_power / shaft_power``

    Mass flow is evaluated from:

        ``mass_flow = density * volumetric_flow``

    If upstream total enthalpy is assigned, discharge total enthalpy is
    evaluated from:

        ``discharge_total_enthalpy = upstream_total_enthalpy
        + shaft_power / mass_flow``
    """

    def __init__(self,
                 name: str, 
                 network: Network,
                 rotor_speed: State,
                 head_rise: State,
                 volumetric_flow: State,
                 density: State,
                 torque: State,
                 upstream_total_pressure: State,
                 discharge_total_pressure: State,
                 upstream_total_enthalpy: State | None = None,
                 discharge_total_enthalpy: State | None = None,
                 gravitational_acceleration: float = 9.80665,
                 efficiency: State | None = None,
                 shaft_power: State | None = None,
                 mass_flow: State | None = None):
        self.setup()

        self._predicted_discharge_total_pressure = None
        self._predicted_discharge_total_enthalpy = None

    def evaluate_states(self):
        H = self.head_rise.value 
        Q = self.volumetric_flow.value
        g = self.gravitational_acceleration.value
        T = self.torque.value
        rho = self.density.value
        po_in = self.upstream_total_pressure.value
        N = self.rotor_speed.value

        omega = (np.pi / 30.0) * N

        if abs(Q) < 1e-12:
            raise ValueError(f"{self.name}: volumetric_flow is too close to zero.")

        if abs(rho * Q) < 1e-12:
            raise ValueError(f"{self.name}: mass flow is too close to zero.")

        if abs(T * omega) < 1e-12:
            raise ValueError(f"{self.name}: shaft power is too close to zero.")

        shaft_power = T * omega
        hydraulic_power = rho * g * H * Q

        po_out = po_in + rho * g * H
        eta = hydraulic_power / shaft_power
        mdot = rho * Q

        self._predicted_discharge_total_pressure = po_out

        self.efficiency.value = eta
        self.shaft_power.value = shaft_power
        self.mass_flow.value = mdot

        if self.upstream_total_enthalpy.is_assigned:
            ho_in = self.upstream_total_enthalpy.value
            dho = shaft_power / mdot
            ho_out = ho_in + dho

            self._predicted_discharge_total_enthalpy = ho_out

            self.discharge_total_enthalpy.value = ho_out
        else:
            self._predicted_discharge_total_enthalpy = None

    @property
    def iteration_variables(self) -> list[State]:
        return [self.volumetric_flow]
    
    @property
    def residuals(self) -> list[float]:
        return [
            self._predicted_discharge_total_pressure
            - self.discharge_total_pressure.value
        ]




class PolytropicPump(Component):
    """
    Polytropic pump pressure-rise component.

    `PolytropicPump` models a pump with changing density between the inlet and
    discharge states. The component solves mass flow as an iteration variable by
    matching a polytropic pressure-rise estimate to the assigned discharge total
    pressure.

    Residuals
    ---------
    pressure_balance : float
        Enforces consistency between predicted and assigned discharge pressure

        ``predicted_discharge_total_pressure - discharge_total_pressure = 0``

    Iteration Variables
    -------------------
    mass_flow : State
        Pump mass flow rate

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    rotor_speed : State
        Pump rotor speed
    head_rise : State
        Pump head rise
    mass_flow : State
        Pump mass flow rate
    upstream_density : State
        Upstream fluid density
    downstream_density : State
        Downstream fluid density
    torque : State
        Shaft torque
    upstream_total_pressure : State
        Upstream total pressure
    discharge_total_pressure : State
        Discharge total pressure
    upstream_total_enthalpy : State
        Upstream total enthalpy
    gravitational_acceleration : float, optional
        Gravitational acceleration

    Outputs
    -------
    discharge_total_enthalpy : State, optional
        Discharge total enthalpy
    efficiency : State, optional
        Pump efficiency
    shaft_power : State, optional
        Shaft power
    inlet_volumetric_flow : State, optional
        Inlet volumetric flow rate
    outlet_volumetric_flow : State, optional
        Outlet volumetric flow rate

    Notes
    -----
    Angular speed is evaluated from:

        ``omega = pi * rotor_speed / 30``

    Shaft power is evaluated from:

        ``shaft_power = torque * omega``

    Specific head work is evaluated from:

        ``H_specific = gravity * head_rise``

    Hydraulic power is evaluated from:

        ``hydraulic_power = mass_flow * H_specific``

    Efficiency is evaluated from:

        ``efficiency = hydraulic_power / shaft_power``

    Discharge total enthalpy is evaluated from:

        ``discharge_total_enthalpy = upstream_total_enthalpy
        + H_specific / efficiency``

    The polytropic exponent factor is evaluated from:

        ``beta = 1 / (1 - log(downstream_density / upstream_density)
        / log(discharge_total_pressure / upstream_total_pressure))``

    Predicted discharge total pressure is evaluated from:

        ``predicted_discharge_total_pressure = downstream_density
        * (H_specific / beta + upstream_total_pressure / upstream_density)``

    Inlet and outlet volumetric flow rates are evaluated from:

        ``inlet_volumetric_flow = mass_flow / upstream_density``

        ``outlet_volumetric_flow = mass_flow / downstream_density``
    """

    def __init__(self,
                 name: str, 
                 network: Network,
                 rotor_speed: State,
                 head_rise: State,  # meters
                 mass_flow: State,
                 upstream_density: State,
                 downstream_density: State,
                 torque: State,
                 upstream_total_pressure: State,
                 discharge_total_pressure: State,
                 upstream_total_enthalpy: State,
                 discharge_total_enthalpy: State | None = None,
                 gravitational_acceleration: float = 9.80665,
                 efficiency: State | None = None,
                 shaft_power: State | None = None,
                 inlet_volumetric_flow: State | None = None,
                 outlet_volumetric_flow: State | None = None):
        self.setup()
    
        self._predicted_discharge_total_pressure = None

    def evaluate_states(self):
        H_m = self.head_rise.value
        mdot = self.mass_flow.value
        g = self.gravitational_acceleration.value
        T = self.torque.value
        rho1 = self.upstream_density.value
        rho2 = self.downstream_density.value
        po_in = self.upstream_total_pressure.value
        po_out = self.discharge_total_pressure.value
        ho_in = self.upstream_total_enthalpy.value
        N = self.rotor_speed.value

        omega = (np.pi / 30.0) * N
        shaft_power = T * omega

        if abs(mdot) < 1e-12:
            raise ValueError(f"{self.name}: mass_flow is too close to zero.")

        if abs(shaft_power) < 1e-12:
            raise ValueError(f"{self.name}: shaft_power is too close to zero.")

        if rho1 <= 0.0 or rho2 <= 0.0:
            raise ValueError(f"{self.name}: densities must be positive.")

        if po_in <= 0.0 or po_out <= 0.0:
            raise ValueError(f"{self.name}: pressures must be positive.")

        # Pump maps usually report head in meters.
        # ROCETS polytropic headrise uses specific work units: J/kg = m^2/s^2.
        H_specific = g * H_m

        hydraulic_power = mdot * H_specific
        eta = hydraulic_power / shaft_power

        if abs(eta) < 1e-12:
            raise ValueError(f"{self.name}: efficiency is too close to zero.")

        dho = H_specific / eta
        ho_out = ho_in + dho

        pressure_ratio = po_out / po_in
        density_ratio = rho2 / rho1

        log_pressure_ratio = np.log(pressure_ratio)

        #if abs(log_pressure_ratio) < 1e-12:
        #    raise ValueError(f"{self.name}: pressure ratio is too close to 1 for beta calculation.")

        beta = 1.0 / (
            1.0 - np.log(density_ratio) / log_pressure_ratio
        )

        self._predicted_discharge_total_pressure = rho2 * (
            H_specific / beta + po_in / rho1
        )

        self.discharge_total_enthalpy.value = ho_out
        self.efficiency.value = eta
        self.shaft_power.value = shaft_power
        self.inlet_volumetric_flow.value = mdot / rho1
        self.outlet_volumetric_flow.value = mdot / rho2

    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]
    
    @property
    def residuals(self) -> list[float]:
        return [
            self._predicted_discharge_total_pressure
            - self.discharge_total_pressure.value
        ]
    



class SimpleEulerCentrifugalPump(Component):
    """
    Simple Euler centrifugal pump map.

    `SimpleEulerCentrifugalPump` computes centrifugal pump head rise, torque,
    shaft power, stagnation pressure rise, and mass flow using ideal Euler
    turbomachinery velocity triangles with hydraulic, mechanical, volumetric,
    and slip-factor corrections.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    rotor_speed : State
        Pump rotor speed
    volumetric_flow : State
        Delivered volumetric flow rate
    density : State
        Fluid density
    impeller_inlet_tip_radius : float
        Impeller inlet tip radius
    impeller_outlet_tip_radius : float
        Impeller outlet tip radius
    inlet_annular_flow_area : float
        Inlet annular flow area
    outlet_annular_flow_area : float
        Outlet annular flow area
    inlet_blade_angle : float
        Inlet blade angle
    outlet_blade_angle : float
        Outlet blade angle
    angle_units : str, optional
        Blade angle units
    gravitational_acceleration : float, optional
        Gravitational acceleration
    slip_factor : float, optional
        Slip factor
    hydraulic_efficiency : float or State, optional
        Hydraulic efficiency
    mechanical_efficiency : float or State, optional
        Mechanical efficiency
    volumetric_efficiency : float or State, optional
        Volumetric efficiency

    Outputs
    -------
    head_rise : State, optional
        Delivered pump head rise
    torque : State, optional
        Required shaft torque
    shaft_power : State, optional
        Required shaft power
    stagnation_pressure_rise : State, optional
        Stagnation pressure rise
    mass_flow : State, optional
        Pump mass flow rate

    Notes
    -----
    Angular speed is evaluated from:

        ``omega = pi * rotor_speed / 30``

    Blade speeds are evaluated from:

        ``U1 = omega * impeller_inlet_tip_radius``

        ``U2 = omega * impeller_outlet_tip_radius``

    Internal impeller flow is evaluated from:

        ``Q_impeller = volumetric_flow / volumetric_efficiency``

    Meridional through-flow velocities are evaluated from:

        ``Cm1 = Q_impeller / inlet_annular_flow_area``

        ``Cm2 = Q_impeller / outlet_annular_flow_area``

    Tangential velocity components are evaluated from:

        ``V_theta1 = U1 - Cm1 / tan(inlet_blade_angle)``

        ``V_theta2 = slip_factor * U2 - Cm2 / tan(outlet_blade_angle)``

    Ideal Euler specific work is evaluated from:

        ``specific_work_euler = U2 * V_theta2 - U1 * V_theta1``

    Ideal Euler head is evaluated from:

        ``H_euler = specific_work_euler / gravity``

    Delivered head is evaluated from:

        ``head_rise = hydraulic_efficiency * H_euler``

    Shaft power is evaluated from:

        ``shaft_power = density * gravity * H_euler * volumetric_flow
        / (mechanical_efficiency * volumetric_efficiency)``

    Shaft torque is evaluated from:

        ``torque = shaft_power / omega``

    Stagnation pressure rise is evaluated from:

        ``stagnation_pressure_rise = density * gravity * head_rise``

    Mass flow is evaluated from:

        ``mass_flow = density * volumetric_flow```

    The efficiency computed by `ConstantDensityPump` should evaluate
    approximately to:

        ``efficiency = hydraulic_efficiency
        * mechanical_efficiency
        * volumetric_efficiency``
    """
    def __init__(self, 
                 name: str, 
                 network: Network,
                 rotor_speed: State,
                 volumetric_flow: State,
                 density: State,
                 impeller_inlet_tip_radius: float,
                 impeller_outlet_tip_radius: float,
                 inlet_annular_flow_area: float,
                 outlet_annular_flow_area: float,
                 inlet_blade_angle: float,
                 outlet_blade_angle: float,
                 angle_units: str = "degrees",
                 gravitational_acceleration: float = 9.80665,
                 slip_factor: float = 1.0,
                 hydraulic_efficiency: float | State = 1.0,
                 mechanical_efficiency: float | State = 1.0,
                 volumetric_efficiency: float | State = 1.0,
                 head_rise: State | None = None,
                 torque: State | None = None,
                 shaft_power: State | None = None,
                 stagnation_pressure_rise: State | None = None,
                 mass_flow: State | None = None):
        self.setup()

    def pre_evaluation(self):
        self.evaluate_states()

    def evaluate_states(self):
        N = self.rotor_speed.value
        Q = self.volumetric_flow.value
        rho = self.density.value

        r1 = self.impeller_inlet_tip_radius.value
        r2 = self.impeller_outlet_tip_radius.value
        A1 = self.inlet_annular_flow_area.value
        A2 = self.outlet_annular_flow_area.value

        beta1 = self.inlet_blade_angle.value
        beta2 = self.outlet_blade_angle.value

        sigma = self.slip_factor.value
        g = self.gravitational_acceleration.value

        eta_h = self.hydraulic_efficiency.value
        eta_m = self.mechanical_efficiency.value
        eta_v = self.volumetric_efficiency.value

        if self.angle_units.lower() in {"degree", "degrees", "deg"}:
            beta1 = np.deg2rad(beta1)
            beta2 = np.deg2rad(beta2)
        elif self.angle_units.lower() in {"radian", "radians", "rad"}:
            pass
        else:
            raise ValueError(f"{self.name}: angle_units must be 'degrees' or 'radians'.")

        if abs(N) < 1e-12:
            raise ValueError(f"{self.name}: rotor_speed must be nonzero.")

        if eta_h <= 0.0 or eta_m <= 0.0 or eta_v <= 0.0:
            raise ValueError(f"{self.name}: pump efficiencies must be positive.")

        if abs(A1) < 1e-12 or abs(A2) < 1e-12:
            raise ValueError(f"{self.name}: annular flow areas must be nonzero.")

        omega = np.pi * N / 30.0

        # Blade speeds.
        U1 = omega * r1
        U2 = omega * r2

        # The impeller internally processes more flow than delivered if eta_v < 1.
        Q_impeller = Q / eta_v

        # Meridional through-flow velocities.
        Cm1 = Q_impeller / A1
        Cm2 = Q_impeller / A2

        # Tangential velocity components from velocity triangles.
        V_theta1 = U1 - Cm1 / np.tan(beta1)
        V_theta2 = sigma * U2 - Cm2 / np.tan(beta2)

        # Ideal Euler work and ideal head.
        specific_work_euler = U2 * V_theta2 - U1 * V_theta1
        H_euler = specific_work_euler / g

        # Actual delivered head after hydraulic losses.
        H_actual = eta_h * H_euler

        # Shaft power is based on ideal Euler power, then mechanical/volumetric losses.
        # This makes hydraulic_output_power / shaft_power = eta_h * eta_m * eta_v.
        ideal_euler_power = rho * g * H_euler * Q
        shaft_power = ideal_euler_power / (eta_m * eta_v)

        torque = shaft_power / omega
        stagnation_pressure_rise = rho * g * H_actual
        mass_flow = rho * Q

        self.head_rise.value = H_actual
        self.shaft_power.value = shaft_power
        self.torque.value = torque
        self.stagnation_pressure_rise.value = stagnation_pressure_rise
        self.mass_flow.value = mass_flow
