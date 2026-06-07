from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class ConstantDensityPump(Component):

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
    Simple centrifugal pump map based on Euler turbomachinery velocity triangles.

    Process
    -------
    1. Compute blade speeds from rotor speed and impeller radii.
    2. Use volumetric efficiency to estimate internal impeller flow.
    3. Use blade angles to estimate tangential velocity components.
    4. Compute ideal Euler head.
    5. Apply hydraulic efficiency to get actual delivered head.
    6. Compute shaft power from ideal Euler power plus mechanical/volumetric losses.

    Notes
    -----
    - head_rise is the actual delivered head.
    - torque is the required shaft torque.
    - shaft_power is the required shaft input power.
    - ConstantDensityPump.efficiency should now evaluate approximately to:

          eta_h * eta_m * eta_v
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
