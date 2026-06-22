from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class Rotor(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        rotor_speed: State, # rpm
        polar_moment_of_inertia: float | None = None,
        net_torque: State | None = None,
    ):
        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.rotor_speed]

    @property
    def residuals(self) -> list[State | float]:
        return [self.net_torque.value]

    @property
    def transient_variables(self) -> list[State]:
        return [self.rotor_speed]
    
    @property
    def transient_derivatives(self) -> list[State | float]:
        return [self.net_torque.value / self.polar_moment_of_inertia.value * 30.0 / math.pi]




class GasTurbine(Component):
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
        self.setup()


    def evaluate_states(self):
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
    def __init__(self,
                 name: str, 
                 network: Network,
                 mass_flow: State,
                 rotor_speed: State,
                 head_rise: State,
                 density: State,
                 torque: State,
                 upstream_pressure: State,
                 discharge_pressure: State,
                 upstream_total_enthalpy: State | None = None,
                 discharge_total_enthalpy: State | None = None,
                 gravitational_acceleration: float = 9.80665,
                 efficiency: State | None = None,
                 shaft_power: State | None = None,
                 volumetric_flow: State | None = None,):
        self.setup()

    def evaluate_states(self):
        mdot = self.mass_flow.value
        H = self.head_rise.value 
        g = self.gravitational_acceleration.value
        T = self.torque.value
        rho = self.density.value
        Q = mdot / rho
        po_in = self.upstream_pressure.value
        N = self.rotor_speed.value

        omega = (math.pi / 30.0) * N

        shaft_power = T * omega
        hydraulic_power = rho * g * H * Q
        eta = hydraulic_power / shaft_power

        self.po_out = po_in + rho * g * H

        self.efficiency.value = eta
        self.shaft_power.value = shaft_power
        self.volumetric_flow.value = Q

        if self.upstream_total_enthalpy.is_assigned:
            ho_in = self.upstream_total_enthalpy.value
            dho = shaft_power / mdot
            ho_out = ho_in + dho
            self.discharge_total_enthalpy.value = ho_out

    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]
    
    @property
    def residuals(self) -> list[State | float]:
        return [self.po_out - self.discharge_pressure.value]








class PolytropicPump(Component):

    def __init__(self,
                 name: str, 
                 network: Network,
                 mass_flow: State,
                 rotor_speed: State,
                 head_rise: State, 
                 torque: State,
                 upstream_pressure: State,
                 discharge_pressure: State,
                 upstream_density: State,
                 discharge_density: State,
                 upstream_total_enthalpy: State,
                 gravitational_acceleration: float = 9.80665,

                 discharge_total_enthalpy: State | None = None,
                 efficiency: State | None = None,
                 shaft_power: State | None = None):
        self.setup()

    def evaluate_states(self):
        H = self.head_rise.value
        mdot = self.mass_flow.value
        g = self.gravitational_acceleration.value
        T = self.torque.value
        rho1 = self.upstream_density.value
        rho2 = self.discharge_density.value
        p_in = self.upstream_pressure.value
        p_out = self.discharge_pressure.value
        ho_in = self.upstream_total_enthalpy.value
        N = self.rotor_speed.value

        omega = (math.pi / 30.0) * N
        shaft_power = T * omega

        # Pump maps usually report head in distance units.
        # ROCETS polytropic headrise uses specific work units.
        H_specific = g * H

        hydraulic_power = mdot * H_specific
        eta = hydraulic_power / shaft_power

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

        self._predicted_discharge_pressure = rho2 * (H_specific / beta + p_in / rho1)

        if self.upstream_total_enthalpy.is_assigned:
            ho_in = self.upstream_total_enthalpy.value
            dho = H_specific / eta
            ho_out = ho_in + dho
            self.discharge_total_enthalpy.value = ho_out
        
        self.efficiency.value = eta
        self.shaft_power.value = shaft_power

    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]
    
    @property
    def residuals(self) -> list[State | float]:
        return [self._predicted_discharge_pressure - self.discharge_pressure.value]