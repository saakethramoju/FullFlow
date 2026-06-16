from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State
from ._flow_math import divide_or_nan

if TYPE_CHECKING:
    from fullflow.System import Network


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
    def residuals(self) -> list[float]:
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
    
        self._predicted_discharge_pressure = None

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

        beta = 1.0 / (1.0 - divide_or_nan(math.log(density_ratio), log_pressure_ratio))

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
    def residuals(self) -> list[float]:
        return [self._predicted_discharge_pressure - self.discharge_pressure.value]