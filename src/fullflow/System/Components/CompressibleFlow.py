from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class CompressibleOrifice(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        upstream_total_pressure: State,
        upstream_total_temperature: State,
        downstream_pressure: State,
        discharge_coefficient: float,
        cross_sectional_area: float,
        gas_constant: float,
        specific_heat_ratio: State,
        upstream_static_enthalpy: State | None = None,
        upstream_static_temperature: State | None = None,
        total_enthalpy: State | None = None,
        mass_flow: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        P1 = self.upstream_total_pressure.value
        T0 = self.upstream_total_temperature.value
        P2 = self.downstream_pressure.value

        Cd = self.discharge_coefficient.value
        A = self.cross_sectional_area.value
        R = self.gas_constant.value
        g = self.specific_heat_ratio.value

        CdA = Cd * A
        cp = g * R / (g - 1.0)

        if self.upstream_static_enthalpy.is_assigned and self.upstream_static_temperature.is_assigned:
            h_static = self.upstream_static_enthalpy.value
            T_static = self.upstream_static_temperature.value
            self.total_enthalpy.value = h_static + cp * (T0 - T_static)

        if abs(P1 - P2) <= (1e-8 + 1e-5 * abs(P2)):
            self.mass_flow.value = 0.0
            return

        flow_sign = 1.0 if P1 > P2 else -1.0
        Po = max(P1, P2)
        Pb = min(P1, P2)
        pressure_ratio = Pb / Po
        critical_pressure_ratio = (2.0 / (g + 1.0)) ** (g / (g - 1.0))

        if pressure_ratio <= critical_pressure_ratio:
            flow_function = math.sqrt((g / (R * T0)) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))
        else:
            flow_function = math.sqrt((2.0 * g / (R * T0 * (g - 1.0))) * (pressure_ratio ** (2.0 / g) - pressure_ratio ** ((g + 1.0) / g)))

        self.mass_flow.value = flow_sign * CdA * Po * flow_function






class IsentropicDiffuser(Component):
    """Ideal-gas isentropic area-change branch. Positive mass flow is 1 -> 2."""

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_static_pressure: State,
        upstream_static_temperature: State,
        downstream_static_pressure: State,
        inlet_cross_sectional_area: float,
        outlet_cross_sectional_area: float,
        specific_heat_ratio: float,
        gas_constant: float,
        upstream_static_enthalpy: State | None = None,
        mass_flow: State | None = None,

        total_enthalpy: State | None = None,
        downstream_static_temperature: State | None = None,
        upstream_mach_number: State | None = None,
        downstream_mach_number: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        P1 = self.upstream_static_pressure.value
        T1 = self.upstream_static_temperature.value
        P2 = self.downstream_static_pressure.value
        A1 = self.inlet_cross_sectional_area.value
        A2 = self.outlet_cross_sectional_area.value
        g = self.specific_heat_ratio.value
        R = self.gas_constant.value

        if abs(P1 - P2) <= (1e-8 + 1e-5 * abs(P2)):
            self.downstream_static_temperature.value = T1
            self.upstream_mach_number.value = 0.0
            self.downstream_mach_number.value = 0.0
            #self.upstream_velocity.value = 0.0
            #self.downstream_velocity.value = 0.0
            self.mass_flow.value = 0.0

            if self.upstream_static_enthalpy.is_assigned:
                self.total_enthalpy.value = self.upstream_static_enthalpy.value

            return

        T2 = T1 * (P2 / P1) ** ((g - 1.0) / g)
        C = (A1 * P1 * math.sqrt(T2)) / (A2 * P2 * math.sqrt(T1))
        M1 = math.sqrt((T2 - T1) / (0.5 * (g - 1.0) * (T1 - T2 * C**2)))
        M2 = C * M1
        a1 = math.sqrt(g * R * T1)
        a2 = math.sqrt(g * R * T2)
        u1 = M1 * a1
        u2 = M2 * a2

        self.downstream_static_temperature.value = T2
        self.upstream_mach_number.value = M1
        self.downstream_mach_number.value = M2
        #self.upstream_velocity.value = u1
        #self.downstream_velocity.value = u2
        self.mass_flow.value = A1 * P1 * M1 * math.sqrt(g / (R * T1))

        if self.upstream_static_enthalpy.is_assigned:
            self.total_enthalpy.value = self.upstream_static_enthalpy.value + 0.5 * u1**2
