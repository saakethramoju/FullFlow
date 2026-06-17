from __future__ import annotations

import math

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _sqrt_or_nan(value: float) -> float:
    return math.sqrt(value) if value >= 0.0 else math.nan


def _isclose_numpy_default(a: float, b: float) -> bool:
    return abs(a - b) <= (1e-8 + 1e-5 * abs(b))



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
        specific_gas_constant: float,
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
        R = self.specific_gas_constant.value
        g = self.specific_heat_ratio.value

        CdA = Cd * A
        cp = g * R / (g - 1.0)

        if self.upstream_static_enthalpy.is_assigned and self.upstream_static_temperature.is_assigned:
            h_static = self.upstream_static_enthalpy.value
            T_static = self.upstream_static_temperature.value
            self.total_enthalpy.value = h_static + cp * (T0 - T_static)

        if _isclose_numpy_default(P1, P2):
            self.mass_flow.value = 0.0
            return

        flow_sign = _sign(P1 - P2)

        Po = max(P1, P2)
        Pb = min(P1, P2)
        To = T0

        if Po <= 0.0:
            raise ValueError(
                f"{self.name}: reference total pressure must be positive. Got {Po}."
            )

        pressure_ratio = Pb / Po
        critical_pressure_ratio = (2.0 / (g + 1.0)) ** (g / (g - 1.0))

        if pressure_ratio <= critical_pressure_ratio:
            flow_function = _sqrt_or_nan((g / (R * To)) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))
        else:
            flow_function = _sqrt_or_nan((2.0 * g / (R * To * (g - 1.0))) * (pressure_ratio ** (2.0 / g) - pressure_ratio ** ((g + 1.0) / g)))

        self.mass_flow.value = flow_sign * CdA * Po * flow_function
