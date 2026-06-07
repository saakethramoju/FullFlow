from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class LiquidRegulator(Component):


    def __init__(self,
                 name: str,
                 network: Network,
                 upstream_pressure: State,
                 set_pressure: State,
                 density: State,
                 discharge_coefficient: float,
                 cross_sectional_area: float,
                 mass_flow: State | None = None):
        self.setup()

    def evaluate_states(self) -> None:
        P1 = self.upstream_pressure.value
        P2 = self.set_pressure.value
        rho = self.density.value
        Cd = self.discharge_coefficient.value
        A = self.cross_sectional_area.value

        self.mass_flow.value = np.sign(P1 - P2) * Cd * A * np.sqrt(2.0 * rho * np.abs(P1 - P2))


class IsentropicGasRegulator(Component):
    """
    Assumes ideal gas
    """
    def __init__(self,
                 name: str,
                 network: Network,
                 upstream_total_pressure: State,
                 upstream_total_temperature: State,
                 set_pressure: State,
                 discharge_coefficient: float,
                 cross_sectional_area: float,
                 specific_gas_constant: float,
                 specific_heat_ratio: State,
                 mass_flow: State | None = None,
                 total_enthalpy: State | None = None):
        
        self.setup()



    def evaluate_states(self):

        P1 = self.upstream_total_pressure.value
        T1 = self.upstream_total_temperature.value
        P2 = self.set_pressure.value

        CdA = self.discharge_coefficient.value * self.cross_sectional_area.value
        R = self.specific_gas_constant.value
        g = self.specific_heat_ratio.value

        cp = g * R / (g - 1.0)
        self.total_enthalpy.value = cp * T1

        if np.isclose(P1, P2):
            self.mass_flow.value = 0.0
            return

        sign = np.sign(P1 - P2)

        Po = max(P1, P2)
        Pb = min(P1, P2)
        To = T1

        pressure_ratio = Pb / Po
        critical_pressure_ratio = (2 / (g + 1)) ** (g / (g - 1))

        if pressure_ratio <= critical_pressure_ratio:
            flow_function = np.sqrt((g / (R * To)) * (2 / (g + 1)) ** ((g + 1) / (g - 1)))

        else:
            flow_function = np.sqrt((2 * g / (R * To * (g - 1))) * (pressure_ratio ** (2 / g) - pressure_ratio ** ((g + 1) / g)))

        self.mass_flow.value = sign * CdA * Po * flow_function
