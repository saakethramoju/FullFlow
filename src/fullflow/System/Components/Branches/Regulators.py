from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State
from ._flow_math import isclose_numpy_default, pressure_drop_flow_rate, sign

if TYPE_CHECKING:
    from fullflow.System import Network


class LiquidRegulator(Component):
    """
    Incompressible liquid regulator flow model.

    `LiquidRegulator` computes mass flow through a liquid regulator using the
    upstream pressure, set pressure, fluid density, discharge coefficient, and
    flow area. The sign of the mass flow follows the sign of the pressure
    difference, allowing reverse flow when the set pressure exceeds the upstream
    pressure.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_pressure : State
        Upstream pressure
    set_pressure : State
        Regulator set pressure
    density : State
        Fluid density
    discharge_coefficient : float
        Discharge coefficient
    cross_sectional_area : float
        Flow area

    Outputs
    -------
    mass_flow : State, optional
        Computed mass flow rate

    Notes
    -----
    Mass flow is evaluated from:

        ``mass_flow = sign(P1 - P2) * Cd * A * sqrt(2 * rho * abs(P1 - P2))``
    """
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

        self.mass_flow.value = pressure_drop_flow_rate(P1 - P2, rho, Cd, A)





class IsentropicGasRegulator(Component):
    """
    Isentropic ideal-gas regulator flow model.

    `IsentropicGasRegulator` computes mass flow through a gas regulator using
    ideal-gas isentropic flow relations. The component automatically switches
    between unchoked and choked flow based on the set-pressure to upstream-total
    pressure ratio.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_total_pressure : State
        Upstream total pressure
    upstream_total_temperature : State
        Upstream total temperature
    set_pressure : State
        Regulator set pressure
    discharge_coefficient : float
        Discharge coefficient
    cross_sectional_area : float
        Flow area
    specific_gas_constant : float
        Specific gas constant
    specific_heat_ratio : State
        Specific heat ratio

    Outputs
    -------
    mass_flow : State, optional
        Computed mass flow rate
    total_enthalpy : State, optional
        Upstream total enthalpy

    Notes
    -----
    This component assumes ideal-gas flow. It uses upstream total pressure,
    upstream total temperature, and regulator set pressure as the back pressure.

    Total enthalpy is evaluated from:

        ``total_enthalpy = cp * T0``

        ``cp = gamma * R / (gamma - 1)``

    The critical pressure ratio is evaluated from:

        ``P_back / P0 = (2 / (gamma + 1)) ** (gamma / (gamma - 1))``

    Choked mass flow is evaluated from:

        ``mass_flow = sign * CdA * P0
        * sqrt((gamma / (R * T0))
        * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))``

    Unchoked mass flow is evaluated from:

        ``mass_flow = sign * CdA * P0
        * sqrt((2 * gamma / (R * T0 * (gamma - 1)))
        * ((P_back / P0) ** (2 / gamma)
        - (P_back / P0) ** ((gamma + 1) / gamma)))``
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

        if isclose_numpy_default(P1, P2):
            self.mass_flow.value = 0.0
            return

        sign_value = sign(P1 - P2)

        Po = max(P1, P2)
        Pb = min(P1, P2)
        To = T1

        pressure_ratio = Pb / Po
        critical_pressure_ratio = (2 / (g + 1)) ** (g / (g - 1))

        if pressure_ratio <= critical_pressure_ratio:
            flow_function = math.sqrt((g / (R * To)) * (2 / (g + 1)) ** ((g + 1) / (g - 1)))

        else:
            flow_function = math.sqrt((2 * g / (R * To * (g - 1))) * (pressure_ratio ** (2 / g) - pressure_ratio ** ((g + 1) / g)))

        self.mass_flow.value = sign_value * CdA * Po * flow_function
