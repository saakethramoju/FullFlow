from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State
from ._flow_math import isclose_numpy_default, pressure_drop_flow_rate, sign, sqrt_or_nan

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

    Computes mass flow through a gas regulator using ideal-gas isentropic flow
    relations. The component automatically switches between unchoked and choked
    flow based on the set-pressure to upstream-total-pressure ratio.

    Energy Reference
    ----------------
    If `upstream_static_enthalpy` and `upstream_static_temperature` are supplied,
    the branch total enthalpy is computed as:

        h0 = h_static + cp * (T0 - T_static)

    This preserves the thermodynamic reference of the connected property package
    while still accounting for the branch total/static temperature difference.

    If `upstream_static_enthalpy` is supplied but `upstream_static_temperature`
    is not, the component assumes a plenum/stagnation inlet and uses:

        h0 = h_static

    If `upstream_static_enthalpy` is omitted, the component falls back to the
    older ideal-gas absolute estimate:

        h0 = cp * T0

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
        Regulator set/static outlet pressure
    discharge_coefficient : float
        Discharge coefficient
    cross_sectional_area : float
        Flow area
    specific_gas_constant : float
        Specific gas constant
    specific_heat_ratio : State
        Specific heat ratio
    upstream_static_enthalpy : State, optional
        Upstream static enthalpy using the connected property package reference
    upstream_static_temperature : State, optional
        Upstream static temperature corresponding to `upstream_static_enthalpy`
    mass_flow : State, optional
        Computed mass flow rate
    total_enthalpy : State, optional
        Branch total enthalpy

    Outputs
    -------
    mass_flow : State
        Computed mass flow rate
    total_enthalpy : State
        Branch total enthalpy
    """

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_total_pressure: State,
        upstream_total_temperature: State,
        set_pressure: State,
        discharge_coefficient: float,
        cross_sectional_area: float,
        specific_gas_constant: float,
        specific_heat_ratio: State,
        upstream_static_enthalpy: State | None = None,
        upstream_static_temperature: State | None = None,
        mass_flow: State | None = None,
        total_enthalpy: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        P1 = self.upstream_total_pressure.value
        T0 = self.upstream_total_temperature.value
        P2 = self.set_pressure.value

        Cd = self.discharge_coefficient.value
        A = self.cross_sectional_area.value
        R = self.specific_gas_constant.value
        g = self.specific_heat_ratio.value

        if T0 <= 0.0:
            raise ValueError(
                f"{self.name}: upstream_total_temperature must be positive. Got {T0}."
            )

        if R <= 0.0:
            raise ValueError(
                f"{self.name}: specific_gas_constant must be positive. Got {R}."
            )

        if g <= 1.0:
            raise ValueError(
                f"{self.name}: specific_heat_ratio must be greater than 1. Got {g}."
            )

        if A < 0.0:
            raise ValueError(
                f"{self.name}: cross_sectional_area must be nonnegative. Got {A}."
            )

        if Cd < 0.0:
            raise ValueError(
                f"{self.name}: discharge_coefficient must be nonnegative. Got {Cd}."
            )

        CdA = Cd * A
        cp = g * R / (g - 1.0)

        if self.upstream_static_enthalpy.is_assigned:
            h_static = self.upstream_static_enthalpy.value

            if self.upstream_static_temperature.is_assigned:
                T_static = self.upstream_static_temperature.value
                self.total_enthalpy.value = h_static + cp * (T0 - T_static)
            else:
                self.total_enthalpy.value = h_static
        else:
            self.total_enthalpy.value = cp * T0

        if isclose_numpy_default(P1, P2):
            self.mass_flow.value = 0.0
            return

        flow_sign = sign(P1 - P2)

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
            flow_function = sqrt_or_nan(
                (g / (R * To))
                * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
            )
        else:
            flow_function = sqrt_or_nan(
                (2.0 * g / (R * To * (g - 1.0)))
                * (
                    pressure_ratio ** (2.0 / g)
                    - pressure_ratio ** ((g + 1.0) / g)
                )
            )

        self.mass_flow.value = flow_sign * CdA * Po * flow_function