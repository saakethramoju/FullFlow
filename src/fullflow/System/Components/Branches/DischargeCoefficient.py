from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class DischargeCoefficient(Component):
    """
    Incompressible restriction flow model using a discharge coefficient.

    `DischargeCoefficient` computes mass flow through a restriction from the
    pressure difference, fluid density, discharge coefficient, and flow area. The
    sign of the mass flow follows the sign of the pressure difference, allowing
    reverse flow when downstream pressure exceeds upstream pressure.

    Relations
    ---------
    Restriction mass flow:

    `mass_flow = sign(P1 - P2) * Cd * A * sqrt(2 * rho * abs(P1 - P2))`

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_pressure : State
        Upstream pressure
    downstream_pressure : State
        Downstream pressure
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
    """

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_pressure: State,
        downstream_pressure: State,
        density: State,
        discharge_coefficient: float,
        cross_sectional_area: float,
        mass_flow: State | None = None,
    ):
        self.setup()

    def evaluate_states(self) -> None:
        P1 = self.upstream_pressure.value
        P2 = self.downstream_pressure.value
        rho = self.density.value
        Cd = self.discharge_coefficient.value
        A = self.cross_sectional_area.value

        self.mass_flow.value = (
            np.sign(P1 - P2)
            * Cd
            * A
            * np.sqrt(2.0 * rho * np.abs(P1 - P2))
        )


class SeriesCdA(Component):
    """
    Equivalent effective area for restrictions in series.

    `SeriesCdA` combines multiple effective flow areas into a single equivalent
    effective area. This is useful when several restrictions are arranged in
    series and should be represented as one equivalent restriction.

    Relations
    ---------
    Series effective area:

    `1 / effective_area_eq^2 = sum(1 / effective_area_i^2)`

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    effective_areas : list[State or float]
        Effective areas connected in series

    Outputs
    -------
    effective_area : State, optional
        Equivalent effective area
    """

    def __init__(
        self,
        name: str,
        network: Network,
        effective_areas: list[State | float],
        effective_area: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        inverse_area_squared_sum = 0.0

        for effective_area in self.effective_areas:
            CdA = effective_area.value

            if abs(CdA) < 1e-12:
                self.effective_area.value = 0.0
                return

            inverse_area_squared_sum += 1.0 / CdA**2

        self.effective_area.value = 1.0 / inverse_area_squared_sum**0.5


class ParallelCdA(Component):
    """
    Equivalent effective area for restrictions in parallel.

    `ParallelCdA` combines multiple effective flow areas into a single
    equivalent effective area. This is useful when several restrictions are
    arranged in parallel and should be represented as one equivalent
    restriction.

    Relations
    ---------
    Parallel effective area:

    `effective_area_eq = sum(effective_area_i)`

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    effective_areas : list[State or float]
        Effective areas connected in parallel

    Outputs
    -------
    effective_area : State, optional
        Equivalent effective area
    """

    def __init__(
        self,
        name: str,
        network: Network,
        effective_areas: list[State | float],
        effective_area: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        self.effective_area.value = sum(
            effective_area.value
            for effective_area in self.effective_areas
        )


class CavitatingVenturi(Component):
    """
    Cavitating liquid venturi model.

    `CavitatingVenturi` computes mass flow through a liquid venturi using a
    noncavitating restriction model or a cavitating/choked venturi model. The
    active mode is selected using a critical downstream-to-upstream pressure
    ratio.

    In cavitating mode, the throat pressure is assumed to be pinned to the
    vapor pressure corresponding to the upstream fluid state. If upstream
    temperature and critical temperature are both assigned, cavitation is
    disabled above the critical temperature.

    Notes
    -----
    Cavitation onset and stable cavitating flow are not identical. Incipient
    cavitation begins when the throat pressure first reaches saturation
    pressure, while fully established cavitating flow depends on geometry and
    empirical behavior.

    Relations
    ---------
    Noncavitating mass flow:

    `mass_flow = sign(P1 - P2) * Cd_noncav * A_t * sqrt(2 * rho * abs(P1 - P2))`

    Cavitating mass flow:

    `mass_flow = Cd_cav * A_t * sqrt(2 * rho * (P1 - vapor_pressure))`

    Cavitating mode criterion:

    `downstream_pressure / upstream_pressure < critical_pressure_ratio`

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_pressure : State
        Upstream pressure
    downstream_pressure : State
        Downstream pressure
    density : State
        Fluid density
    throat_area : float
        Venturi throat area
    vapor_pressure : State
        Fluid vapor pressure
    critical_pressure_ratio : float, optional
        Pressure ratio below which cavitating mode is activated
    cavitating_discharge_coefficient : float, optional
        Discharge coefficient used in cavitating mode
    noncavitating_discharge_coefficient : float, optional
        Discharge coefficient used in noncavitating mode
    upstream_temperature : State, optional
        Upstream fluid temperature
    critical_temperature : State, optional
        Fluid critical temperature

    Outputs
    -------
    mass_flow : State, optional
        Computed venturi mass flow rate
    is_cavitating : bool, optional
        Whether cavitating mode is active
    """

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_pressure: State,
        downstream_pressure: State,
        density: State,
        throat_area: float,
        vapor_pressure: State,
        critical_pressure_ratio: float = 0.8,
        cavitating_discharge_coefficient: float = 0.94,
        noncavitating_discharge_coefficient: float = 0.6,
        upstream_temperature: State | None = None,
        critical_temperature: State | None = None,
        mass_flow: State | None = None,
        is_cavitating: bool = False,
    ):
        self.setup()

    def evaluate_states(self):
        P1 = self.upstream_pressure.value
        P2 = self.downstream_pressure.value
        rho = self.density.value
        A = self.throat_area.value
        PRcrit = self.critical_pressure_ratio.value
        Cd_cav = self.cavitating_discharge_coefficient.value
        Cd_noncav = self.noncavitating_discharge_coefficient.value

        pressure_ratio = P2 / P1

        above_critical_temperature = False
        if (
            self.upstream_temperature.is_assigned
            and self.critical_temperature.is_assigned
        ):
            above_critical_temperature = (
                self.upstream_temperature.value >= self.critical_temperature.value
            )

        if above_critical_temperature or pressure_ratio >= PRcrit:
            self.is_cavitating = False
            dP = P1 - P2
            self.mass_flow.value = (
                np.sign(dP) * Cd_noncav * A * np.sqrt(2.0 * rho * np.abs(dP))
            )
        else:
            self.is_cavitating = True

            Pvap = self.vapor_pressure.value
            dP = P1 - Pvap

            self.mass_flow.value = Cd_cav * A * np.sqrt(2.0 * rho * dP)