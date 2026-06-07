from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State



class DischargeCoefficient(Component):

    def __init__(self,
                 name: str,
                 network: Network,
                 upstream_pressure: State,
                 downstream_pressure: State,
                 density: State,
                 discharge_coefficient: float,
                 cross_sectional_area: float,
                 mass_flow: State | None = None):
        self.setup()

    def evaluate_states(self) -> None:
        P1 = self.upstream_pressure.value
        P2 = self.downstream_pressure.value
        rho = self.density.value
        Cd = self.discharge_coefficient.value
        A = self.cross_sectional_area.value

        self.mass_flow.value = np.sign(P1 - P2) * Cd * A * np.sqrt(2.0 * rho * np.abs(P1 - P2))




class SeriesCdA(Component):

    def __init__(
        self,
        name: str,
        network: Network,
        effective_areas: list[State | float],
        effective_area: State | None = None,
    ):
        """
        Equivalent effective area for flow restrictions in series.

        Uses:
            1 / CdA_eq^2 = sum(1 / CdA_i^2)
        """
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

    def __init__(
        self,
        name: str,
        network: Network,
        effective_areas: list[State | float],
        effective_area: State | None = None,
    ):
        """
        Equivalent effective area for flow restrictions in parallel.

        Uses:
            CdA_eq = sum(CdA_i)
        """
        self.setup()

    def evaluate_states(self):
        self.effective_area.value = sum(
            effective_area.value
            for effective_area in self.effective_areas
        )






class CavitatingVenturi(Component):
    """
    GFSSP-style cavitating venturi model.

    This component models a cavitating/choked liquid venturi using the
    simplified approach described in the GFSSP 7.02 training notes.

    The venturi operates in one of two modes:

    1) Non-cavitating flow

        mdot = Cd_noncav * A_t * sqrt(2 * rho * abs(P_up - P_down))

    2) Cavitating/choked flow

        mdot = Cd_cav * A_t * sqrt(2 * rho * (P_up - P_sat))

    In cavitating mode, the throat pressure is assumed to be pinned to the
    vapor pressure corresponding to the upstream fluid temperature.

    Notes
    -----
    Cavitation onset and stable cavitating/choked flow are not identical.

    A venturi may begin experiencing incipient cavitation when the throat
    pressure first reaches saturation pressure. However, fully established
    cavitating/choked flow is geometry-dependent and empirical.

    Therefore, this component follows the GFSSP methodology by using an
    empirical critical pressure ratio criterion:

        P_downstream / P_upstream < critical_pressure_ratio

    to determine when the cavitating/choked-flow model should be activated.

    Typical values:
        critical_pressure_ratio ≈ 0.7 - 0.8

    If upstream_temperature and critical_temperature are both assigned,
    cavitation is disabled when upstream_temperature >= critical_temperature
    because fluids above the critical temperature no longer possess a
    liquid-vapor saturation boundary.
    """

    def __init__(self,
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
                 is_cavitating: bool = False):
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
        if self.upstream_temperature.is_assigned and self.critical_temperature.is_assigned:
            above_critical_temperature = self.upstream_temperature.value >= self.critical_temperature.value

        if above_critical_temperature or pressure_ratio >= PRcrit:
            self.is_cavitating = False
            dP = P1 - P2
            self.mass_flow.value = np.sign(dP) * Cd_noncav * A * np.sqrt(2.0 * rho * np.abs(dP))
        else:
            self.is_cavitating = True

            Pvap = self.vapor_pressure.value
            dP = P1 - Pvap

            self.mass_flow.value = Cd_cav * A * np.sqrt(2.0 * rho * dP)