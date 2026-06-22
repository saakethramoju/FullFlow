from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


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

        # Freeze the choked/unchoked branch during transient nonlinear solves.
        is_choked = self.propose("is_choked", pressure_ratio <= critical_pressure_ratio)

        if is_choked:
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
            self.mass_flow.value = 0.0
            if self.upstream_static_enthalpy.is_assigned:
                self.total_enthalpy.value = self.upstream_static_enthalpy.value
            return

        T2 = T1 * (P2 / P1) ** ((g - 1.0) / g)
        C = (A1 * P1 * math.sqrt(T2)) / (A2 * P2 * math.sqrt(T1))
        M1 = math.sqrt((T2 - T1) / (0.5 * (g - 1.0) * (T1 - T2 * C**2)))
        M2 = C * M1
        a1 = math.sqrt(g * R * T1)
        u1 = M1 * a1

        self.downstream_static_temperature.value = T2
        self.upstream_mach_number.value = M1
        self.downstream_mach_number.value = M2
        self.mass_flow.value = A1 * P1 * M1 * math.sqrt(g / (R * T1))

        if self.upstream_static_enthalpy.is_assigned:
            self.total_enthalpy.value = self.upstream_static_enthalpy.value + 0.5 * u1**2






class IsentropicNozzle(Component):

    def __init__(
        self, 
        name: str, 
        network: Network,
        upstream_total_pressure: State,
        upstream_total_temperature: State,
        ambient_pressure: State,
        specific_heat_ratio: float,
        gas_constant: float,
        throat_area: float,
        expansion_ratio: float,
        mass_flow: State | None = None,
        exit_mach_number: State | None = None,
        exit_static_pressure: State | None = None,
        normal_shock: bool | None = None,
        shock_mach_number: State | None = 2.0,
    ):
        self.setup()

    def evaluate_states(self):
        Po = self.upstream_total_pressure.value
        To = self.upstream_total_temperature.value
        Pamb = self.ambient_pressure.value
        g = self.specific_heat_ratio.value
        R = self.gas_constant.value
        At = self.throat_area.value
        eps = self.expansion_ratio.value

        PR = Po / Pamb
        PR_choked = ((g + 1.0) / 2.0) ** (g / (g - 1.0))
        choked_check1 = PR > PR_choked

        FP_choked = self._flow_parameter_from_mach(1.0, g)
        FP_exit = self._flow_parameter_from_pressure_ratio(PR, g)
        FP_throat = FP_exit * eps
        choked_check2 = FP_throat > FP_choked

        # Freeze the choked/unchoked nozzle branch during transient nonlinear solves.
        choked = self.propose("is_choked", choked_check1 or choked_check2)

        if not choked:
            mdot = FP_throat * Po * At / (R * To) ** 0.5
            Me = self._mach_from_pressure_ratio(PR, g)
            Pe = Pamb
            # Keep the normal-shock branch fixed while the timestep is solved.
            self.propose("normal_shock", False)
            Ms = 0.0

        else:
            mdot = FP_choked * Po * At / (R * To) ** 0.5

            if eps <= 1.0:
                Me = 1.0
                Pe = Po / self._pressure_ratio_from_mach(Me, g)
                # Keep the normal-shock branch fixed while the timestep is solved.
                self.propose("normal_shock", False)
                Ms = 0.0

            else:
                Me_test = self._mach_from_flow_parameter(FP_choked / eps, g, 1.0 + 1.0e-12, 50.0)
                Po_over_Ps_exit = self._pressure_ratio_from_mach(Me_test, g)
                Pe_test = Po / Po_over_Ps_exit
                RPR = PR / Po_over_Ps_exit

                if RPR >= 1.0:
                    Me = Me_test
                    Pe = Pe_test
                    # Keep the normal-shock branch fixed while the timestep is solved.
                    self.propose("normal_shock", False)
                    Ms = 0.0

                else:
                    Ps2_Ps1 = self._normal_shock_static_pressure_ratio(Me_test, g)
                    RPRS = Ps2_Ps1 * RPR

                    # Freeze the normal-shock/no-shock branch during transient nonlinear solves.
                    shock = self.propose("normal_shock", RPRS < 1.0)

                    if not shock:
                        Me = Me_test
                        Pe = Pe_test
                        Ms = 0.0

                    else:
                        Ms = self._shock_mach_from_internal_shock_condition(FP_choked, PR, eps, g, 1.0 + 1.0e-12, Me_test - 1.0e-12)
                        Pt2_Pt1 = self._normal_shock_total_pressure_ratio(Ms, g)
                        FP_exit = FP_choked / (Pt2_Pt1 * eps)
                        Me = self._mach_from_flow_parameter(FP_exit, g, 1.0e-12, 1.0 - 1.0e-12)
                        Pe = Po * Pt2_Pt1 / self._pressure_ratio_from_mach(Me, g)

        self.mass_flow.value = mdot
        self.exit_mach_number.value = Me
        self.exit_static_pressure.value = Pe
        self.shock_mach_number.value = Ms

    @staticmethod
    def _normal_shock_static_pressure_ratio(M1: float, g: float) -> float:
        return 2.0 * g * M1**2 / (g + 1.0) - (g - 1.0) / (g + 1.0)

    @staticmethod
    def _normal_shock_total_pressure_ratio(M1: float, g: float) -> float:
        term1 = (((g + 1.0) / 2.0 * M1**2) / (1.0 + 0.5 * (g - 1.0) * M1**2)) ** (g / (g - 1.0))
        term2 = ((2.0 * g / (g + 1.0)) * M1**2 - (g - 1.0) / (g + 1.0)) ** (1.0 / (1.0 - g))
        return term1 * term2

    @staticmethod
    def _shock_mach_from_internal_shock_condition(FP_throat: float, PR: float, eps: float, g: float, low: float, high: float) -> float:
        low_error = IsentropicNozzle._internal_shock_pressure_error(low, FP_throat, PR, eps, g)
        high_error = IsentropicNozzle._internal_shock_pressure_error(high, FP_throat, PR, eps, g)

        if low_error == 0.0:
            return low
        if high_error == 0.0:
            return high
        if low_error * high_error > 0.0:
            raise ValueError("Internal normal shock Mach solve is not bracketed.")

        for _ in range(80):
            mid = 0.5 * (low + high)
            mid_error = IsentropicNozzle._internal_shock_pressure_error(mid, FP_throat, PR, eps, g)

            if abs(mid_error) <= 1.0e-12 * max(1.0, PR):
                return mid

            if low_error * mid_error <= 0.0:
                high = mid
                high_error = mid_error
            else:
                low = mid
                low_error = mid_error

        return 0.5 * (low + high)

    @staticmethod
    def _internal_shock_pressure_error(Ms: float, FP_throat: float, PR: float, eps: float, g: float) -> float:
        Pt2_Pt1 = IsentropicNozzle._normal_shock_total_pressure_ratio(Ms, g)
        FP_exit = FP_throat / (Pt2_Pt1 * eps)
        Me = IsentropicNozzle._mach_from_flow_parameter(FP_exit, g, 1.0e-12, 1.0 - 1.0e-12)
        Po_over_Ps_exit = IsentropicNozzle._pressure_ratio_from_mach(Me, g)
        return Po_over_Ps_exit - PR * Pt2_Pt1

    @staticmethod
    def _mach_from_pressure_ratio(PR: float, g: float) -> float:
        if PR <= 1.0:
            return 0.0
        return math.sqrt(2.0 / (g - 1.0) * (PR ** ((g - 1.0) / g) - 1.0))

    @staticmethod
    def _flow_parameter_from_pressure_ratio(PR: float, g: float) -> float:
        if PR <= 1.0:
            return 0.0
        return ((2.0 * g / (g - 1.0) * (PR ** ((g - 1.0) / g) - 1.0)) / (PR ** ((g + 1.0) / g))) ** 0.5

    @staticmethod
    def _mach_from_flow_parameter(FP: float, g: float, low: float, high: float) -> float:
        for _ in range(80):
            mid = 0.5 * (low + high)
            value = IsentropicNozzle._flow_parameter_from_mach(mid, g)
            if value < FP:
                if high <= 1.0:
                    low = mid
                else:
                    high = mid
            else:
                if high <= 1.0:
                    high = mid
                else:
                    low = mid
        return 0.5 * (low + high)

    @staticmethod
    def _flow_parameter_from_mach(M: float, g: float) -> float:
        return g**0.5 * M / (1.0 + 0.5 * (g - 1.0) * M**2) ** ((g + 1.0) / (2.0 * (g - 1.0)))

    @staticmethod
    def _pressure_ratio_from_mach(M: float, g: float) -> float:
        return (1.0 + 0.5 * (g - 1.0) * M**2) ** (g / (g - 1.0))