from __future__ import annotations

import math

import numpy as np
from scipy.special import lambertw, wrightomega
from scipy.optimize import brentq
from typing import TYPE_CHECKING

from fullflow.System import Component, State
from ._flow_math import isclose_numpy_default, sign, sqrt_or_nan

if TYPE_CHECKING:
    from fullflow.System import Network


class IsentropicCompressibleOrifice(Component):
    """
    Isentropic compressible orifice model for ideal-gas flow.

    Computes mass flow through an orifice using ideal-gas isentropic flow
    relations. The component automatically switches between unchoked and choked
    flow based on the downstream-to-upstream pressure ratio.

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
    downstream_pressure : State
        Downstream static/back pressure
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
        downstream_pressure: State,
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
        P2 = self.downstream_pressure.value

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









class CompressibleFlowTube(Component):
    """
    Compressible flow tube with longitudinal inertia and wall friction.

    `CompressibleFlowTube` represents a constant-area duct connecting two
    compressible-flow nodes. The component solves the tube mass flow rate from a
    steady one-dimensional momentum balance using the supplied endpoint static
    pressures, densities, geometry, and Darcy friction factor.

    The component also computes useful diagnostic flow states such as velocity,
    Mach number, total enthalpy, total temperature, and total pressure when the
    required inputs are provided.

    This component is intended for subsonic, non-choked duct flow. It does not
    enforce Fanno choking, normal shocks, or nozzle choking. If the solution
    produces Mach numbers near or above one, the model should be replaced with a
    choked/Fanno/nozzle component or a regime-switching model.

    Sign Convention
    ---------------
    Positive `mass_flow` means flow from the upstream node to the downstream
    node.

    Negative `mass_flow` means reverse flow from the downstream node to the
    upstream node.

    Residuals
    ---------
    momentum_balance : float
        Enforces steady one-dimensional momentum balance through the tube.

        ``pressure_force - friction_force - inertia_force = 0``

        where:

        ``pressure_force = (P1 - P2) * A``

        ``friction_force = Kf * mass_flow * abs(mass_flow) * A``

        ``inertia_force = mdot * Δu`` with the sign chosen consistently for
        forward or reverse flow.

    Iteration Variables
    -------------------
    mass_flow : State
        Tube mass flow rate.

    Inputs
    ------
    mass_flow : State or float
        Tube mass flow rate. This is the primary iteration variable.
    upstream_static_pressure : State or float
        Upstream static pressure.
    upstream_static_temperature : State or float
        Upstream static temperature.
    upstream_density : State or float
        Upstream density.
    downstream_static_pressure : State or float
        Downstream static pressure.
    downstream_static_temperature : State or float
        Downstream static temperature.
    downstream_density : State or float
        Downstream density.
    length : State or float
        Tube length.
    inner_diameter : State or float
        Tube inner diameter.
    friction_factor : State or float, optional
        Darcy friction factor. If omitted or unassigned, the friction term is
        treated as zero.
    upstream_static_enthalpy : State or float, optional
        Upstream static enthalpy. Required to compute `total_enthalpy`.
    upstream_speed_of_sound : State or float, optional
        Upstream speed of sound. Required to compute upstream Mach number and
        upstream total conditions.
    downstream_speed_of_sound : State or float, optional
        Downstream speed of sound. Required to compute downstream Mach number
        and downstream total conditions.
    specific_heat_ratio : State or float, optional
        Specific heat ratio. Required with speed of sound to compute total
        pressure and total temperature.

    Outputs
    -------
    total_enthalpy : State, optional
        Upstream total enthalpy based on upstream static enthalpy and upstream
        velocity.

        ``h0 = h1 + 0.5 * u1**2``

    upstream_mach_number : State, optional
        Upstream Mach number magnitude.

        ``M1 = abs(u1) / a1``

    downstream_mach_number : State, optional
        Downstream Mach number magnitude.

        ``M2 = abs(u2) / a2``

    upstream_total_pressure : State, optional
        Upstream isentropic total pressure. Assigned only when both
        `upstream_speed_of_sound` and `specific_heat_ratio` are assigned.

        ``P01 = P1 * (1 + 0.5 * (gamma - 1) * M1**2) ** (gamma / (gamma - 1))``

    upstream_total_temperature : State, optional
        Upstream total temperature. Assigned only when both
        `upstream_speed_of_sound` and `specific_heat_ratio` are assigned.

        ``T01 = T1 * (1 + 0.5 * (gamma - 1) * M1**2)``

    downstream_total_pressure : State, optional
        Downstream isentropic total pressure. Assigned only when both
        `downstream_speed_of_sound` and `specific_heat_ratio` are assigned.

        ``P02 = P2 * (1 + 0.5 * (gamma - 1) * M2**2) ** (gamma / (gamma - 1))``

    downstream_total_temperature : State, optional
        Downstream total temperature. Assigned only when both
        `downstream_speed_of_sound` and `specific_heat_ratio` are assigned.

        ``T02 = T2 * (1 + 0.5 * (gamma - 1) * M2**2)``

    Notes
    -----
    Flow area is evaluated from:

        ``A = pi / 4 * D**2``

    Endpoint velocities are evaluated from:

        ``u1 = mass_flow / (rho1 * A)``

        ``u2 = mass_flow / (rho2 * A)``

    The Mach numbers reported by this component are magnitudes. Flow direction
    is carried by the sign of `mass_flow`, not by the sign of Mach number.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        upstream_static_pressure: State,
        upstream_static_temperature: State,
        upstream_density: State,
        downstream_static_pressure: State,
        downstream_static_temperature: State,
        downstream_density: State,
        length: float,
        inner_diameter: float,
        friction_factor: float | None = None,
        upstream_static_enthalpy: State | None = None,
        upstream_speed_of_sound: State | None = None,
        downstream_speed_of_sound: State | None = None,
        specific_heat_ratio: State | None = None,
        total_enthalpy: State | None = None,
        upstream_mach_number: State | None = None,
        downstream_mach_number: State | None = None,
        upstream_total_pressure: State | None = None,
        upstream_total_temperature: State | None = None,
        downstream_total_pressure: State | None = None,
        downstream_total_temperature: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        mdot = self.mass_flow.value

        p1 = self.upstream_static_pressure.value
        T1 = self.upstream_static_temperature.value
        rho1 = self.upstream_density.value

        p2 = self.downstream_static_pressure.value
        T2 = self.downstream_static_temperature.value
        rho2 = self.downstream_density.value

        L = self.length.value
        D = self.inner_diameter.value
        A = (math.pi / 4.0) * D**2

        if D <= 0.0:
            raise ValueError(
                f"{self.name}: inner_diameter must be greater than zero. Got {D}."
            )

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: flow area must be greater than zero. Got {A}."
            )

        if rho1 <= 0.0:
            raise ValueError(
                f"{self.name}: upstream_density must be greater than zero. Got {rho1}."
            )

        if rho2 <= 0.0:
            raise ValueError(
                f"{self.name}: downstream_density must be greater than zero. Got {rho2}."
            )

        u1 = mdot / (rho1 * A)
        u2 = mdot / (rho2 * A)

        M1 = None
        M2 = None

        if self.upstream_static_enthalpy.is_assigned:
            h1 = self.upstream_static_enthalpy.value
            self.total_enthalpy.value = h1 + 0.5 * u1**2

        if self.upstream_speed_of_sound.is_assigned:
            a1 = self.upstream_speed_of_sound.value

            if a1 <= 0.0:
                raise ValueError(
                    f"{self.name}: upstream_speed_of_sound must be greater than zero. "
                    f"Got {a1}."
                )

            M1 = abs(u1) / a1
            self.upstream_mach_number.value = M1

        if self.downstream_speed_of_sound.is_assigned:
            a2 = self.downstream_speed_of_sound.value

            if a2 <= 0.0:
                raise ValueError(
                    f"{self.name}: downstream_speed_of_sound must be greater than zero. "
                    f"Got {a2}."
                )

            M2 = abs(u2) / a2
            self.downstream_mach_number.value = M2

        if self.specific_heat_ratio.is_assigned:
            k = self.specific_heat_ratio.value

            if k <= 1.0:
                raise ValueError(
                    f"{self.name}: specific_heat_ratio must be greater than one. "
                    f"Got {k}."
                )

            if M1 is not None:
                upstream_total_ratio = 1.0 + 0.5 * (k - 1.0) * M1**2

                self.upstream_total_temperature.value = T1 * upstream_total_ratio
                self.upstream_total_pressure.value = (
                    p1 * upstream_total_ratio ** (k / (k - 1.0))
                )

            if M2 is not None:
                downstream_total_ratio = 1.0 + 0.5 * (k - 1.0) * M2**2

                self.downstream_total_temperature.value = T2 * downstream_total_ratio
                self.downstream_total_pressure.value = (
                    p2 * downstream_total_ratio ** (k / (k - 1.0))
                )

        inertia = (
            max(mdot, 0.0) * (u2 - u1)
            - max(-mdot, 0.0) * (u1 - u2)
        )

        pressure = (p1 - p2) * A

        if not self.friction_factor.is_assigned:
            friction = 0.0
        else:
            f = self.friction_factor.value

            if f < 0.0:
                raise ValueError(
                    f"{self.name}: friction_factor must be nonnegative. Got {f}."
                )

            Kf = 8.0 * f * L / (rho1 * math.pi**2 * D**5)
            friction = Kf * mdot * abs(mdot) * A

        self._residual = pressure - friction - inertia

    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]

    @property
    def residuals(self) -> list[float]:
        return [self._residual]






'''
class IsentropicAreaChange(Component):
    """
    Explicit isentropic area-change model for ideal-gas flow.

    The upstream state is fully supplied by the user. The component then computes
    the downstream state using constant-gamma isentropic relations.

    Modes
    -----
    Pressure mode
        Used when `downstream_static_pressure` is supplied.

        Pressure mode takes priority over area mode. The downstream Mach number
        is computed from the pressure ratio, then the isentropically consistent
        downstream area is computed and assigned to `downstream_area`.

    Area mode
        Used when `downstream_static_pressure` is not supplied and
        `downstream_area` is supplied.

        The upstream Mach and upstream area define A*. The downstream Mach is
        computed from A2/A* using `exit_mach_regime`.

    Notes
    -----
    This is an explicit component. It does not add residuals.

    Exact zero upstream Mach is not allowed because A/A* is singular at M = 0.
    Use a small positive Mach number for nearly stagnant inlet conditions.

    If a subsonic upstream state is connected to a supersonic downstream branch,
    the component reports `is_choked = "choked"` because the isentropic path must
    pass through M = 1 somewhere between the stations.

    If `upstream_static_enthalpy` is supplied, total enthalpy is computed on the
    same reference basis as the connected property model:

        h0 = h1 + cp * (T0 - T1)

    Otherwise, the fallback is:

        h0 = cp * T0
    """

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_mach_number: State,
        upstream_static_pressure: State,
        upstream_static_temperature: State,
        specific_gas_constant: State | float,
        specific_heat_ratio: State | float,
        upstream_area: State | float,
        downstream_static_pressure: State | float | None = None,
        downstream_area: State | float | None = None,
        upstream_static_enthalpy: State | float | None = None,
        exit_mach_regime: str = "subsonic",
        pressure_mach_regime: str = "auto",
        downstream_mach_number: State | None = None,
        downstream_static_temperature: State | None = None,
        downstream_density: State | None = None,
        downstream_speed_of_sound: State | None = None,
        downstream_area_required: State | None = None,
        mass_flow: State | None = None,
        total_enthalpy: State | None = None,
        upstream_total_pressure: State | None = None,
        upstream_total_temperature: State | None = None,
        downstream_total_pressure: State | None = None,
        downstream_total_temperature: State | None = None,
        static_temperature_ratio: State | None = None,
        velocity_ratio: State | None = None,
        density_ratio: State | None = None,
        critical_area: State | None = None,
        area_ratio: State | None = None,
        is_choked: State | None = None,
        flow_regime: State | None = None,
    ):
        self._use_downstream_pressure = downstream_static_pressure is not None
        self._use_downstream_area = downstream_area is not None

        self.setup()

        self.exit_mach_regime = str(self.exit_mach_regime).lower().strip()
        self.pressure_mach_regime = str(self.pressure_mach_regime).lower().strip()

        if self.exit_mach_regime not in ("subsonic", "supersonic"):
            raise ValueError(
                f"{self.name}: exit_mach_regime must be 'subsonic' or "
                f"'supersonic'. Got {self.exit_mach_regime!r}."
            )

        if self.pressure_mach_regime not in ("auto", "subsonic", "supersonic"):
            raise ValueError(
                f"{self.name}: pressure_mach_regime must be 'auto', 'subsonic', "
                f"or 'supersonic'. Got {self.pressure_mach_regime!r}."
            )

        if self._use_downstream_pressure:
            self._mode = "pressure"
        elif self._use_downstream_area:
            self._mode = "area"
        else:
            raise ValueError(
                f"{self.name}: requires either downstream_static_pressure or "
                "downstream_area."
            )

    def evaluate_states(self):
        M1 = self.upstream_mach_number.value
        p1 = self.upstream_static_pressure.value
        T1 = self.upstream_static_temperature.value
        A1 = self.upstream_area.value
        R = self.specific_gas_constant.value
        k = self.specific_heat_ratio.value

        self._validate_positive("upstream_static_pressure", p1)
        self._validate_positive("upstream_static_temperature", T1)
        self._validate_positive("upstream_area", A1)
        self._validate_positive("specific_gas_constant", R)

        if M1 <= 0.0:
            raise ValueError(
                f"{self.name}: upstream_mach_number must be greater than zero. "
                f"Got {M1}. Exact zero Mach is singular for the area-Mach relation."
            )

        if k <= 1.0:
            raise ValueError(
                f"{self.name}: specific_heat_ratio must be greater than one. "
                f"Got {k}."
            )

        gm1 = k - 1.0
        half_gm1 = 0.5 * gm1
        pressure_exponent = k / gm1

        total_ratio_1 = 1.0 + half_gm1 * M1 * M1
        T0 = T1 * total_ratio_1
        p0 = p1 * total_ratio_1 ** pressure_exponent

        A1_Astar = self._area_mach_function(M1, k)
        Astar = A1 / A1_Astar

        if self._mode == "pressure":
            M2, p2, A2_required = self._pressure_mode(
                p0=p0,
                Astar=Astar,
                k=k,
            )

            self.downstream_area.value = A2_required

        else:
            M2, p2, A2_required = self._area_mode(
                p0=p0,
                Astar=Astar,
                k=k,
            )

            self.downstream_static_pressure.value = p2

        total_ratio_2 = 1.0 + half_gm1 * M2 * M2
        T2 = T0 / total_ratio_2
        rho2 = p2 / (R * T2)
        a2 = math.sqrt(k * R * T2)

        mdot = p1 * math.sqrt(k / (R * T1)) * A1 * M1

        cp = k * R / gm1
        if self.upstream_static_enthalpy.is_assigned:
            h0 = self.upstream_static_enthalpy.value + cp * (T0 - T1)
        else:
            h0 = cp * T0

        T2_T1 = T2 / T1
        p2_p1 = p2 / p1
        rho2_rho1 = p2_p1 / T2_T1
        v2_v1 = (M2 / M1) * math.sqrt(T2_T1)

        area_ratio = A2_required / Astar
        choked = self._is_choked(M1, M2)

        self.downstream_mach_number.value = M2
        self.downstream_static_temperature.value = T2
        self.downstream_density.value = rho2
        self.downstream_speed_of_sound.value = a2
        self.downstream_area_required.value = A2_required

        self.mass_flow.value = mdot
        self.total_enthalpy.value = h0

        self.upstream_total_pressure.value = p0
        self.upstream_total_temperature.value = T0
        self.downstream_total_pressure.value = p0
        self.downstream_total_temperature.value = T0

        self.static_temperature_ratio.value = T2_T1
        self.velocity_ratio.value = v2_v1
        self.density_ratio.value = rho2_rho1

        self.critical_area.value = Astar
        self.area_ratio.value = area_ratio
        self.is_choked.value = "choked" if choked else "unchoked"
        self.flow_regime.value = self._flow_regime(M1, M2, choked)

    def _pressure_mode(
        self,
        *,
        p0: float,
        Astar: float,
        k: float,
    ) -> tuple[float, float, float]:
        p2 = self.downstream_static_pressure.value
        self._validate_positive("downstream_static_pressure", p2)

        if p2 > p0 and not self._isclose(p2, p0):
            raise ValueError(
                f"{self.name}: downstream_static_pressure cannot exceed upstream "
                f"total pressure. Got p2={p2}, p0={p0}."
            )

        M2 = self._mach_from_pressure_ratio(p2 / p0, k)

        if self.pressure_mach_regime == "subsonic" and M2 > 1.0:
            raise ValueError(
                f"{self.name}: pressure-mode solution is supersonic "
                f"(M2={M2}), but pressure_mach_regime='subsonic'."
            )

        if self.pressure_mach_regime == "supersonic" and M2 < 1.0:
            raise ValueError(
                f"{self.name}: pressure-mode solution is subsonic "
                f"(M2={M2}), but pressure_mach_regime='supersonic'."
            )

        if M2 <= 0.0:
            A2_required = math.inf
        else:
            A2_required = Astar * self._area_mach_function(M2, k)

        return M2, p2, A2_required

    def _area_mode(
        self,
        *,
        p0: float,
        Astar: float,
        k: float,
    ) -> tuple[float, float, float]:
        A2 = self.downstream_area.value
        self._validate_positive("downstream_area", A2)

        A2_Astar = A2 / Astar

        if A2_Astar < 1.0 and not self._isclose(A2_Astar, 1.0):
            raise ValueError(
                f"{self.name}: A2/A* must be >= 1. Got {A2_Astar}. "
                "The supplied upstream Mach, upstream area, and downstream area "
                "are not isentropically compatible."
            )

        if self._isclose(A2_Astar, 1.0):
            M2 = 1.0
        else:
            M2 = self._inverse_area_mach_function(
                A2_Astar,
                k,
                self.exit_mach_regime,
            )

        gm1 = k - 1.0
        p2 = p0 / (1.0 + 0.5 * gm1 * M2 * M2) ** (k / gm1)

        return M2, p2, A2

    def _mach_from_pressure_ratio(self, pressure_ratio: float, gamma: float) -> float:
        if pressure_ratio <= 0.0:
            raise ValueError(
                f"{self.name}: static-to-total pressure ratio must be positive. "
                f"Got {pressure_ratio}."
            )

        if pressure_ratio > 1.0 and not self._isclose(pressure_ratio, 1.0):
            raise ValueError(
                f"{self.name}: static-to-total pressure ratio cannot exceed one. "
                f"Got {pressure_ratio}."
            )

        gm1 = gamma - 1.0
        M2_squared = (2.0 / gm1) * (
            pressure_ratio ** (-gm1 / gamma) - 1.0
        )

        if M2_squared < -1e-12:
            raise ValueError(
                f"{self.name}: pressure relation produced negative M^2. "
                f"M^2={M2_squared}."
            )

        return math.sqrt(max(M2_squared, 0.0))

    def _is_choked(self, M1: float, M2: float) -> bool:
        if self._isclose(M1, 1.0) or self._isclose(M2, 1.0):
            return True

        if M1 < 1.0 < M2:
            return True

        if M1 > 1.0 or M2 > 1.0:
            return True

        return False

    def _flow_regime(self, M1: float, M2: float, choked: bool) -> str:
        if choked:
            if self._isclose(M2, 1.0):
                return "choked_downstream_sonic"
            if M1 < 1.0 < M2:
                return "choked_subsonic_to_supersonic"
            if M1 > 1.0 and M2 > 1.0:
                return "supersonic_streamtube"
            return "choked"

        return "subsonic_streamtube"

    def _area_mach_function(self, M: float, gamma: float) -> float:
        if M <= 0.0:
            raise ValueError(
                f"{self.name}: Mach number must be positive. Got {M}."
            )

        if gamma <= 1.0:
            raise ValueError(
                f"{self.name}: specific_heat_ratio must be greater than one. "
                f"Got {gamma}."
            )

        gm1 = gamma - 1.0

        return (
            (1.0 / M)
            * (
                (2.0 / (gamma + 1.0))
                * (1.0 + 0.5 * gm1 * M * M)
            )
            ** ((gamma + 1.0) / (2.0 * gm1))
        )

    def _inverse_area_mach_function(
        self,
        area_ratio: float,
        gamma: float,
        branch: str = "subsonic",
    ) -> float:
        if area_ratio < 1.0 and not self._isclose(area_ratio, 1.0):
            raise ValueError(
                f"{self.name}: A/A* must be >= 1. Got {area_ratio}."
            )

        if self._isclose(area_ratio, 1.0):
            return 1.0

        def residual(M: float) -> float:
            return self._area_mach_function(M, gamma) - area_ratio

        if branch == "subsonic":
            return float(brentq(residual, 1e-12, 1.0 - 1e-12))

        if branch == "supersonic":
            lo = 1.0 + 1e-12
            hi = 2.0

            while residual(hi) < 0.0:
                hi *= 2.0
                if hi > 1e6:
                    raise RuntimeError(
                        f"{self.name}: could not bracket supersonic Mach solution."
                    )

            return float(brentq(residual, lo, hi))

        raise ValueError(
            f"{self.name}: branch must be 'subsonic' or 'supersonic'. "
            f"Got {branch!r}."
        )

    def _validate_positive(self, name: str, value: float) -> None:
        if value <= 0.0:
            raise ValueError(
                f"{self.name}: {name} must be greater than zero. Got {value}."
            )

    @staticmethod
    def _isclose(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)



class ChokedFannoFlow(Component):
    """
    Choked Fanno flow model for ideal-gas duct flow.

    `ChokedFannoFlow` computes the upstream state required for a constant-area
    adiabatic duct with friction to choke at the downstream end. The downstream
    state is treated as the Fanno star state.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_density : State
        Upstream density
    upstream_speed_of_sound : State
        Upstream speed of sound
    specific_heat_ratio : State
        Specific heat ratio
    friction_factor : State
        Darcy friction factor
    length : float
        Tube length
    inner_diameter : float
        Tube inner diameter
    upstream_static_enthalpy : State, optional
        Upstream static enthalpy
    regime : str, optional
        Fanno branch to use
    upstream_mach_number : State, optional
        Upstream Mach number

    Outputs
    -------
    mass_flux : State, optional
        Computed mass flux
    mass_flow : State, optional
        Computed mass flow rate
    total_enthalpy : State, optional
        Upstream total enthalpy
    downstream_mach_number : State or float, optional
        Downstream Mach number
    static_temperature_ratio : State, optional
        Downstream-to-upstream static temperature ratio
    static_pressure_ratio : State, optional
        Downstream-to-upstream static pressure ratio
    density_ratio : State, optional
        Downstream-to-upstream density ratio
    velocity_ratio : State, optional
        Downstream-to-upstream velocity ratio
    total_pressure_ratio : State, optional
        Downstream-to-upstream total pressure ratio
    total_temperature_ratio : State or float, optional
        Downstream-to-upstream total temperature ratio
    friction_factor_to_choke : State, optional
        Friction factor required to choke for the current geometry
    fL_over_D_to_choke : State, optional
        Fanno friction length required to choke

    Notes
    -----
    This component assumes forward flow only, constant friction factor, ideal-gas
    behavior, and a circular duct. Ratios are downstream to upstream.

    If upstream Mach number is provided, it is used to calculate ratios. If it
    is not provided, it is calculated from friction factor, length, and diameter.

    Flow area is evaluated from:

        ``A = pi / 4 * D**2``

    Friction length is evaluated from:

        ``fL_over_D = f * L / D``

    Mass flux is evaluated from:

        ``mass_flux = rho1 * M1 * a1``

    Mass flow is evaluated from:

        ``mass_flow = mass_flux * A``

    Total enthalpy is evaluated from:

        ``total_enthalpy = upstream_static_enthalpy + 0.5 * velocity1**2``

    Upstream velocity is evaluated from:

        ``velocity1 = M1 * a1``

    Static temperature ratio is evaluated from:

        ``T2 / T1 = (1 + 0.5 * (gamma - 1) * M1**2)
        / (1 + 0.5 * (gamma - 1) * M2**2)``

    Density ratio is evaluated from:

        ``rho2 / rho1 = (M1 / M2) * sqrt(1 / (T2 / T1))``

    Static pressure ratio is evaluated from:

        ``P2 / P1 = (rho2 / rho1) * (T2 / T1)``

    Velocity ratio is evaluated from:

        ``v2 / v1 = 1 / (rho2 / rho1)``

    Total pressure ratio is evaluated from:

        ``P02 / P01 = (M1 / M2)
        * (T2 / T1) ** ((gamma + 1) / (2 * (1 - gamma)))``

    The Fanno function is evaluated from:

        ``fL_over_D_to_choke = (1 - M**2) / (gamma * M**2)
        + (gamma + 1) / (2 * gamma)
        * log(((gamma + 1) * M**2) / (2 + (gamma - 1) * M**2))``
    """
    def __init__(
        self,
        name: str,
        network: Network,
        upstream_density: State,
        upstream_speed_of_sound: State,
        specific_heat_ratio: State,
        friction_factor: State,
        length: float,
        inner_diameter: float,
        upstream_static_enthalpy: State | None = None,
        regime: str = "subsonic",
        upstream_mach_number: State | None = None,

        mass_flux: State | None = None,
        mass_flow: State | None = None,
        total_enthalpy: State | None = None,
        downstream_mach_number=1.0,
        static_temperature_ratio: State | None = None,
        static_pressure_ratio: State | None = None,
        density_ratio: State | None = None,
        velocity_ratio: State | None = None,
        total_pressure_ratio: State | None = None,
        total_temperature_ratio=1.0,
        friction_factor_to_choke: State | None = None,
        fL_over_D_to_choke: State | None = None
    ):
        self._use_given_mach = upstream_mach_number is not None
        self.setup()

        temp = self.regime
        self.regime = self.regime.lower()

        if self.regime not in ("subsonic", "supersonic"):
            raise ValueError(
                f"Regime must be 'subsonic' or 'supersonic', got {temp}"
            )
    

    def evaluate_states(self):
        k = self.specific_heat_ratio.value
        rho1 = self.upstream_density.value
        a1 = self.upstream_speed_of_sound.value
        L = self.length.value
        D = self.inner_diameter.value
        f = self.friction_factor.value
        A = (math.pi / 4.0) * D**2

        if self._use_given_mach:
            M1 = self.upstream_mach_number.value
            self._validate_mach_regime(M1)
        else:
            fL_D = f * L / D
            M1 = self._inverse_fanno_function(fL_D, k, self.regime)
            self.upstream_mach_number.value = M1

        fL_D_to_choke = self._fanno_function(M1, k)
        self.fL_over_D_to_choke.value = fL_D_to_choke
        self.friction_factor_to_choke.value = fL_D_to_choke * D / L

        G = rho1 * M1 * a1
        mdot = G * A

        self.mass_flux.value = G
        self.mass_flow.value = mdot
        self.upstream_mach_number.value = M1

        if self.upstream_static_enthalpy.is_assigned:
            h1 = self.upstream_static_enthalpy.value
            v1 = M1 * a1
            self.total_enthalpy.value = h1 + 0.5 * v1**2

        M2 = self.downstream_mach_number.value

        T2_T1 = (1 + 0.5 * (k - 1) * M1**2) / (1 + 0.5 * (k - 1) * M2**2)
        rho2_rho1 = (M1 / M2) * sqrt_or_nan(1 / T2_T1)
        p2_p1 = rho2_rho1 * T2_T1
        v2_v1 = 1 / rho2_rho1
        po2_po1 = (M1 / M2) * T2_T1**((k + 1) / (2 * (1 - k)))

        self.static_temperature_ratio.value = T2_T1
        self.static_pressure_ratio.value = p2_p1
        self.density_ratio.value = rho2_rho1
        self.velocity_ratio.value = v2_v1
        self.total_pressure_ratio.value = po2_po1
        self.total_temperature_ratio.value = 1.0

    def _validate_mach_regime(self, M: float) -> None:
        if self.regime == "subsonic" and M >= 1.0:
            raise ValueError(f"Subsonic Fanno flow requires M1 < 1. Got M1={M:.6g}.")

        if self.regime == "supersonic" and M <= 1.0:
            raise ValueError(f"Supersonic Fanno flow requires M1 > 1. Got M1={M:.6g}.")

    def _fanno_function(self, M: float, k: float) -> float:
        return (
            (1.0 - M**2) / (k * M**2)
            + (k + 1.0) / (2.0 * k)
            * math.log(((k + 1.0) * M**2) / (2.0 + (k - 1.0) * M**2))
        )

    def _valid_fanno_geometry_message(self, target: float, k: float, branch: str) -> str:
        f = self.friction_factor.value
        L = self.length.value
        D = self.inner_diameter.value
        current = f * L / D

        if branch == "supersonic":
            M_limit = 10.0
            target_limit = self._fanno_function(M_limit, k)
            direction = "shorter tube, larger diameter, or lower friction factor"
        elif branch == "subsonic":
            M_limit = 1e-6
            target_limit = self._fanno_function(M_limit, k)
            direction = "longer tube, smaller diameter, or higher friction factor"
        else:
            raise ValueError("branch must be 'subsonic' or 'supersonic'")

        valid_L = target_limit * D / f
        valid_D = f * L / target_limit

        return (
            f"No valid {branch} Fanno solution for fL/D={current:.6g}, k={k:.6g}.\n"
            f"Current geometry: L={L:.6g} m, D={D:.6g} m, f={f:.6g}.\n"
            f"Try a {direction}.\n"
            f"At current D and f, use approximately L <= {valid_L:.6g} m.\n"
            f"At current L and f, use approximately D >= {valid_D:.6g} m."
        )

    def _inverse_fanno_function(self, target: float, k: float, branch: str = "subsonic") -> float:
        if target <= 0.0:
            return 1.0

        c = (k + 1.0) / 2.0
        B = 1.0 + (k / c) * target

        if branch == "subsonic":
            u = wrightomega(B)
        elif branch == "supersonic":
            u = -lambertw(-math.exp(-B), k=0).real
        else:
            raise ValueError("branch must be 'subsonic' or 'supersonic'")

        x = c * u - (k - 1.0) / 2.0

        if x <= 0.0 or not math.isfinite(x):
            raise ValueError(self._valid_fanno_geometry_message(target, k, branch))

        return float(1.0 / sqrt_or_nan(x))
    






class ChokedRayleighFlow(Component):
    """
    Choked Rayleigh flow model for ideal-gas heat addition.

    `ChokedRayleighFlow` computes the Rayleigh choking state for frictionless
    constant-area flow with heat addition. The downstream state is treated as the
    Rayleigh star state, so the downstream Mach number is one.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_density : State
        Upstream density
    upstream_speed_of_sound : State
        Upstream speed of sound
    upstream_static_temperature : State
        Upstream static temperature
    specific_heat_ratio : State
        Specific heat ratio
    specific_gas_constant : State
        Specific gas constant
    inner_diameter : float
        Duct inner diameter
    heat_rate : State or float, optional
        Heat rate added to the flow
    upstream_static_enthalpy : State, optional
        Upstream static enthalpy
    regime : str, optional
        Rayleigh branch to use
    upstream_mach_number : State, optional
        Upstream Mach number

    Outputs
    -------
    mass_flux : State, optional
        Computed mass flux
    mass_flow : State, optional
        Computed mass flow rate
    total_enthalpy_in : State, optional
        Upstream total enthalpy
    total_enthalpy_out : State, optional
        Downstream total enthalpy
    downstream_mach_number : State or float, optional
        Downstream Mach number
    static_temperature_ratio : State, optional
        Downstream-to-upstream static temperature ratio
    static_pressure_ratio : State, optional
        Downstream-to-upstream static pressure ratio
    density_ratio : State, optional
        Downstream-to-upstream density ratio
    velocity_ratio : State, optional
        Downstream-to-upstream velocity ratio
    total_temperature_ratio : State, optional
        Downstream-to-upstream total temperature ratio
    total_pressure_ratio : State, optional
        Downstream-to-upstream total pressure ratio
    heat_per_mass : State, optional
        Heat addition per unit mass
    heat_rate_to_choke : State, optional
        Heat rate required to reach choking
    heat_per_mass_to_choke : State, optional
        Heat addition per unit mass required to reach choking

    Notes
    -----
    This component assumes forward flow only, frictionless Rayleigh flow,
    ideal-gas behavior, and a constant-area duct. Positive heat rate adds energy
    to the flow. Ratios are downstream to upstream, where downstream corresponds
    to the Rayleigh star state.

    If upstream Mach number is provided, the heat rate input is treated
    diagnostically and does not determine the Mach number. If upstream Mach
    number is not provided, heat rate determines the upstream Mach number
    through the Rayleigh choking relation.

    Flow area is evaluated from:

        ``A = pi / 4 * D**2``

    Specific heat at constant pressure is evaluated from:

        ``cp = gamma * R / (gamma - 1)``

    Mass flux is evaluated from:

        ``mass_flux = rho1 * M1 * a1``

    Mass flow is evaluated from:

        ``mass_flow = mass_flux * A``

    Upstream total temperature is evaluated from:

        ``T01 = T1 * (1 + 0.5 * (gamma - 1) * M1**2)``

    Total temperature ratio to star is evaluated from:

        ``T01 / T0star = ((1 + gamma) / (1 + gamma * M1**2))**2
        * M1**2
        * ((1 + (gamma - 1) / 2 * M1**2)
        / (1 + (gamma - 1) / 2))``

    Heat per mass to choke is evaluated from:

        ``heat_per_mass_to_choke = cp * (T0star - T01)``

    Heat rate to choke is evaluated from:

        ``heat_rate_to_choke = mass_flow * heat_per_mass_to_choke``

    Heat per mass is evaluated from:

        ``heat_per_mass = heat_rate / mass_flow``

    Total enthalpy out is evaluated from:

        ``total_enthalpy_out = total_enthalpy_in + heat_per_mass``

    Static pressure ratio is evaluated from:

        ``Pstar / P1 = (1 + gamma * M1**2) / (1 + gamma)``

    Static temperature ratio is evaluated from:

        ``Tstar / T1 = (((1 + gamma) * M1) / (1 + gamma * M1**2)) ** 2``

    Density ratio is evaluated from:

        ``rhostar / rho1 = (1 + gamma) * M1**2 / (1 + gamma * M1**2)``

    Velocity ratio is evaluated from:

        ``vstar / v1 = 1 / (rhostar / rho1)``

    Total pressure ratio to star is evaluated from:

        ``P01 / P0star = ((2 + (gamma - 1) * M1**2) / (1 + gamma))
        ** (gamma / (gamma - 1))
        * ((1 + gamma) / (1 + gamma * M1**2))``

    Choking heat rate is evaluated from:

        ``heat_rate = C * (1 - M1**2)**2 / M1``

        ``C = rho1 * a1 * A * cp * T1 / (2 * (1 + gamma))``
    """
    def __init__(
        self,
        name: str,
        network: Network,
        upstream_density: State,
        upstream_speed_of_sound: State,
        upstream_static_temperature: State,
        specific_heat_ratio: State,
        specific_gas_constant: State,
        inner_diameter: float,
        heat_rate: State | float | None = None,
        upstream_static_enthalpy: State | None = None,
        regime: str = "subsonic",
        upstream_mach_number: State | None = None,

        mass_flux: State | None = None,
        mass_flow: State | None = None,
        total_enthalpy_in: State | None = None,
        total_enthalpy_out: State | None = None,
        downstream_mach_number: State | float = 1.0,
        static_temperature_ratio: State | None = None,
        static_pressure_ratio: State | None = None,
        density_ratio: State | None = None,
        velocity_ratio: State | None = None,
        total_temperature_ratio: State | None = None,
        total_pressure_ratio: State | None = None,
        heat_per_mass: State | None = None,
        heat_rate_to_choke: State | None = None,
        heat_per_mass_to_choke: State | None = None,
    ):
        self._use_given_mach = upstream_mach_number is not None
        self.setup()

        original_regime = self.regime
        self.regime = self.regime.lower()

        if self.regime not in ("subsonic", "supersonic"):
            raise ValueError(
                f"Regime must be 'subsonic' or 'supersonic', got {original_regime}"
            )

    def evaluate_states(self):
        k = self.specific_heat_ratio.value
        R = self.specific_gas_constant.value
        cp = k * R / (k - 1.0)

        rho1 = self.upstream_density.value
        a1 = self.upstream_speed_of_sound.value
        T1 = self.upstream_static_temperature.value

        D = self.inner_diameter.value
        A = (math.pi / 4.0) * D**2

        if self._use_given_mach:
            M1 = self.upstream_mach_number.value
            self._validate_mach_regime(M1)
        else:
            if not self.heat_rate.is_assigned:
                raise ValueError(
                    f"{self.name}: heat_rate is required when "
                    "upstream_mach_number is not provided."
                )

            M1 = self._mach_from_choking_heat_rate(
                qdot=self.heat_rate.value,
                rho1=rho1,
                a1=a1,
                A=A,
                T1=T1,
                cp=cp,
                k=k,
                regime=self.regime,
            )

            self.upstream_mach_number.value = M1

        G = rho1 * M1 * a1
        mdot = G * A

        self.mass_flux.value = G
        self.mass_flow.value = mdot
        self.downstream_mach_number.value = 1.0

        T01 = T1 * (1.0 + 0.5 * (k - 1.0) * M1**2)
        T01_T0star = self._rayleigh_total_temperature_ratio_to_star(M1, k)
        T0star = T01 / T01_T0star

        q_to_choke = cp * (T0star - T01)
        qdot_to_choke = mdot * q_to_choke

        self.heat_per_mass_to_choke.value = q_to_choke
        self.heat_rate_to_choke.value = qdot_to_choke

        if self.heat_rate.is_assigned:
            q = self.heat_rate.value / mdot
        else:
            q = q_to_choke

        self.heat_per_mass.value = q

        if self.upstream_static_enthalpy.is_assigned:
            h1 = self.upstream_static_enthalpy.value
            u1 = M1 * a1
            h01 = h1 + 0.5 * u1**2

            self.total_enthalpy_in.value = h01
            self.total_enthalpy_out.value = h01 + q

        pstar_p1 = (1.0 + k * M1**2) / (1.0 + k)

        Tstar_T1 = (((1.0 + k) * M1) / (1.0 + k * M1**2)) ** 2

        rhostar_rho1 = (1.0 + k) * M1**2/ (1.0 + k * M1**2)

        vstar_v1 = 1.0 / rhostar_rho1

        T0star_T01 = 1.0 / T01_T0star

        p01_p0star = self._rayleigh_total_pressure_ratio_to_star(M1, k)
        p0star_p01 = 1.0 / p01_p0star

        self.static_pressure_ratio.value = pstar_p1
        self.static_temperature_ratio.value = Tstar_T1
        self.density_ratio.value = rhostar_rho1
        self.velocity_ratio.value = vstar_v1
        self.total_temperature_ratio.value = T0star_T01
        self.total_pressure_ratio.value = p0star_p01

    def _validate_mach_regime(self, M: float) -> None:
        if self.regime == "subsonic" and not (0.0 < M < 1.0):
            raise ValueError(
                f"Subsonic Rayleigh flow requires 0 < M1 < 1. Got M1={M:.6g}."
            )

        if self.regime == "supersonic" and M <= 1.0:
            raise ValueError(
                f"Supersonic Rayleigh flow requires M1 > 1. Got M1={M:.6g}."
            )

    def _rayleigh_total_temperature_ratio_to_star(self, M: float, k: float) -> float:
        return ((1+k)/(1+k*M**2))**2 * M**2 * ((1+(k-1)/2 * M**2)/(1+(k-1)/2))

    def _rayleigh_total_pressure_ratio_to_star(self, M: float, k: float) -> float:
        return (
            ((2.0 + (k - 1.0) * M**2) / (1.0 + k)) ** (k / (k - 1.0))
            * ((1.0 + k) / (1.0 + k * M**2))
        )

    def _q_to_choke_per_mass(self, M: float, T1: float, cp: float, k: float) -> float:
        T01 = T1 * (1.0 + 0.5 * (k - 1.0) * M**2)
        T01_T0star = self._rayleigh_total_temperature_ratio_to_star(M, k)
        T0star = T01 / T01_T0star

        return cp * (T0star - T01)
        
    def _valid_rayleigh_heat_message(
        self,
        qdot: float,
        rho1: float,
        a1: float,
        A: float,
        T1: float,
        cp: float,
        k: float,
        regime: str,
    ) -> str:
        C = rho1 * a1 * A * cp * T1 / (2.0 * (1.0 + k))

        examples = []
        for M in [0.2, 0.4, 0.6, 0.8, 0.95]:
            if regime == "supersonic":
                M = 1.0 / M

            qdot_to_choke = C * ((1.0 - M**2) ** 2 / M)
            examples.append((M, qdot_to_choke))

        lines = [
            f"No valid {regime} Rayleigh choking solution for heat_rate={qdot:.6g} W.",
            f"Current upstream state gives C={C:.6g} W, where:",
            "    heat_rate = C * (1 - M1^2)^2 / M1",
            "",
            "Compatible example heat rates:",
        ]

        for M, q in examples:
            lines.append(f"    M1={M:.4g} -> heat_rate={q:.6g} W")

        return "\n".join(lines)

    def _mach_from_choking_heat_rate(
        self,
        qdot: float,
        rho1: float,
        a1: float,
        A: float,
        T1: float,
        cp: float,
        k: float,
        regime: str,
    ) -> float:
        if qdot <= 0.0:
            raise ValueError(
                f"{self.name}: Choked Rayleigh flow requires positive heat_rate. "
                f"Got {qdot:.6g}."
            )

        C = rho1 * a1 * A * cp * T1 / (2.0 * (1.0 + k))
        H = qdot / C

        # M^4 - 2 M^2 - H M + 1 = 0
        coeffs = [1.0, 0.0, -2.0, -H, 1.0]
        roots = np.roots(coeffs)

        real_roots = [
            float(r.real)
            for r in roots
            if abs(r.imag) < 1e-10 and r.real > 0.0
        ]

        if regime == "subsonic":
            candidates = [M for M in real_roots if 0.0 < M < 1.0]
        elif regime == "supersonic":
            candidates = [M for M in real_roots if M > 1.0]
        else:
            raise ValueError("regime must be 'subsonic' or 'supersonic'")

        if not candidates:
            raise ValueError(
                self._valid_rayleigh_heat_message(
                    qdot=qdot,
                    rho1=rho1,
                    a1=a1,
                    A=A,
                    T1=T1,
                    cp=cp,
                    k=k,
                    regime=regime,
                )
            )

        return candidates[0]
    








class StationaryNormalShock(Component):
    """
    Stationary normal shock model for ideal-gas flow.

    `StationaryNormalShock` computes downstream Mach number and normal-shock
    property ratios for a stationary shock in an inertial reference frame. The
    component can use either upstream Mach number or static pressure ratio as
    the defining input.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    specific_heat_ratio : State
        Specific heat ratio
    upstream_mach_number : State, optional
        Upstream Mach number
    static_pressure_ratio : State, optional
        Upstream-to-downstream static pressure ratio

    Outputs
    -------
    downstream_mach_number : State, optional
        Downstream Mach number
    static_temperature_ratio : State, optional
        Downstream-to-upstream static temperature ratio
    density_ratio : State, optional
        Downstream-to-upstream density ratio
    velocity_ratio : State, optional
        Downstream-to-upstream velocity ratio
    total_pressure_ratio : State, optional
        Downstream-to-upstream total pressure ratio
    total_temperature_ratio : State, optional
        Downstream-to-upstream total temperature ratio
    sonic_area_ratio : State, optional
        Downstream-to-upstream sonic area ratio

    Notes
    -----
    This component assumes forward flow only and ideal-gas behavior. It applies
    only if the shock is stationary in the inertial reference frame. Otherwise,
    the inlet Mach number must be transformed with a Galilean velocity
    transformation.

    Downstream-to-upstream static pressure ratio is evaluated from:

        ``P2 / P1 = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)``

    Upstream-to-downstream static pressure ratio is evaluated from:

        ``P1 / P2 = 1 / (P2 / P1)``

    Upstream Mach number from static pressure ratio is evaluated from:

        ``M1 = sqrt(1 + ((gamma + 1) / (2 * gamma)) * (P2 / P1 - 1))``

    Downstream Mach number is evaluated from:

        ``M2 = sqrt((M1**2 + 2 / (gamma - 1))
        / ((2 * gamma / (gamma - 1)) * M1**2 - 1))``

    Static temperature ratio is evaluated from:

        ``T2 / T1 = (1 + 0.5 * (gamma - 1) * M1**2)
        / (1 + 0.5 * (gamma - 1) * M2**2)``

    Density ratio is evaluated from:

        ``rho2 / rho1 = (P2 / P1) / (T2 / T1)``

    Velocity ratio is evaluated from:

        ``v2 / v1 = 1 / (rho2 / rho1)``

    Total pressure ratio is evaluated from:

        ``P02 / P01 = (((gamma + 1) * M1**2)
        / (2 + (gamma - 1) * M1**2)) ** (gamma / (gamma - 1))
        * ((gamma + 1) / (2 * gamma * M1**2 - (gamma - 1)))
        ** (1 / (gamma - 1))``

    Total temperature ratio is evaluated from:

        ``T02 / T01 = 1``

    Sonic area ratio is evaluated from:

        ``A2_star / A1_star = 1 / (P02 / P01)``
    """
    def __init__(self, 
                 name: str, 
                 network: Network,
                 specific_heat_ratio: State,
                 upstream_mach_number: State | None = None,
                 static_pressure_ratio: State | None = None,
                 
                 downstream_mach_number: State | None = None,
                 static_temperature_ratio: State | None = None,
                 density_ratio: State | None = None,
                 velocity_ratio: State | None = None,
                 total_pressure_ratio: State | None = None,
                 total_temperature_ratio: State | None = 1.0,
                 sonic_area_ratio: State | None = None):
        self.setup()

        if self.upstream_mach_number.is_assigned:
            self._mach_mode = True

            if self.upstream_mach_number.value <= 1.0:
                raise ValueError(
                    "Upstream Mach number must be greater than 1.0. "
                    f"Got {self.upstream_mach_number.value}."
                )

        elif self.static_pressure_ratio.is_assigned:
            self._mach_mode = False

            if not (0.0 < self.static_pressure_ratio.value < 1.0):
                raise ValueError(
                    "Static pressure ratio must satisfy 0.0 < p1 / p2 < 1.0. "
                    f"Got {self.static_pressure_ratio.value}."
                )

        else:
            raise ValueError(
                "Either upstream Mach number or static pressure ratio must be assigned."
            )

    def evaluate_states(self):
        k = self.specific_heat_ratio.value

        if self._mach_mode:
            M1 = self.upstream_mach_number.value

            p2_p1 = 1.0 + (2.0 * k / (k + 1.0)) * (M1**2 - 1.0)
            p1_p2 = 1.0 / p2_p1

            self.static_pressure_ratio.value = p1_p2

        else:
            p1_p2 = self.static_pressure_ratio.value
            p2_p1 = 1.0 / p1_p2

            M1 = sqrt_or_nan(1.0 + ((k + 1.0) / (2.0 * k)) * (p2_p1 - 1.0))

            self.upstream_mach_number.value = M1

        M2 = sqrt_or_nan((M1**2 + 2.0 / (k - 1.0)) / ((2.0 * k / (k - 1.0)) * M1**2 - 1.0))

        T2_T1 = (1.0 + 0.5 * (k - 1.0) * M1**2) / (1.0 + 0.5 * (k - 1.0) * M2**2)

        rho2_rho1 = p2_p1 / T2_T1
        v2_v1 = 1.0 / rho2_rho1

        p02_p01 = (((k + 1.0) * M1**2) / (2.0 + (k - 1.0) * M1**2))**(k / (k - 1.0)) * ((k + 1.0) / (2.0 * k * M1**2 - (k - 1.0)))**(1.0 / (k - 1.0))

        T02_T01 = 1.0
        A2star_A1star = 1.0 / p02_p01

        self.downstream_mach_number.value = M2
        self.static_temperature_ratio.value = T2_T1
        self.density_ratio.value = rho2_rho1
        self.velocity_ratio.value = v2_v1
        self.total_pressure_ratio.value = p02_p01
        self.total_temperature_ratio.value = T02_T01
        self.sonic_area_ratio.value = A2star_A1star
'''