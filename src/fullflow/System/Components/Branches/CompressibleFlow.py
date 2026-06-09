from __future__ import annotations

import numpy as np
from scipy.special import lambertw, wrightomega
from scipy.optimize import brentq
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network



class IsentropicCompressibleOrifice(Component):
    """
    Isentropic compressible orifice model for ideal-gas flow.

    `IsentropicCompressibleOrifice` computes mass flow through an orifice using
    ideal-gas isentropic flow relations. The component automatically switches
    between unchoked and choked flow based on the downstream-to-upstream
    pressure ratio.

    Notes
    -----
    1) Assumes ideal-gas flow.
    2) Uses upstream total pressure and upstream total temperature.
    3) Uses downstream static pressure as the back pressure.
    4) Supports reverse flow by applying the sign of the pressure difference.
    5) Total enthalpy is calculated from ideal-gas constant-cp behavior.

    Relations
    ---------
    Total enthalpy:

    `total_enthalpy = cp * T0`

    `cp = gamma * R / (gamma - 1)`

    Critical pressure ratio:

    `P_back / P0 = (2 / (gamma + 1)) ** (gamma / (gamma - 1))`

    Choked mass flow:

    `mass_flow = sign * CdA * P0 * sqrt((gamma / (R * T0))
    * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))`

    Unchoked mass flow:

    `mass_flow = sign * CdA * P0 * sqrt((2 * gamma / (R * T0 * (gamma - 1)))
    * ((P_back / P0) ** (2 / gamma)
    - (P_back / P0) ** ((gamma + 1) / gamma)))`

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
        Downstream static pressure
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
    """
    def __init__(self,
                 name: str,
                 network: Network,
                 upstream_total_pressure: State,
                 upstream_total_temperature: State,
                 downstream_pressure: State,
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
        P2 = self.downstream_pressure.value

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








class IsentropicAreaChange(Component):
    """
    Isentropic area-change model for ideal-gas flow.

    `IsentropicAreaChange` computes the downstream Mach number, mass flow, total
    enthalpy, and isentropic property ratios for a change in flow area. The
    component can operate in pressure mode or area mode depending on which
    downstream quantity is provided.

    Notes
    -----
    1) Forward flow only.
    2) Isentropic, adiabatic, inviscid area change.
    3) Ideal gas with constant specific heat ratio.
    4) Ratios are downstream to upstream.
    5) If downstream static pressure is provided, this component uses pressure mode.
    6) If downstream area is provided and downstream static pressure is not
       provided, this component uses area mode.
    7) If both downstream static pressure and downstream area are provided,
       pressure mode takes priority and downstream area is overwritten with the
       isentropically consistent area.
    8) In pressure mode, downstream Mach is calculated from the isentropic
       pressure relation.
    9) In area mode, downstream Mach is calculated from the area-Mach relation
       using the selected subsonic or supersonic branch.
    10) Mass flow is calculated from the upstream static state, upstream Mach,
        and upstream area.
    11) Total enthalpy is calculated from the ideal-gas stagnation temperature.
    12) This is an explicit component; it does not add residuals to force
        consistency with an independently solved downstream node.
    13) To ensure consistency with a downstream node when solving based on exit
        area, assign the downstream node directly from this component's outputs
        or ratio outputs.
    14) This component does not model shocks, friction, heat transfer, choking
        losses, or non-isentropic pressure loss.

    Relations
    ---------
    Total temperature:

    `T0 = T * (1 + (gamma - 1) / 2 * M**2)`

    Total pressure:

    `P0 = P * (1 + (gamma - 1) / 2 * M**2) ** (gamma / (gamma - 1))`

    Area-Mach relation:

    `A / A_star = (1 / M) * ((2 / (gamma + 1))
    * (1 + (gamma - 1) / 2 * M**2))
    ** ((gamma + 1) / (2 * (gamma - 1)))`

    Mass flow:

    `mass_flow = P1 * sqrt(gamma / (R * T1)) * A1 * M1`

    Total enthalpy:

    `total_enthalpy = gamma * R * T0 / (gamma - 1)`

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_mach_number : State
        Upstream Mach number
    upstream_static_pressure : State
        Upstream static pressure
    upstream_static_temperature : State
        Upstream static temperature
    specific_gas_constant : float
        Specific gas constant
    specific_heat_ratio : float
        Specific heat ratio
    upstream_area : float
        Upstream flow area
    downstream_static_pressure : State, optional
        Downstream static pressure
    downstream_area : State or float, optional
        Downstream flow area
    exit_mach_regime : str, optional
        Mach branch used in area mode

    Outputs
    -------
    downstream_mach_number : State, optional
        Downstream Mach number
    mass_flow : State, optional
        Computed mass flow rate
    total_enthalpy : State, optional
        Total enthalpy
    static_temperature_ratio : State, optional
        Downstream-to-upstream static temperature ratio
    velocity_ratio : State, optional
        Downstream-to-upstream velocity ratio
    density_ratio : State, optional
        Downstream-to-upstream density ratio
    """
    def __init__(
        self,
        name: str,
        network: Network,
        upstream_mach_number: State,
        upstream_static_pressure: State,
        upstream_static_temperature: State,
        specific_gas_constant: float,
        specific_heat_ratio: float,
        upstream_area: float,
        downstream_static_pressure: State | None = None,
        downstream_area: State | float | None = None,
        exit_mach_regime: str = "subsonic",
        downstream_mach_number: State | None = None,
        mass_flow: State | None = None,
        total_enthalpy: State | None = None,
        static_temperature_ratio: State | None = None,
        velocity_ratio: State | None = None,
        density_ratio: State | None = None
    ):
        
        self._use_downstream_pressure = downstream_static_pressure is not None
        self._use_downstream_area = downstream_area is not None

        self.setup()

        temp = self.exit_mach_regime
        self.exit_mach_regime = self.exit_mach_regime.lower()

        if self.exit_mach_regime not in ("subsonic", "supersonic"):
            raise ValueError(
                f"Regime must be 'subsonic' or 'supersonic', got {temp}"
            )

        if self._use_downstream_pressure and self._use_downstream_area:
            self._mode = "pressure"
        elif self._use_downstream_pressure:
            self._mode = "pressure"
        elif self._use_downstream_area:
            self._mode = "area"
        else:
            raise ValueError(
                "IsentropicAreaChange requires either downstream_static_pressure "
                "or downstream_area."
            )

    def evaluate_states(self):
        M1 = self.upstream_mach_number.value
        p1 = self.upstream_static_pressure.value
        T1 = self.upstream_static_temperature.value
        A1 = self.upstream_area.value
        R = self.specific_gas_constant.value
        k = self.specific_heat_ratio.value

        T0 = T1 * (1.0 + (k - 1.0) / 2.0 * M1**2)
        p0 = p1 * (1.0 + (k - 1.0) / 2.0 * M1**2) ** (k / (k - 1.0))

        if self._mode == "pressure":
            p2 = self.downstream_static_pressure.value

            p2_p1 = p2 / p1
            C = (1.0 + (k - 1.0) / 2.0 * M1**2) ** (k / (k - 1.0))

            M2_squared = (2.0 / (k - 1.0)) * (
                (C / p2_p1) ** ((k - 1.0) / k) - 1.0
            )

            M2 = np.sqrt(max(M2_squared, 0.0))

            A1_Astar = self._area_mach_function(M1, k)
            Astar = A1 / A1_Astar
            A2_consistent = Astar * self._area_mach_function(M2, k)

            self.downstream_area.value = A2_consistent

        elif self._mode == "area":
            A2 = self.downstream_area.value

            A1_Astar = self._area_mach_function(M1, k)
            Astar = A1 / A1_Astar

            A2_Astar = A2 / Astar
            M2 = self._inverse_area_mach_function(A2_Astar, k, self.exit_mach_regime)

            p2 = p0 / (
                1.0 + (k - 1.0) / 2.0 * M2**2
            ) ** (k / (k - 1.0))

            self.downstream_static_pressure.value = p2

        else:
            raise ValueError(
                "IsentropicAreaChange requires either downstream_static_pressure "
                "or downstream_area to be assigned."
            )

        mdot = p1 * np.sqrt(k / (R * T1)) * A1 * M1
        h0 = k * R * T0 / (k - 1.0)

        T2_T1 = (1 + (k-1)/2 * M1**2) / (1 + (k-1)/2 * M2**2)
        v2_v1 = (M2/M1) * np.sqrt(T2_T1)
        rho2_rho1 = (p2/p1)**(1/k)

        self.downstream_mach_number.value = M2
        self.mass_flow.value = mdot
        self.total_enthalpy.value = h0
        self.static_temperature_ratio.value = T2_T1
        self.velocity_ratio.value = v2_v1
        self.density_ratio.value = rho2_rho1

    def _area_mach_function(self, M: float, gamma: float) -> float:
        M = float(M)
        gamma = float(gamma)

        if M <= 0.0:
            raise ValueError("Mach number must be positive.")

        return (
            1.0 / M
            * (
                (2.0 / (gamma + 1.0))
                * (1.0 + (gamma - 1.0) / 2.0 * M**2)
            )
            ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )

    def _inverse_area_mach_function(
        self,
        area_ratio: float,
        gamma: float,
        branch: str = "subsonic",
    ) -> float:
        area_ratio = float(area_ratio)
        gamma = float(gamma)

        if area_ratio < 1.0:
            raise ValueError("A/A* must be >= 1.")

        if np.isclose(area_ratio, 1.0):
            return 1.0

        def residual(M):
            return self._area_mach_function(M, gamma) - area_ratio

        if branch == "subsonic":
            return float(brentq(residual, 1e-12, 1.0 - 1e-12))

        elif branch == "supersonic":
            lo = 1.0 + 1e-12
            hi = 2.0

            while residual(hi) < 0.0:
                hi *= 2.0
                if hi > 1e6:
                    raise RuntimeError("Could not bracket supersonic Mach solution.")

            return float(brentq(residual, lo, hi))

        else:
            raise ValueError("branch must be 'subsonic' or 'supersonic'")
        








class CompressibleFlowTube(Component):
    """
    Compressible flow tube with longitudinal inertia and friction.

    `CompressibleFlowTube` solves mass flow as an iteration variable using a
    steady momentum residual between upstream and downstream nodes. The branch
    supports forward and reverse flow through the longitudinal inertia sign
    convention.

    Notes
    -----
    1) Forward and reverse flow are supported by the longitudinal inertia sign
       convention.
    2) Constant friction factor.
    3) Ideal gas or weakly compressible fluid properties supplied externally.
    4) Circular duct.
    5) This component solves mass flow as an iteration variable using a steady
       momentum residual.
    6) Upstream and downstream static pressures, densities, and temperatures are
       provided by the connected nodes.
    7) Static temperatures are only used for total temperature calculation.
    8) If upstream static enthalpy is provided, total enthalpy is calculated from
       upstream static enthalpy and upstream velocity.
    9) If upstream speed of sound is provided, upstream Mach number is calculated.
    10) If downstream speed of sound is provided, downstream Mach number is
        calculated.
    11) If specific heat ratio and upstream speed of sound are provided,
        upstream total pressure and total temperature are calculated.
    12) If specific heat ratio and downstream speed of sound are provided,
        downstream total pressure and total temperature are calculated.
    13) Total pressure calculations assume locally isentropic conversion between
        static and stagnation properties.
    14) The friction term is based on the supplied friction factor, length,
        diameter, density, and mass flow.
    15) This is not a choked-flow model; choking should be handled by a separate
        component or regime switch.
    16) To ensure consistency, upstream and downstream node properties should be
        solved by the network, while this branch supplies the momentum residual
        connecting those nodes.

    Relations
    ---------
    Flow area:

    `A = pi / 4 * D**2`

    Velocity:

    `u = mass_flow / (density * A)`

    Total enthalpy:

    `total_enthalpy = static_enthalpy + 0.5 * u**2`

    Mach number:

    `M = u / a`

    Total temperature:

    `T0 = T * (1 + 0.5 * (gamma - 1) * M**2)`

    Total pressure:

    `P0 = P * (1 + 0.5 * (gamma - 1) * M**2) ** (gamma / (gamma - 1))`

    Momentum residual:

    `residual = pressure_force - friction_force - inertia_force`

    Pressure force:

    `pressure_force = (P1 - P2) * A`

    Friction coefficient:

    `Kf = 8 * f * L / (rho * pi**2 * D**5)`

    Friction force:

    `friction_force = Kf * mass_flow * abs(mass_flow) * A`

    Iteration Variables
    -------------------
    mass_flow : State
        Tube mass flow rate

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Tube mass flow rate
    upstream_static_pressure : State
        Upstream static pressure
    upstream_static_temperature : State
        Upstream static temperature
    upstream_density : State
        Upstream density
    downstream_static_pressure : State
        Downstream static pressure
    downstream_static_temperature : State
        Downstream static temperature
    downstream_density : State
        Downstream density
    length : float
        Tube length
    inner_diameter : float
        Tube inner diameter
    friction_factor : float, optional
        Darcy friction factor
    upstream_static_enthalpy : State, optional
        Upstream static enthalpy
    upstream_speed_of_sound : State, optional
        Upstream speed of sound
    downstream_speed_of_sound : State, optional
        Downstream speed of sound
    specific_heat_ratio : State, optional
        Specific heat ratio

    Outputs
    -------
    total_enthalpy : State, optional
        Upstream total enthalpy
    upstream_mach_number : State, optional
        Upstream Mach number
    downstream_mach_number : State, optional
        Downstream Mach number
    upstream_total_pressure : State, optional
        Upstream total pressure
    upstream_total_temperature : State, optional
        Upstream total temperature
    downstream_total_pressure : State, optional
        Downstream total pressure
    downstream_total_temperature : State, optional
        Downstream total temperature
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
        downstream_total_temperature: State | None = None
    ):
        self.setup()

    def evaluate_states(self):
        mdot = self.mass_flow.value
        p1 = self.upstream_static_pressure.value
        rho1 = self.upstream_density.value
        p2 = self.downstream_static_pressure.value
        rho2 = self.downstream_density.value
        L = self.length.value
        D = self.inner_diameter.value
        A = (np.pi/4) * (D**2)

        u1 = mdot / (rho1 * A)
        u2 = mdot / (rho2 * A)

        if self.upstream_static_enthalpy.is_assigned:
            h1 = self.upstream_static_enthalpy.value
            self.total_enthalpy.value = h1 + 0.5 * u1**2

        if self.upstream_speed_of_sound.is_assigned:
            a1 = self.upstream_speed_of_sound.value
            M1 = u1 / a1
            self.upstream_mach_number.value = M1

        if self.downstream_speed_of_sound.is_assigned:
            a2 = self.downstream_speed_of_sound.value
            M2 = u2 / a2
            self.downstream_mach_number.value = M2

        if (
            self.specific_heat_ratio.is_assigned
            and self.upstream_speed_of_sound.is_assigned
        ):
            k = self.specific_heat_ratio.value
            self.upstream_total_temperature.value = self.upstream_static_temperature.value * (1 + 0.5 * (k - 1) * M1**2)
            self.upstream_total_pressure.value = p1 * (1 + 0.5 * (k - 1) * M1**2)**(k / (k - 1))

        if (
            self.specific_heat_ratio.is_assigned
            and self.downstream_speed_of_sound.is_assigned
        ):
            k = self.specific_heat_ratio.value
            self.downstream_total_temperature.value = self.downstream_static_temperature.value * (1 + 0.5 * (k - 1) * M2**2)
            self.downstream_total_pressure.value = p2 * (1 + 0.5 * (k - 1) * M2**2)**(k / (k - 1))      


        inertia = max(mdot, 0.0) * (u2 - u1) - max(-mdot, 0.0) * (u1 - u2)
        pressure = (p1 - p2) * A

        if not self.friction_factor.is_assigned:
            friction = 0
        else:
            f = self.friction_factor.value
            Kf = 8 * f * L / (rho1 * np.pi**2 * D**5)
            friction = Kf * mdot * np.abs(mdot) * A


        self._residual = pressure - friction - inertia


    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]
    
    @property
    def residuals(self) -> list[float]:
        return [self._residual]






class ChokedFannoFlow(Component):
    """
    Choked Fanno flow model for ideal-gas duct flow.

    `ChokedFannoFlow` computes the upstream state required for a constant-area
    adiabatic duct with friction to choke at the downstream end. The downstream
    state is treated as the Fanno star state.

    Notes
    -----
    1) Forward flow only.
    2) Constant friction factor.
    3) Ideal gas.
    4) Circular duct.
    5) If M1 is provided, it will be used to calculate ratios.
    6) If M1 is not provided, it will be calculated from friction factor,
       length, and diameter.
    7) Ratios are all downstream to upstream.
    8) Supersonic flow tends to be biased towards shorter tubes, larger
       diameters, or smaller friction factors.
    9) Total enthalpy can only be given if static enthalpy is provided.
    10) To ensure consistency with upstream node, the downstream node should be
        directly assigned based on the ratio outputs from this class and should
        not be an iterative node.

    Relations
    ---------
    Flow area:

    `A = pi / 4 * D**2`

    Friction length:

    `fL_over_D = f * L / D`

    Mass flux:

    `mass_flux = rho1 * M1 * a1`

    Mass flow:

    `mass_flow = mass_flux * A`

    Total enthalpy:

    `total_enthalpy = upstream_static_enthalpy + 0.5 * velocity1**2`

    Upstream velocity:

    `velocity1 = M1 * a1`

    Static temperature ratio:

    `T2 / T1 = (1 + 0.5 * (gamma - 1) * M1**2)
    / (1 + 0.5 * (gamma - 1) * M2**2)`

    Density ratio:

    `rho2 / rho1 = (M1 / M2) * sqrt(1 / (T2 / T1))`

    Static pressure ratio:

    `P2 / P1 = (rho2 / rho1) * (T2 / T1)`

    Velocity ratio:

    `v2 / v1 = 1 / (rho2 / rho1)`

    Total pressure ratio:

    `P02 / P01 = (M1 / M2) * (T2 / T1) ** ((gamma + 1) / (2 * (1 - gamma)))`

    Fanno function:

    `fL_over_D_to_choke = (1 - M**2) / (gamma * M**2)
    + (gamma + 1) / (2 * gamma)
    * log(((gamma + 1) * M**2) / (2 + (gamma - 1) * M**2))`

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
        A = (np.pi / 4.0) * D**2

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
        rho2_rho1 = (M1 / M2) * np.sqrt(1 / T2_T1)
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
            * np.log(((k + 1.0) * M**2) / (2.0 + (k - 1.0) * M**2))
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
            u = -np.real(lambertw(-np.exp(-B), k=0))
        else:
            raise ValueError("branch must be 'subsonic' or 'supersonic'")

        x = c * u - (k - 1.0) / 2.0

        if x <= 0.0 or not np.isfinite(x):
            raise ValueError(self._valid_fanno_geometry_message(target, k, branch))

        return float(1.0 / np.sqrt(x))
    







class ChokedRayleighFlow(Component):
    """
    Choked Rayleigh flow model for ideal-gas heat addition.

    `ChokedRayleighFlow` computes the Rayleigh choking state for frictionless
    constant-area flow with heat addition. The downstream state is treated as the
    Rayleigh star state, so the downstream Mach number is one.

    Notes
    -----
    1) Forward flow only.
    2) Frictionless Rayleigh flow.
    3) Ideal gas with constant specific heat ratio.
    4) Constant-area duct.
    5) Downstream state is the Rayleigh star state, so M2 = 1.
    6) If upstream Mach number is provided, the component explicitly computes
       the Rayleigh choking ratios and required heat addition to reach the star
       state.
    7) If upstream Mach number is not provided, the specified heat_rate is used
       to infer the upstream Mach number that would choke at the downstream
       state.
    8) This component models Rayleigh choking through heat addition. Positive
       heat_rate adds energy to the flow.
    9) Ratios are downstream-to-upstream, where downstream corresponds to the
       Rayleigh star state.
    10) Total enthalpy changes according to `q = heat_rate / mass_flow`.
    11) Total pressure decreases through Rayleigh flow because heat transfer
        generates entropy.
    12) For subsonic flow, heat addition accelerates the flow toward choking.
    13) For supersonic flow, heat addition decelerates the flow toward choking.
    14) If upstream Mach number is provided, the heat_rate input is treated
        diagnostically and does not determine the Mach number.
    15) If upstream Mach number is not provided, the heat_rate determines the
        upstream Mach number through the Rayleigh choking relation.
    16) To ensure consistency with upstream node, the downstream node should be
        directly assigned based on the ratio outputs from this class and should
        not be an iterative node.

    Relations
    ---------
    Flow area:

    `A = pi / 4 * D**2`

    Specific heat at constant pressure:

    `cp = gamma * R / (gamma - 1)`

    Mass flux:

    `mass_flux = rho1 * M1 * a1`

    Mass flow:

    `mass_flow = mass_flux * A`

    Upstream total temperature:

    `T01 = T1 * (1 + 0.5 * (gamma - 1) * M1**2)`

    Total temperature ratio to star:

    `T01 / T0star = ((1 + gamma) / (1 + gamma * M1**2))**2
    * M1**2
    * ((1 + (gamma - 1) / 2 * M1**2)
    / (1 + (gamma - 1) / 2))`

    Heat per mass to choke:

    `heat_per_mass_to_choke = cp * (T0star - T01)`

    Heat rate to choke:

    `heat_rate_to_choke = mass_flow * heat_per_mass_to_choke`

    Heat per mass:

    `heat_per_mass = heat_rate / mass_flow`

    Total enthalpy in:

    `total_enthalpy_in = upstream_static_enthalpy + 0.5 * velocity1**2`

    Total enthalpy out:

    `total_enthalpy_out = total_enthalpy_in + heat_per_mass`

    Static pressure ratio:

    `Pstar / P1 = (1 + gamma * M1**2) / (1 + gamma)`

    Static temperature ratio:

    `Tstar / T1 = (((1 + gamma) * M1) / (1 + gamma * M1**2)) ** 2`

    Density ratio:

    `rhostar / rho1 = (1 + gamma) * M1**2 / (1 + gamma * M1**2)`

    Velocity ratio:

    `vstar / v1 = 1 / (rhostar / rho1)`

    Total pressure ratio to star:

    `P01 / P0star = ((2 + (gamma - 1) * M1**2) / (1 + gamma))
    ** (gamma / (gamma - 1))
    * ((1 + gamma) / (1 + gamma * M1**2))`

    Choking heat-rate relation used when upstream Mach number is not provided:

    `heat_rate = C * (1 - M1**2)**2 / M1`

    `C = rho1 * a1 * A * cp * T1 / (2 * (1 + gamma))`

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
        A = (np.pi / 4.0) * D**2

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

    Notes
    -----
    1) Forward flow only.
    2) Only applies if the shock is stationary in the inertial reference frame.
       Otherwise, the inlet Mach number must be transformed with a Galilean
       velocity transformation.
    3) Assumed to be ideal gas.
    4) If Mach number is assigned, it is better to simply use the outputs from
       this class instead of iteration on the downstream node.

    Relations
    ---------
    Downstream-to-upstream static pressure ratio:

    `P2 / P1 = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)`

    Upstream-to-downstream static pressure ratio:

    `P1 / P2 = 1 / (P2 / P1)`

    Upstream Mach number from static pressure ratio:

    `M1 = sqrt(1 + ((gamma + 1) / (2 * gamma)) * (P2 / P1 - 1))`

    Downstream Mach number:

    `M2 = sqrt((M1**2 + 2 / (gamma - 1))
    / ((2 * gamma / (gamma - 1)) * M1**2 - 1))`

    Static temperature ratio:

    `T2 / T1 = (1 + 0.5 * (gamma - 1) * M1**2)
    / (1 + 0.5 * (gamma - 1) * M2**2)`

    Density ratio:

    `rho2 / rho1 = (P2 / P1) / (T2 / T1)`

    Velocity ratio:

    `v2 / v1 = 1 / (rho2 / rho1)`

    Total pressure ratio:

    `P02 / P01 = (((gamma + 1) * M1**2)
    / (2 + (gamma - 1) * M1**2)) ** (gamma / (gamma - 1))
    * ((gamma + 1) / (2 * gamma * M1**2 - (gamma - 1)))
    ** (1 / (gamma - 1))`

    Total temperature ratio:

    `T02 / T01 = 1`

    Sonic area ratio:

    `A2_star / A1_star = 1 / (P02 / P01)`

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

            M1 = np.sqrt(1.0 + ((k + 1.0) / (2.0 * k)) * (p2_p1 - 1.0))

            self.upstream_mach_number.value = M1

        M2 = np.sqrt((M1**2 + 2.0 / (k - 1.0)) / ((2.0 * k / (k - 1.0)) * M1**2 - 1.0))

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
