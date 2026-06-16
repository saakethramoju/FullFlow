from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component
from ._flow_math import pressure_drop_flow_rate, sqrt_or_nan, _effective_area_from_mass_flow

if TYPE_CHECKING:
    from fullflow.System import Network, State




class FlowTube(Component):

    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        upstream_static_pressure: State,
        downstream_static_pressure: State,
        length: float,
        hydraulic_diameter: float,
        cross_sectional_area: float,
        upstream_density: State | None = None,
        downstream_density: State | None = None,
        friction_factor: float | None = None,
        gravitational_acceleration: float = 9.80665,
        height_change: float | None = None,
        upstream_static_enthalpy: State | None = None,
        total_enthalpy: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        mdot = self.mass_flow.value

        p1 = self.upstream_static_pressure.value
        p2 = self.downstream_static_pressure.value
        g = self.gravitational_acceleration.value

        L = self.length.value
        D = self.hydraulic_diameter.value
        if self.cross_sectional_area.is_assigned:
            A = self.cross_sectional_area.value
        else:
            A = (math.pi / 4.0) * D**2

        pressure = (p1 - p2) * A
        friction = 0.0
        inertia = 0.0
        gravity = 0.0

        if self.upstream_density.is_assigned:
            rho1 = self.upstream_density.value 
            u1 = mdot / (rho1 * A)

            if self.height_change.is_assigned:
                dh = self.height_change.value
                gravity = rho1 * g * dh

            if self.friction_factor.is_assigned:
                f = self.friction_factor.value
                Kf = 8.0 * f * L / (rho1 * math.pi**2 * D**5)
                friction = Kf * mdot * abs(mdot) * A

            if self.upstream_static_enthalpy.is_assigned:
                h1 = self.upstream_static_enthalpy.value
                self.total_enthalpy.value = h1 + 0.5 * u1**2

            if self.downstream_density.is_assigned:
                rho2 = self.downstream_density.value
                u2 = mdot / (rho2 * A)
                inertia = max(mdot, 0.0) * (u2 - u1) - max(-mdot, 0.0) * (u1 - u2)

        self._residual = pressure - friction - inertia - gravity

    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]

    @property
    def residuals(self) -> list[float]:
        return [self._residual]
    





class DarcyWeisbach(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        upstream_pressure: State,
        downstream_pressure: State,
        length: float,
        hydraulic_diameter: float,
        density: State,
        cross_sectional_area: State | float | None = None,
        friction_factor: State | float | None = None,
        gravitational_acceleration: State | float = 9.80665,
        height_change: State | float | None = None,
        effective_area: float | None = None
    ):
        self.setup()

    def evaluate_states(self):
        mdot = self.mass_flow.value

        p1 = self.upstream_pressure.value
        p2 = self.downstream_pressure.value
        g = self.gravitational_acceleration.value

        L = self.length.value
        D = self.hydraulic_diameter.value
        rho = self.density.value
        
        if self.cross_sectional_area.is_assigned:
            A = self.cross_sectional_area.value
        else:
            A = (math.pi / 4.0) * D**2

        if self.friction_factor.is_assigned:
            f = self.friction_factor.value
        else:
            f = 0.0

        if self.height_change.is_assigned:
            dh = self.height_change.value
        else:
            dh = 0.0

        Kf = 8.0 * f * L / (rho * math.pi**2 * D**5)

        self.effective_area.value = _effective_area_from_mass_flow(mdot, p1-p2, rho)

        pressure = p1 - p2
        friction = Kf * mdot * abs(mdot) * A
        gravity = rho * g * dh

        self._residual = pressure - friction - gravity

    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]

    @property
    def residuals(self) -> list[float]:
        return [self._residual]







class DischargeCoefficient(Component):
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

        self.mass_flow.value = pressure_drop_flow_rate(P1 - P2, rho, Cd, A)





class CavitatingVenturi(Component):
    """
    Cavitating liquid venturi model.

    `CavitatingVenturi` computes mass flow through a liquid venturi using a
    noncavitating restriction model or a cavitating venturi model. The active
    mode is selected using a critical downstream-to-upstream pressure ratio.

    In cavitating mode, the throat pressure is assumed to be pinned to the
    vapor pressure corresponding to the upstream fluid state. If upstream
    temperature and critical temperature are both assigned, cavitation is
    disabled above the critical temperature.

    Cavitation onset and stable cavitating flow are not identical. Incipient
    cavitation begins when the throat pressure first reaches saturation
    pressure, while fully established cavitating flow depends on geometry and
    empirical behavior.

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

    Notes
    -----
    Noncavitating mass flow is evaluated from:

        ``mass_flow = sign(P1 - P2) * Cd_noncav * A_t * sqrt(2 * rho * abs(P1 - P2))``

    Cavitating mass flow is evaluated from:

        ``mass_flow = Cd_cav * A_t * sqrt(2 * rho * (P1 - vapor_pressure))``

    Cavitating mode is activated when:

        ``downstream_pressure / upstream_pressure < critical_pressure_ratio``
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
            self.mass_flow.value = pressure_drop_flow_rate(dP, rho, Cd_noncav, A)
        else:
            self.is_cavitating = True

            Pvap = self.vapor_pressure.value
            dP = P1 - Pvap

            self.mass_flow.value = Cd_cav * A * sqrt_or_nan(2.0 * rho * dP)






class SeriesCdA(Component):
    """
    Equivalent effective area for restrictions in series.

    `SeriesCdA` combines multiple effective flow areas into a single equivalent
    effective area. This is useful when several restrictions are arranged in
    series and should be represented as one equivalent restriction.

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

    Notes
    -----
    Series effective area is evaluated from:

        ``1 / effective_area_eq^2 = sum(1 / effective_area_i^2)``
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

    Notes
    -----
    Parallel effective area is evaluated from:

        ``effective_area_eq = sum(effective_area_i)``
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





class RectanglePoiseuille(Component):
    """
    Poiseuille number correlation for rectangular ducts.

    `RectanglePoiseuille` computes an approximate Poiseuille number for a
    rectangular duct from its height and width. The result can be used by
    laminar duct-flow pressure-loss or friction-factor calculations.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    height : float
        Rectangle height
    width : float
        Rectangle width

    Outputs
    -------
    poiseuille_number : State, optional
        Computed Poiseuille number

    Notes
    -----
    The aspect ratio is evaluated from the smaller half-dimension divided by
    the larger half-dimension:

        ``x = b / a``

    The Poiseuille number is evaluated from:

        ``Po = A0 + A1 * x + A2 * x^2 + A3 * x^3 + A4 * x^4``
    """
    def __init__(
        self,
        name: str,
        network: Network,
        height: float,
        width: float,
        poiseuille_number: float | None = None,
    ):
        if height <= 0.0:
            raise ValueError(f"Rectangle height must be positive. Got length={height}.")
        
        if width <= 0.0:
            raise ValueError(f"Rectangle width must be positive. Got length={width}.")
        
        self.setup()
        a = max(height/2, width/2)
        b = min(height/2, width/2)
        x = b/a
        A0 = 23.9201
        A1 = -29.436
        A2 = 30.3872
        A3 = -10.7128
        A4 = 0.0
        self.poiseuille_number.value = A0 + A1*x + A2*x**2 + A3*x**3 + A4*x**4





class EllipsePoiseuille(Component):
    """
    Poiseuille number correlation for elliptical ducts.

    `EllipsePoiseuille` computes an approximate Poiseuille number for an
    elliptical duct from its semi-major and semi-minor axes. The result can be
    used by laminar duct-flow pressure-loss or friction-factor calculations.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    semi_major_axis : float
        Ellipse semi-major axis
    semi_minor_axis : float
        Ellipse semi-minor axis

    Outputs
    -------
    poiseuille_number : State, optional
        Computed Poiseuille number

    Notes
    -----
    The aspect ratio is evaluated from the smaller semi-axis divided by the
    larger semi-axis:

        ``x = b / a``

    The Poiseuille number is evaluated from:

        ``Po = A0 + A1 * x + A2 * x^2 + A3 * x^3 + A4 * x^4``
    """
    def __init__(
        self,
        name: str,
        network: Network,
        semi_major_axis: float,
        semi_minor_axis: float,
        poiseuille_number: float | None = None,
    ):
        if semi_major_axis <= 0.0:
            raise ValueError(f"Ellipse semi-major axis must be positive. Got length={semi_major_axis}.")
        
        if semi_minor_axis <= 0.0:
            raise ValueError(f"Ellipse semi-minor axis must be positive. Got length={semi_minor_axis}.")

        self.setup()
        a = max(semi_major_axis, semi_minor_axis)
        b = min(semi_minor_axis, semi_major_axis)
        x = b/a
        A0 = 19.7669
        A1 = -4.53458
        A2 = -11.5239
        A3 = 22.3709
        A4 = -10.0874
        self.poiseuille_number.value = A0 + A1*x + A2*x**2 + A3*x**3 + A4*x**4






class CircularAnnulusPoiseuille(Component):
    """
    Poiseuille number correlation for circular annuli.

    `CircularAnnulusPoiseuille` computes an approximate Poiseuille number for a circular
    annulus from its inner and outer diameters. The result can be used by
    laminar annular duct-flow pressure-loss or friction-factor calculations.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    inner_diameter : float
        Annulus inner diameter
    outer_diameter : float
        Annulus outer diameter

    Outputs
    -------
    poiseuille_number : State, optional
        Computed Poiseuille number

    Notes
    -----
    The diameter ratio is evaluated from:

        ``x = inner_diameter / outer_diameter``

    For small diameter ratios, the Poiseuille number is evaluated from:

        ``Po = A0 * x^A1``

    Otherwise, the Poiseuille number is evaluated from:

        ``Po = A0 + A1 * x + A2 * x^2 + A3 * x^3 + A4 * x^4``
    """
    def __init__(
        self,
        name: str,
        network: Network,
        inner_diameter: float,
        outer_diameter: float,
        poiseuille_number: float | None = None,
    ):
        if inner_diameter <= 0.0:
            raise ValueError(f"Annulus inner diameter must be positive. Got length={inner_diameter}.")
        
        if outer_diameter <= 0.0:
            raise ValueError(f"Annulus outer_diameter must be positive. Got length={outer_diameter}.")

        self.setup()
        a = outer_diameter
        b = inner_diameter
        x = b/a

        if x < 0.2508:
            A0 = 24.8272
            A1 = 0.0479888
            self.poiseuille_number.value = A0 * x**A1
        else:
            A0 = 22.0513
            A1 = 6.44473
            A2 = -7.35451
            A3 = 2.78999
            A4 = 0
            self.poiseuille_number.value = A0 + A1*x + A2*x**2 + A3*x**3 + A4*x**4







class HydraulicDiameter(Component):
    """
    Hydraulic diameter from flow area and wetted perimeter.

    `HydraulicDiameter` computes hydraulic diameter from cross-sectional flow
    area and wetted perimeter. The result is commonly used as the characteristic
    diameter for Reynolds number, Nusselt number, and duct-flow correlations.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    cross_sectional_area : State or float
        Flow cross-sectional area
    wetted_perimeter : State or float
        Wetted perimeter

    Outputs
    -------
    hydraulic_diameter : State, optional
        Hydraulic diameter

    Notes
    -----
    Hydraulic diameter is evaluated from:

        ``hydraulic_diameter = 4 * cross_sectional_area / wetted_perimeter``
    """
    def __init__(
        self,
        name: str,
        network: Network,
        cross_sectional_area: State | float,
        wetted_perimeter: State | float,
        hydraulic_diameter: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        A = self.cross_sectional_area.value
        P = self.wetted_perimeter.value

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: cross_sectional_area must be greater than zero. Got {A}."
            )

        if P <= 0.0:
            raise ValueError(
                f"{self.name}: wetted_perimeter must be greater than zero. Got {P}."
            )

        self.hydraulic_diameter.value = 4.0 * A / P