from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component

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
        cross_sectional_area: float | None = None,
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
            self.cross_sectional_area.value = A

        pressure = (p1 - p2) * A
        friction = 0.0
        inertia = 0.0
        gravity = 0.0

        if self.upstream_density.is_assigned:
            rho1 = self.upstream_density.value
            u1 = mdot / (rho1 * A)

            if self.height_change.is_assigned:
                dh = self.height_change.value
                gravity = rho1 * g * dh * A

            if self.friction_factor.is_assigned:
                f = self.friction_factor.value
                Kf = f * L / (2.0 * rho1 * D * A**2)
                friction = Kf * mdot * abs(mdot) * A

            if self.upstream_static_enthalpy.is_assigned:
                h1 = self.upstream_static_enthalpy.value
                self.total_enthalpy.value = h1 + 0.5 * u1**2

            if self.downstream_density.is_assigned:
                rho2 = self.downstream_density.value
                u2 = mdot / (rho2 * A)
                inertia = max(mdot, 0.0) * (u2 - u1) - max(-mdot, 0.0) * (u1 - u2)

        self._residual = pressure - friction - inertia - gravity
        self._mass_flow_dot = self._residual / L

    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]

    @property
    def residuals(self) -> list[State | float]:
        return [self._residual]

    @property
    def transient_variables(self) -> list[State]:
        return [self.mass_flow]

    @property
    def transient_derivatives(self) -> list[State | float]:
        return [self._mass_flow_dot]





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
        effective_area: float | None = None,
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
            self.cross_sectional_area.value = A

        if self.friction_factor.is_assigned:
            f = self.friction_factor.value
        else:
            f = 0.0

        if self.height_change.is_assigned:
            dh = self.height_change.value
        else:
            dh = 0.0

        Kf = f * L / (2.0 * rho * D * A**2)

        pressure_drop = p1 - p2

        if abs(pressure_drop) < 1e-12:
            self.effective_area.value = 0.0
        else:
            self.effective_area.value = abs(mdot) / math.sqrt(2.0 * rho * abs(pressure_drop))

        pressure = (p1 - p2) * A
        friction = Kf * mdot * abs(mdot) * A
        gravity = rho * g * dh * A

        self._residual = pressure - friction - gravity
        self._mass_flow_dot = self._residual / L

    @property
    def iteration_variables(self) -> list[State]:
        return [self.mass_flow]

    @property
    def residuals(self) -> list[State | float]:
        return [self._residual]

    @property
    def transient_variables(self) -> list[State]:
        return [self.mass_flow]

    @property
    def transient_derivatives(self) -> list[State | float]:
        return [self._mass_flow_dot]





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
        length: float | None = None,
        mass_flow: State | None = None,
    ):
        self.setup()

    def evaluate_states(self) -> None:
        P1 = self.upstream_pressure.value
        P2 = self.downstream_pressure.value
        rho = self.density.value
        Cd = self.discharge_coefficient.value
        A = self.cross_sectional_area.value

        dP = P1 - P2

        if self.length.is_assigned:
            L = self.length.value
            mdot = self.mass_flow.value

            R = 1.0 / (2.0 * (Cd * A)**2)
            Z = L / A

            self._mass_flow_dot = (dP - (R / rho) * mdot * abs(mdot)) / Z

        else:
            if dP > 0.0:
                sign = 1.0
            elif dP < 0.0:
                sign = -1.0
            else:
                sign = 0.0

            value = 2.0 * rho * abs(dP)

            if value >= 0.0:
                self.mass_flow.value = sign * Cd * A * math.sqrt(value)
            else:
                self.mass_flow.value = math.nan

    @property
    def transient_variables(self) -> list[State]:
        if self.length.is_assigned:
            return [self.mass_flow]
        return []

    @property
    def transient_derivatives(self) -> list[State | float]:
        if self.length.is_assigned:
            return [self._mass_flow_dot]
        return []





class CavitatingVenturi(Component):
    """
    Direct-calculation cavitating liquid venturi.

    The component uses pressure recovery factor to estimate the downstream
    pressure where cavitation begins.

    Recovery factor:

        R = (P2 - Pt) / (P1 - Pt)

    At incipient cavitation:

        Pt = Pvap

    Therefore:

        P2_critical = Pvap + R * (P1 - Pvap)

    If:

        P2 <= P2_critical

    the venturi is cavitating and the throat pressure is assumed to be vapor
    pressure:

        Pt = Pvap

    Cavitating flow:

        mdot = Cd_cav A sqrt(2 rho (P1 - Pvap))

    Otherwise, the venturi is treated as a normal restriction:

        mdot = sign(P1 - P2) Cd_noncav A sqrt(2 rho |P1 - P2|)

    This is a direct calculator. It does not add residuals or iteration variables.
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
        pressure_recovery_factor: float = 0.85,
        cavitating_discharge_coefficient: float = 0.94,
        noncavitating_discharge_coefficient: float = 0.6,
        mass_flow: State | None = None,
        is_cavitating: bool = False,
    ):
        self.setup()

    def evaluate_states(self):
        P1 = self.upstream_pressure.value
        P2 = self.downstream_pressure.value
        rho = self.density.value
        A = self.throat_area.value
        Pvap = self.vapor_pressure.value
        R = self.pressure_recovery_factor.value
        Cd_cav = self.cavitating_discharge_coefficient.value
        Cd_noncav = self.noncavitating_discharge_coefficient.value

        dP = P1 - P2

        if dP > 0.0:
            sign = 1.0
        elif dP < 0.0:
            sign = -1.0
        else:
            sign = 0.0

        P2_critical = Pvap + R * (P1 - Pvap)
        self.critical_downstream_pressure = P2_critical

        is_cavitating = self.is_cavitating.propose(dP > 0.0 and P2 <= P2_critical)

        if is_cavitating:
            self.throat_pressure = Pvap

            dP = P1 - Pvap

            if dP > 0.0:
                self.mass_flow.value = Cd_cav * A * math.sqrt(2.0 * rho * dP)
            else:
                self.mass_flow.value = 0.0

        else:
            if dP > 0.0 and R < 1.0:
                self.throat_pressure = P1 - dP / (1.0 - R)
            else:
                self.throat_pressure = P1

            if dP != 0.0:
                self.mass_flow.value = sign * Cd_noncav * A * math.sqrt(2.0 * rho * abs(dP))
            else:
                self.mass_flow.value = 0.0

    @property
    def ignored_export_attributes(self):
        return {"critical_downstream_pressure"}





class SeriesCdA(Component):
    """Equivalent effective area for restrictions in series."""
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

        for effective_area in self.effective_areas.value:
            if hasattr(effective_area, "value"):
                CdA = effective_area.value
            else:
                CdA = effective_area

            if abs(CdA) < 1e-12:
                self.effective_area.value = 0.0
                return

            inverse_area_squared_sum += 1.0 / CdA**2

        self.effective_area.value = 1.0 / inverse_area_squared_sum**0.5





class ParallelCdA(Component):
    """Equivalent effective area for restrictions in parallel."""
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
            effective_area.value if hasattr(effective_area, "value") else effective_area
            for effective_area in self.effective_areas.value
        )





class RectanglePoiseuille(Component):
    """Poiseuille number correlation for rectangular ducts."""
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
    """Poiseuille number correlation for elliptical ducts."""
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
    """Poiseuille number correlation for circular annuli."""
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
    """Hydraulic diameter from flow area and wetted perimeter."""
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