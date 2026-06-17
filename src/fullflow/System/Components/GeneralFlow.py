from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _sqrt_or_nan(value: float) -> float:
    return math.sqrt(value) if value >= 0.0 else math.nan


def _pressure_drop_flow_rate(
    pressure_drop: float,
    density: float,
    discharge_coefficient: float,
    area: float,
) -> float:
    return (
        _sign(pressure_drop)
        * discharge_coefficient
        * area
        * _sqrt_or_nan(2.0 * density * abs(pressure_drop))
    )


def _effective_area_from_mass_flow(
    mass_flow: float,
    pressure_drop: float,
    density: float,
) -> float:
    if abs(pressure_drop) < 1e-12:
        return 0.0
    if density <= 0.0:
        raise ValueError("density must be positive.")
    return abs(mass_flow) / math.sqrt(2.0 * density * abs(pressure_drop))



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

        self.mass_flow.value = _pressure_drop_flow_rate(P1 - P2, rho, Cd, A)





class CavitatingVenturi(Component):
    """Cavitating liquid venturi model."""

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
            self.mass_flow.value = _pressure_drop_flow_rate(dP, rho, Cd_noncav, A)
        else:
            self.is_cavitating = True

            Pvap = self.vapor_pressure.value
            dP = P1 - Pvap

            self.mass_flow.value = Cd_cav * A * _sqrt_or_nan(2.0 * rho * dP)






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

        for effective_area in self.effective_areas:
            CdA = effective_area.value

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
            effective_area.value
            for effective_area in self.effective_areas
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
