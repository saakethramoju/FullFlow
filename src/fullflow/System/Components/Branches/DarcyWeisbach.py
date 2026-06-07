from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network




def _effective_area_from_mass_flow(
    mass_flow: float,
    pressure_drop: float,
    density: float,
) -> float:
    """
    Computes an equivalent CdA/effective area from:

        mdot = CdA * sqrt(2 * rho * abs(dP))
    """
    if abs(pressure_drop) < 1e-12:
        return 0.0

    if density <= 0.0:
        raise ValueError("density must be positive.")

    return abs(mass_flow) / math.sqrt(2.0 * density * abs(pressure_drop))


class GravityPressureChange(Component):

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_pressure: State | float,
        density: State | float,
        elevation_change: State | float,
        gravitional_acceleration: State | float = 9.80665,
        downstream_pressure: State | None = None,
        mass_flow: State | None = None,
        effective_area: State | None = None,
    ):
        """
        Computes pressure change from elevation.

        elevation_change is positive upward.

        If mass_flow is assigned, this also computes an equivalent
        effective area using:

            mdot = CdA * sqrt(2 * rho * abs(dP))
        """
        self.setup()

        self.evaluate_states()

    def evaluate_states(self) -> None:
        self.downstream_pressure.value = (
            self.upstream_pressure.value
            - self.density.value
            * self.gravitional_acceleration.value
            * self.elevation_change.value
        )

        if self.mass_flow.is_assigned:
            dP = self.upstream_pressure.value - self.downstream_pressure.value
            self.effective_area.value = _effective_area_from_mass_flow(
                self.mass_flow.value,
                dP,
                self.density.value,
            )

class DarcyWeisbach(Component):

    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        upstream_pressure: State,
        downstream_pressure: State,
        length: float,
        cross_sectional_area: float,
        hydraulic_diameter: float,
        density: State,
        friction_factor: State | float | None = None,
        effective_area: State | None = None,
    ):
        self.setup()
        self._predicted_mass_flow = None

    def evaluate_states(self):
        pressure_drop = self.upstream_pressure.value - self.downstream_pressure.value

        Kf = 8.0 * self.friction_factor.value * self.length.value / (
            self.density.value
            * math.pi**2
            * self.hydraulic_diameter.value**5
        )

        if abs(pressure_drop) < 1e-12:
            self._predicted_mass_flow = 0.0
        else:
            self._predicted_mass_flow = math.copysign(
                math.sqrt(abs(pressure_drop) / Kf),
                pressure_drop,
            )

        self.effective_area.value = _effective_area_from_mass_flow(
            self.mass_flow.value,
            pressure_drop,
            self.density.value,
        )

    @property
    def iteration_variables(self):
        return [self.mass_flow]

    @property
    def residuals(self):
        return [self.mass_flow.value - self._predicted_mass_flow]





class RectanglePoiseuille(Component):

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




class CircularAnnulus(Component):

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
    Computes hydraulic diameter from flow area and wetted perimeter.

    Dh = 4A / Pw

    This is the geometry diameter used for Reynolds number, Nusselt number,
    and most duct-flow correlations. Positive hydraulic_diameter is required.
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