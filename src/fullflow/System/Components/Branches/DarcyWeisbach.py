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
    Compute an equivalent effective area from mass flow.

    `_effective_area_from_mass_flow` computes the equivalent `CdA` or effective
    flow area that would produce the supplied mass flow for a given pressure
    drop and density.

    Parameters
    ----------
    mass_flow : float
        Mass flow rate
    pressure_drop : float
        Pressure drop
    density : float
        Fluid density

    Returns
    -------
    effective_area : float
        Equivalent effective area

    Notes
    -----
    Effective area is evaluated from:

        ``mass_flow = effective_area * sqrt(2 * density * abs(pressure_drop))``
    """
    if abs(pressure_drop) < 1e-12:
        return 0.0

    if density <= 0.0:
        raise ValueError("density must be positive.")

    return abs(mass_flow) / math.sqrt(2.0 * density * abs(pressure_drop))


class GravityPressureChange(Component):
    """
    Hydrostatic pressure change from elevation.

    `GravityPressureChange` computes downstream pressure from upstream pressure,
    fluid density, gravitational acceleration, and elevation change. Positive
    elevation change is upward, so pressure decreases as elevation increases.

    If mass flow is assigned, the component also computes an equivalent
    effective area for the resulting pressure drop.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_pressure : State or float
        Upstream pressure
    density : State or float
        Fluid density
    elevation_change : State or float
        Elevation change
    gravitional_acceleration : State or float, optional
        Gravitational acceleration
    downstream_pressure : State, optional
        Downstream pressure
    mass_flow : State, optional
        Mass flow rate
    effective_area : State, optional
        Equivalent effective area

    Outputs
    -------
    downstream_pressure : State, optional
        Downstream pressure
    effective_area : State, optional
        Equivalent effective area

    Notes
    -----
    Downstream pressure is evaluated from:

        ``downstream_pressure = upstream_pressure
        - density * gravitational_acceleration * elevation_change``

    Equivalent effective area is evaluated from:

        ``mass_flow = effective_area * sqrt(2 * density * abs(pressure_drop))``
    """

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
    """
    Darcy-Weisbach pressure-loss branch.

    `DarcyWeisbach` solves mass flow as an iteration variable using the
    Darcy-Weisbach pressure-loss relation. The predicted mass flow is computed
    from the pressure drop, density, friction factor, length, and hydraulic
    diameter, and the residual drives the solved mass flow toward that value.

    Residuals
    ---------
    mass_flow_balance : float
        Enforces consistency between solved and predicted mass flow

        ``mass_flow - predicted_mass_flow = 0``

    Iteration Variables
    -------------------
    mass_flow : State
        Branch mass flow rate

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Branch mass flow rate
    upstream_pressure : State
        Upstream pressure
    downstream_pressure : State
        Downstream pressure
    length : float
        Branch length
    cross_sectional_area : float
        Flow cross-sectional area
    hydraulic_diameter : float
        Hydraulic diameter
    density : State
        Fluid density
    friction_factor : State or float, optional
        Darcy friction factor
    effective_area : State, optional
        Equivalent effective area

    Outputs
    -------
    effective_area : State, optional
        Equivalent effective area

    Notes
    -----
    The Darcy-Weisbach coefficient is evaluated from:

        ``Kf = 8 * f * L / (density * pi^2 * hydraulic_diameter^5)``

    Predicted mass flow is evaluated from:

        ``predicted_mass_flow = sign(pressure_drop)
        * sqrt(abs(pressure_drop) / Kf)``

    Effective area is evaluated from:

        ``mass_flow = effective_area * sqrt(2 * density * abs(pressure_drop))``
    """

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