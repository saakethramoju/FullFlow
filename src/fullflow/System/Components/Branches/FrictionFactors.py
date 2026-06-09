from __future__ import annotations

import numpy as np
from scipy.special import wrightomega
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network



class Colebrook(Component):
    """
    Colebrook-White Darcy friction factor correlation.

    `Colebrook` computes a Darcy friction factor from mass flow, viscosity,
    hydraulic diameter, flow area, and roughness. Laminar flow uses a
    Poiseuille-number fallback, while turbulent flow uses an explicit
    Colebrook-White solution.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Fluid mass flow rate. The absolute value is used
    friction_factor : State
        Output Darcy friction factor
    hydraulic_diameter : State or float
        Hydraulic diameter
    dynamic_viscosity : State
        Fluid dynamic viscosity
    cross_sectional_area : State or float
        Flow cross-sectional area
    poiseuille_number : float, optional
        Poiseuille number used for laminar flow
    roughness : State or float, optional
        Absolute wall roughness
    reynolds_number : State or float, optional
        Output Reynolds number
    reynolds_number_threshold : State or float, optional
        Reynolds number threshold for laminar fallback

    Outputs
    -------
    friction_factor : State
        Darcy friction factor
    reynolds_number : State or float, optional
        Reynolds number

    Notes
    -----
    The hydraulic-diameter Reynolds number is evaluated from:

        ``Re_Dh = mdot * Dh / (mu * A)``

    The effective laminar diameter is evaluated from:

        ``Deff = 16 * Dh / Po``

    The effective Reynolds number is evaluated from:

        ``Re_eff = mdot * Deff / (mu * A)``

    For laminar flow, the Darcy friction factor is evaluated from:

        ``f = 4 * Po / Re_Dh``

    For turbulent flow, the explicit Colebrook-White solution is used.

    The Poiseuille number input is only used for the incompressible laminar
    fallback.
    """
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        friction_factor: State,
        hydraulic_diameter: State | float,
        dynamic_viscosity: State,
        cross_sectional_area: State | float,
        poiseuille_number: float = 16,
        roughness: State | float = 0.0,
        reynolds_number: State | float | None = None,
        reynolds_number_threshold: State | float = 2300.0,
    ):
        self.setup()

    def evaluate_states(self):
        mdot = abs(self.mass_flow.value)
        mu = self.dynamic_viscosity.value
        A = self.cross_sectional_area.value
        Dh = self.hydraulic_diameter.value
        Po = self.poiseuille_number.value
        e = self.roughness.value

        Re_Dh = mdot * Dh / (mu * A)
        Deff = 16.0 * Dh / Po
        Re_eff = mdot * Deff / (mu * A)

        Re_Dh = max(Re_Dh, 1e-12)
        Re_eff = max(Re_eff, 1e-12)

        self.Deff = Deff

        if Re_Dh <= self.reynolds_number_threshold.value:
            self.reynolds_number.value = Re_Dh
            f = 4*Po/Re_Dh
        else:
            self.reynolds_number.value = Re_eff
            f = self._colebrook_explicit(Re_eff, e, Deff)

        self.friction_factor.value = f

    def _colebrook_explicit(self, Re, roughness, hydraulic_diameter):
        a = 2.51 / Re
        b = roughness / (3.7 * hydraulic_diameter)
        c = 0.5 * np.log(10.0)

        y = np.log(c / a) + c * b / a
        x = wrightomega(y) / c - b / a

        return 1.0 / x**2




class Churchill(Component):
    """
    Churchill Darcy friction factor correlation.

    `Churchill` computes a Darcy friction factor from mass flow, viscosity,
    hydraulic diameter, flow area, roughness, and Poiseuille number. The
    Churchill correlation provides a smooth transition across laminar,
    transitional, and turbulent Reynolds numbers.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Fluid mass flow rate. The absolute value is used
    friction_factor : State
        Output Darcy friction factor
    hydraulic_diameter : State or float
        Hydraulic diameter
    dynamic_viscosity : State
        Fluid dynamic viscosity
    cross_sectional_area : State or float
        Flow cross-sectional area
    roughness : State or float, optional
        Absolute wall roughness
    poiseuille_number : float, optional
        Poiseuille number used for incompressible laminar flow
    reynolds_number : State or float, optional
        Output Reynolds number

    Outputs
    -------
    friction_factor : State
        Darcy friction factor
    reynolds_number : State or float, optional
        Reynolds number

    Notes
    -----
    The effective laminar diameter is evaluated from:

        ``Deff = 16 * Dh / Po``

    The Reynolds number is evaluated from:

        ``Re = mdot * Deff / (mu * A)``

    The relative roughness is evaluated from:

        ``relative_roughness = roughness / Deff``

    The Churchill auxiliary terms are evaluated from:

        ``A = (2.457 * log(1 / ((7 / Re)^0.9 + 0.27 * relative_roughness)))^16``

        ``B = (37530 / Re)^16``

    The Darcy friction factor is evaluated from:

        ``f = 8 * ((8 / Re)^12 + (A + B)^(-1.5))^(1 / 12)``

    The Poiseuille number input is only used for incompressible laminar flow.
    """
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        friction_factor: State,
        hydraulic_diameter: State | float,
        dynamic_viscosity: State,
        cross_sectional_area: State | float,
        roughness: State | float = 0.0,
        poiseuille_number: float = 16,
        reynolds_number: State | float | None = None,
    ):
        self.setup()
        self.Deff = 16*self.hydraulic_diameter.value / self.poiseuille_number.value

    def evaluate_states(self):
        self.Deff = 16 * self.hydraulic_diameter.value / self.poiseuille_number.value
        self.reynolds_number.value = (
            abs(self.mass_flow.value)
            * self.Deff
            / (self.dynamic_viscosity.value * self.cross_sectional_area.value)
        )

        self.reynolds_number.value = max(self.reynolds_number.value, 1e-12)
        Re = self.reynolds_number.value
        relative_roughness = self.roughness.value / self.Deff

        A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * relative_roughness))) ** 16
        B = (37530.0 / Re) ** 16
        f = 8.0 * ((8.0 / Re) ** 12 + (A + B) ** (-1.5)) ** (1.0 / 12.0)

        self.friction_factor.value = f


    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {
            "Deff",
        }






class PetukhovFriction(Component):
    """
    Petukhov smooth-pipe turbulent Darcy friction factor correlation.

    `PetukhovFriction` computes a Darcy friction factor from mass flow,
    viscosity, hydraulic diameter, and flow area. Laminar flow uses a
    Poiseuille-number fallback, while turbulent flow uses the Petukhov
    smooth-pipe correlation.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Fluid mass flow rate. The absolute value is used
    friction_factor : State
        Output Darcy friction factor
    hydraulic_diameter : State or float
        Hydraulic diameter
    dynamic_viscosity : State
        Fluid dynamic viscosity
    cross_sectional_area : State or float
        Flow cross-sectional area
    poiseuille_number : float, optional
        Poiseuille number used for laminar flow
    reynolds_number : State or float, optional
        Output Reynolds number
    reynolds_number_threshold : State or float, optional
        Reynolds number threshold for laminar fallback

    Outputs
    -------
    friction_factor : State
        Darcy friction factor
    reynolds_number : State or float, optional
        Reynolds number

    Notes
    -----
    The Reynolds number is evaluated from:

        ``Re = mdot * Dh / (mu * A)``

    For laminar flow, the Darcy friction factor is evaluated from:

        ``f = 4 * Po / Re``

    For turbulent smooth-pipe flow, the Darcy friction factor is evaluated from:

        ``f = (0.79 * ln(Re) - 1.64)^(-2)``

    This correlation returns the Darcy friction factor. The roughness input is
    intentionally omitted because this correlation does not include relative
    roughness. Use Colebrook or Churchill when wall roughness should be modeled.
    """
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        friction_factor: State,
        hydraulic_diameter: State | float,
        dynamic_viscosity: State,
        cross_sectional_area: State | float,
        poiseuille_number: float = 16,
        reynolds_number: State | float | None = None,
        reynolds_number_threshold: State | float = 2300.0,
    ):
        self.setup()

    def evaluate_states(self):
        mdot = abs(self.mass_flow.value)
        mu = self.dynamic_viscosity.value
        A = self.cross_sectional_area.value
        Dh = self.hydraulic_diameter.value
        Po = self.poiseuille_number.value

        Re = mdot * Dh / (mu * A)
        Re = max(Re, 1e-12)

        self.reynolds_number.value = Re

        if Re <= self.reynolds_number_threshold.value:
            f = 4.0 * Po / Re
        else:
            f = (0.79 * np.log(Re) - 1.64) ** -2

        self.friction_factor.value = f