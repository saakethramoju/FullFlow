from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Conduction(Component):
    """
    One-dimensional conduction heat transfer between two temperature nodes.

    `Conduction` computes conductive heat transfer between two thermal nodes
    using a one-dimensional Fourier-law resistance. Positive heat rate means
    heat is added to `temperature1` from `temperature2`.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    temperature1 : State
        Receiving-side temperature
    temperature2 : State
        Source-side temperature
    thermal_conductivity : State
        Thermal conductivity
    length : float
        Conduction length
    conductive_area : float
        Conductive area

    Outputs
    -------
    heat_rate : State, optional
        Conductive heat transfer rate

    Notes
    -----
    Conductive heat transfer is evaluated from:

        ``heat_rate = thermal_conductivity * conductive_area
        / length * (temperature2 - temperature1)``
    """

    def __init__(
        self,
        name: str,
        network: Network,
        temperature1: State,
        temperature2: State,
        thermal_conductivity: State,
        length: float,
        conductive_area: float,
        heat_rate: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        k = self.thermal_conductivity.value
        A = self.conductive_area.value
        L = self.length.value
        T1 = self.temperature1.value
        T2 = self.temperature2.value

        if k <= 0.0:
            raise ValueError(f"{self.name}: thermal_conductivity must be positive. Got {k}.")

        if A <= 0.0:
            raise ValueError(f"{self.name}: conductive_area must be positive. Got {A}.")

        if L <= 0.0:
            raise ValueError(f"{self.name}: length must be positive. Got {L}.")

        self.heat_rate.value = k * A / L * (T2 - T1)



class Radiation(Component):
    """
    Diffuse-gray radiation exchange between two temperature nodes.

    `Radiation` computes radiative heat transfer between two diffuse-gray
    surfaces using emissivities, radiating areas, and a view factor. Positive
    heat rate indicates net radiative heat transfer from `temperature2` to
    `temperature1`.

    This component can be used for surface-to-surface radiation or vacuum
    jacketed tube radiation.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    temperature1 : State
        Receiving-side surface temperature
    temperature2 : State
        Source-side surface temperature
    emissivity1 : float
        Receiving-side surface emissivity
    emissivity2 : float
        Source-side surface emissivity
    radiative_area1 : float
        Receiving-side radiative area
    radiative_area2 : float, optional
        Source-side radiative area
    view_factor12 : float, optional
        View factor from surface 1 to surface 2

    Outputs
    -------
    heat_rate : State, optional
        Radiative heat transfer rate

    Notes
    -----
    The radiation denominator is evaluated from:

        ``denominator = (1 - emissivity1) / (emissivity1 * radiative_area1)
        + 1 / (radiative_area1 * view_factor12)
        + (1 - emissivity2) / (emissivity2 * radiative_area2)``

    Radiative heat transfer is evaluated from:

        ``heat_rate = sigma * (temperature2^4 - temperature1^4)
        / denominator``
    """
    SIGMA = 5.670374419e-8  # W/m^2-K^4

    def __init__(
        self,
        name: str,
        network: Network,
        temperature1: State,
        temperature2: State,
        emissivity1: float,
        emissivity2: float,
        radiative_area1: float,
        radiative_area2: float | None = None,
        view_factor12: float = 1.0,
        heat_rate: State | None = None,
    ):
        self.setup()

        if radiative_area2 is None:
            self.radiative_area2.value = self.radiative_area1.value

    def evaluate_states(self):
        T1 = self.temperature1.value
        T2 = self.temperature2.value

        eps1 = self.emissivity1.value
        eps2 = self.emissivity2.value

        A1 = self.radiative_area1.value
        A2 = self.radiative_area2.value

        F12 = self.view_factor12.value

        if eps1 <= 0.0 or eps1 > 1.0:
            raise ValueError(
                f"{self.name}: emissivity1 must be in (0, 1]. Got {eps1}."
            )

        if eps2 <= 0.0 or eps2 > 1.0:
            raise ValueError(
                f"{self.name}: emissivity2 must be in (0, 1]. Got {eps2}."
            )

        if A1 <= 0.0:
            raise ValueError(
                f"{self.name}: radiative_area1 must be greater than zero. Got {A1}."
            )

        if A2 <= 0.0:
            raise ValueError(
                f"{self.name}: radiative_area2 must be greater than zero. Got {A2}."
            )

        if F12 <= 0.0 or F12 > 1.0:
            raise ValueError(
                f"{self.name}: view_factor12 must be in (0, 1]. Got {F12}."
            )

        denominator = (
            (1.0 - eps1) / (eps1 * A1)
            + 1.0 / (A1 * F12)
            + (1.0 - eps2) / (eps2 * A2)
        )

        if denominator <= 0.0:
            raise ValueError(
                f"{self.name}: invalid radiation denominator ({denominator})."
            )

        self.heat_rate.value = (
            self.SIGMA
            * (T2**4 - T1**4)
            / denominator
        )





class AmbientRadiation(Component):
    """
    Radiation exchange between a surface and an ambient enclosure.

    `AmbientRadiation` computes radiative heat transfer between a solid surface
    and a surrounding ambient enclosure. Positive heat rate indicates net
    radiative heat transfer to the solid surface from the ambient surroundings.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    solid_temperature : State
        Solid surface temperature
    ambient_temperature : State or float
        Ambient enclosure temperature
    emissivity : State or float
        Solid surface emissivity
    radiative_area : State or float
        Radiative area
    ambient_emissivity : State or float, optional
        Ambient enclosure emissivity

    Outputs
    -------
    heat_rate : State, optional
        Radiative heat transfer rate

    Notes
    -----
    The radiation denominator is evaluated from:

        ``denominator = 1 / emissivity + 1 / ambient_emissivity - 1``

    Radiative heat transfer is evaluated from:

        ``heat_rate = sigma * radiative_area
        * (ambient_temperature^4 - solid_temperature^4)
        / denominator``
    """
    SIGMA = 5.670374419e-8  # W/m^2-K^4

    def __init__(
        self,
        name: str,
        network: Network,
        solid_temperature: State,
        ambient_temperature: State | float,
        emissivity: State | float,
        radiative_area: State | float,
        ambient_emissivity: State | float = 1.0,
        heat_rate: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        Ts = self.solid_temperature.value
        Tamb = self.ambient_temperature.value

        eps_s = self.emissivity.value
        eps_amb = self.ambient_emissivity.value

        A = self.radiative_area.value

        if eps_s <= 0.0 or eps_s > 1.0:
            raise ValueError(
                f"{self.name}: emissivity must be in (0, 1]. Got {eps_s}."
            )

        if eps_amb <= 0.0 or eps_amb > 1.0:
            raise ValueError(
                f"{self.name}: ambient_emissivity must be in (0, 1]. Got {eps_amb}."
            )

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: radiative_area must be greater than zero. Got {A}."
            )

        denominator = (
            1.0 / eps_s
            + 1.0 / eps_amb
            - 1.0
        )

        if denominator <= 0.0:
            raise ValueError(
                f"{self.name}: invalid radiation denominator ({denominator})."
            )

        self.heat_rate.value = (
            self.SIGMA
            * A
            * (Tamb**4 - Ts**4)
            / denominator
        )







class Convection(Component):
    """
    Convective heat transfer between a surface and a fluid.

    `Convection` computes heat transfer between a surface and a surrounding
    fluid using a prescribed convection coefficient. Positive heat rate means
    heat is added to the surface from the fluid.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    surface_temperature : State
        Surface temperature
    fluid_temperature : State or float
        Fluid temperature
    convective_area : State or float
        Convective area
    convection_coefficient : State or float
        Convective heat transfer coefficient

    Outputs
    -------
    heat_rate : State, optional
        Convective heat transfer rate

    Notes
    -----
    Convective heat transfer is evaluated from:

        ``heat_rate = convection_coefficient * convective_area
        * (fluid_temperature - surface_temperature)``
    """
    def __init__(
        self,
        name: str,
        network: Network,
        surface_temperature: State,
        fluid_temperature: State | float,
        convective_area: State | float,
        convection_coefficient: State | float,
        heat_rate: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        Ts = self.surface_temperature.value
        Tf = self.fluid_temperature.value
        h = self.convection_coefficient.value
        A = self.convective_area.value

        if h <= 0.0:
            raise ValueError(f"{self.name}: convection_coefficient must be positive. Got {h}.")

        if A <= 0.0:
            raise ValueError(f"{self.name}: convective_area must be positive. Got {A}.")

        self.heat_rate.value = h * A * (Tf - Ts)






class TemperatureRecoveryFactor(Component):
    """
    Compressible boundary-layer temperature recovery factor.

    `TemperatureRecoveryFactor` computes the recovery factor used to estimate
    adiabatic wall temperature in compressible boundary-layer heat transfer. If
    no Prandtl number is provided, the recovery factor defaults to one.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    prandtl_number : State, optional
        Prandtl number
    turbulent : bool, optional
        Whether to use the turbulent boundary-layer exponent

    Outputs
    -------
    recovery_factor : State, optional
        Temperature recovery factor

    Notes
    -----
    The adiabatic wall temperature relation is:

        ``T_aw = T + r * (T0 - T)``

    For turbulent boundary layers, the recovery factor is evaluated from:

        ``r = Pr^(1/3)``

    For laminar boundary layers, the recovery factor is evaluated from:

        ``r = Pr^(1/2)``

    If no Prandtl number is provided, the recovery factor is:

        ``r = 1``
    """
    def __init__(
        self,
        name: str,
        network: Network,
        prandtl_number: State | None = None,
        recovery_factor: State | None = None,
        turbulent: bool = True,
    ):
        self.setup()

    def evaluate_states(self):
        if self.prandtl_number is None or not self.prandtl_number.is_assigned:
            self.recovery_factor.value = 1.0
            return

        Pr = self.prandtl_number.value

        if Pr <= 0.0:
            raise ValueError(
                f"{self.name}: prandtl_number must be greater than zero. Got {Pr}."
            )

        if self.turbulent:
            self.recovery_factor.value = Pr ** (1.0 / 3.0)
        else:
            self.recovery_factor.value = Pr ** 0.5







class AdiabaticWallTemperature(Component):
    """
    Adiabatic wall temperature for compressible flow.

    `AdiabaticWallTemperature` computes the temperature an insulated wall would
    attain when exposed to a compressible flow, using total temperature, static
    temperature, and a recovery factor.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    total_temperature : State
        Total temperature
    static_temperature : State
        Static temperature
    recovery_factor : State
        Temperature recovery factor

    Outputs
    -------
    adiabatic_wall_temperature : State, optional
        Adiabatic wall temperature

    Notes
    -----
    Adiabatic wall temperature is evaluated from:

        ``T_aw = T + r * (T0 - T)``

    where `T_aw` is adiabatic wall temperature, `T` is static temperature, `T0`
    is total temperature, and `r` is the recovery factor.
    """
    def __init__(
        self,
        name: str,
        network: Network,
        total_temperature: State,
        static_temperature: State,
        recovery_factor: State,
        adiabatic_wall_temperature: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        T0 = self.total_temperature.value
        T = self.static_temperature.value
        r = self.recovery_factor.value

        self.adiabatic_wall_temperature.value = (
            T + r * (T0 - T)
        )







class EckertReferenceTemperature(Component):
    """
    Calculates Eckert's reference (film) temperature.

    The reference temperature is used to evaluate fluid properties
    within a compressible turbulent boundary layer:

        T_f = 0.5 T_w + 0.28 T + 0.22 T_aw

    where

        T_f  = Eckert reference temperature
        T_w  = wall temperature
        T    = static fluid temperature
        T_aw = adiabatic wall temperature
    """

    def __init__(
        self,
        name: str,
        network: Network,
        wall_temperature: State,
        static_temperature: State,
        adiabatic_wall_temperature: State,
        reference_temperature: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        Tw = self.wall_temperature.value
        T = self.static_temperature.value
        Taw = self.adiabatic_wall_temperature.value

        self.reference_temperature.value = (
            0.5 * Tw
            + 0.28 * T
            + 0.22 * Taw
        )