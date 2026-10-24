from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Conduction(Component):
    """One-dimensional conduction heat transfer between two temperature nodes."""

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
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        k = self.thermal_conductivity.value
        A = self.conductive_area.value
        L = self.length.value
        T1 = self.temperature1.value
        T2 = self.temperature2.value
        self.heat_rate.value = k * A / L * (T2 - T1)



class Radiation(Component):
    """Diffuse-gray radiation exchange between two surfaces."""
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
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

        if radiative_area2 is None:
            self.radiative_area2.value = self.radiative_area1.value

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        T1 = self.temperature1.value
        T2 = self.temperature2.value

        eps1 = self.emissivity1.value
        eps2 = self.emissivity2.value

        A1 = self.radiative_area1.value
        A2 = self.radiative_area2.value

        F12 = self.view_factor12.value

        denominator = (1.0 - eps1) / (eps1 * A1) + 1.0 / (A1 * F12) + (1.0 - eps2) / (eps2 * A2)

        self.heat_rate.value = self.SIGMA * (T2**4 - T1**4) / denominator





class AmbientRadiation(Component):
    """Radiation exchange between a surface and a large ambient enclosure."""
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
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        Ts = self.solid_temperature.value
        Tamb = self.ambient_temperature.value

        eps_s = self.emissivity.value
        eps_amb = self.ambient_emissivity.value

        A = self.radiative_area.value

        denominator = 1.0 / eps_s + 1.0 / eps_amb - 1.0

        self.heat_rate.value = self.SIGMA * A * (Tamb**4 - Ts**4) / denominator







class Convection(Component):
    """Convective heat transfer between a wall/surface state and a fluid state."""
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
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        Ts = self.surface_temperature.value
        Tf = self.fluid_temperature.value
        h = self.convection_coefficient.value
        A = self.convective_area.value

        self.heat_rate.value = h * A * (Tf - Ts)






class TemperatureRecoveryFactor(Component):
    """Boundary-layer recovery factor from Prandtl number."""
    def __init__(
        self,
        name: str,
        network: Network,
        prandtl_number: State | None = None,
        recovery_factor: State | None = None,
        turbulent: bool = True,
    ):
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        if self.prandtl_number is None or not self.prandtl_number.is_assigned:
            self.recovery_factor.value = 1.0
            return

        Pr = self.prandtl_number.value

        if self.turbulent.value:
            self.recovery_factor.value = Pr ** (1.0 / 3.0)
        else:
            self.recovery_factor.value = Pr ** 0.5







class AdiabaticWallTemperature(Component):
    """Adiabatic wall temperature for compressible boundary-layer heat transfer."""
    def __init__(
        self,
        name: str,
        network: Network,
        total_temperature: State,
        static_temperature: State,
        recovery_factor: State,
        adiabatic_wall_temperature: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        T0 = self.total_temperature.value
        T = self.static_temperature.value
        r = self.recovery_factor.value

        self.adiabatic_wall_temperature.value = T + r * (T0 - T)







class EckertReferenceTemperature(Component):
    """Eckert reference temperature used for evaluating gas-side transport properties."""

    def __init__(
        self,
        name: str,
        network: Network,
        wall_temperature: State,
        static_temperature: State,
        adiabatic_wall_temperature: State,
        reference_temperature: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state."""
        Tw = self.wall_temperature.value
        T = self.static_temperature.value
        Taw = self.adiabatic_wall_temperature.value

        self.reference_temperature.value = 0.5 * Tw + 0.28 * T + 0.22 * Taw
