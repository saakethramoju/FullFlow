from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State



class Solid(Component):
    """
    Lumped solid thermal node.

    Residual
    --------
        q_net = 0

    Optional Biot Number
    --------------------
        Bi = h * Lc / k

    where Bi << 1 indicates that the lumped-temperature assumption is
    reasonable. A common rule of thumb is Bi < 0.1.

    Parameters
    ----------
    temperature : State
        Lumped solid temperature [K].

    mass : float, optional
        Solid mass [kg].

    specific_heat : State, optional
        Solid specific heat [J/kg-K].

    heat_rate : State | float
        Net heat rate into the solid node [W].

    characteristic_length : State | float, optional
        Solid characteristic length Lc [m], usually volume/surface area.

    thermal_conductivity : State | float, optional
        Solid thermal conductivity [W/m-K].

    convection_coefficient : State | float, optional
        Representative external convection coefficient [W/m²-K].

    biot_number : State, optional
        Output Biot number [-].
    """

    def __init__(
        self,
        name: str,
        network: Network,
        temperature: State,
        mass: float | None = None,
        specific_heat: State | None = None,
        characteristic_length: State | float | None = None,
        thermal_conductivity: State | float | None = None,
        convection_coefficient: State | float | None = None,
        biot_number: State | None = None,
        heat_rate: State | float = 0.0,
    ):
        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.temperature]

    @property
    def residuals(self) -> list[float]:
        return [self.heat_rate.value]

    def evaluate_states(self):
        if (
            self.characteristic_length is None
            or self.thermal_conductivity is None
            or self.convection_coefficient is None
        ):
            return

        Lc = self.characteristic_length.value
        k = self.thermal_conductivity.value
        h = self.convection_coefficient.value

        if Lc <= 0.0:
            raise ValueError(
                f"{self.name}: characteristic_length must be greater than zero. Got {Lc}."
            )

        self.biot_number.value = h * Lc / k