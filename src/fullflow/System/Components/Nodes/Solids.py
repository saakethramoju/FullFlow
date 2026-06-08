from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Solid(Component):
    """
    Lumped solid thermal node.

    `Solid` represents a lumped-capacitance thermal mass whose temperature is
    solved from a steady-state energy balance. The component is intended for
    conjugate heat transfer networks where conduction, convection, radiation,
    and other thermal components contribute heat to a common solid node.

    Positive heat rates add energy to the solid. Negative heat rates remove
    energy from the solid.

    Residuals
    ---------
    energy_balance : float
        Enforces steady-state thermal equilibrium:

        `heat_rate = 0`

        The heat rate is typically formed by summing all heat transfer
        mechanisms connected to the node.

    Relations
    ---------
    Optional Biot number calculation:

    `Bi = h * Lc / k`

    where:

    * `Bi` is the Biot number [-]
    * `h` is the convection coefficient [W/m²-K]
    * `Lc` is the characteristic length [m]
    * `k` is the thermal conductivity [W/m-K]

    As a general guideline, `Bi < 0.1` indicates that the lumped-temperature
    assumption is likely reasonable.

    Iteration Variables
    -------------------
    temperature : State
        Solid temperature.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    temperature : State
        Solid temperature [K].
    mass : float, optional
        Solid mass [kg].
    specific_heat : State, optional
        Solid specific heat capacity [J/kg-K].
    characteristic_length : State or float, optional
        Characteristic length used for Biot number evaluation [m].
    thermal_conductivity : State or float, optional
        Solid thermal conductivity used for Biot number evaluation [W/m-K].
    convection_coefficient : State or float, optional
        Representative convection coefficient used for Biot number evaluation
        [W/m²-K].
    biot_number : State, optional
        Output Biot number [-].
    heat_rate : State or float, optional
        Net heat rate into the solid node [W]. Positive values add heat to the
        solid. Defaults to 0.
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
        # Store whether the user actually requested Biot-number evaluation.
        self._has_biot_inputs = (
            characteristic_length is not None
            and thermal_conductivity is not None
            and convection_coefficient is not None
        )

        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.temperature]

    @property
    def residuals(self) -> list[float]:
        return [self.heat_rate.value]

    def evaluate_states(self):
        if not self._has_biot_inputs:
            return

        Lc = self.characteristic_length.value
        k = self.thermal_conductivity.value
        h = self.convection_coefficient.value

        if Lc <= 0.0:
            raise ValueError(
                f"{self.name}: characteristic_length must be greater than zero. Got {Lc}."
            )

        if k <= 0.0:
            raise ValueError(
                f"{self.name}: thermal_conductivity must be greater than zero. Got {k}."
            )

        if h < 0.0:
            raise ValueError(
                f"{self.name}: convection_coefficient must be nonnegative. Got {h}."
            )

        self.biot_number.value = h * Lc / k