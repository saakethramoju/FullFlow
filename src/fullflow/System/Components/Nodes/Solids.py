from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Solid(Component):
    """
    Lumped solid thermal node.

    Positive heat_rate means net heat added to the solid node [W].

    Optional Biot number:
        Bi = h * Lc / k
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