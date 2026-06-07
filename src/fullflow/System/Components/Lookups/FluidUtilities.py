from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class ReferenceAdjustment(Component):
    """
    Shift an energy-like State from one reference basis to another.

    Useful for enthalpy/internal_energy reference corrections.
    Reference values should be at the same temperature (and pressure, ideally).

    Formula
    -------
    adjusted_value = input_value - old_reference_value + new_reference_value

    Example
    -------
    h_coolprop_basis = State()

    ReferenceAdjustment(
        "Shift h to CoolProp basis",
        network,
        input_value=ideal_gas_h,
        output_value=h_coolprop_basis,
        old_reference_value=ideal_gas_h_ref,
        new_reference_value=coolprop_h_ref,
    )
    """

    def __init__(
        self,
        name: str,
        network: Network,
        input_value: State | float,
        old_reference_value: State | float,
        new_reference_value: State | float,
        output_value: State | None = None,
    ):
        self.setup()

        if not isinstance(self.output_value, State):
            raise TypeError(
                "output_value must be a State or None. "
                "If None is passed, setup() should convert it to State()."
            )

    def pre_evaluation(self):
        self.evaluate_states()

    def evaluate_states(self) -> None:
        self.output_value.value = (
            self.input_value.value
            - self.old_reference_value.value
            + self.new_reference_value.value
        )

    @property
    def adjusted_value(self) -> State:
        return self.output_value