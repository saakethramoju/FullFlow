from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class ReferenceAdjustment(Component):
    """
    Reference-basis adjustment for an energy-like state.

    `ReferenceAdjustment` shifts an input value from one reference basis to
    another by subtracting the old reference value and adding the new reference
    value. This is useful for enthalpy, internal energy, entropy, or other
    reference-dependent quantities when different property backends use
    different reference states.

    The old and new reference values should correspond to the same reference
    condition.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    input_value : State or float
        Value on the original reference basis
    old_reference_value : State or float
        Reference value on the original basis
    new_reference_value : State or float
        Reference value on the target basis

    Outputs
    -------
    output_value : State, optional
        Value shifted to the target reference basis

    Notes
    -----
    The adjusted value is evaluated from:

        ``output_value = input_value - old_reference_value + new_reference_value``

    The adjusted value can also be accessed through the `adjusted_value`
    property.
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