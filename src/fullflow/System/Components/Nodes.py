from __future__ import annotations

from typing import TYPE_CHECKING
from numbers import Real

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
        Enforces steady-state thermal equilibrium.

        ``heat_rate = 0``

        The heat rate is typically formed by summing all heat transfer
        mechanisms connected to the node.

    Relations
    ---------
    biot_number : State
        Computes the Biot number used to assess the validity of the
        lumped-capacitance assumption.

        ``Bi = h * Lc / k``

        where:

        * `Bi` is the Biot number
        * `h` is the convection coefficient
        * `Lc` is the characteristic length
        * `k` is the thermal conductivity

        As a general guideline, `Bi < 0.1` indicates that the
        lumped-temperature assumption is likely reasonable.

    Iteration Variables
    -------------------
    temperature : State
        Solid temperature

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    temperature : State
        Solid temperature
    mass : float, optional
        Solid mass
    specific_heat : State, optional
        Solid specific heat capacity
    characteristic_length : State or float, optional
        Characteristic length used for Biot number evaluation
    thermal_conductivity : State or float, optional
        Solid thermal conductivity used for Biot number evaluation
    convection_coefficient : State or float, optional
        Representative convection coefficient used for Biot number evaluation
    biot_number : State, optional
        Output Biot number
    heat_rate : State or float, optional
        Net heat rate into the solid node. Positive values add heat to the
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








class Volume(Component):
    """
    Lumped steady-state fluid control volume.

    `Volume` enforces mass conservation. If `enthalpy` is provided, it also
    enforces steady-state energy conservation.

    Modes
    -----
    Mass-only mode
        Used when `enthalpy` is not provided.

        Residual:

            mass_flow_in - mass_flow_out = 0

        Iteration variable:

            pressure

    Mass + energy mode
        Used when `enthalpy` is provided.

        Residuals:

            mass_flow_in - mass_flow_out = 0

            mass_flow_in * total_enthalpy_in
            - mass_flow_out * h_out
            + heat_rate = 0

        where `h_out` is `total_enthalpy_out` if assigned, otherwise `enthalpy`.

        Iteration variables:

            pressure
            enthalpy

    Sign Convention
    ---------------
    `mass_flow_in` is positive into the volume.

    `mass_flow_out` is positive out of the volume.

    `heat_rate` is positive into the volume.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    pressure : State
        Volume pressure.
    volume : State or float
        Physical control volume. Required.
    enthalpy : State or float, optional
        Volume/static outlet enthalpy. Providing this turns on the energy
        residual.
    total_enthalpy_in : State or float, optional
        Total specific enthalpy entering the volume. Required when `enthalpy`
        is provided.
    total_enthalpy_out : State or float, optional
        Total specific enthalpy leaving the volume. If omitted, `enthalpy` is
        used as the outlet enthalpy.
    heat_rate : State or float, optional
        Net heat transfer rate into the volume. Defaults to zero if omitted.
    temperature : State, optional
        Stored volume temperature.
    density : State, optional
        Stored volume density.
    internal_energy : State, optional
        Stored volume internal energy.
    mass_flow_in : State or float, optional
        Mass flow entering the volume.
    mass_flow_out : State or float, optional
        Mass flow leaving the volume.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        volume: State | float,
        enthalpy: State | float | None = None,
        total_enthalpy_in: State | float | None = None,
        total_enthalpy_out: State | float | None = None,
        heat_rate: State | float | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        mass_flow_in: State | float | None = None,
        mass_flow_out: State | float | None = None,
    ):

        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        if (self.enthalpy.is_assigned and self.total_enthalpy_in.is_assigned):
            return [self.pressure, self.enthalpy]

        return [self.pressure]

    @property
    def residuals(self) -> list[float]:
        mass_balance = self.mass_flow_in.value - self.mass_flow_out.value

        if not (self.enthalpy.is_assigned and self.total_enthalpy_in.is_assigned):
            return [mass_balance]

        qdot = self.heat_rate.value if self.heat_rate.is_assigned else 0.0

        h_out = (
            self.total_enthalpy_out.value
            if self.total_enthalpy_out.is_assigned
            else self.enthalpy.value
        )

        energy_balance = (
            self.mass_flow_in.value * self.total_enthalpy_in.value
            - self.mass_flow_out.value * h_out
            + qdot
        )

        return [
            mass_balance,
            energy_balance,
        ]