from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, Composition

if TYPE_CHECKING:
    from fullflow.System import Network, State



class SimpleVolume(Component):
    """
    Lumped fluid volume with mass conservation only.

    `SimpleVolume` represents an internal fluid control volume whose pressure is
    solved from steady-state continuity. It is useful when the node pressure must
    be an iteration variable, but the node energy equation is handled elsewhere
    or intentionally neglected.

    Residuals
    ---------
    mass_balance : float
        Enforces steady-state mass conservation:

        `mass_flow_in - mass_flow_out = 0`

    Iteration Variables
    -------------------
    pressure : State
        Internal volume pressure.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    pressure : State
        Internal volume pressure [Pa].
    volume : float
        Volume of the control volume [m^3].
    density : State, optional
        Fluid density in the volume [kg/m^3].
    temperature : State, optional
        Fluid temperature in the volume [K].
    enthalpy : State, optional
        Fluid specific enthalpy in the volume [J/kg].
    composition : Composition, optional
        Current fluid composition in the volume.
    composition_in : Composition, optional
        Incoming fluid composition.
    mass_flow_in : State, optional
        Total mass flow rate entering the volume [kg/s].
    mass_flow_out : State, optional
        Total mass flow rate leaving the volume [kg/s].
    """

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        volume: float,
        density: State | None = None,
        temperature: State | None = None,
        enthalpy: State | None = None,
        composition: Composition = Composition(),
        composition_in: Composition = Composition(),
        mass_flow_in: State | None = None,
        mass_flow_out: State | None = None,
    ):
        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.pressure]

    @property
    def residuals(self) -> list[float]:
        return [self.mass_flow_in.value - self.mass_flow_out.value]






class Volume(Component):
    """
    Lumped fluid volume with steady-state mass and energy conservation.

    `Volume` represents an internal fluid control volume whose pressure and
    specific enthalpy are solved from continuity and energy balance. It is useful
    for compressible, heated, cooled, or mixed-flow nodes where both pressure and
    thermodynamic state must be solved.

    Residuals
    ---------
    mass_balance : float
        Enforces steady-state mass conservation:

        `mass_flow_in - mass_flow_out = 0`

    energy_balance : float
        Enforces steady-state flow energy conservation:

        `mass_flow_in * total_enthalpy_in
        - mass_flow_out * total_enthalpy_out
        + heat_rate = 0`

        If `total_enthalpy_out` is not assigned, the volume enthalpy is used as
        the outlet total enthalpy. If `heat_rate` is not assigned, it is treated
        as zero.

    Iteration Variables
    -------------------
    pressure : State
        Internal volume pressure.
    enthalpy : State
        Internal volume specific enthalpy.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    pressure : State
        Internal volume pressure [Pa].
    enthalpy : State
        Internal volume specific enthalpy [J/kg].
    volume : float
        Volume of the control volume [m^3].
    total_enthalpy_in : State
        Total specific enthalpy entering the volume [J/kg].
    total_enthalpy_out : State, optional
        Total specific enthalpy leaving the volume [J/kg]. If omitted, `enthalpy`
        is used.
    heat_rate : State or float, optional
        Net heat rate into the volume [W]. Positive values add energy.
    temperature : State, optional
        Fluid temperature in the volume [K].
    density : State, optional
        Fluid density in the volume [kg/m^3].
    internal_energy : State, optional
        Fluid specific internal energy in the volume [J/kg].
    mass_flow_in : State, optional
        Total mass flow rate entering the volume [kg/s].
    mass_flow_out : State, optional
        Total mass flow rate leaving the volume [kg/s].
    """
    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        enthalpy: State,
        volume: float,
        total_enthalpy_in: State,
        total_enthalpy_out: State | None = None,
        heat_rate: State | float | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        mass_flow_in: State | None = None,
        mass_flow_out: State | None = None,
    ):
        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.pressure, self.enthalpy]

    @property
    def residuals(self) -> list[float]:
        qdot = self.heat_rate.value if self.heat_rate.is_assigned else 0.0

        h_out = (
            self.total_enthalpy_out.value
            if self.total_enthalpy_out.is_assigned
            else self.enthalpy.value
        )

        return [
            self.mass_flow_in.value - self.mass_flow_out.value,
            (
                self.mass_flow_in.value * self.total_enthalpy_in.value
                - self.mass_flow_out.value * h_out
                + qdot
            ),
        ]