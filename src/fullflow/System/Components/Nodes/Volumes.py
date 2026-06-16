from __future__ import annotations

from typing import TYPE_CHECKING
from numbers import Real

from fullflow.System import Component, Composition

if TYPE_CHECKING:
    from fullflow.System import Network, State



class SimpleVolume(Component):
    """
    Lumped fluid control volume with mass conservation only.

    `SimpleVolume` represents a zero-dimensional fluid node whose pressure is
    solved from a steady-state continuity equation. It is useful when the node
    pressure must be an iteration variable, but energy storage, enthalpy mixing,
    and temperature evolution are either handled elsewhere or intentionally
    neglected.

    This component is commonly used for simple junctions, chambers, tanks, or
    internal nodes where only mass balance is required.

    Sign Convention
    ---------------
    `mass_flow_in` is positive into the volume.

    `mass_flow_out` is positive out of the volume.

    Residuals
    ---------
    mass_balance : float
        Enforces steady-state mass conservation.

        ``mass_flow_in - mass_flow_out = 0``

    Iteration Variables
    -------------------
    pressure : State
        Volume pressure solved by the steady-state solver.

    Inputs
    ------
    pressure : State
        Volume pressure. This is the only iteration variable.
    volume : float
        Physical volume of the control volume. Currently stored for model
        completeness and future transient use; it is not used in the
        steady-state mass residual.
    density : State, optional
        Fluid density in the volume. Stored for downstream components or future
        extensions; not used by the mass-only residual.
    temperature : State, optional
        Fluid temperature in the volume. Stored for downstream components or
        future extensions; not used by the mass-only residual.
    enthalpy : State, optional
        Fluid specific enthalpy in the volume. Stored for downstream components
        or future extensions; not used by the mass-only residual.
    composition : Composition, optional
        Fluid composition inside the volume.
    composition_in : Composition, optional
        Incoming fluid composition.

    Flow Inputs
    -----------
    mass_flow_in : State, optional
        Total mass flow rate entering the volume.
    mass_flow_out : State, optional
        Total mass flow rate leaving the volume.

    Outputs
    -------
    pressure : State
        Solved volume pressure.
    mass_flow_out : State
        Outlet mass flow state when not externally assigned.
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
        composition: Composition | None = None,
        composition_in: Composition | None = None,
        mass_flow_in: State | None = None,
        mass_flow_out: State | None = None,
    ):
        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        return [self.pressure]

    @property
    def residuals(self) -> list[float]:
        return [
            self.mass_flow_in.value - self.mass_flow_out.value
        ]







class Volume(Component):
    """
    Lumped fluid control volume with optional steady-state energy balance.

    `Volume` represents a zero-dimensional fluid node. It can operate in either
    mass-only mode or mass-plus-energy mode.

    By default, the mode is selected automatically:

    * If no enthalpy inputs are provided, `Volume` behaves like
      `SimpleVolume` and enforces only mass conservation.
    * If `enthalpy`, `total_enthalpy_in`, or `total_enthalpy_out` is provided,
      `Volume` also enforces a steady-state energy balance.

    This allows the same component to be used as a simple pressure node or as an
    enthalpy-solving control volume.

    Sign Convention
    ---------------
    `mass_flow_in` is positive into the volume.

    `mass_flow_out` is positive out of the volume.

    `heat_rate` is positive when heat is added to the volume.

    Modes
    -----
    Mass-only mode
        Used when `energy_balance=False`, or when `energy_balance=None` and no
        enthalpy inputs are provided.

        Residual:

        ``mass_flow_in - mass_flow_out = 0``

        Iteration variable:

        ``pressure``

    Mass + energy mode
        Used when `energy_balance=True`, or when `energy_balance=None` and
        enthalpy inputs are provided.

        Residuals:

        ``mass_flow_in - mass_flow_out = 0``

        ``mass_flow_in * total_enthalpy_in - mass_flow_out * h_out + heat_rate = 0``

        where `h_out` is `total_enthalpy_out` when assigned, otherwise
        `enthalpy`.

        Iteration variables:

        ``pressure`` and ``enthalpy``

    Residuals
    ---------
    mass_balance : float
        Enforces steady-state mass conservation.

        ``mass_flow_in - mass_flow_out = 0``

    energy_balance : float
        Enforces steady-state energy conservation when energy balance mode is
        enabled.

        ``mass_flow_in * total_enthalpy_in - mass_flow_out * h_out + heat_rate = 0``

    Iteration Variables
    -------------------
    pressure : State
        Volume pressure solved by the steady-state solver.

    enthalpy : State
        Volume outlet/static enthalpy solved by the steady-state solver when
        energy balance mode is enabled.

    Inputs
    ------
    pressure : State
        Volume pressure.
    enthalpy : State or float, optional
        Volume enthalpy. In energy-balance mode, this becomes an iteration
        variable and is used as the outlet enthalpy if `total_enthalpy_out` is
        not assigned.
    volume : float, optional
        Physical volume of the control volume. Stored for model completeness and
        future transient use; it is not used directly in the current
        steady-state residuals.
    total_enthalpy_in : State or float, optional
        Total specific enthalpy entering the volume.
    total_enthalpy_out : State or float, optional
        Total specific enthalpy leaving the volume. If omitted or unassigned in
        energy-balance mode, `enthalpy` is used instead.
    heat_rate : State or float, optional
        Net heat transfer rate into the volume. Positive values add energy.
        Defaults to zero when omitted.
    temperature : State, optional
        Volume temperature. Stored for downstream components or future
        extensions; not used directly in the residuals.
    density : State, optional
        Volume density. Stored for downstream components or future extensions;
        not used directly in the residuals.
    internal_energy : State, optional
        Volume specific internal energy. Stored for downstream components or
        future extensions; not used directly in the residuals.
    composition : Composition, optional
        Fluid composition inside the volume.
    composition_in : Composition, optional
        Incoming fluid composition.
    mass_flow_in : State, optional
        Total mass flow rate entering the volume.
    mass_flow_out : State, optional
        Total mass flow rate leaving the volume.
    energy_balance : bool, optional
        Controls whether the energy equation is included.

        * `None`: automatically enable energy balance when enthalpy inputs are
          provided.
        * `False`: force mass-only mode.
        * `True`: force mass + energy mode.

    Outputs
    -------
    pressure : State
        Solved volume pressure.

    enthalpy : State
        Solved volume enthalpy when energy balance mode is enabled.

    mass_flow_out : State
        Outlet mass flow state when not externally assigned.

    Examples
    --------
    Mass-only pressure node::

        node = Volume(
            "Node",
            network,
            pressure=node_pressure,
            volume=1.0,
            mass_flow_in=inlet.mass_flow,
            mass_flow_out=outlet.mass_flow,
        )

    Energy-balancing volume::

        chamber = Volume(
            "Chamber",
            network,
            pressure=chamber_pressure,
            enthalpy=chamber_enthalpy,
            volume=0.01,
            total_enthalpy_in=inlet.total_enthalpy,
            mass_flow_in=inlet.mass_flow,
            mass_flow_out=nozzle.mass_flow,
            heat_rate=0.0,
        )
    """

    def __init__(
        self,
        name: str,
        network: Network,
        pressure: State,
        enthalpy: State | float | None = None,
        volume: float | None = None,
        total_enthalpy_in: State | float | None = None,
        total_enthalpy_out: State | float | None = None,
        heat_rate: State | float | None = None,
        temperature: State | None = None,
        density: State | None = None,
        internal_energy: State | None = None,
        composition: Composition | None = None,
        composition_in: Composition | None = None,
        mass_flow_in: State | None = None,
        mass_flow_out: State | None = None,
        energy_balance: bool | None = None,
    ):
        # Backward/ergonomic positional support:
        #
        # New mass-only positional form:
        #     Volume(name, network, pressure, volume)
        #
        # Old energy positional form still works:
        #     Volume(name, network, pressure, enthalpy, volume, total_enthalpy_in)
        #
        # If the fourth positional argument is numeric and volume was not
        # provided, treat it as volume, not enthalpy.
        if volume is None and isinstance(enthalpy, Real):
            volume = float(enthalpy)
            enthalpy = None

        if volume is None:
            raise ValueError(
                "Volume requires `volume`. Use either "
                "Volume(name, network, pressure, volume) for mass-only mode, "
                "or Volume(name, network, pressure, enthalpy, volume, ...) "
                "for energy-balance mode."
            )

        if energy_balance is None:
            energy_balance = (
                enthalpy is not None
                or total_enthalpy_in is not None
                or total_enthalpy_out is not None
            )

        if not energy_balance and heat_rate is not None:
            raise ValueError(
                "heat_rate was provided, but no enthalpy inputs were provided. "
                "Either provide enthalpy/total_enthalpy_in or set "
                "energy_balance=True."
            )

        self.setup()

    @property
    def iteration_variables(self) -> list[State]:
        if self.energy_balance:
            return [self.pressure, self.enthalpy]

        return [self.pressure]

    @property
    def residuals(self) -> list[float]:
        mass_balance = self.mass_flow_in.value - self.mass_flow_out.value

        if not self.energy_balance:
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