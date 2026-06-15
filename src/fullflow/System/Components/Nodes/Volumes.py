from __future__ import annotations

from typing import TYPE_CHECKING
from numbers import Real

from fullflow.System import Component, Composition

if TYPE_CHECKING:
    from fullflow.System import Network, State


class SimpleVolume(Component):
    """
    Lumped fluid volume with mass conservation only.

    Residual:
        mass_flow_in - mass_flow_out = 0

    Iteration variable:
        pressure
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
    Lumped fluid volume with optional steady-state energy balance.

    By default, Volume automatically chooses its mode:

    1. Mass-only mode
       Used when no enthalpy inputs are provided.

       Residual:
           mass_flow_in - mass_flow_out = 0

       Iteration variable:
           pressure

    2. Mass + energy mode
       Used when `enthalpy`, `total_enthalpy_in`, or `energy_balance=True`
       is provided.

       Residuals:
           mass_flow_in - mass_flow_out = 0

           mass_flow_in * total_enthalpy_in
           - mass_flow_out * total_enthalpy_out
           + heat_rate = 0

       Iteration variables:
           pressure
           enthalpy

    Notes
    -----
    This keeps old behavior when enthalpy arguments are supplied, but allows:

        Volume(
            "Node",
            network,
            pressure=node_pressure,
            volume=1.0,
            mass_flow_in=inlet.mass_flow,
            mass_flow_out=outlet.mass_flow,
        )

    to behave like SimpleVolume.

    Parameters
    ----------
    energy_balance : bool | None
        If None, automatically enables energy balance when enthalpy inputs are
        provided. If False, forces mass-only behavior. If True, forces the
        mass + energy formulation.
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