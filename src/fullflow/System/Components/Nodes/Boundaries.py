from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Boundary(Component):
    """
    Fixed pressure and temperature boundary condition.

    `Boundary` represents a thermodynamic boundary where both pressure and
    temperature are prescribed. It is commonly used as a source, sink, ambient
    condition, reservoir, or external system interface.

    This component introduces no iteration variables and contributes no
    residual equations.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    pressure : State
        Boundary pressure [Pa].
    temperature : State
        Boundary temperature [K].
    """
    def __init__(self, 
                 name: str,
                 network: Network,
                 pressure: State,
                 temperature: State):
        self.setup()

class PressureBoundary(Component):
    """
    Fixed pressure boundary condition.

    `PressureBoundary` prescribes a pressure while allowing other thermodynamic
    properties to be determined elsewhere in the network.

    This component introduces no iteration variables and contributes no
    residual equations.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    pressure : State
        Boundary pressure [Pa].
    """
    def __init__(self, 
                 name: str,
                 network: Network,
                 pressure: State):
        self.setup()


class TemperatureBoundary(Component):
    """
    Fixed temperature boundary condition.

    `TemperatureBoundary` prescribes a temperature while allowing other
    thermodynamic properties to be determined elsewhere in the network.

    This component introduces no iteration variables and contributes no
    residual equations.

    Parameters
    ----------
    name : str
        Component name.
    network : Network
        Network that owns this component.
    temperature : State
        Boundary temperature [K].
    """
    def __init__(self, 
                 name: str,
                 network: Network,
                 temperature: State):
        self.setup()