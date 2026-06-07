from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Boundary(Component):
    
    def __init__(self, 
                 name: str,
                 network: Network,
                 pressure: State,
                 temperature: State):
        self.setup()

class PressureBoundary(Component):

    def __init__(self, 
                 name: str,
                 network: Network,
                 pressure: State):
        self.setup()


class TemperatureBoundary(Component):

    def __init__(self, 
                 name: str,
                 network: Network,
                 temperature: State):
        self.setup()