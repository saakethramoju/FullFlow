from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class Sensor(Component):
    """Named virtual sensor reading for HDF5 export.

    In this first version, a sensor is just a named view of a State, similar to
    ``Network.track(...)``, but exported under the run's ``/sensors`` group.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        reading: State,
        # variable: State | None = None,
        # data=None,
    ) -> None:
        self.setup()

    @property
    def value(self):
        return self.reading.value
