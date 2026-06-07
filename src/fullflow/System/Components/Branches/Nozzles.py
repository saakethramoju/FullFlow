from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State
from fullflow.Utilities import create_SI_CEA_object

if TYPE_CHECKING:
    from fullflow.System import Network


class RocketCEAChokedNozzle(Component):

    def __init__(self, 
                 name: str,
                 network: Network,
                 fuel: str,
                 oxidizer: str,
                 chamber_pressure: State,
                 throat_area: float,
                 expansion_ratio: float,
                 ambient_pressure: State,
                 mixture_ratio: State,
                 characterstic_velocity_efficiency: float,
                 thrust_coefficient_efficiency: float,
                 thrust: State | None = None,
                 mass_flow: State | None = None):
        self.setup()
        self._cea_obj = create_SI_CEA_object(self.fuel, self.oxidizer)


    def evaluate_states(self) -> None:
        Pc = self.chamber_pressure.value
        MR = self.mixture_ratio.value
        At = self.throat_area.value

        cstar_ideal = self._cea_obj.get_Cstar(Pc, MR)
        _, Cf_ideal, _ = self._cea_obj.get_PambCf(self.ambient_pressure.value, Pc, MR, self.expansion_ratio.value)

        self.mass_flow.value = Pc * At / (self.characterstic_velocity_efficiency.value * cstar_ideal)
        self.thrust.value = self.thrust_coefficient_efficiency.value * Cf_ideal * Pc * At