from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State
from fullflow.Utilities import create_SI_CEA_object

if TYPE_CHECKING:
    from fullflow.System import Network


class RocketCEAChokedNozzle(Component):
    """
    RocketCEA choked nozzle performance model.

    `RocketCEAChokedNozzle` computes choked nozzle mass flow and thrust using
    RocketCEA ideal performance outputs with user-supplied characteristic
    velocity and thrust coefficient efficiencies. The component uses chamber
    pressure, throat area, expansion ratio, ambient pressure, and mixture ratio
    to evaluate the nozzle operating point.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    fuel : str
        Fuel name passed to RocketCEA
    oxidizer : str
        Oxidizer name passed to RocketCEA
    chamber_pressure : State
        Chamber pressure
    throat_area : float
        Nozzle throat area
    expansion_ratio : float
        Nozzle expansion ratio
    ambient_pressure : State
        Ambient pressure
    mixture_ratio : State
        Oxidizer-to-fuel mixture ratio
    characterstic_velocity_efficiency : float
        Characteristic velocity efficiency
    thrust_coefficient_efficiency : float
        Thrust coefficient efficiency

    Outputs
    -------
    thrust : State, optional
        Nozzle thrust
    mass_flow : State, optional
        Nozzle mass flow rate

    Notes
    -----
    Ideal characteristic velocity is obtained from RocketCEA:

        ``cstar_ideal = CEA.get_Cstar(Pc, MR)``

    Ideal thrust coefficient is obtained from RocketCEA:

        ``Cf_ideal = CEA.get_PambCf(Pamb, Pc, MR, expansion_ratio)``

    Choked nozzle mass flow is evaluated from:

        ``mass_flow = Pc * At / (cstar_efficiency * cstar_ideal)``

    Nozzle thrust is evaluated from:

        ``thrust = Cf_efficiency * Cf_ideal * Pc * At``
    """
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