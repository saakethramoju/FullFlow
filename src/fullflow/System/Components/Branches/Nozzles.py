from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component, State
from ._flow_math import sqrt_or_nan

if TYPE_CHECKING:
    from fullflow.System import Network


 

class FrozenFlowRocketNozzle(Component):
    """
    Frozen-flow nozzle mass flow and thrust model.

    `FrozenFlowRocketNozzle` computes nozzle mass flow and thrust using
    one-dimensional, isentropic, frozen-composition perfect-gas relations.
    The component automatically switches between unchoked and choked flow
    based on the ratio of back pressure to upstream total pressure.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    upstream_total_pressure : State
        Upstream total pressure
    upstream_total_temperature : State
        Upstream total temperature
    back_pressure : State
        Nozzle back pressure
    throat_area : float
        Nozzle throat area
    specific_heat_ratio : State
        Frozen specific heat ratio
    specific_gas_constant : State | float
        Frozen specific gas constant
    characteristic_velocity_efficiency : State | float
        Characteristic velocity efficiency
    thrust_coefficient_efficiency : State | float
        Thrust coefficient efficiency

    Outputs
    -------
    characteristic_velocity_ideal : State, optional
        Ideal characteristic velocity for choked flow
    thrust_coefficient_ideal : State, optional
        Ideal thrust coefficient
    characteristic_velocity : State, optional
        Efficiency-corrected characteristic velocity for choked flow
    thrust_coefficient : State, optional
        Efficiency-corrected thrust coefficient
    thrust : State, optional
        Nozzle thrust
    mass_flow : State, optional
        Nozzle mass flow rate

    Notes
    -----
    Critical pressure ratio is evaluated from:

        ``critical_pressure_ratio = (2 / (gamma + 1))**(gamma / (gamma - 1))``

    Unchoked Mach number is evaluated from:

        ``M = sqrt((2 / (gamma - 1)) * ((Pb / P0)**(-(gamma - 1) / gamma) - 1))``

    Unchoked mass flow is evaluated from:

        ``mass_flow = A * P0 * sqrt(gamma / (R * T0)) * M * (1 + 0.5 * (gamma - 1) * M**2)**(-(gamma + 1) / (2 * (gamma - 1)))``

    Choked ideal mass flow is evaluated from:

        ``mass_flow_ideal = A * P0 * sqrt(gamma / (R * T0)) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))``

    Choked ideal characteristic velocity is evaluated from:

        ``cstar_ideal = P0 * A / mass_flow_ideal``

    Choked corrected mass flow is evaluated from:

        ``mass_flow = P0 * A / (eta_cstar * cstar_ideal)``

    Thrust is evaluated from:

        ``thrust = Cf * P0 * A``

    For unchoked flow, characteristic velocity is set to `None` because
    characteristic velocity is a choked rocket-performance quantity.
    """
    def __init__(
        self,
        name: str,
        network: Network,
        upstream_total_pressure: State,
        upstream_total_temperature: State,
        back_pressure: State,
        throat_area: float,
        specific_heat_ratio: State,
        specific_gas_constant: float,
        characteristic_velocity_efficiency: State | float = 1.0,
        thrust_coefficient_efficiency: State | float = 1.0,
        characteristic_velocity_ideal: State | None = None,
        thrust_coefficient_ideal: State | None = None,
        characteristic_velocity: State | None = None,
        thrust_coefficient: State | None = None,
        thrust: State | None = None,
        mass_flow: State | None = None
    ):
        self.setup()

    def evaluate_states(self):
        P0 = self.upstream_total_pressure.value
        T0 = self.upstream_total_temperature.value
        Pb = self.back_pressure.value
        A = self.throat_area.value
        gamma = self.specific_heat_ratio.value
        R = self.specific_gas_constant.value
        eta_cstar = self.characteristic_velocity_efficiency.value
        eta_Cf = self.thrust_coefficient_efficiency.value

        critical_pressure_ratio = (2.0 / (gamma + 1.0))**(gamma / (gamma - 1.0))
        pressure_ratio = Pb / P0
        choked = pressure_ratio <= critical_pressure_ratio

        if choked:
            mass_flow_ideal = A * P0 * sqrt_or_nan(gamma / (R * T0)) * (2.0 / (gamma + 1.0))**((gamma + 1.0) / (2.0 * (gamma - 1.0)))
            cstar_ideal = P0 * A / mass_flow_ideal
            cstar = eta_cstar * cstar_ideal
            mass_flow = P0 * A / cstar

            T_star = T0 * (2.0 / (gamma + 1.0))
            P_star = P0 * critical_pressure_ratio
            u_star = sqrt_or_nan(gamma * R * T_star)
            thrust_ideal = mass_flow_ideal * u_star + (P_star - Pb) * A
            Cf_ideal = thrust_ideal / (P0 * A)
            Cf = eta_Cf * Cf_ideal
            thrust = Cf * P0 * A

            self.characteristic_velocity_ideal.value = cstar_ideal
            self.characteristic_velocity.value = cstar
            self.thrust_coefficient_ideal.value = Cf_ideal
            self.thrust_coefficient.value = Cf
            self.mass_flow.value = mass_flow
            self.thrust.value = thrust

        else:
            M = sqrt_or_nan((2.0 / (gamma - 1.0)) * (pressure_ratio**(-(gamma - 1.0) / gamma) - 1.0))
            T = T0 / (1.0 + 0.5 * (gamma - 1.0) * M**2)
            u = M * sqrt_or_nan(gamma * R * T)
            mass_flow = A * P0 * sqrt_or_nan(gamma / (R * T0)) * M * (1.0 + 0.5 * (gamma - 1.0) * M**2)**(-(gamma + 1.0) / (2.0 * (gamma - 1.0)))
            thrust = mass_flow * u
            Cf_ideal = thrust / (P0 * A)
            Cf = eta_Cf * Cf_ideal

            self.characteristic_velocity_ideal.value = None
            self.characteristic_velocity.value = None
            self.thrust_coefficient_ideal.value = Cf_ideal
            self.thrust_coefficient.value = Cf
            self.mass_flow.value = mass_flow
            self.thrust.value = thrust


