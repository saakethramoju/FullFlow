from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network



class SpecificImpulse(Component):

    """Specific impulse from thrust and propellant mass flow.

        The component computes ``Isp = thrust / (mass_flow * g0)``.  It is a simple
        diagnostic helper for rocket-engine examples.  The user is responsible for
        passing consistent SI units unless a different unit convention is used
        consistently across the model."""
    def __init__(
        self, 
        name: str, 
        network: Network,
        thrust: State,
        mass_flow: State,
        gravitational_acceleration: float = 9.80665,
        specific_impulse: State | None = None
        ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        self.specific_impulse.value = self.thrust.value / (self.mass_flow.value * self.gravitational_acceleration.value)





class IdealCharacteristicVelocity(Component):

    """Ideal-gas characteristic velocity diagnostic.

        The component estimates ideal ``c*`` from chamber total temperature, gas
        constant, and specific-heat ratio using the standard isentropic throat-flow
        expression.  It is intended for quick rocket-performance checks and should
        not be treated as a full CEA performance model."""
    def __init__(
        self, 
        name: str, 
        network: Network,
        total_temperature: State,
        gas_constant: float,
        specific_heat_ratio: float,
        characteristic_velocity: State | None = None
        ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        g = self.specific_heat_ratio.value
        numerator = math.sqrt(g * self.gas_constant.value* self.total_temperature.value)
        denominator = g * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        self.characteristic_velocity.value = numerator / denominator
