from __future__ import annotations

import math
from scipy.special import wrightomega
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State



class Colebrook(Component):
    """Colebrook-White Darcy friction factor with laminar fallback.

        The component computes Reynolds number from mass flow, area, diameter, and
        dynamic viscosity.  Below ``reynolds_number_threshold`` it uses the supplied
        laminar Poiseuille-number relation.  Above the threshold it solves/estimates
        the turbulent Colebrook-White rough-pipe friction factor.  The output is a
        Darcy friction factor suitable for ``DarcyWeisbach`` and convection
        correlations."""
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        friction_factor: State,
        hydraulic_diameter: State | float,
        dynamic_viscosity: State,
        cross_sectional_area: State | float,
        poiseuille_number: float = 16,
        roughness: State | float = 0.0,
        reynolds_number: State | float | None = None,
        reynolds_number_threshold: State | float = 2300.0,
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
        mdot = abs(self.mass_flow.value)
        mu = self.dynamic_viscosity.value
        A = self.cross_sectional_area.value
        Dh = self.hydraulic_diameter.value
        Po = self.poiseuille_number.value
        e = self.roughness.value

        Re_Dh = mdot * Dh / (mu * A)
        Deff = 16.0 * Dh / Po
        Re_eff = mdot * Deff / (mu * A)

        Re_Dh = max(Re_Dh, 1e-12)
        Re_eff = max(Re_eff, 1e-12)

        self.Deff = Deff

        is_turbulent = Re_Dh > self.reynolds_number_threshold.value

        if not is_turbulent:
            self.reynolds_number.value = Re_Dh
            f = 4*Po/Re_Dh
        else:
            self.reynolds_number.value = Re_eff
            f = self._colebrook_explicit(Re_eff, e, Deff)

        self.friction_factor.value = f

    def _colebrook_explicit(self, Re, roughness, hydraulic_diameter):
        a = 2.51 / Re
        b = roughness / (3.7 * hydraulic_diameter)
        c = 0.5 * math.log(10.0)

        y = math.log(c / a) + c * b / a
        x = wrightomega(y) / c - b / a

        return 1.0 / x**2




class Churchill(Component):
    """Churchill all-Reynolds-number Darcy friction factor correlation.

        The Churchill correlation provides a continuous friction-factor estimate
        from laminar through turbulent regimes and includes roughness effects.  It is
        useful when a smooth transition is more important than explicitly switching
        between laminar and turbulent formulas."""
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        friction_factor: State,
        hydraulic_diameter: State | float,
        dynamic_viscosity: State,
        cross_sectional_area: State | float,
        roughness: State | float = 0.0,
        poiseuille_number: float = 16,
        reynolds_number: State | float | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()
        self.Deff = 16*self.hydraulic_diameter.value / self.poiseuille_number.value

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        self.Deff = 16 * self.hydraulic_diameter.value / self.poiseuille_number.value
        self.reynolds_number.value = (
            abs(self.mass_flow.value)
            * self.Deff
            / (self.dynamic_viscosity.value * self.cross_sectional_area.value)
        )

        self.reynolds_number.value = max(self.reynolds_number.value, 1e-12)
        Re = self.reynolds_number.value
        relative_roughness = self.roughness.value / self.Deff

        A = (2.457 * math.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * relative_roughness))) ** 16
        B = (37530.0 / Re) ** 16
        f = 8.0 * ((8.0 / Re) ** 12 + (A + B) ** (-1.5)) ** (1.0 / 12.0)

        self.friction_factor.value = f


    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {
            "Deff",
        }





class PetukhovFriction(Component):
    """Petukhov smooth-pipe turbulent Darcy friction factor with laminar fallback.

        This component computes Reynolds number and then applies a laminar
        Poiseuille relation below the threshold and the Petukhov turbulent relation
        above it.  It is intended for smooth tubes used with Petukhov or similar
        heat-transfer correlations."""
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        friction_factor: State,
        hydraulic_diameter: State | float,
        dynamic_viscosity: State,
        cross_sectional_area: State | float,
        poiseuille_number: float = 16,
        reynolds_number: State | float | None = None,
        reynolds_number_threshold: State | float = 2300.0,
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
        mdot = abs(self.mass_flow.value)
        mu = self.dynamic_viscosity.value
        A = self.cross_sectional_area.value
        Dh = self.hydraulic_diameter.value
        Po = self.poiseuille_number.value

        Re = mdot * Dh / (mu * A)
        Re = max(Re, 1e-12)

        self.reynolds_number.value = Re

        is_turbulent = Re > self.reynolds_number_threshold.value

        if not is_turbulent:
            f = 4.0 * Po / Re
        else:
            f = (0.79 * math.log(Re) - 1.64) ** -2

        self.friction_factor.value = f
