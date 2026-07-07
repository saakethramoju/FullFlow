from __future__ import annotations

from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Conduction(Component):
    """One-dimensional conduction heat transfer between two temperature nodes.

        ``heat_rate`` is positive from ``temperature1`` to ``temperature2`` and is
        computed as ``k * A / L * (T1 - T2)``.  Use this component for wall segments,
        simple solids, thermal-resistance ladders, and lumped thermal networks."""

    def __init__(
        self,
        name: str,
        network: Network,
        temperature1: State,
        temperature2: State,
        thermal_conductivity: State,
        length: float,
        conductive_area: float,
        heat_rate: State | None = None,
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
        k = self.thermal_conductivity.value
        A = self.conductive_area.value
        L = self.length.value
        T1 = self.temperature1.value
        T2 = self.temperature2.value
        self.heat_rate.value = k * A / L * (T2 - T1)



class Radiation(Component):
    """Diffuse-gray radiation exchange between two surfaces.

        The component computes net radiative heat rate using two emissivities,
        surface areas, and a view factor.  If ``radiative_area2`` is omitted the two
        surfaces are treated as having the same area.  Positive heat rate is from
        ``temperature1`` to ``temperature2``."""
    SIGMA = 5.670374419e-8  # W/m^2-K^4

    def __init__(
        self,
        name: str,
        network: Network,
        temperature1: State,
        temperature2: State,
        emissivity1: float,
        emissivity2: float,
        radiative_area1: float,
        radiative_area2: float | None = None,
        view_factor12: float = 1.0,
        heat_rate: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

        if radiative_area2 is None:
            self.radiative_area2.value = self.radiative_area1.value

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        T1 = self.temperature1.value
        T2 = self.temperature2.value

        eps1 = self.emissivity1.value
        eps2 = self.emissivity2.value

        A1 = self.radiative_area1.value
        A2 = self.radiative_area2.value

        F12 = self.view_factor12.value

        denominator = (1.0 - eps1) / (eps1 * A1) + 1.0 / (A1 * F12) + (1.0 - eps2) / (eps2 * A2)

        self.heat_rate.value = self.SIGMA * (T2**4 - T1**4) / denominator





class AmbientRadiation(Component):
    """Radiation exchange between a surface and a large ambient enclosure.

        The surface temperature, ambient temperature, emissivity, radiative area, and
        optional ambient emissivity determine net heat transfer.  Positive heat rate
        leaves the surface when the surface is hotter than the ambient."""
    SIGMA = 5.670374419e-8  # W/m^2-K^4

    def __init__(
        self,
        name: str,
        network: Network,
        solid_temperature: State,
        ambient_temperature: State | float,
        emissivity: State | float,
        radiative_area: State | float,
        ambient_emissivity: State | float = 1.0,
        heat_rate: State | None = None,
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
        Ts = self.solid_temperature.value
        Tamb = self.ambient_temperature.value

        eps_s = self.emissivity.value
        eps_amb = self.ambient_emissivity.value

        A = self.radiative_area.value

        denominator = 1.0 / eps_s + 1.0 / eps_amb - 1.0

        self.heat_rate.value = self.SIGMA * A * (Tamb**4 - Ts**4) / denominator







class Convection(Component):
    """Convective heat transfer between a wall/surface state and a fluid state.

        The component computes ``heat_rate = h * A * (surface_temperature -
        fluid_temperature)``.  Positive heat rate is from the surface into the fluid.
        It is usually coupled to ``Solid`` and ``Volume`` components or to explicit
        heat-rate states in a thermal-fluid network."""
    def __init__(
        self,
        name: str,
        network: Network,
        surface_temperature: State,
        fluid_temperature: State | float,
        convective_area: State | float,
        convection_coefficient: State | float,
        heat_rate: State | None = None,
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
        Ts = self.surface_temperature.value
        Tf = self.fluid_temperature.value
        h = self.convection_coefficient.value
        A = self.convective_area.value

        self.heat_rate.value = h * A * (Tf - Ts)






class TemperatureRecoveryFactor(Component):
    """Boundary-layer recovery factor from Prandtl number.

        The component computes a turbulent recovery factor ``Pr^(1/3)`` by default
        or a laminar recovery factor ``sqrt(Pr)`` when ``turbulent=False``.  The
        result is typically used by ``AdiabaticWallTemperature`` and gas-side heat
        transfer correlations."""
    def __init__(
        self,
        name: str,
        network: Network,
        prandtl_number: State | None = None,
        recovery_factor: State | None = None,
        turbulent: bool = True,
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
        if self.prandtl_number is None or not self.prandtl_number.is_assigned:
            self.recovery_factor.value = 1.0
            return

        Pr = self.prandtl_number.value

        if self.turbulent.value:
            self.recovery_factor.value = Pr ** (1.0 / 3.0)
        else:
            self.recovery_factor.value = Pr ** 0.5







class AdiabaticWallTemperature(Component):
    """Adiabatic wall temperature for compressible boundary-layer heat transfer.

        The component combines total temperature, static temperature, and recovery
        factor to estimate the wall temperature that would produce zero convective
        heat flux.  It is commonly used before a gas-side convection coefficient or
        wall heat-flux calculation."""
    def __init__(
        self,
        name: str,
        network: Network,
        total_temperature: State,
        static_temperature: State,
        recovery_factor: State,
        adiabatic_wall_temperature: State | None = None,
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
        T0 = self.total_temperature.value
        T = self.static_temperature.value
        r = self.recovery_factor.value

        self.adiabatic_wall_temperature.value = T + r * (T0 - T)







class EckertReferenceTemperature(Component):
    """Eckert reference temperature used for evaluating gas-side transport properties.

        The component estimates a representative film/reference temperature from
        wall temperature, static temperature, and adiabatic wall temperature.  Users
        can pass the output into property lookups for viscosity, conductivity, and
        specific heat."""

    def __init__(
        self,
        name: str,
        network: Network,
        wall_temperature: State,
        static_temperature: State,
        adiabatic_wall_temperature: State,
        reference_temperature: State | None = None,
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
        Tw = self.wall_temperature.value
        T = self.static_temperature.value
        Taw = self.adiabatic_wall_temperature.value

        self.reference_temperature.value = 0.5 * Tw + 0.28 * T + 0.22 * Taw
