from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Gnielinski(Component):
    """Gnielinski turbulent internal-flow heat-transfer coefficient correlation.

        The component computes Reynolds number, Prandtl number, Nusselt number,
        Stanton number, and convection coefficient from mass flow, diameter, area,
        fluid properties, and Darcy friction factor.  It is intended for turbulent
        single-phase internal convection in pipes and ducts."""

    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        hydraulic_diameter: State | float,
        friction_factor: State | float,
        fluid_conductivity: State,
        fluid_specific_heat: State,
        fluid_dynamic_viscosity: State,
        cross_sectional_area: State | float,
        reynolds_number: State | float | None = None,
        prandtl_number: State | float | None = None,
        nusselt_number: State | float | None = None,
        stanton_number: State | float | None = None,
        convection_coefficient: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        Dh = self.hydraulic_diameter.value
        f = self.friction_factor.value
        k = self.fluid_conductivity.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        A = self.cross_sectional_area.value
        mdot = abs(self.mass_flow.value)

        if self.Re_given:
            Re = self.reynolds_number.value
        else:
            Re = mdot * Dh / (mu * A)
            self.reynolds_number.value = Re

        if self.Pr_given:
            Pr = self.prandtl_number.value
        else:
            Pr = mu * Cp / k
            self.prandtl_number.value = Pr

        Nu = (f / 8.0) * (Re - 1000.0) * Pr / (1.0 + 12.7 * (f / 8.0) ** 0.5 * (Pr ** (2.0 / 3.0) - 1.0))

        self.convection_coefficient.value = Nu * k / Dh

        self.nusselt_number.value = Nu
        self.stanton_number.value = Nu / (Re*Pr)




class Miropolskii(Component):
    """Miropolskii film-boiling/two-phase convection coefficient helper.

        The component evaluates a vapor-property based heat-transfer coefficient
        with density-ratio and quality corrections.  It is useful for cryogenic
        chilldown and boiling examples where vapor-side transport dominates the heat
        transfer estimate."""

    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        hydraulic_diameter: State | float,
        cross_sectional_area: State | float,
        quality: State | float,
        vapor_density: State,
        vapor_specific_heat: State,
        vapor_dynamic_viscosity: State,
        vapor_conductivity: State,
        liquid_density: State,
        reynolds_number: State | float | None = None,
        prandtl_number: State | float | None = None,
        correction_factor: State | float | None = None,
        nusselt_number: State | float | None = None,
        stanton_number: State | float | None = None,
        convection_coefficient: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        mdot = abs(self.mass_flow.value)
        Dh = self.hydraulic_diameter.value
        A = self.cross_sectional_area.value
        x = self.quality.value
        rho_v = self.vapor_density.value
        Cp_v = self.vapor_specific_heat.value
        mu_v = self.vapor_dynamic_viscosity.value
        k_v = self.vapor_conductivity.value
        rho_l = self.liquid_density.value

        G = mdot / A

        if self.Re_given:
            Re = self.reynolds_number.value
        else:
            Re = (G * Dh / mu_v) * (x + (rho_v / rho_l) * (1.0 - x))
            self.reynolds_number.value = Re

        if self.Pr_given:
            Pr = self.prandtl_number.value
        else:
            Pr = Cp_v * mu_v / k_v
            self.prandtl_number.value = Pr

        Y = 1.0 - 0.1 * (rho_l / rho_v)**0.4 * (1.0 - x)**0.4

        if Y <= 0.0:
            raise ValueError(
                f"{self.name}: Miropolskii correction factor became non-positive. Y={Y}."
            )

        Nu = 0.023 * Re**0.8 * Pr**0.4 * Y

        self.correction_factor.value = Y
        self.nusselt_number.value = Nu
        self.stanton_number.value = Nu / (Re * Pr)
        self.convection_coefficient.value = Nu * k_v / Dh







class Petukhov(Component):
    """Petukhov turbulent forced-convection heat-transfer coefficient correlation.

        Similar to ``Gnielinski``, this component uses mass flux, hydraulic diameter,
        fluid transport properties, and Darcy friction factor to compute Reynolds,
        Prandtl, Nusselt, Stanton, and ``h``.  It is intended for smooth-pipe
        turbulent internal-flow estimates."""
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        hydraulic_diameter: State | float,
        friction_factor: State | float,
        fluid_conductivity: State,
        fluid_specific_heat: State,
        fluid_dynamic_viscosity: State,
        cross_sectional_area: State | float,
        reynolds_number: State | float | None = None,
        prandtl_number: State | float | None = None,
        nusselt_number: State | float | None = None,
        stanton_number: State | float | None = None,
        convection_coefficient: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        Dh = self.hydraulic_diameter.value
        f = self.friction_factor.value
        k = self.fluid_conductivity.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        A = self.cross_sectional_area.value
        mdot = abs(self.mass_flow.value)

        if self.Re_given:
            Re = self.reynolds_number.value
        else:
            Re = mdot * Dh / (mu * A)
            self.reynolds_number.value = Re

        if self.Pr_given:
            Pr = self.prandtl_number.value
        else:
            Pr = mu * Cp / k
            self.prandtl_number.value = Pr

        Nu = ((f / 8.0) * Re * Pr) / (
            1.07
            + 12.7 * (f / 8.0) ** 0.5 * (Pr ** (2.0 / 3.0) - 1.0)
        )

        self.convection_coefficient.value = Nu * k / Dh
        self.nusselt_number.value = Nu
        self.stanton_number.value = Nu / (Re * Pr)



class SiederTate(Component):
    """Sieder-Tate turbulent convection coefficient with viscosity correction.

        The component uses bulk and wall viscosities so the heat-transfer coefficient
        accounts for temperature-dependent viscosity.  This is useful when wall and
        bulk temperatures differ strongly, such as regenerative cooling channels or
        heated/cold transfer lines."""
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        hydraulic_diameter: State | float,
        fluid_conductivity: State,
        fluid_specific_heat: State,
        bulk_fluid_dynamic_viscosity: State,
        wall_fluid_dynamic_viscosity: State,
        cross_sectional_area: State | float,
        reynolds_number: State | float | None = None,
        prandtl_number: State | float | None = None,
        nusselt_number: State | float | None = None,
        stanton_number: State | float | None = None,
        convection_coefficient: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        Dh = self.hydraulic_diameter.value
        k = self.fluid_conductivity.value
        Cp = self.fluid_specific_heat.value
        mu = self.bulk_fluid_dynamic_viscosity.value
        mu_w = self.wall_fluid_dynamic_viscosity.value
        A = self.cross_sectional_area.value
        mdot = abs(self.mass_flow.value)

        if self.Re_given:
            Re = self.reynolds_number.value
        else:
            Re = mdot * Dh / (mu * A)
            self.reynolds_number.value = Re

        if self.Pr_given:
            Pr = self.prandtl_number.value
        else:
            Pr = mu * Cp / k
            self.prandtl_number.value = Pr


        Nu = 0.027 * Re**0.8 * Pr**(1.0 / 3.0) * (mu / mu_w)**0.14

        self.convection_coefficient.value = Nu * k / Dh

        self.nusselt_number.value = Nu
        self.stanton_number.value = Nu / (Re*Pr)




class DittusBoelter(Component):
    """Dittus-Boelter turbulent internal-flow convection coefficient.

        The component computes a simple power-law Nusselt number and corresponding
        convection coefficient from flow rate, hydraulic diameter, area, and fluid
        properties.  It is a compact engineering correlation for fully developed
        turbulent pipe flow."""
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        hydraulic_diameter: State | float,
        fluid_conductivity: State,
        fluid_specific_heat: State,
        fluid_dynamic_viscosity: State,
        cross_sectional_area: State | float,
        reynolds_number: State | float | None = None,
        prandtl_number: State | float | None = None,
        nusselt_number: State | float | None = None,
        stanton_number: State | float | None = None,
        convection_coefficient: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        Dh = self.hydraulic_diameter.value
        k = self.fluid_conductivity.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        A = self.cross_sectional_area.value
        mdot = abs(self.mass_flow.value)

        if self.Re_given:
            Re = self.reynolds_number.value
        else:
            Re = mdot * Dh / (mu * A)
            self.reynolds_number.value = Re

        if self.Pr_given:
            Pr = self.prandtl_number.value
        else:
            Pr = mu * Cp / k
            self.prandtl_number.value = Pr

        
        Nu = 0.023 * Re**0.8 * Pr**(1.0 / 3.0)

        self.convection_coefficient.value = Nu * k / Dh

        self.nusselt_number.value = Nu
        self.stanton_number.value = Nu / (Re*Pr)






class Bartz(Component):
    """Bartz rocket-thrust-chamber gas-side convection coefficient correlation.

        ``Bartz`` estimates convective heat-transfer coefficient for hot gas flow in
        a chamber/nozzle environment from mass flow, hydraulic diameter, chamber gas
        properties, mean-temperature correction terms, and optional throat-converging
        radius.  It is intended for preliminary engine heat-transfer models."""
    def __init__(self, 
                 name: str, 
                 network: Network,
                 mass_flow: State,
                 hydraulic_diameter: State | float,
                 chamber_specific_heat_cp: State,
                 chamber_prandtl_number: State,
                 chamber_dynamic_viscosity: State,
                 local_freestream_density: State,
                 mean_temperature_density: State,
                 mean_temperature_dynamic_viscosity: State,
                 throat_converging_radius: float | None = None,
                 convection_coefficient: State | None = None):
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
        D = self.hydraulic_diameter.value
        Cp0 = self.chamber_specific_heat_cp.value
        Pr0 = self.chamber_prandtl_number.value
        mu0 = self.chamber_dynamic_viscosity.value
        rho = self.local_freestream_density.value
        rho_am = self.mean_temperature_density.value
        mu_am = self.mean_temperature_dynamic_viscosity.value
        A = (math.pi/4) * D**2

        if self.throat_converging_radius.is_assigned:
            rc = self.throat_converging_radius.value
            
            geometric_correction = D/rc
        else:
            geometric_correction = 1

        X = (0.026/(D**0.2)) * (mu0**0.2 * Cp0 / Pr0**0.6) * (mdot/A)**0.8
        sigma = (rho_am/rho)**0.8 * (mu_am/mu0)**0.2
        hg = X * sigma * geometric_correction

        self.convection_coefficient.value = hg





class NaturalConvection(Component):
    """Generic natural-convection heat-transfer coefficient from dimensionless groups.

        The component computes Grashof, Prandtl, Rayleigh, Nusselt, and convection
        coefficient using fluid density, viscosity, conductivity, specific heat,
        thermal expansion coefficient, gravity, characteristic length, and wall/fluid
        temperature difference."""
    def __init__(
        self,
        name: str,
        network: Network,
        wall_temperature: State,
        fluid_temperature: State,
        characteristic_length: State | float,
        fluid_density: State,
        fluid_specific_heat: State,
        fluid_dynamic_viscosity: State,
        fluid_conductivity: State,
        thermal_expansion_coefficient: State,
        gravity: State | float = 9.80665,
        grashof_number: State | float | None = None,
        prandtl_number: State | float | None = None,
        rayleigh_number: State | float | None = None,
        nusselt_number: State | float | None = None,
        convection_coefficient: State | None = None,
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
        Tf = self.fluid_temperature.value
        L = self.characteristic_length.value
        rho = self.fluid_density.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        k = self.fluid_conductivity.value
        beta = self.thermal_expansion_coefficient.value
        g = self.gravity.value

        Gr = g * beta * abs(Tw - Tf) * L**3 * rho**2 / mu**2
        Pr = Cp * mu / k
        Ra = Gr * Pr

        high_rayleigh = Ra >= 1.0e9

        if not high_rayleigh:
            c = 0.59
            n = 0.25
        else:
            c = 0.13
            n = 0.33

        Nu = c * Ra**n

        self.grashof_number.value = Gr
        self.prandtl_number.value = Pr
        self.rayleigh_number.value = Ra
        self.nusselt_number.value = Nu
        self.convection_coefficient.value = Nu * k / L



class ChurchillChu(Component):
    """Churchill-Chu natural-convection correlation.

        This component evaluates the Churchill-Chu vertical-plate style correlation
        and writes the same diagnostic dimensionless groups as ``NaturalConvection``.
        It is convenient when a more specific natural-convection relation is desired
        without writing a custom component."""
    def __init__(
        self,
        name: str,
        network: Network,
        wall_temperature: State,
        fluid_temperature: State,
        characteristic_length: State | float,
        fluid_density: State,
        fluid_specific_heat: State,
        fluid_dynamic_viscosity: State,
        fluid_conductivity: State,
        thermal_expansion_coefficient: State,
        gravity: State | float = 9.80665,
        grashof_number: State | float | None = None,
        prandtl_number: State | float | None = None,
        rayleigh_number: State | float | None = None,
        nusselt_number: State | float | None = None,
        convection_coefficient: State | None = None,
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
        Tf = self.fluid_temperature.value
        L = self.characteristic_length.value
        rho = self.fluid_density.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        k = self.fluid_conductivity.value
        beta = self.thermal_expansion_coefficient.value
        g = self.gravity.value

        Gr = g * beta * abs(Tw - Tf) * L**3 * rho**2 / mu**2
        Pr = Cp * mu / k
        Ra = Gr * Pr

        Nu = (0.825 + 0.387 * Ra**(1.0 / 6.0) / (1.0 + (0.492 / Pr)**(9.0 / 16.0))**(8.0 / 27.0))**2

        self.grashof_number.value = Gr
        self.prandtl_number.value = Pr
        self.rayleigh_number.value = Ra
        self.nusselt_number.value = Nu
        self.convection_coefficient.value = Nu * k / L
