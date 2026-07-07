from __future__ import annotations

import math
import numpy as np
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class FlowTube(Component):

    """One-dimensional branch with pressure, friction, gravity, inertia, and optional compressible-flow diagnostics.

        ``FlowTube`` represents a finite-length flow path between an upstream and a
        downstream static pressure.  It is more general than a simple restriction:
        the component can include Darcy friction, elevation change, density changes
        between the ends of the tube, and a flow-inertia dynamic term.  Positive
        ``mass_flow`` is from upstream to downstream.

        Solver behavior
        ---------------
        In steady state the component exposes a momentum residual and the solver
        varies ``mass_flow`` until pressure forces, losses, gravity, and optional
        momentum terms balance.  In transient mode ``mass_flow`` is treated as a real
        dynamic state, so the residual integrates flow acceleration instead of
        instantly enforcing the steady momentum equation.

        Outputs
        -------
        The component writes diagnostic states such as total enthalpy when enthalpy
        inputs are supplied.  If speed of sound and area data are available, the
        component can also flag approximate normal-shock behavior for diagnostic
        model development."""
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        upstream_static_pressure: State,
        downstream_static_pressure: State,
        length: float,
        hydraulic_diameter: float,
        cross_sectional_area: float | None = None,
        upstream_density: State | None = None,
        downstream_density: State | None = None,
        friction_factor: float | None = None,
        upstream_speed_of_sound: float | None = None,
        downstream_speed_of_sound: float | None = None,
        gravitational_acceleration: float = 9.80665,
        height_change: float | None = None,
        upstream_static_enthalpy: State | None = None,
        total_enthalpy: State | None = None,
        normal_shock: State | bool | None = False,
        shock_mach_number: State | None = 0.0,
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
        mdot = self.mass_flow.value

        p1 = self.upstream_static_pressure.value
        p2 = self.downstream_static_pressure.value
        g = self.gravitational_acceleration.value

        L = self.length.value
        D = self.hydraulic_diameter.value

        if self.cross_sectional_area.is_assigned:
            A = self.cross_sectional_area.value
        else:
            A = (math.pi / 4.0) * D**2
            self.cross_sectional_area.value = A

        pressure = (p1 - p2) * A
        friction = 0.0
        inertia = 0.0
        gravity = 0.0

        self.normal_shock.value = False
        self.shock_mach_number.value = 0.0

        if self.upstream_density.is_assigned:
            rho1 = self.upstream_density.value
            u1 = mdot / (rho1 * A)

            if self.height_change.is_assigned:
                dh = self.height_change.value
                gravity = rho1 * g * dh * A

            if self.friction_factor.is_assigned:
                f = self.friction_factor.value
                Kf = f * L / (2.0 * rho1 * D * A**2)
                friction = Kf * mdot * abs(mdot) * A

            if self.upstream_static_enthalpy.is_assigned:
                h1 = self.upstream_static_enthalpy.value
                self.total_enthalpy.value = h1 + 0.5 * u1**2

            if self.downstream_density.is_assigned:
                rho2 = self.downstream_density.value
                u2 = mdot / (rho2 * A)
                inertia = max(mdot, 0.0) * (u2 - u1) - max(-mdot, 0.0) * (u1 - u2)

                if (
                    friction == 0.0
                    and gravity == 0.0
                    and self.upstream_speed_of_sound.is_assigned
                    and self.downstream_speed_of_sound.is_assigned
                ):
                    a1 = self.upstream_speed_of_sound.value
                    a2 = self.downstream_speed_of_sound.value

                    if mdot > 0.0:
                        p_pre = p1
                        rho_pre = rho1
                        u_pre = abs(u1)
                        a_pre = a1

                        p_post = p2
                        rho_post = rho2
                        u_post = abs(u2)
                        a_post = a2

                    elif mdot < 0.0:
                        p_pre = p2
                        rho_pre = rho2
                        u_pre = abs(u2)
                        a_pre = a2

                        p_post = p1
                        rho_post = rho1
                        u_post = abs(u1)
                        a_post = a1

                    else:
                        p_pre = p1
                        rho_pre = rho1
                        u_pre = 0.0
                        a_pre = a1

                        p_post = p2
                        rho_post = rho2
                        u_post = 0.0
                        a_post = a2

                    if a_pre > 0.0 and a_post > 0.0:
                        M_pre = u_pre / a_pre
                        M_post = u_post / a_post

                        if (
                            p_post > p_pre
                            and rho_post > rho_pre
                            and M_pre > 1.0
                            and M_post < 1.0
                        ):
                            self.normal_shock.value = True
                            self.shock_mach_number.value = M_pre

        self.momentum_error = pressure - friction - inertia - gravity
        self.mass_flow_dot = self.momentum_error / L

    @property
    def dynamics(self):
        # Flow inertia is a real dynamic equation.  SteadyState drives
        # mass_flow_dot to zero; Transient integrates mass_flow.
        """Return dynamic equations contributed by this component.
        
                A two-item tuple ``(state, derivative)`` means the solver integrates that
                state directly.  A three-item tuple ``(iteration_state, stored_state,
                derivative)`` means the nonlinear solver iterates a convenient state but
                conserves/integrates a different stored quantity.  Steady-state solves
                drive the derivative to zero."""
        return [(self.mass_flow, self.mass_flow_dot)]








class AdiabaticFlow(Component):
    """Adiabatic branch relation based on total enthalpy conservation.

        The component relates upstream and downstream static enthalpy, density,
        area, and mass flow through the assumption that total enthalpy is conserved
        across the branch.  It is intended for simple gas-flow bookkeeping where the
        user wants an energy relation without detailed losses, heat transfer, or a
        full compressible nozzle solution.

        Positive mass flow is from upstream to downstream.  Optional upstream area
        and density let the component include velocity on both sides; otherwise only
        the downstream kinetic-energy correction is used."""

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_static_enthalpy: State,
        downstream_static_enthalpy: State,
        downstream_density: State,
        downstream_cross_sectional_area: State | float,
        upstream_density: State | None = None,
        upstream_cross_sectional_area: State | float | None = None,
        mass_flow: State | None = 0.0,
        total_enthalpy: State | None = 0.0,
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
        h1 = self.upstream_static_enthalpy.value
        h2 = self.downstream_static_enthalpy.value
        rho2 = self.downstream_density.value
        A2 = self.downstream_cross_sectional_area.value

        self.mass_flow.value = 0.0
        self.total_enthalpy.value = h1

        if rho2 <= 0.0 or A2 <= 0.0:
            return

        upstream_finite_area = (
            self.upstream_density.is_assigned
            and self.upstream_cross_sectional_area.is_assigned
        )

        if upstream_finite_area:
            rho1 = self.upstream_density.value
            A1 = self.upstream_cross_sectional_area.value

            if rho1 <= 0.0 or A1 <= 0.0:
                return

            denominator = 1.0/(rho2*A2)**2 - 1.0/(rho1*A1)**2

            if denominator == 0.0:
                return

            mass_flow_squared = 2.0*(h1 - h2)/denominator

            if mass_flow_squared <= 0.0:
                return

            self.mass_flow.value = mass_flow_squared**0.5

            upstream_velocity = self.mass_flow.value/(rho1*A1)
            self.total_enthalpy.value = h1 + 0.5*upstream_velocity**2

        else:
            velocity_squared = 2.0*(h1 - h2)

            if velocity_squared <= 0.0:
                return

            downstream_velocity = velocity_squared**0.5

            self.mass_flow.value = rho2*A2*downstream_velocity
            self.total_enthalpy.value = h1





class DarcyWeisbach(Component):
    """Incompressible Darcy-Weisbach branch with optional flow inertia.

        ``DarcyWeisbach`` computes the mass flow through a pipe-like branch from the
        pressure difference, density, hydraulic diameter, length, area, friction
        factor, and optional height change.  It uses the Darcy friction factor, not
        the Fanning factor.  Positive ``mass_flow`` is from ``upstream_pressure`` to
        ``downstream_pressure``.

        If ``effective_area`` is supplied the branch includes inertia and returns a
        transient dynamic equation for ``mass_flow``.  Without inertia it behaves as
        an algebraic restriction.  This makes the same component useful for quick
        steady pipe losses and for water-hammer-style transient examples where flow
        momentum storage matters."""
    def __init__(
        self,
        name: str,
        network: Network,
        mass_flow: State,
        upstream_pressure: State,
        downstream_pressure: State,
        length: float,
        hydraulic_diameter: float,
        density: State,
        cross_sectional_area: State | float | None = None,
        friction_factor: State | float | None = None,
        gravitational_acceleration: State | float = 9.80665,
        height_change: State | float | None = None,
        effective_area: float | None = None,
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
        mdot = self.mass_flow.value

        p1 = self.upstream_pressure.value
        p2 = self.downstream_pressure.value
        g = self.gravitational_acceleration.value

        L = self.length.value
        D = self.hydraulic_diameter.value
        rho = self.density.value
        
        if self.cross_sectional_area.is_assigned:
            A = self.cross_sectional_area.value
        else:
            A = (math.pi / 4.0) * D**2
            self.cross_sectional_area.value = A

        if self.friction_factor.is_assigned:
            f = self.friction_factor.value
        else:
            f = 0.0

        if self.height_change.is_assigned:
            dh = self.height_change.value
        else:
            dh = 0.0

        Kf = f * L / (2.0 * rho * D * A**2)

        pressure_drop = p1 - p2

        effective_area_value = 2.0 * rho * abs(pressure_drop)

        if effective_area_value > 0.0:
            self.effective_area.value = abs(mdot) / math.sqrt(effective_area_value)
        else:
            self.effective_area.value = 0.0

        pressure = (p1 - p2) * A
        friction = Kf * mdot * abs(mdot) * A
        gravity = rho * g * dh * A

        self.momentum_error = pressure - friction - gravity
        self.mass_flow_dot = self.momentum_error / L

    @property
    def dynamics(self):
        # Pipe inertia is represented by mass_flow_dot.  In steady state this
        # derivative is driven to zero, which recovers the usual pressure-loss
        # equation.
        """Return dynamic equations contributed by this component.
        
                A two-item tuple ``(state, derivative)`` means the solver integrates that
                state directly.  A three-item tuple ``(iteration_state, stored_state,
                derivative)`` means the nonlinear solver iterates a convenient state but
                conserves/integrates a different stored quantity.  Steady-state solves
                drive the derivative to zero."""
        return [(self.mass_flow, self.mass_flow_dot)]





class DischargeCoefficient(Component):
    """Reversible incompressible orifice/restriction relation using CdA.

        This component computes ``mass_flow`` from upstream pressure, downstream
        pressure, density, discharge coefficient, and flow area.  It supports reverse
        flow by preserving the sign of the pressure difference.  When a length is
        supplied, the component also exposes a flow-inertia dynamic equation so
        transient solves can integrate acceleration through the restriction."""
    def __init__(
        self,
        name: str,
        network: Network,
        upstream_pressure: State,
        downstream_pressure: State,
        density: State,
        discharge_coefficient: float,
        cross_sectional_area: float,
        length: float | None = None,
        mass_flow: State | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        self.setup()


    def evaluate_states(self) -> None:
        """Evaluate the component for the current network state.
        
                Solvers call this method repeatedly while settling derived states and
                assembling residuals.  It should read input ``State.value`` fields, write
                output states, and update any residual or derivative attributes exposed
                through ``balances`` or ``dynamics``.  The method does not advance time;
                transient integration is handled by the solver."""
        P1 = self.upstream_pressure.value
        P2 = self.downstream_pressure.value
        rho = self.density.value
        Cd = self.discharge_coefficient.value
        A = self.cross_sectional_area.value

        dP = P1 - P2

        if self.length.is_assigned:
            L = self.length.value
            mdot = self.mass_flow.value

            R = 1.0 / (2.0 * (Cd * A)**2)
            Z = L / A

            self.mass_flow_dot = (dP - (R / rho) * mdot * abs(mdot)) / Z

        else:
            sign = np.sign(dP)
            value = max(0.0, 2.0 * rho * abs(dP))

            self.mass_flow.value = sign * Cd * A * math.sqrt(value)

    @property
    def dynamics(self):
        # Without length this is a direct algebraic calculator: mass_flow is
        # written explicitly from pressure drop, so no solver equation is added.
        # With length, the branch has flow inertia and mass_flow_dot is a real
        # dynamic equation.
        """Return dynamic equations contributed by this component.
        
                A two-item tuple ``(state, derivative)`` means the solver integrates that
                state directly.  A three-item tuple ``(iteration_state, stored_state,
                derivative)`` means the nonlinear solver iterates a convenient state but
                conserves/integrates a different stored quantity.  Steady-state solves
                drive the derivative to zero."""
        if self.length.is_assigned:
            return [(self.mass_flow, self.mass_flow_dot)]
        return []



class CavitatingVenturi(Component):
    """Liquid venturi/orifice relation with a cavitating and noncavitating branch.

        The component compares the recovered throat pressure against vapor pressure
        using ``pressure_recovery_factor``.  In the noncavitating regime it uses the
        supplied noncavitating discharge coefficient and the actual downstream
        pressure drop.  In the cavitating regime it limits the effective downstream
        pressure by vapor pressure and uses the cavitating discharge coefficient.

        Outputs include ``mass_flow`` and ``is_cavitating`` so users can track or
        plot when the model switches regimes."""

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_pressure: State,
        downstream_pressure: State,
        density: State,
        throat_area: float,
        vapor_pressure: State,
        pressure_recovery_factor: float = 0.85,
        cavitating_discharge_coefficient: float = 0.94,
        noncavitating_discharge_coefficient: float = 0.6,
        mass_flow: State | None = None,
        is_cavitating: bool = False,
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
        P1 = self.upstream_pressure.value
        P2 = self.downstream_pressure.value
        rho = self.density.value
        A = self.throat_area.value
        Pvap = self.vapor_pressure.value
        R = self.pressure_recovery_factor.value
        Cd_cav = self.cavitating_discharge_coefficient.value
        Cd_noncav = self.noncavitating_discharge_coefficient.value

        dP = P1 - P2
        sign = np.sign(dP)

        P2_critical = Pvap + R * (P1 - Pvap)
        self.critical_downstream_pressure = P2_critical

        is_cavitating = dP > 0.0 and P2 <= P2_critical

        if is_cavitating:
            self.throat_pressure = Pvap

            dP_cav = P1 - Pvap
            value = max(0.0, 2.0 * rho * dP_cav)

            self.mass_flow.value = Cd_cav * A * math.sqrt(value)

        else:
            if dP > 0.0 and R < 1.0:
                self.throat_pressure = P1 - dP / (1.0 - R)
            else:
                self.throat_pressure = P1

            value = max(0.0, 2.0 * rho * abs(dP))

            self.mass_flow.value = sign * Cd_noncav * A * math.sqrt(value)
            
    @property
    def ignored_export_attributes(self):
        return super().ignored_export_attributes | {"critical_downstream_pressure"}





class SeriesCdA(Component):
    """Equivalent effective area for multiple restrictions in series.

        The input list contains individual effective areas, normally ``Cd * A``
        values or already-computed effective area states.  The equivalent area is
        computed with the reciprocal-square relation used for pressure-drop devices
        in series.  This is useful when a line contains several small restrictions
        but the surrounding model only needs one lumped effective area."""
    def __init__(
        self,
        name: str,
        network: Network,
        effective_areas: list[State | float],
        effective_area: State | None = None,
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
        inverse_area_squared_sum = 0.0

        for effective_area in self.effective_areas.value:
            if hasattr(effective_area, "value"):
                CdA = effective_area.value
            else:
                CdA = effective_area

            if abs(CdA) < 1e-12:
                self.effective_area.value = 0.0
                return

            inverse_area_squared_sum += 1.0 / CdA**2

        self.effective_area.value = 1.0 / inverse_area_squared_sum**0.5





class ParallelCdA(Component):
    """Equivalent effective area for multiple restrictions in parallel.

        The component sums effective areas from parallel branches.  It is intended
        for injector elements, parallel orifices, or manifolded restrictions whose
        individual pressure drops are approximately the same."""
    def __init__(
        self,
        name: str,
        network: Network,
        effective_areas: list[State | float],
        effective_area: State | None = None,
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
        self.effective_area.value = sum(
            effective_area.value if hasattr(effective_area, "value") else effective_area
            for effective_area in self.effective_areas.value
        )





class RectanglePoiseuille(Component):
    """Poiseuille-number helper for laminar rectangular ducts.

        The component computes a geometry-only laminar Poiseuille number from duct
        height and width.  The result is normally passed into friction-factor
        components so non-circular laminar friction can be handled without hardcoding
        the value in the flow model."""
    def __init__(
        self,
        name: str,
        network: Network,
        height: float,
        width: float,
        poiseuille_number: float | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        if height <= 0.0:
            raise ValueError(f"Rectangle height must be positive. Got length={height}.")
        
        if width <= 0.0:
            raise ValueError(f"Rectangle width must be positive. Got length={width}.")
        
        self.setup()
        a = max(height/2, width/2)
        b = min(height/2, width/2)
        x = b/a
        A0 = 23.9201
        A1 = -29.436
        A2 = 30.3872
        A3 = -10.7128
        A4 = 0.0
        self.poiseuille_number.value = A0 + A1*x + A2*x**2 + A3*x**3 + A4*x**4





class EllipsePoiseuille(Component):
    """Poiseuille-number helper for laminar elliptical ducts.

        The component computes the geometry factor for an elliptical passage using
        the semi-major and semi-minor axes.  The output can be fed into laminar or
        transitional friction-factor components."""
    def __init__(
        self,
        name: str,
        network: Network,
        semi_major_axis: float,
        semi_minor_axis: float,
        poiseuille_number: float | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        if semi_major_axis <= 0.0:
            raise ValueError(f"Ellipse semi-major axis must be positive. Got length={semi_major_axis}.")
        
        if semi_minor_axis <= 0.0:
            raise ValueError(f"Ellipse semi-minor axis must be positive. Got length={semi_minor_axis}.")

        self.setup()
        a = max(semi_major_axis, semi_minor_axis)
        b = min(semi_minor_axis, semi_major_axis)
        x = b/a
        A0 = 19.7669
        A1 = -4.53458
        A2 = -11.5239
        A3 = 22.3709
        A4 = -10.0874
        self.poiseuille_number.value = A0 + A1*x + A2*x**2 + A3*x**3 + A4*x**4






class CircularAnnulusPoiseuille(Component):
    """Poiseuille-number helper for laminar circular annuli.

        The component calculates the laminar annular-duct Poiseuille number from
        inner and outer diameters.  It is useful for bearings, cooling gaps, seals,
        and any annular hydraulic passage."""
    def __init__(
        self,
        name: str,
        network: Network,
        inner_diameter: float,
        outer_diameter: float,
        poiseuille_number: float | None = None,
    ):
        """Initialize the object and register any FullFlow state wiring.
        
                Constructor parameters are documented on the class docstring and in the
                function signature.  Component constructors normally call
                ``Component.setup()``, which converts plain scalars to ``State`` objects,
                preserves supplied state-like objects, creates output states for optional
                ``None`` arguments, stores metadata, and registers the component with its
                network."""
        if inner_diameter <= 0.0:
            raise ValueError(f"Annulus inner diameter must be positive. Got length={inner_diameter}.")
        
        if outer_diameter <= 0.0:
            raise ValueError(f"Annulus outer_diameter must be positive. Got length={outer_diameter}.")

        self.setup()
        a = outer_diameter
        b = inner_diameter
        x = b/a

        if x < 0.2508:
            A0 = 24.8272
            A1 = 0.0479888
            self.poiseuille_number.value = A0 * x**A1
        else:
            A0 = 22.0513
            A1 = 6.44473
            A2 = -7.35451
            A3 = 2.78999
            A4 = 0
            self.poiseuille_number.value = A0 + A1*x + A2*x**2 + A3*x**3 + A4*x**4







class HydraulicDiameter(Component):
    """Hydraulic diameter computed from area and wetted perimeter.

        The component writes ``hydraulic_diameter = 4 * area / wetted_perimeter``.
        It is a convenience node for examples and for models where geometry is built
        from reusable states rather than constants."""
    def __init__(
        self,
        name: str,
        network: Network,
        cross_sectional_area: State | float,
        wetted_perimeter: State | float,
        hydraulic_diameter: State | None = None,
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
        A = self.cross_sectional_area.value
        P = self.wetted_perimeter.value

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: cross_sectional_area must be greater than zero. Got {A}."
            )

        if P <= 0.0:
            raise ValueError(
                f"{self.name}: wetted_perimeter must be greater than zero. Got {P}."
            )

        self.hydraulic_diameter.value = 4.0 * A / P
