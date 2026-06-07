from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Gnielinski(Component):
    """
    Gnielinski turbulent forced-convection heat transfer coefficient.

    Correlation
    -----------
        Nu = ((f / 8) (Re - 1000) Pr)
             / (1 + 12.7 sqrt(f / 8) (Pr^(2/3) - 1))

        Re = mdot * Dh / (mu * A)

        Pr = cp * mu / k

        h = Nu * k / Dh

    Parameters
    ----------
    hydraulic_diameter : State | float
        Hydraulic diameter of the flow passage [m].

    friction_factor : State | float
        Darcy friction factor [-].

    fluid_conductivity : State
        Fluid thermal conductivity [W/m-K].

    fluid_specific_heat : State
        Fluid specific heat capacity [J/kg-K].

    fluid_dynamic_viscosity : State
        Fluid dynamic viscosity [Pa-s].

    cross_sectional_area : State | float
        Flow cross-sectional area [m²].

    mass_flow : State
        Fluid mass flow rate [kg/s]. The absolute value is used.

    convection_coefficient : State, optional
        Output convection coefficient h [W/m²-K].
        If omitted, a new State is created.

    Outputs
    -------
    convection_coefficient : State
        Convective heat transfer coefficient [W/m²-K].

    Assumptions
    -----------
    * Single-phase internal flow.
    * Fully developed flow.
    * Uses the Darcy friction factor.
    * Fluid properties evaluated at the bulk fluid temperature.

    Recommended Validity Range
    --------------------------
    * 3,000 <= Re <= 5×10⁶
    * 0.5 <= Pr <= 2,000

    Notes
    -----
    Gnielinski is generally more accurate than Dittus-Boelter and
    Sieder-Tate because it incorporates the friction factor and remains
    applicable through much of the transitional and turbulent regimes.
    """

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
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        Dh = self.hydraulic_diameter.value
        f = self.friction_factor.value
        k = self.fluid_conductivity.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        A = self.cross_sectional_area.value
        mdot = abs(self.mass_flow.value)

        if Dh <= 0.0:
            raise ValueError(
                f"{self.name}: hydraulic_diameter must be greater than zero. Got {Dh}."
            )

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: cross_sectional_area must be greater than zero. Got {A}."
            )

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
    """
    Miropolskii film-boiling heat transfer coefficient for two-phase flow.

    Correlation
    -----------
        Nu = 0.023 Re^0.8 Pr^0.4 Y

        Re = (G Dh / mu_v) [x + (rho_v / rho_l)(1 - x)]

        Pr = Cp_v mu_v / k_v

        Y = 1 - 0.1 (rho_l / rho_v)^0.4 (1 - x)^0.4

        h = Nu k_v / Dh

    Notes
    -----
    x is vapor quality.

    This correlation is intended for film-boiling two-phase flow and is useful
    for chilldown problems. It is not intended for nucleate boiling.
    """

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
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        mdot = abs(self.mass_flow.value)
        Dh = self.hydraulic_diameter.value
        A = self.cross_sectional_area.value
        x = self.quality.value
        rho_v = self.vapor_density.value
        Cp_v = self.vapor_specific_heat.value
        mu_v = self.vapor_dynamic_viscosity.value
        k_v = self.vapor_conductivity.value
        rho_l = self.liquid_density.value

        if Dh <= 0.0:
            raise ValueError(
                f"{self.name}: hydraulic_diameter must be greater than zero. Got {Dh}."
            )

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: cross_sectional_area must be greater than zero. Got {A}."
            )

        if not (0.0 <= x <= 1.0):
            raise ValueError(
                f"{self.name}: quality must be between 0 and 1. Got {x}."
            )

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
    """
    Petukhov turbulent forced-convection heat transfer coefficient.

    Correlation
    -----------
        Nu = ((f / 8) Re Pr)
             / (1.07 + 12.7 sqrt(f / 8) (Pr^(2/3) - 1))

        Re = mdot * Dh / (mu * A)

        Pr = cp * mu / k

        h = Nu * k / Dh

    Notes
    -----
    This correlation uses the Darcy friction factor, not the Fanning
    friction factor.

    Recommended Validity Range
    --------------------------
    * 10,000 <= Re <= 5×10⁶
    * 0.5 <= Pr <= 2,000
    """

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
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        Dh = self.hydraulic_diameter.value
        f = self.friction_factor.value
        k = self.fluid_conductivity.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        A = self.cross_sectional_area.value
        mdot = abs(self.mass_flow.value)

        if Dh <= 0.0:
            raise ValueError(
                f"{self.name}: hydraulic_diameter must be greater than zero. Got {Dh}."
            )

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: cross_sectional_area must be greater than zero. Got {A}."
            )

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
    """
    Sieder-Tate turbulent forced-convection heat transfer coefficient.

    Correlation
    -----------
        Nu = 0.027 Re^0.8 Pr^(1/3) (mu / mu_w)^0.14

        Re = mdot * Dh / (mu * A)

        Pr = cp * mu / k

        h = Nu * k / Dh

    Parameters
    ----------
    hydraulic_diameter : State | float
        Hydraulic diameter of the flow passage [m].

    fluid_conductivity : State
        Bulk fluid thermal conductivity [W/m-K].

    fluid_specific_heat : State
        Bulk fluid specific heat capacity [J/kg-K].

    bulk_fluid_dynamic_viscosity : State
        Dynamic viscosity evaluated at the bulk fluid temperature [Pa-s].

    wall_fluid_dynamic_viscosity : State
        Dynamic viscosity evaluated at the wall temperature [Pa-s].

    cross_sectional_area : State | float
        Flow cross-sectional area [m²].

    mass_flow : State
        Fluid mass flow rate [kg/s]. The absolute value is used.

    convection_coefficient : State, optional
        Output convection coefficient h [W/m²-K].
        If omitted, a new State is created.

    Outputs
    -------
    convection_coefficient : State
        Convective heat transfer coefficient [W/m²-K].

    Assumptions
    -----------
    * Single-phase flow.
    * Fully developed turbulent internal flow.
    * Bulk fluid properties are evaluated at the bulk fluid temperature.
    * Wall viscosity is evaluated at the wall temperature.
    * Preferred over Dittus-Boelter when wall and fluid temperatures differ
      enough for viscosity variation to matter.

    Recommended Validity Range
    --------------------------
    * Re >= 10,000
    * 0.7 <= Pr <= 16,700
    * L / Dh > 10

    Notes
    -----
    Sieder-Tate corrects Dittus-Boelter-like turbulent convection for
    temperature-dependent viscosity using (mu / mu_w)^0.14.
    """

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
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        Dh = self.hydraulic_diameter.value
        k = self.fluid_conductivity.value
        Cp = self.fluid_specific_heat.value
        mu = self.bulk_fluid_dynamic_viscosity.value
        mu_w = self.wall_fluid_dynamic_viscosity.value
        A = self.cross_sectional_area.value
        mdot = abs(self.mass_flow.value)

        if Dh <= 0.0:
            raise ValueError(
                f"{self.name}: hydraulic_diameter must be greater than zero. Got {Dh}."
            )

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: cross_sectional_area must be greater than zero. Got {A}."
            )

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
    """
    Dittus-Boelter (Colburn form) turbulent forced-convection heat transfer
    coefficient for fully developed internal flow.

    Correlation
    -----------
        Nu = 0.023 Re^0.8 Pr^(1/3)

        Re = mdot * Dh / (mu * A)

        Pr = cp * mu / k

        h = Nu * k / Dh

    Parameters
    ----------
    hydraulic_diameter : State | float
        Hydraulic diameter of the flow passage [m].

    fluid_conductivity : State
        Fluid thermal conductivity [W/m-K].

    fluid_specific_heat : State
        Fluid specific heat capacity [J/kg-K].

    fluid_dynamic_viscosity : State
        Fluid dynamic viscosity [Pa-s].

    cross_sectional_area : State | float
        Flow cross-sectional area [m²].

    mass_flow : State
        Fluid mass flow rate [kg/s].

    convection_coefficient : State, optional
        Output convection coefficient h [W/m²-K].
        If omitted, a new State is created.

    Outputs
    -------
    convection_coefficient : State
        Convective heat transfer coefficient [W/m²-K].

    Derived Quantities
    ------------------
    Reynolds number:

        Re = mdot * Dh / (mu * A)

    Prandtl number:

        Pr = cp * mu / k

    Nusselt number:

        Nu = 0.023 Re^0.8 Pr^(1/3)

    Assumptions
    -----------
    * Single-phase flow.
    * Fully developed turbulent internal flow.
    * Fluid properties evaluated at the bulk fluid temperature.
    * Uses the Colburn form with a fixed Prandtl exponent of 1/3.

    Recommended Validity Range
    --------------------------
    * Re >= 10,000
    * 0.7 <= Pr <= 160

    Recommended Temperature Difference Limits
    -----------------------------------------
    For best accuracy, the wall-to-fluid temperature difference should be:

    * |Twall - Tfluid| < 260.928 K (10 °F) for liquids.
    * |Twall - Tfluid| < 310.928 K (100 °F) for gases.

    Notes
    -----
    For large temperature differences, developing flow, rough tubes,
    transitional flow, or strong property variation, more advanced
    correlations such as Sieder-Tate or Gnielinski are generally preferred.
    """
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
        self.setup()

        self.Re_given = reynolds_number is not None
        self.Pr_given = prandtl_number is not None

    def evaluate_states(self):
        Dh = self.hydraulic_diameter.value
        k = self.fluid_conductivity.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        A = self.cross_sectional_area.value
        mdot = abs(self.mass_flow.value)

        if Dh <= 0.0:
            raise ValueError(
                f"{self.name}: hydraulic_diameter must be greater than zero. Got {Dh}."
            )

        if A <= 0.0:
            raise ValueError(
                f"{self.name}: cross_sectional_area must be greater than zero. Got {A}."
            )

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
    """
    Bartz convective heat transfer coefficient correlation for
    compressible flow in rocket thrust chambers and nozzles.

    Correlation
    -----------
        h_g = X * σ * G

        X = (0.026 / D^0.2)
            * (μ0^0.2 * Cp0 / Pr0^0.6)
            * (mdot / A)^0.8

        σ = (ρ_am / ρ)^0.8
            * (μ_am / μ0)^0.2

    Optional geometric correction:

        G = D / rc

    where:

        A = π D² / 4

    Parameters
    ----------
    mass_flow : State
        Local mass flow rate [kg/s].

    hydraulic_diameter : State | float
        Local hydraulic diameter or equivalent nozzle diameter [m].

    chamber_specific_heat_cp : State
        Specific heat capacity evaluated at stagnation conditions [J/kg-K].

    chamber_prandtl_number : State
        Prandtl number evaluated at stagnation conditions [-].

    chamber_dynamic_viscosity : State
        Dynamic viscosity evaluated at stagnation conditions [Pa-s].

    local_freestream_density : State
        Local gas density at the evaluation location [kg/m³].

    mean_temperature_density : State
        Gas density evaluated at the arithmetic mean temperature
        T_am = (T + T_w) / 2 [kg/m³]. T is the local freestream 
        static temperature.

    mean_temperature_dynamic_viscosity : State
        Dynamic viscosity evaluated at the arithmetic mean temperature
        T_am = (T + T_w) / 2 [Pa-s]. T is the local freestream
        static temperature.

    throat_converging_radius : float, optional
        Radius of curvature of the throat converging section [m].
        When supplied, the geometric correction D/rc is applied.

    convection_coefficient : State, optional
        Output convection coefficient h_g [W/m²-K].
        If omitted, a new State is created.

    Outputs
    -------
    convection_coefficient : State
        Gas-side convective heat transfer coefficient [W/m²-K].

    Derived Quantities
    ------------------
    Mass flux:

        G_m = mdot / A

    Property correction factor:

        σ = (ρ_am / ρ)^0.8
            * (μ_am / μ0)^0.2

    Base Bartz coefficient:

        X = (0.026 / D^0.2)
            * (μ0^0.2 * Cp0 / Pr0^0.6)
            * G_m^0.8

    Final heat transfer coefficient:

        h_g = X * σ * G

    Assumptions
    -----------
    * Turbulent, high-Reynolds-number compressible flow.
    * Developed for rocket combustion gases.
    * Chamber properties are evaluated at stagnation conditions.
    * Local property variation in the boundary layer is
      represented through the Bartz correction factor σ.
    * Hydraulic diameter is used as the characteristic length scale.
    * The nozzle cross section is circular.
    * Radiation heat transfer is neglected.

    Recommended Use
    ---------------
    * Rocket thrust chambers.
    * Converging-diverging nozzles.
    * Regeneratively cooled rocket engines.
    * Preliminary and system-level thermal analyses.

    Notes
    -----
    This implementation follows the classical Bartz engineering
    correlation using chamber (stagnation) transport properties and
    the mean-temperature correction factor:

        T_am = (T_g + T_w) / 2

    The Bartz correlation is empirical and is most accurate for
    chemically reacting rocket exhaust gases.

    Bartz tends to underpredict when the effects of radiation are
    strong, when there is a lot dissociation/recombination in the 
    boundary layer, or when there are significant combustion 
    instabilities. 

    Bartz tends to overpredict when soot deposition on the walls is 
    significant or when the combustion is incomplete.
    """
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
        self.setup()

    def evaluate_states(self):
        mdot = abs(self.mass_flow.value)
        D = self.hydraulic_diameter.value
        Cp0 = self.chamber_specific_heat_cp.value
        Pr0 = self.chamber_prandtl_number.value
        mu0 = self.chamber_dynamic_viscosity.value
        rho = self.local_freestream_density.value
        rho_am = self.mean_temperature_density.value
        mu_am = self.mean_temperature_dynamic_viscosity.value
        A = (np.pi/4) * D**2

        if self.throat_converging_radius.is_assigned:
            rc = self.throat_converging_radius.value

            if rc <= 0.0:
                raise ValueError(
                    f"{self.name}: throat_converging_radius must be greater than zero. Got {rc}."
                )
            
            geometric_correction = D/rc
        else:
            geometric_correction = 1

        if D <= 0.0:
            raise ValueError(
                f"{self.name}: hydraulic_diameter must be greater than zero. Got {D}."
            )
        
        X = (0.026/(D**0.2)) * (mu0**0.2 * Cp0 / Pr0**0.6) * (mdot/A)**0.8
        sigma = (rho_am/rho)**0.8 * (mu_am/mu0)**0.2
        hg = X * sigma * geometric_correction

        self.convection_coefficient.value = hg







class NaturalConvection(Component):
    """
    Empirical natural-convection heat transfer coefficient.

    Correlation
    -----------
        Gr = g beta |Tw - Tf| L^3 rho^2 / mu^2

        Pr = Cp mu / k

        Ra = Gr Pr

        Nu = c Ra^n

        h = Nu k / L

    Coefficients
    ------------
        Laminar:   Ra < 1e9   -> c = 0.59, n = 0.25
        Turbulent: Ra >= 1e9  -> c = 0.13, n = 0.33

    Notes
    -----
    Fluid properties should be evaluated at the film temperature:

        Tfilm = 0.5 * (Tw + Tf)

    beta is the volumetric thermal expansion coefficient [1/K].
    For an ideal gas, beta = 1 / Tfilm.

    Recommended Validity Range
    --------------------------
    * 1e4 <= Ra <= 1e13
    """

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
        self.setup()

    def evaluate_states(self):
        Tw = self.wall_temperature.value
        Tf = self.fluid_temperature.value
        L = self.characteristic_length.value
        rho = self.fluid_density.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        k = self.fluid_conductivity.value
        beta = self.thermal_expansion_coefficient.value
        g = self.gravity.value

        if L <= 0.0:
            raise ValueError(
                f"{self.name}: characteristic_length must be greater than zero. Got {L}."
            )

        Gr = g * beta * abs(Tw - Tf) * L**3 * rho**2 / mu**2
        Pr = Cp * mu / k
        Ra = Gr * Pr

        if Ra < 1.0e9:
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
    """
    Churchill-Chu natural-convection heat transfer coefficient.

    Correlation
    -----------
        Gr = g beta |Tw - Tf| L^3 rho^2 / mu^2

        Pr = Cp mu / k

        Ra = Gr Pr

        Nu = (0.825 + 0.387 Ra^(1/6) / (1 + (0.492 / Pr)^(9/16))^(8/27))^2

        h = Nu k / L

    Notes
    -----
    Fluid properties should be evaluated at the film temperature:

        Tfilm = 0.5 * (Tw + Tf)

    beta is the volumetric thermal expansion coefficient [1/K].
    For an ideal gas, beta = 1 / Tfilm.

    Recommended Validity Range
    --------------------------
    * 1e-1 <= Ra <= 1e12
    """

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
        self.setup()

    def evaluate_states(self):
        Tw = self.wall_temperature.value
        Tf = self.fluid_temperature.value
        L = self.characteristic_length.value
        rho = self.fluid_density.value
        Cp = self.fluid_specific_heat.value
        mu = self.fluid_dynamic_viscosity.value
        k = self.fluid_conductivity.value
        beta = self.thermal_expansion_coefficient.value
        g = self.gravity.value

        if L <= 0.0:
            raise ValueError(
                f"{self.name}: characteristic_length must be greater than zero. Got {L}."
            )

        Gr = g * beta * abs(Tw - Tf) * L**3 * rho**2 / mu**2
        Pr = Cp * mu / k
        Ra = Gr * Pr

        Nu = (0.825 + 0.387 * Ra**(1.0 / 6.0) / (1.0 + (0.492 / Pr)**(9.0 / 16.0))**(8.0 / 27.0))**2

        self.grashof_number.value = Gr
        self.prandtl_number.value = Pr
        self.rayleigh_number.value = Ra
        self.nusselt_number.value = Nu
        self.convection_coefficient.value = Nu * k / L
