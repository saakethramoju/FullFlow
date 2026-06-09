from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State


class Gnielinski(Component):
    """
    Gnielinski turbulent forced-convection heat transfer coefficient.

    `Gnielinski` computes the convective heat transfer coefficient for turbulent
    internal flow using the Gnielinski correlation. The correlation incorporates
    the Darcy friction factor and is generally more accurate than simpler
    Dittus-Boelter-style correlations over much of the transitional and
    turbulent regime.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Fluid mass flow rate. The absolute value is used
    hydraulic_diameter : State or float
        Hydraulic diameter of the flow passage
    friction_factor : State or float
        Darcy friction factor
    fluid_conductivity : State
        Fluid thermal conductivity
    fluid_specific_heat : State
        Fluid specific heat capacity
    fluid_dynamic_viscosity : State
        Fluid dynamic viscosity
    cross_sectional_area : State or float
        Flow cross-sectional area
    reynolds_number : State or float, optional
        Reynolds number. If omitted, it is calculated
    prandtl_number : State or float, optional
        Prandtl number. If omitted, it is calculated
    nusselt_number : State or float, optional
        Output Nusselt number
    stanton_number : State or float, optional
        Output Stanton number

    Outputs
    -------
    convection_coefficient : State, optional
        Convective heat transfer coefficient. If omitted, a new State is created

    Notes
    -----
    The Reynolds number is evaluated from:

        ``Re = mdot * Dh / (mu * A)``

    The Prandtl number is evaluated from:

        ``Pr = cp * mu / k``

    The Nusselt number is evaluated from:

        ``Nu = ((f / 8) * (Re - 1000) * Pr)
        / (1 + 12.7 * sqrt(f / 8) * (Pr^(2/3) - 1))``

    The convection coefficient is evaluated from:

        ``h = Nu * k / Dh``

    The Stanton number is evaluated from:

        ``St = Nu / (Re * Pr)``

    This correlation assumes single-phase, fully developed internal flow. It
    uses the Darcy friction factor, and fluid properties should be evaluated at
    the bulk fluid temperature.

    Recommended validity range:

    * 3,000 <= Re <= 5e6
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

        Nu = (f / 8.0) * (Re - 1000.0) * Pr / (1.0 + 12.7 * (f / 8.0) ** 0.5 * (Pr ** (2.0 / 3.0) - 1.0))

        self.convection_coefficient.value = Nu * k / Dh

        self.nusselt_number.value = Nu
        self.stanton_number.value = Nu / (Re*Pr)




class Miropolskii(Component):
    """
    Miropolskii film-boiling heat transfer coefficient for two-phase flow.

    `Miropolskii` computes a convective heat transfer coefficient for
    film-boiling two-phase flow. It is useful for chilldown-style problems where
    vapor quality and separate liquid and vapor properties are available.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Fluid mass flow rate. The absolute value is used
    hydraulic_diameter : State or float
        Hydraulic diameter of the flow passage
    cross_sectional_area : State or float
        Flow cross-sectional area
    quality : State or float
        Vapor quality
    vapor_density : State
        Vapor density
    vapor_specific_heat : State
        Vapor specific heat capacity
    vapor_dynamic_viscosity : State
        Vapor dynamic viscosity
    vapor_conductivity : State
        Vapor thermal conductivity
    liquid_density : State
        Liquid density
    reynolds_number : State or float, optional
        Reynolds number. If omitted, it is calculated
    prandtl_number : State or float, optional
        Prandtl number. If omitted, it is calculated
    correction_factor : State or float, optional
        Output Miropolskii correction factor
    nusselt_number : State or float, optional
        Output Nusselt number
    stanton_number : State or float, optional
        Output Stanton number

    Outputs
    -------
    convection_coefficient : State, optional
        Convective heat transfer coefficient. If omitted, a new State is created

    Notes
    -----
    This correlation is intended for film-boiling two-phase flow. It is not
    intended for nucleate boiling.

    `x` is vapor quality.

    Mass flux is evaluated from:

        ``G = mdot / A``

    The Reynolds number is evaluated from:

        ``Re = (G * Dh / mu_v) * (x + (rho_v / rho_l) * (1 - x))``

    The Prandtl number is evaluated from:

        ``Pr = Cp_v * mu_v / k_v``

    The Miropolskii correction factor is evaluated from:

        ``Y = 1 - 0.1 * (rho_l / rho_v)^0.4 * (1 - x)^0.4``

    The Nusselt number is evaluated from:

        ``Nu = 0.023 * Re^0.8 * Pr^0.4 * Y``

    The convection coefficient is evaluated from:

        ``h = Nu * k_v / Dh``

    The Stanton number is evaluated from:

        ``St = Nu / (Re * Pr)``
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

    `Petukhov` computes the convective heat transfer coefficient for turbulent
    internal flow using the Petukhov correlation. The correlation uses the Darcy
    friction factor and is intended for single-phase turbulent flow.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Fluid mass flow rate. The absolute value is used
    hydraulic_diameter : State or float
        Hydraulic diameter of the flow passage
    friction_factor : State or float
        Darcy friction factor
    fluid_conductivity : State
        Fluid thermal conductivity
    fluid_specific_heat : State
        Fluid specific heat capacity
    fluid_dynamic_viscosity : State
        Fluid dynamic viscosity
    cross_sectional_area : State or float
        Flow cross-sectional area
    reynolds_number : State or float, optional
        Reynolds number. If omitted, it is calculated
    prandtl_number : State or float, optional
        Prandtl number. If omitted, it is calculated
    nusselt_number : State or float, optional
        Output Nusselt number
    stanton_number : State or float, optional
        Output Stanton number

    Outputs
    -------
    convection_coefficient : State, optional
        Convective heat transfer coefficient. If omitted, a new State is created

    Notes
    -----
    This correlation uses the Darcy friction factor, not the Fanning friction
    factor.

    The Reynolds number is evaluated from:

        ``Re = mdot * Dh / (mu * A)``

    The Prandtl number is evaluated from:

        ``Pr = cp * mu / k``

    The Nusselt number is evaluated from:

        ``Nu = ((f / 8) * Re * Pr)
        / (1.07 + 12.7 * sqrt(f / 8) * (Pr^(2/3) - 1))``

    The convection coefficient is evaluated from:

        ``h = Nu * k / Dh``

    The Stanton number is evaluated from:

        ``St = Nu / (Re * Pr)``

    Recommended validity range:

    * 10,000 <= Re <= 5e6
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

    `SiederTate` computes the convective heat transfer coefficient for turbulent
    internal flow while correcting for temperature-dependent viscosity between
    the bulk fluid and the wall.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Fluid mass flow rate. The absolute value is used
    hydraulic_diameter : State or float
        Hydraulic diameter of the flow passage
    fluid_conductivity : State
        Bulk fluid thermal conductivity
    fluid_specific_heat : State
        Bulk fluid specific heat capacity
    bulk_fluid_dynamic_viscosity : State
        Dynamic viscosity evaluated at the bulk fluid temperature
    wall_fluid_dynamic_viscosity : State
        Dynamic viscosity evaluated at the wall temperature
    cross_sectional_area : State or float
        Flow cross-sectional area
    reynolds_number : State or float, optional
        Reynolds number. If omitted, it is calculated
    prandtl_number : State or float, optional
        Prandtl number. If omitted, it is calculated
    nusselt_number : State or float, optional
        Output Nusselt number
    stanton_number : State or float, optional
        Output Stanton number

    Outputs
    -------
    convection_coefficient : State, optional
        Convective heat transfer coefficient. If omitted, a new State is created

    Notes
    -----
    Sieder-Tate corrects Dittus-Boelter-like turbulent convection for
    temperature-dependent viscosity using the wall-to-bulk viscosity ratio. It
    is preferred over Dittus-Boelter when wall and bulk fluid temperatures
    differ enough for viscosity variation to matter.

    The Reynolds number is evaluated from:

        ``Re = mdot * Dh / (mu * A)``

    The Prandtl number is evaluated from:

        ``Pr = cp * mu / k``

    The Nusselt number is evaluated from:

        ``Nu = 0.027 * Re^0.8 * Pr^(1/3) * (mu / mu_w)^0.14``

    The convection coefficient is evaluated from:

        ``h = Nu * k / Dh``

    The Stanton number is evaluated from:

        ``St = Nu / (Re * Pr)``

    This correlation assumes single-phase, fully developed turbulent internal
    flow. Bulk fluid properties should be evaluated at the bulk fluid
    temperature, and wall viscosity should be evaluated at the wall temperature.

    Recommended validity range:

    * Re >= 10,000
    * 0.7 <= Pr <= 16,700
    * L / Dh > 10
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
    Dittus-Boelter turbulent forced-convection heat transfer coefficient.

    `DittusBoelter` computes the convective heat transfer coefficient for fully
    developed turbulent internal flow using the Colburn form of the
    Dittus-Boelter correlation.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Fluid mass flow rate. The absolute value is used
    hydraulic_diameter : State or float
        Hydraulic diameter of the flow passage
    fluid_conductivity : State
        Fluid thermal conductivity
    fluid_specific_heat : State
        Fluid specific heat capacity
    fluid_dynamic_viscosity : State
        Fluid dynamic viscosity
    cross_sectional_area : State or float
        Flow cross-sectional area
    reynolds_number : State or float, optional
        Reynolds number. If omitted, it is calculated
    prandtl_number : State or float, optional
        Prandtl number. If omitted, it is calculated
    nusselt_number : State or float, optional
        Output Nusselt number
    stanton_number : State or float, optional
        Output Stanton number

    Outputs
    -------
    convection_coefficient : State, optional
        Convective heat transfer coefficient. If omitted, a new State is created

    Notes
    -----
    The Reynolds number is evaluated from:

        ``Re = mdot * Dh / (mu * A)``

    The Prandtl number is evaluated from:

        ``Pr = cp * mu / k``

    The Nusselt number is evaluated from:

        ``Nu = 0.023 * Re^0.8 * Pr^(1/3)``

    The convection coefficient is evaluated from:

        ``h = Nu * k / Dh``

    The Stanton number is evaluated from:

        ``St = Nu / (Re * Pr)``

    This correlation assumes single-phase, fully developed turbulent internal
    flow. Fluid properties should be evaluated at the bulk fluid temperature. It
    uses the Colburn form with a fixed Prandtl exponent of `1/3`.

    Recommended validity range:

    * Re >= 10,000
    * 0.7 <= Pr <= 160

    For large temperature differences, developing flow, rough tubes,
    transitional flow, or strong property variation, more advanced correlations
    such as Sieder-Tate or Gnielinski are generally preferred.
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
    Bartz gas-side convective heat transfer coefficient correlation.

    `Bartz` computes the gas-side convective heat transfer coefficient for
    compressible flow in rocket thrust chambers and nozzles. The implementation
    uses chamber transport properties and a mean-temperature correction factor
    to account for local property variation near the wall.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    mass_flow : State
        Local mass flow rate. The absolute value is used
    hydraulic_diameter : State or float
        Local hydraulic diameter or equivalent nozzle diameter
    chamber_specific_heat_cp : State
        Specific heat capacity evaluated at stagnation conditions
    chamber_prandtl_number : State
        Prandtl number evaluated at stagnation conditions
    chamber_dynamic_viscosity : State
        Dynamic viscosity evaluated at stagnation conditions
    local_freestream_density : State
        Local gas density at the evaluation location
    mean_temperature_density : State
        Gas density evaluated at the arithmetic mean temperature
    mean_temperature_dynamic_viscosity : State
        Dynamic viscosity evaluated at the arithmetic mean temperature
    throat_converging_radius : float, optional
        Radius of curvature of the throat converging section
    convection_coefficient : State, optional
        Gas-side convective heat transfer coefficient

    Outputs
    -------
    convection_coefficient : State, optional
        Gas-side convective heat transfer coefficient. If omitted, a new State is created

    Notes
    -----
    The mean temperature is evaluated from:

        ``T_am = (T_g + T_w) / 2``

    The flow area is evaluated from:

        ``A = pi * D^2 / 4``

    The mass flux is evaluated from:

        ``G_m = mdot / A``

    The property correction factor is evaluated from:

        ``sigma = (rho_am / rho)^0.8 * (mu_am / mu0)^0.2``

    The base Bartz coefficient is evaluated from:

        ``X = (0.026 / D^0.2)
        * (mu0^0.2 * Cp0 / Pr0^0.6)
        * G_m^0.8``

    The optional geometric correction is evaluated from:

        ``G = D / rc``

    The final heat transfer coefficient is evaluated from:

        ``h_g = X * sigma * G``

    This implementation follows the classical Bartz engineering correlation
    using chamber stagnation transport properties and the mean-temperature
    correction factor.

    The Bartz correlation is empirical and is most accurate for chemically
    reacting rocket exhaust gases in thrust chambers and converging-diverging
    nozzles.

    Bartz tends to underpredict when radiation is strong, when there is
    significant dissociation or recombination in the boundary layer, or when
    combustion instabilities are significant.

    Bartz tends to overpredict when soot deposition on the walls is significant
    or when combustion is incomplete.
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

    `NaturalConvection` computes a natural-convection heat transfer coefficient
    using a simple Rayleigh-number power-law correlation. The coefficient changes
    depending on whether the estimated Rayleigh number is below or above the
    turbulent transition threshold.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    wall_temperature : State
        Wall temperature
    fluid_temperature : State
        Fluid temperature
    characteristic_length : State or float
        Characteristic length
    fluid_density : State
        Fluid density
    fluid_specific_heat : State
        Fluid specific heat capacity
    fluid_dynamic_viscosity : State
        Fluid dynamic viscosity
    fluid_conductivity : State
        Fluid thermal conductivity
    thermal_expansion_coefficient : State
        Volumetric thermal expansion coefficient
    gravity : State or float, optional
        Gravitational acceleration
    grashof_number : State or float, optional
        Output Grashof number
    prandtl_number : State or float, optional
        Output Prandtl number
    rayleigh_number : State or float, optional
        Output Rayleigh number
    nusselt_number : State or float, optional
        Output Nusselt number

    Outputs
    -------
    convection_coefficient : State, optional
        Convective heat transfer coefficient. If omitted, a new State is created

    Notes
    -----
    Fluid properties should be evaluated at the film temperature:

        ``T_film = 0.5 * (T_w + T_f)``

    For an ideal gas, the volumetric thermal expansion coefficient is commonly
    approximated from:

        ``beta = 1 / T_film``

    The Grashof number is evaluated from:

        ``Gr = g * beta * abs(Tw - Tf) * L^3 * rho^2 / mu^2``

    The Prandtl number is evaluated from:

        ``Pr = Cp * mu / k``

    The Rayleigh number is evaluated from:

        ``Ra = Gr * Pr``

    The Nusselt number is evaluated from:

        ``Nu = c * Ra^n``

    The convection coefficient is evaluated from:

        ``h = Nu * k / L``

    The correlation coefficients are:

        ``Ra < 1e9: c = 0.59, n = 0.25``

        ``Ra >= 1e9: c = 0.13, n = 0.33``

    Recommended validity range:

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

    `ChurchillChu` computes a natural-convection heat transfer coefficient using
    the Churchill-Chu correlation. The component evaluates Grashof, Prandtl,
    Rayleigh, and Nusselt numbers before computing the convection coefficient.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    wall_temperature : State
        Wall temperature
    fluid_temperature : State
        Fluid temperature
    characteristic_length : State or float
        Characteristic length
    fluid_density : State
        Fluid density
    fluid_specific_heat : State
        Fluid specific heat capacity
    fluid_dynamic_viscosity : State
        Fluid dynamic viscosity
    fluid_conductivity : State
        Fluid thermal conductivity
    thermal_expansion_coefficient : State
        Volumetric thermal expansion coefficient
    gravity : State or float, optional
        Gravitational acceleration
    grashof_number : State or float, optional
        Output Grashof number
    prandtl_number : State or float, optional
        Output Prandtl number
    rayleigh_number : State or float, optional
        Output Rayleigh number
    nusselt_number : State or float, optional
        Output Nusselt number

    Outputs
    -------
    convection_coefficient : State, optional
        Convective heat transfer coefficient. If omitted, a new State is created

    Notes
    -----
    Fluid properties should be evaluated at the film temperature:

        ``T_film = 0.5 * (T_w + T_f)``

    For an ideal gas, the volumetric thermal expansion coefficient is commonly
    approximated from:

        ``beta = 1 / T_film``

    The Grashof number is evaluated from:

        ``Gr = g * beta * abs(Tw - Tf) * L^3 * rho^2 / mu^2``

    The Prandtl number is evaluated from:

        ``Pr = Cp * mu / k``

    The Rayleigh number is evaluated from:

        ``Ra = Gr * Pr``

    The Nusselt number is evaluated from:

        ``Nu = (0.825 + 0.387 * Ra^(1/6)
        / (1 + (0.492 / Pr)^(9/16))^(8/27))^2``

    The convection coefficient is evaluated from:

        ``h = Nu * k / L``

    Recommended validity range:

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
