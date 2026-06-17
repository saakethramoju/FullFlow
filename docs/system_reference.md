# System Reference

Long API notes moved out of the source files to keep the implementation compact.


## `State`

**Source:** `System/State.py` (class)


Lightweight value container used throughout FullFlow.

A ``State`` can hold an assignable value or a derived expression. Numeric
states are used by solvers as iteration variables and residual inputs;
object states are useful for passing backend objects between components.

Parameters
----------
value : object, optional
    Initial value. Real numeric values are stored as ``float``. ``None``
    creates an unassigned state.
bounds : tuple[float or None, float or None], optional
    Lower and upper solver bounds. ``None`` maps to an infinite bound.
keep_feasible : bool, optional
    Passed through to SciPy bounded solvers.


## `Gnielinski`

**Source:** `System/Components/ConvectionCoefficients.py` (class)


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


## `Miropolskii`

**Source:** `System/Components/ConvectionCoefficients.py` (class)


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


## `Petukhov`

**Source:** `System/Components/ConvectionCoefficients.py` (class)


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


## `SiederTate`

**Source:** `System/Components/ConvectionCoefficients.py` (class)


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


## `DittusBoelter`

**Source:** `System/Components/ConvectionCoefficients.py` (class)


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


## `Bartz`

**Source:** `System/Components/ConvectionCoefficients.py` (class)


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


## `NaturalConvection`

**Source:** `System/Components/ConvectionCoefficients.py` (class)


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


## `ChurchillChu`

**Source:** `System/Components/ConvectionCoefficients.py` (class)


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


## `Colebrook`

**Source:** `System/Components/FrictionFactors.py` (class)


Colebrook-White Darcy friction factor correlation.

`Colebrook` computes a Darcy friction factor from mass flow, viscosity,
hydraulic diameter, flow area, and roughness. Laminar flow uses a
Poiseuille-number fallback, while turbulent flow uses an explicit
Colebrook-White solution.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
mass_flow : State
    Fluid mass flow rate. The absolute value is used
friction_factor : State
    Output Darcy friction factor
hydraulic_diameter : State or float
    Hydraulic diameter
dynamic_viscosity : State
    Fluid dynamic viscosity
cross_sectional_area : State or float
    Flow cross-sectional area
poiseuille_number : float, optional
    Poiseuille number used for laminar flow
roughness : State or float, optional
    Absolute wall roughness
reynolds_number : State or float, optional
    Output Reynolds number
reynolds_number_threshold : State or float, optional
    Reynolds number threshold for laminar fallback

Outputs
-------
friction_factor : State
    Darcy friction factor
reynolds_number : State or float, optional
    Reynolds number

Notes
-----
The hydraulic-diameter Reynolds number is evaluated from:

    ``Re_Dh = mdot * Dh / (mu * A)``

The effective laminar diameter is evaluated from:

    ``Deff = 16 * Dh / Po``

The effective Reynolds number is evaluated from:

    ``Re_eff = mdot * Deff / (mu * A)``

For laminar flow, the Darcy friction factor is evaluated from:

    ``f = 4 * Po / Re_Dh``

For turbulent flow, the explicit Colebrook-White solution is used.

The Poiseuille number input is only used for the incompressible laminar
fallback.


## `Churchill`

**Source:** `System/Components/FrictionFactors.py` (class)


Churchill Darcy friction factor correlation.

`Churchill` computes a Darcy friction factor from mass flow, viscosity,
hydraulic diameter, flow area, roughness, and Poiseuille number. The
Churchill correlation provides a smooth transition across laminar,
transitional, and turbulent Reynolds numbers.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
mass_flow : State
    Fluid mass flow rate. The absolute value is used
friction_factor : State
    Output Darcy friction factor
hydraulic_diameter : State or float
    Hydraulic diameter
dynamic_viscosity : State
    Fluid dynamic viscosity
cross_sectional_area : State or float
    Flow cross-sectional area
roughness : State or float, optional
    Absolute wall roughness
poiseuille_number : float, optional
    Poiseuille number used for incompressible laminar flow
reynolds_number : State or float, optional
    Output Reynolds number

Outputs
-------
friction_factor : State
    Darcy friction factor
reynolds_number : State or float, optional
    Reynolds number

Notes
-----
The effective laminar diameter is evaluated from:

    ``Deff = 16 * Dh / Po``

The Reynolds number is evaluated from:

    ``Re = mdot * Deff / (mu * A)``

The relative roughness is evaluated from:

    ``relative_roughness = roughness / Deff``

The Churchill auxiliary terms are evaluated from:

    ``A = (2.457 * log(1 / ((7 / Re)^0.9 + 0.27 * relative_roughness)))^16``

    ``B = (37530 / Re)^16``

The Darcy friction factor is evaluated from:

    ``f = 8 * ((8 / Re)^12 + (A + B)^(-1.5))^(1 / 12)``

The Poiseuille number input is only used for incompressible laminar flow.


## `PetukhovFriction`

**Source:** `System/Components/FrictionFactors.py` (class)


Petukhov smooth-pipe turbulent Darcy friction factor correlation.

`PetukhovFriction` computes a Darcy friction factor from mass flow,
viscosity, hydraulic diameter, and flow area. Laminar flow uses a
Poiseuille-number fallback, while turbulent flow uses the Petukhov
smooth-pipe correlation.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
mass_flow : State
    Fluid mass flow rate. The absolute value is used
friction_factor : State
    Output Darcy friction factor
hydraulic_diameter : State or float
    Hydraulic diameter
dynamic_viscosity : State
    Fluid dynamic viscosity
cross_sectional_area : State or float
    Flow cross-sectional area
poiseuille_number : float, optional
    Poiseuille number used for laminar flow
reynolds_number : State or float, optional
    Output Reynolds number
reynolds_number_threshold : State or float, optional
    Reynolds number threshold for laminar fallback

Outputs
-------
friction_factor : State
    Darcy friction factor
reynolds_number : State or float, optional
    Reynolds number

Notes
-----
The Reynolds number is evaluated from:

    ``Re = mdot * Dh / (mu * A)``

For laminar flow, the Darcy friction factor is evaluated from:

    ``f = 4 * Po / Re``

For turbulent smooth-pipe flow, the Darcy friction factor is evaluated from:

    ``f = (0.79 * ln(Re) - 1.64)^(-2)``

This correlation returns the Darcy friction factor. The roughness input is
intentionally omitted because this correlation does not include relative
roughness. Use Colebrook or Churchill when wall roughness should be modeled.


## `CavitatingVenturi`

**Source:** `System/Components/GeneralFlow.py` (class)


Cavitating liquid venturi model.

`CavitatingVenturi` computes mass flow through a liquid venturi using a
noncavitating restriction model or a cavitating venturi model. The active
mode is selected using a critical downstream-to-upstream pressure ratio.

In cavitating mode, the throat pressure is assumed to be pinned to the
vapor pressure corresponding to the upstream fluid state. If upstream
temperature and critical temperature are both assigned, cavitation is
disabled above the critical temperature.

Cavitation onset and stable cavitating flow are not identical. Incipient
cavitation begins when the throat pressure first reaches saturation
pressure, while fully established cavitating flow depends on geometry and
empirical behavior.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
upstream_pressure : State
    Upstream pressure
downstream_pressure : State
    Downstream pressure
density : State
    Fluid density
throat_area : float
    Venturi throat area
vapor_pressure : State
    Fluid vapor pressure
critical_pressure_ratio : float, optional
    Pressure ratio below which cavitating mode is activated
cavitating_discharge_coefficient : float, optional
    Discharge coefficient used in cavitating mode
noncavitating_discharge_coefficient : float, optional
    Discharge coefficient used in noncavitating mode
upstream_temperature : State, optional
    Upstream fluid temperature
critical_temperature : State, optional
    Fluid critical temperature

Outputs
-------
mass_flow : State, optional
    Computed venturi mass flow rate
is_cavitating : bool, optional
    Whether cavitating mode is active

Notes
-----
Noncavitating mass flow is evaluated from:

    ``mass_flow = sign(P1 - P2) * Cd_noncav * A_t * sqrt(2 * rho * abs(P1 - P2))``

Cavitating mass flow is evaluated from:

    ``mass_flow = Cd_cav * A_t * sqrt(2 * rho * (P1 - vapor_pressure))``

Cavitating mode is activated when:

    ``downstream_pressure / upstream_pressure < critical_pressure_ratio``


## `SeriesCdA`

**Source:** `System/Components/GeneralFlow.py` (class)


Equivalent effective area for restrictions in series.

`SeriesCdA` combines multiple effective flow areas into a single equivalent
effective area. This is useful when several restrictions are arranged in
series and should be represented as one equivalent restriction.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
effective_areas : list[State or float]
    Effective areas connected in series

Outputs
-------
effective_area : State, optional
    Equivalent effective area

Notes
-----
Series effective area is evaluated from:

    ``1 / effective_area_eq^2 = sum(1 / effective_area_i^2)``


## `ParallelCdA`

**Source:** `System/Components/GeneralFlow.py` (class)


Equivalent effective area for restrictions in parallel.

`ParallelCdA` combines multiple effective flow areas into a single
equivalent effective area. This is useful when several restrictions are
arranged in parallel and should be represented as one equivalent
restriction.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
effective_areas : list[State or float]
    Effective areas connected in parallel

Outputs
-------
effective_area : State, optional
    Equivalent effective area

Notes
-----
Parallel effective area is evaluated from:

    ``effective_area_eq = sum(effective_area_i)``


## `RectanglePoiseuille`

**Source:** `System/Components/GeneralFlow.py` (class)


Poiseuille number correlation for rectangular ducts.

`RectanglePoiseuille` computes an approximate Poiseuille number for a
rectangular duct from its height and width. The result can be used by
laminar duct-flow pressure-loss or friction-factor calculations.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
height : float
    Rectangle height
width : float
    Rectangle width

Outputs
-------
poiseuille_number : State, optional
    Computed Poiseuille number

Notes
-----
The aspect ratio is evaluated from the smaller half-dimension divided by
the larger half-dimension:

    ``x = b / a``

The Poiseuille number is evaluated from:

    ``Po = A0 + A1 * x + A2 * x^2 + A3 * x^3 + A4 * x^4``


## `EllipsePoiseuille`

**Source:** `System/Components/GeneralFlow.py` (class)


Poiseuille number correlation for elliptical ducts.

`EllipsePoiseuille` computes an approximate Poiseuille number for an
elliptical duct from its semi-major and semi-minor axes. The result can be
used by laminar duct-flow pressure-loss or friction-factor calculations.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
semi_major_axis : float
    Ellipse semi-major axis
semi_minor_axis : float
    Ellipse semi-minor axis

Outputs
-------
poiseuille_number : State, optional
    Computed Poiseuille number

Notes
-----
The aspect ratio is evaluated from the smaller semi-axis divided by the
larger semi-axis:

    ``x = b / a``

The Poiseuille number is evaluated from:

    ``Po = A0 + A1 * x + A2 * x^2 + A3 * x^3 + A4 * x^4``


## `CircularAnnulusPoiseuille`

**Source:** `System/Components/GeneralFlow.py` (class)


Poiseuille number correlation for circular annuli.

`CircularAnnulusPoiseuille` computes an approximate Poiseuille number for a circular
annulus from its inner and outer diameters. The result can be used by
laminar annular duct-flow pressure-loss or friction-factor calculations.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
inner_diameter : float
    Annulus inner diameter
outer_diameter : float
    Annulus outer diameter

Outputs
-------
poiseuille_number : State, optional
    Computed Poiseuille number

Notes
-----
The diameter ratio is evaluated from:

    ``x = inner_diameter / outer_diameter``

For small diameter ratios, the Poiseuille number is evaluated from:

    ``Po = A0 * x^A1``

Otherwise, the Poiseuille number is evaluated from:

    ``Po = A0 + A1 * x + A2 * x^2 + A3 * x^3 + A4 * x^4``


## `HydraulicDiameter`

**Source:** `System/Components/GeneralFlow.py` (class)


Hydraulic diameter from flow area and wetted perimeter.

`HydraulicDiameter` computes hydraulic diameter from cross-sectional flow
area and wetted perimeter. The result is commonly used as the characteristic
diameter for Reynolds number, Nusselt number, and duct-flow correlations.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
cross_sectional_area : State or float
    Flow cross-sectional area
wetted_perimeter : State or float
    Wetted perimeter

Outputs
-------
hydraulic_diameter : State, optional
    Hydraulic diameter

Notes
-----
Hydraulic diameter is evaluated from:

    ``hydraulic_diameter = 4 * cross_sectional_area / wetted_perimeter``


## `Conduction`

**Source:** `System/Components/HeatTransfer.py` (class)


One-dimensional conduction heat transfer between two temperature nodes.

`Conduction` computes conductive heat transfer between two thermal nodes
using a one-dimensional Fourier-law resistance. Positive heat rate means
heat is added to `temperature1` from `temperature2`.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
temperature1 : State
    Receiving-side temperature
temperature2 : State
    Source-side temperature
thermal_conductivity : State
    Thermal conductivity
length : float
    Conduction length
conductive_area : float
    Conductive area

Outputs
-------
heat_rate : State, optional
    Conductive heat transfer rate

Notes
-----
Conductive heat transfer is evaluated from:

    ``heat_rate = thermal_conductivity * conductive_area
    / length * (temperature2 - temperature1)``


## `Radiation`

**Source:** `System/Components/HeatTransfer.py` (class)


Diffuse-gray radiation exchange between two temperature nodes.

`Radiation` computes radiative heat transfer between two diffuse-gray
surfaces using emissivities, radiating areas, and a view factor. Positive
heat rate indicates net radiative heat transfer from `temperature2` to
`temperature1`.

This component can be used for surface-to-surface radiation or vacuum
jacketed tube radiation.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
temperature1 : State
    Receiving-side surface temperature
temperature2 : State
    Source-side surface temperature
emissivity1 : float
    Receiving-side surface emissivity
emissivity2 : float
    Source-side surface emissivity
radiative_area1 : float
    Receiving-side radiative area
radiative_area2 : float, optional
    Source-side radiative area
view_factor12 : float, optional
    View factor from surface 1 to surface 2

Outputs
-------
heat_rate : State, optional
    Radiative heat transfer rate

Notes
-----
The radiation denominator is evaluated from:

    ``denominator = (1 - emissivity1) / (emissivity1 * radiative_area1)
    + 1 / (radiative_area1 * view_factor12)
    + (1 - emissivity2) / (emissivity2 * radiative_area2)``

Radiative heat transfer is evaluated from:

    ``heat_rate = sigma * (temperature2^4 - temperature1^4)
    / denominator``


## `AmbientRadiation`

**Source:** `System/Components/HeatTransfer.py` (class)


Radiation exchange between a surface and an ambient enclosure.

`AmbientRadiation` computes radiative heat transfer between a solid surface
and a surrounding ambient enclosure. Positive heat rate indicates net
radiative heat transfer to the solid surface from the ambient surroundings.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
solid_temperature : State
    Solid surface temperature
ambient_temperature : State or float
    Ambient enclosure temperature
emissivity : State or float
    Solid surface emissivity
radiative_area : State or float
    Radiative area
ambient_emissivity : State or float, optional
    Ambient enclosure emissivity

Outputs
-------
heat_rate : State, optional
    Radiative heat transfer rate

Notes
-----
The radiation denominator is evaluated from:

    ``denominator = 1 / emissivity + 1 / ambient_emissivity - 1``

Radiative heat transfer is evaluated from:

    ``heat_rate = sigma * radiative_area
    * (ambient_temperature^4 - solid_temperature^4)
    / denominator``


## `Convection`

**Source:** `System/Components/HeatTransfer.py` (class)


Convective heat transfer between a surface and a fluid.

`Convection` computes heat transfer between a surface and a surrounding
fluid using a prescribed convection coefficient. Positive heat rate means
heat is added to the surface from the fluid.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
surface_temperature : State
    Surface temperature
fluid_temperature : State or float
    Fluid temperature
convective_area : State or float
    Convective area
convection_coefficient : State or float
    Convective heat transfer coefficient

Outputs
-------
heat_rate : State, optional
    Convective heat transfer rate

Notes
-----
Convective heat transfer is evaluated from:

    ``heat_rate = convection_coefficient * convective_area
    * (fluid_temperature - surface_temperature)``


## `TemperatureRecoveryFactor`

**Source:** `System/Components/HeatTransfer.py` (class)


Compressible boundary-layer temperature recovery factor.

`TemperatureRecoveryFactor` computes the recovery factor used to estimate
adiabatic wall temperature in compressible boundary-layer heat transfer. If
no Prandtl number is provided, the recovery factor defaults to one.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
prandtl_number : State, optional
    Prandtl number
turbulent : bool, optional
    Whether to use the turbulent boundary-layer exponent

Outputs
-------
recovery_factor : State, optional
    Temperature recovery factor

Notes
-----
The adiabatic wall temperature relation is:

    ``T_aw = T + r * (T0 - T)``

For turbulent boundary layers, the recovery factor is evaluated from:

    ``r = Pr^(1/3)``

For laminar boundary layers, the recovery factor is evaluated from:

    ``r = Pr^(1/2)``

If no Prandtl number is provided, the recovery factor is:

    ``r = 1``


## `AdiabaticWallTemperature`

**Source:** `System/Components/HeatTransfer.py` (class)


Adiabatic wall temperature for compressible flow.

`AdiabaticWallTemperature` computes the temperature an insulated wall would
attain when exposed to a compressible flow, using total temperature, static
temperature, and a recovery factor.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
total_temperature : State
    Total temperature
static_temperature : State
    Static temperature
recovery_factor : State
    Temperature recovery factor

Outputs
-------
adiabatic_wall_temperature : State, optional
    Adiabatic wall temperature

Notes
-----
Adiabatic wall temperature is evaluated from:

    ``T_aw = T + r * (T0 - T)``

where `T_aw` is adiabatic wall temperature, `T` is static temperature, `T0`
is total temperature, and `r` is the recovery factor.


## `Lookup`

**Source:** `System/Components/Lookup.py` (class)


Wrap a function, class, or external model as a FullFlow component.

``Lookup`` is the bridge between a FullFlow network and external code such
as ThermoProp objects, CoolProp wrappers, interpolation functions, property
packages, correlations, or user-defined callables. Inputs may be constants,
:class:`State` objects, derived states, other ``Lookup`` objects, or
attributes from other lookups.

The wrapped callable is evaluated during network state evaluation. The
returned object is stored in ``output`` and its attributes are exposed as
state-like proxies. For example, after creating ``gas = Lookup(...)``,
expressions like ``gas.pressure``, ``gas.temperature``, and
``gas.enthalpy`` can be passed directly into other components.

Priority Inputs
---------------
Some property packages allow multiple state-input pairs, but only one value
from each interchangeable group should be passed at a time. ``priority``
handles this without requiring full input-mode definitions.

For each priority group, only the first available value is passed to the
wrapped callable. A value may come from an existing lookup input or from an
already evaluated output attribute when the callable accepts that attribute
name as an input. All inputs not listed in a priority group are passed
normally.

Example: initialize an ideal gas with pressure-temperature, then switch to
pressure-enthalpy once enthalpy becomes a solver variable::

    gas = Lookup(
        "Gas",
        network,
        IdealGas,
        fluid="gn2",
        pressure=3e7,
        temperature=300,
        priority=("enthalpy", "temperature"),
    )

Initial evaluation passes::

    IdealGas(fluid="gn2", pressure=3e7, temperature=300)

After ``gas.enthalpy`` becomes assigned, evaluation passes::

    IdealGas(fluid="gn2", pressure=..., enthalpy=...)

Multiple independent priority groups are supported::

    priority=[
        ("enthalpy", "temperature"),
        ("density", "pressure"),
    ]

Common Usage
------------
Property object or constructor::

    fuel = Lookup(
        "Fuel",
        network,
        Propellant,
        "rp-1",
        temperature=298.15,
    )

Chained lookup::

    reactants = Lookup(
        "Reactants",
        network,
        Reactants,
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=2.0,
    )

    eq = Lookup(
        "Combustion",
        network,
        Equilibrium,
        reactants=reactants,
        pressure=chamber.pressure,
    )

Parameters
----------
name : str
    Component name shown in reports and diagnostics.
network : Network
    Network that owns this lookup.
callable_ : Callable[..., T]
    Function, class, constructor, or callable object to evaluate.
*args : Any
    Positional arguments passed to ``callable_``. Nested FullFlow states and
    lookup attributes are resolved before evaluation.
output : State, optional
    State used to store the returned object. Most users can omit this.
evaluate_on_set : bool, default=False
    If True, changing an input immediately evaluates the lookup.
strict_inputs : bool, default=False
    If True, assigning an unknown attribute raises instead of creating an
    output guess.
strict_outputs : bool, default=False
    Reserved for stricter output checking.
wrap_errors : bool, default=False
    If True, exceptions from the wrapped callable are wrapped with this
    lookup's name.
evaluate_in_pre_evaluation : bool, default=True
    If True, evaluate during the network pre-evaluation pass.
lazy : bool, optional
    Convenience inverse of ``evaluate_in_pre_evaluation``.
defer_until_inputs_available : bool, default=True
    If True, unavailable inputs defer evaluation instead of raising
    immediately.
cache : bool, default=True
    If True, skip evaluation when resolved inputs and callable structure are
    unchanged.
cache_tol : float, default=0.0
    Optional numeric tolerance used when building cache fingerprints.
reuse_existing : bool, default=True
    If True, try to update an existing output object via ``update(...)``
    before constructing a new one.
memo_size : int, default=1
    Number of recent input/output combinations to retain.
output_guesses : dict[str, Any], optional
    Initial guesses for output attributes before first evaluation.
input_guesses : dict[str, Any], optional
    Fallback values for temporarily unavailable inputs.
priority : tuple[str, ...] or sequence of tuple[str, ...], optional
    Input priority group or groups. For each group, only the first available
    input is passed.
**kwargs : Any
    Keyword arguments passed to ``callable_``. Keyword arguments also become
    named lookup inputs.

Notes
-----
``Lookup`` intentionally does not call ``Component.setup()`` because it must
preserve raw positional and keyword dependency objects. Calling setup would
wrap constructor arguments in ways that break dynamic lookup inputs.


## `Map1D`

**Source:** `System/Components/Maps.py` (class)


Generic one-dimensional map lookup.

`Map1D` interpolates one or more output values from a single independent
variable. The map input may be a list, tuple, NumPy array, pandas Series,
or any array-like object accepted by `np.asarray`.

Multiple output maps can be evaluated simultaneously from the same
independent variable.

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
x_value : State
    Current independent-variable value
x_map : array-like
    Independent-variable map coordinates
y_maps : dict[str, array-like]
    Dictionary of dependent-variable maps

Outputs
-------
<map name> : State
    One output State is automatically created for every key in `y_maps`.

Notes
-----
Each output map is evaluated from:

    ``y = interp(x, x_map, y_map)``

where `interp` is linear interpolation.

All maps are automatically sorted by `x_map` during initialization.


## `Map2D`

**Source:** `System/Components/Maps.py` (class)


Generic two-dimensional map lookup.

`Map2D` interpolates one or more output maps from two input values. Inputs
can be lists, tuples, NumPy arrays, pandas Series, or any array-like object
accepted by `np.asarray`.

The `z_maps` values must be 2D arrays with shape:

    (len(y_map), len(x_map))

Example
-------
PumpMap = Map2D(
    "Pump Map",
    network,
    x_value=volumetric_flow,
    y_value=rotor_speed,
    x_map=[0.01, 0.02, 0.03],
    y_map=[10000, 20000, 30000],
    z_maps={
        "head_rise": [
            [100, 90, 80],
            [150, 140, 120],
            [200, 180, 160],
        ],
        "torque": [
            [1.0, 1.2, 1.4],
            [2.0, 2.4, 2.8],
            [3.0, 3.6, 4.2],
        ],
    },
)

Outputs are created automatically:

    PumpMap.head_rise
    PumpMap.torque


## `Solid`

**Source:** `System/Components/Nodes.py` (class)


Lumped solid thermal node.

`Solid` represents a lumped-capacitance thermal mass whose temperature is
solved from a steady-state energy balance. The component is intended for
conjugate heat transfer networks where conduction, convection, radiation,
and other thermal components contribute heat to a common solid node.

Positive heat rates add energy to the solid. Negative heat rates remove
energy from the solid.

Residuals
---------
energy_balance : float
    Enforces steady-state thermal equilibrium.

    ``heat_rate = 0``

    The heat rate is typically formed by summing all heat transfer
    mechanisms connected to the node.

Relations
---------
biot_number : State
    Computes the Biot number used to assess the validity of the
    lumped-capacitance assumption.

    ``Bi = h * Lc / k``

    where:

    * `Bi` is the Biot number
    * `h` is the convection coefficient
    * `Lc` is the characteristic length
    * `k` is the thermal conductivity

    As a general guideline, `Bi < 0.1` indicates that the
    lumped-temperature assumption is likely reasonable.

Iteration Variables
-------------------
temperature : State
    Solid temperature

Parameters
----------
name : str
    Component name
network : Network
    Network that owns this component
temperature : State
    Solid temperature
mass : float, optional
    Solid mass
specific_heat : State, optional
    Solid specific heat capacity
characteristic_length : State or float, optional
    Characteristic length used for Biot number evaluation
thermal_conductivity : State or float, optional
    Solid thermal conductivity used for Biot number evaluation
convection_coefficient : State or float, optional
    Representative convection coefficient used for Biot number evaluation
biot_number : State, optional
    Output Biot number
heat_rate : State or float, optional
    Net heat rate into the solid node. Positive values add heat to the
    solid. Defaults to 0.


## `Volume`

**Source:** `System/Components/Nodes.py` (class)


Lumped steady-state fluid control volume.

`Volume` enforces mass conservation. If `enthalpy` is provided, it also
enforces steady-state energy conservation.

Modes
-----
Mass-only mode
    Used when `enthalpy` is not provided.

    Residual:

        mass_flow_in - mass_flow_out = 0

    Iteration variable:

        pressure

Mass + energy mode
    Used when `enthalpy` is provided.

    Residuals:

        mass_flow_in - mass_flow_out = 0

        mass_flow_in * total_enthalpy_in
        - mass_flow_out * h_out
        + heat_rate = 0

    where `h_out` is `total_enthalpy_out` if assigned, otherwise `enthalpy`.

    Iteration variables:

        pressure
        enthalpy

Sign Convention
---------------
`mass_flow_in` is positive into the volume.

`mass_flow_out` is positive out of the volume.

`heat_rate` is positive into the volume.

Parameters
----------
name : str
    Component name.
network : Network
    Network that owns this component.
pressure : State
    Volume pressure.
volume : State or float
    Physical control volume. Required.
enthalpy : State or float, optional
    Volume/static outlet enthalpy. Providing this turns on the energy
    residual.
total_enthalpy_in : State or float, optional
    Total specific enthalpy entering the volume. Required when `enthalpy`
    is provided.
total_enthalpy_out : State or float, optional
    Total specific enthalpy leaving the volume. If omitted, `enthalpy` is
    used as the outlet enthalpy.
heat_rate : State or float, optional
    Net heat transfer rate into the volume. Defaults to zero if omitted.
temperature : State, optional
    Stored volume temperature.
density : State, optional
    Stored volume density.
internal_energy : State, optional
    Stored volume internal energy.
mass_flow_in : State or float, optional
    Mass flow entering the volume.
mass_flow_out : State or float, optional
    Mass flow leaving the volume.


## `ModelOption`

**Source:** `System/Model.py` (class)


Deferred component option used by `Model`.

`ModelOption` stores enough information to build either one component or a
group of components later. Unlike a normal `Component`, a `ModelOption` does
not register itself with a `Network` when it is created.

Model options are useful when a solver should be able to try alternate
component implementations without constructing all of them at once.

Parameters
----------
name : str
    Model option name
*model_options : ModelOption
    Grouped model options
component_class : type, optional
    Component class to construct
kwargs : dict, optional
    Keyword arguments passed to the component constructor
components : list[ModelOption], optional
    Grouped model options

Notes
-----
A single-component option stores a component class and constructor keyword
arguments:

    ``ModelOption("Choked", component_class=ChokedFlow, kwargs={...})``

A grouped option stores multiple `ModelOption` objects that should be built
and removed together:

    ``ModelOption("Full Model", option1, option2, option3)``

A `ModelOption` must define either a single `component_class` or grouped
component options, but not both.

Grouped options return a list of components when built.


## `Model`

**Source:** `System/Model.py` (class)


Collection of alternative component implementations.

`Model` stores one or more `ModelOption` objects and builds one selected
option into a `Network`. Only the active option is converted into real
components.

Models are useful for trying alternate physical regimes, component
formulations, or grouped component implementations between solve attempts.

Parameters
----------
name : str
    Model name
network : Network
    Network the selected option will be added to
*model_options : ModelOption
    Model options passed positionally
components : list[ModelOption], optional
    Model options passed by keyword
order : list[str], optional
    Option names defining the try order

Notes
-----
`Model` does not build automatically during initialization. The selected
option is built when `build()` is called:

    ``model.build("Choked")``

If no option name is supplied, `build()` uses the first option in `order`:

    ``model.build()``

The active option can be removed from the network with:

    ``model.clear()``

The active option can be replaced with another option using:

    ``model.replace("Unchoked")``

The next option in the try order can be built with:

    ``model.build_next()``

Switching options should happen between solve attempts, not during a Newton
iteration.

Option names must be unique, and every name in `order` must correspond to a
valid option.
