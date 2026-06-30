"""
Shock-free equilibrium nozzle example using an SP combustion-products map.

Physical layout
---------------

    RP-1 / LOX chamber equilibrium state
    Pc, MR, h0, s0
          |
          v

    +-------------------------+
    |      Injector Face      |
    |  HP equilibrium state   |
    |  Pc = fixed in example  |
    +-------------------------+
          |
          |  Converging Section
          |  AdiabaticFlow:
          |      h0 = h_chamber
          |      mdot = rho_t * A_t * u_t
          v

    +-------------------------+
    |        Throat Node      |
    |  SP(Pt, s0, MR) lookup  |
    |  Pt solved by mass      |
    |  conservation           |
    +-------------------------+
          |
          |  Diverging Section
          |  AdiabaticFlow:
          |      h_t + 0.5*u_t^2 = h_e + 0.5*u_e^2
          |      mdot_t = mdot_e
          v

    +-------------------------+
    |        Exit Plane       |
    |  SP(Pe, s0, MR) lookup  |
    |  Pe solved by choked    |
    |  nozzle closure         |
    +-------------------------+
          |
          v

    Ambient pressure, Pamb
    Pamb is not forced onto Pe for this choked no-shock example.
    Pamb only appears in the pressure-thrust term.

Model notes
-----------
This example assumes the nozzle is shock-free and isentropic inside the gas path.

The chamber state is generated with HP equilibrium. The nozzle stations use
SP lookups with the chamber entropy.

For the current 400 psia chamber / 1 atm ambient setup, the nozzle is choked.
The internal nozzle closure is therefore:

    throat_mach = 1

The exit pressure is solved internally. It is not set equal to ambient pressure.
If Pe and Pamb do not match, that mismatch is handled outside the nozzle as
plume expansion/compression, or by a shock model in a more advanced example.
"""


from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Constants and options
# ---------------------------------------------------------------------------

psia_to_pa = 6894.76
g0 = 9.80665

filename = "equilibrium_nozzle"

# Set to True only when the SP map needs to be regenerated.
# Normal example runs should use the existing HDF5 map.
make_map = False


# ---------------------------------------------------------------------------
# Thermochemistry map function
# ---------------------------------------------------------------------------

def rp1_lox_sp_map(pressure, entropy, mixture_ratio, guess_temperature):
    """
    Equilibrium combustion-products SP map for RP-1 / LOX.

    Inputs
    ------
    pressure:
        Static pressure at the nozzle station.

    entropy:
        Static entropy at the nozzle station. For this shock-free isentropic
        example, this is the chamber entropy.

    mixture_ratio:
        Oxidizer-to-fuel mixture ratio.

    guess_temperature:
        Temperature guess used by the SP equilibrium solve.

    Returns
    -------
    dict
        Properties needed by the nozzle model:
            gamma
            gas constant
            density
            enthalpy
            entropy
            speed of sound
            temperature
    """

    fuel = Propellant("rp-1", temperature=298.15)
    oxidizer = Propellant("lox", temperature=90.17)

    reactants = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=reactants,
        mode="sp",
        pressure=pressure,
        entropy=entropy,
        guess_temperature=guess_temperature,
    )

    return {
        "specific_heat_ratio": eq.gamma,
        "gas_constant": eq.gas_constant,
        "density": eq.density,
        "enthalpy": eq.enthalpy,
        "entropy": eq.entropy,
        "speed_of_sound": eq.speed_of_sound,
        "temperature": eq.temperature,
    }


# ---------------------------------------------------------------------------
# Chamber / injector-face state
# ---------------------------------------------------------------------------

# Mixture ratio and chamber pressure are fixed in this standalone nozzle
# example. In a full engine network, chamber pressure would usually be solved
# by injector, chamber, and feed-system balances.
mixture_ratio = State(2.3)
chamber_pressure = State(400 * psia_to_pa)

# Ambient pressure is not used to force the internal exit pressure for the
# choked no-shock solution. It is used later in the pressure-thrust term.
ambient_pressure = State(101325.0)

fuel = Propellant("rp-1", temperature=298.15)
oxidizer = Propellant("lox", temperature=90.17)

Props = Reactants(
    fuels=fuel,
    oxidizers=oxidizer,
    mixture_ratio=mixture_ratio.value,
)

# HP equilibrium gives the injector-face / chamber stagnation-like state.
# For this simplified nozzle model, chamber velocity is assumed negligible, so
# chamber static enthalpy is used as the nozzle total enthalpy.
InjectorFace = Equilibrium(
    reactants=Props,
    mode="hp",
    pressure=chamber_pressure.value,
)


# ---------------------------------------------------------------------------
# Optional SP map generation
# ---------------------------------------------------------------------------

if make_map:
    generate_map(
        filename=filename,
        group="products_sp",
        axes=[
            Axis.log(
                "pressure",
                start=1 * psia_to_pa,
                stop=500 * psia_to_pa,
                count=18,
                units="Pa",
            ),
            Axis.linear(
                "entropy",
                start=0.90 * InjectorFace.entropy,
                stop=1.10 * InjectorFace.entropy,
                count=10,
                units="J/kg-K",
            ),
            Axis.linear(
                "mixture_ratio",
                start=1,
                stop=4,
                count=9,
                units="",
            ),
        ],
        constants={
            "guess_temperature": InjectorFace.temperature,
        },
        evaluate=rp1_lox_sp_map,
        overwrite=True,
        raise_errors=True,
    )


# ---------------------------------------------------------------------------
# Network setup
# ---------------------------------------------------------------------------

EquilibriumNozzle = Network("Equilibrium Nozzle")


# ---------------------------------------------------------------------------
# Nozzle geometry and station guesses
# ---------------------------------------------------------------------------

# These are solver guesses, not prescribed final answers.
throat_pressure = State(300 * psia_to_pa)
exit_pressure = State(10 * psia_to_pa)

# The shock-free isentropic nozzle uses chamber entropy at every station.
throat_entropy = State(InjectorFace.entropy)

# Nozzle geometry.
# Areas are in m^2. The conversion 1550 approximately converts in^2 to m^2.
throat_area = 6 / 1550
expansion_ratio = 6
exit_area = throat_area * expansion_ratio


# ---------------------------------------------------------------------------
# Thermodynamic station maps
# ---------------------------------------------------------------------------

# Throat state from SP(Pt, s0, MR).
ThroatMap = Map.from_hdf5(
    "Throat Map",
    EquilibriumNozzle,
    filename=filename,
    group="products_sp",
    inputs={
        "pressure": throat_pressure,
        "entropy": throat_entropy,
        "mixture_ratio": mixture_ratio,
    },
)

# Exit state from SP(Pe, s0, MR).
ExitMap = Map.from_hdf5(
    "Exit Map",
    EquilibriumNozzle,
    filename=filename,
    group="products_sp",
    inputs={
        "pressure": exit_pressure,
        "entropy": throat_entropy,
        "mixture_ratio": mixture_ratio,
    },
)


# ---------------------------------------------------------------------------
# Converging section: chamber to throat
# ---------------------------------------------------------------------------

# The upstream density and upstream area are intentionally omitted here.
# That tells AdiabaticFlow to treat InjectorFace.enthalpy as total enthalpy:
#
#     h0 = h_chamber
#
# Then the branch computes:
#
#     u_t = sqrt(2*(h0 - h_t))
#     mdot = rho_t * A_t * u_t
Conv = AdiabaticFlow(
    "Converging Section",
    EquilibriumNozzle,
    upstream_static_enthalpy=InjectorFace.enthalpy,
    downstream_static_enthalpy=ThroatMap.enthalpy,
    downstream_density=ThroatMap.density,
    downstream_cross_sectional_area=throat_area,
)


# ---------------------------------------------------------------------------
# Throat mass-conservation node
# ---------------------------------------------------------------------------

# This algebraic Volume supplies the unknown outflow state Throat.mass_flow_out.
# Its balance is:
#
#     Conv.mass_flow - Throat.mass_flow_out = 0
#
# That balance solves throat_pressure.
Throat = Volume(
    "Throat",
    EquilibriumNozzle,
    pressure=throat_pressure,
    mass_flow_in=Conv.mass_flow,
)


# ---------------------------------------------------------------------------
# Diverging section: throat to exit
# ---------------------------------------------------------------------------

# This section uses both station areas and densities, so AdiabaticFlow applies:
#
#     h_t + 0.5*u_t^2 = h_e + 0.5*u_e^2
#     rho_t*u_t*A_t = rho_e*u_e*A_e
#
# The mass flow is tied to Throat.mass_flow_out, so the throat Volume and
# diverging branch must agree on the same nozzle flow rate.
Div = AdiabaticFlow(
    "Diverging Section",
    EquilibriumNozzle,
    upstream_static_enthalpy=ThroatMap.enthalpy,
    downstream_static_enthalpy=ExitMap.enthalpy,
    upstream_density=ThroatMap.density,
    downstream_density=ExitMap.density,
    upstream_cross_sectional_area=throat_area,
    downstream_cross_sectional_area=exit_area,
    mass_flow=Throat.mass_flow_out,
)


# ---------------------------------------------------------------------------
# Derived velocities and Mach numbers
# ---------------------------------------------------------------------------

throat_velocity = Conv.mass_flow / (ThroatMap.density * throat_area)
exit_velocity = Div.mass_flow / (ExitMap.density * exit_area)

throat_mach = abs(throat_velocity) / ThroatMap.speed_of_sound
exit_mach = abs(exit_velocity) / ExitMap.speed_of_sound


# ---------------------------------------------------------------------------
# Jet boundary / nozzle closure
# ---------------------------------------------------------------------------

# For this specific example, the chamber-to-ambient pressure ratio is low enough
# that the nozzle is choked. The no-shock choked closure is:
#
#     throat_mach = 1
#
# This closure solves exit_pressure. Ambient pressure is not forced onto the
# internal exit pressure in the choked no-shock branch.
JetBoundary = Balance(
    "Jet Boundary",
    EquilibriumNozzle,
    variable=exit_pressure,
    function=throat_mach - 1.0,
)


# ---------------------------------------------------------------------------
# Thrust and performance
# ---------------------------------------------------------------------------

# Ideal one-dimensional thrust:
#
#     F = mdot*ue + (Pe - Pamb)*Ae
#
# The pressure term can be positive or negative depending on whether the nozzle
# is underexpanded or overexpanded relative to ambient.
thrust = Conv.mass_flow * exit_velocity + (exit_pressure - ambient_pressure) * exit_area

# Specific impulse:
#
#     Isp = F/(mdot*g0)
specific_impulse = thrust / (Conv.mass_flow * g0)


# ---------------------------------------------------------------------------
# Tracked outputs
# ---------------------------------------------------------------------------

EquilibriumNozzle.track("Chamber Pressure [psia]", InjectorFace.pressure / psia_to_pa)
EquilibriumNozzle.track("Throat Pressure [psia]", throat_pressure / psia_to_pa)
EquilibriumNozzle.track("Exit Pressure [psia]", exit_pressure / psia_to_pa)
EquilibriumNozzle.track("Ambient Pressure [psia]", ambient_pressure / psia_to_pa)

EquilibriumNozzle.track("Chamber Temperature [K]", InjectorFace.temperature)
EquilibriumNozzle.track("Throat Temperature [K]", ThroatMap.temperature)
EquilibriumNozzle.track("Exit Temperature [K]", ExitMap.temperature)

EquilibriumNozzle.track("Chamber Gamma", InjectorFace.gamma)
EquilibriumNozzle.track("Throat Gamma", ThroatMap.specific_heat_ratio)
EquilibriumNozzle.track("Exit Gamma", ExitMap.specific_heat_ratio)

EquilibriumNozzle.track("Throat Mach", throat_mach)
EquilibriumNozzle.track("Exit Mach", exit_mach)

EquilibriumNozzle.track("Mass Flow [kg/s]", Conv.mass_flow)

EquilibriumNozzle.track("Thrust [lbf]", thrust / 4.448)
EquilibriumNozzle.track("Specific Impulse [s]", specific_impulse)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

SteadyState(EquilibriumNozzle).solve(verbose=True)