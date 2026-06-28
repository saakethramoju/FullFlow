from fullflow import *
from thermoprop import *


psia_to_pa = 6894.76

filename = "equilibrium_nozzle"

mixture_ratio = 2.3
chamber_pressure = 400 * psia_to_pa

fuel = Propellant("rp-1", temperature=298.15)
oxidizer = Propellant("lox", temperature=90.17)


Props = Reactants(
    fuels=fuel,
    oxidizers=oxidizer,
    mixture_ratio=mixture_ratio,
)

InjectorFace = Equilibrium(
    reactants=Props,
    mode="hp",
    pressure=chamber_pressure,
)


# Use the ideal chamber entropy only to choose a reasonable entropy-axis range.
# The SP map itself will not be tied to this one chamber state.
reference_entropy = InjectorFace.entropy


def rp1_lox_tp_map(pressure, temperature, mixture_ratio):

    r = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=r,
        mode="tp",
        pressure=pressure,
        temperature=temperature,
    )

    return {
        "specific_heat_ratio": eq.gamma,
        "gas_constant": eq.gas_constant,
        "density": eq.density,
        "enthalpy": eq.enthalpy,
        "entropy": eq.entropy,
        "speed_of_sound": eq.speed_of_sound,
    }


def rp1_lox_sp_map(pressure, entropy, mixture_ratio):

    r = Reactants(
        fuels=fuel,
        oxidizers=oxidizer,
        mixture_ratio=mixture_ratio,
    )

    eq = Equilibrium(
        reactants=r,
        mode="sp",
        pressure=pressure,
        entropy=entropy,
        guess_temperature=InjectorFace.temperature,
    )

    return {
        "specific_heat_ratio": eq.gamma,
        "gas_constant": eq.gas_constant,
        "density": eq.density,
        "enthalpy": eq.enthalpy,
        "entropy": eq.entropy,
        "speed_of_sound": eq.speed_of_sound,
    }


generate_map(
    filename=filename,
    group="products_tp",
    axes=[
        Axis.log(
            "pressure",
            start=1 * psia_to_pa,
            stop=500 * psia_to_pa,
            count=30,
            units="Pa",
        ),
        Axis.linear(
            "temperature",
            start=500,
            stop=4500,
            count=50,
            units="K",
        ),
        Axis.linear(
            "mixture_ratio",
            start=1,
            stop=4,
            count=25,
            units="",
        ),
    ],
    evaluate=rp1_lox_tp_map,
    overwrite=True,
    raise_errors=True,
)


generate_map(
    filename=filename,
    group="products_sp",
    axes=[
        Axis.log(
            "pressure",
            start=1 * psia_to_pa,
            stop=500 * psia_to_pa,
            count=50,
            units="Pa",
        ),
        Axis.linear(
            "entropy",
            start=0.80 * reference_entropy,
            stop=1.20 * reference_entropy,
            count=40,
            units="J/kg-K",
        ),
        Axis.linear(
            "mixture_ratio",
            start=1,
            stop=4,
            count=25,
            units="",
        ),
    ],
    evaluate=rp1_lox_sp_map,
    overwrite=True,
    raise_errors=True,
)