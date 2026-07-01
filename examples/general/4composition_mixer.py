"""
Composition mixer example.

Composition conserves labels carried by mass flow. The labels can be species,
mixture fractions, contaminants, dye fractions, or any other scalar carried by
a stream.

For steady algebraic mixing, Composition solves outlet values from inlet and
outlet mass flows:

    sum(mdot_in*x_in) - sum(mdot_out*x_out) = 0

This example mixes two inlet streams and solves the outlet water/alcohol mass
fractions.
"""

from fullflow import *


MixerNetwork = Network("Composition Mixer Example")

stream_a_mass_flow = State(2.0)     # kg/s
stream_b_mass_flow = State(1.0)     # kg/s
mixed_mass_flow = State(3.0)        # kg/s

stream_a_composition = {
    "water": State(0.90),
    "alcohol": State(0.10),
}

stream_b_composition = {
    "water": State(0.10),
    "alcohol": State(0.90),
}

mixed_composition = {
    "water": State(0.50),
    "alcohol": State(0.50),
}

Mixer = Composition(
    "Mixer Composition",
    MixerNetwork,
    inlets=[
        (stream_a_mass_flow, stream_a_composition),
        (stream_b_mass_flow, stream_b_composition),
    ],
    outlets=[
        (mixed_mass_flow, mixed_composition),
    ],
    solve=mixed_composition,
)

MixerNetwork.track("Mixed Water Fraction", mixed_composition["water"])
MixerNetwork.track("Mixed Alcohol Fraction", mixed_composition["alcohol"])

SteadyState(MixerNetwork).solve(verbose=True)
