"""
Composition splitter example.

Composition can also solve the unknown composition of one outlet stream. Passing
None for a stream composition means:

    use the values in solve={...} for this stream

This is useful for splitters, separators, filters, and any case where one stream
composition is unknown.

This example has a feed stream, a clean stream, and a concentrate stream. The
clean stream composition is known, so FullFlow solves the concentrate stream
composition from conservation.
"""

from fullflow import *


SplitterNetwork = Network("Composition Splitter Example")

feed_mass_flow = State(1.0)         # kg/s
clean_mass_flow = State(0.6)        # kg/s
concentrate_mass_flow = State(0.4)  # kg/s

feed_composition = {
    "water": State(0.70),
    "salt": State(0.30),
}

clean_composition = {
    "water": State(0.98),
    "salt": State(0.02),
}

concentrate_composition = {
    "water": State(0.50),
    "salt": State(0.50),
}

Splitter = Composition(
    "Splitter Composition",
    SplitterNetwork,
    inlets=[
        (feed_mass_flow, feed_composition),
    ],
    outlets=[
        (clean_mass_flow, clean_composition),
        (concentrate_mass_flow, None),
    ],
    solve=concentrate_composition,
)

SplitterNetwork.track("Concentrate Water Fraction", concentrate_composition["water"])
SplitterNetwork.track("Concentrate Salt Fraction", concentrate_composition["salt"])

SteadyState(SplitterNetwork).solve(verbose=True)
