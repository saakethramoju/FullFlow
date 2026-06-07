# FullFlow

[![PyPI version](https://img.shields.io/pypi/v/fullflow)](https://pypi.org/project/fullflow/)
[![Python](https://img.shields.io/pypi/pyversions/fullflow)](https://pypi.org/project/fullflow/)
[![Downloads](https://img.shields.io/pypi/dm/fullflow)](https://pypi.org/project/fullflow/)
[![License](https://img.shields.io/github/license/saakethramoju/FullFlow)](https://github.com/saakethramoju/FullFlow)
[![Build](https://github.com/saakethramoju/FullFlow/actions/workflows/check.yml/badge.svg)](https://github.com/saakethramoju/FullFlow/actions)
[![Release](https://github.com/saakethramoju/FullFlow/actions/workflows/release.yml/badge.svg)](https://github.com/saakethramoju/FullFlow/actions)

FullFlow is a Python framework for fluid, thermal, and propulsion network simulation.

It provides a component-based architecture for constructing and solving engineering systems composed of interconnected fluid, thermal, and control elements.

FullFlow is inspired by tools such as:

* ROCETS
* GFSSP
* NPSS
* EcosimPro

while remaining fully open-source and Python-native.

## Why FullFlow?

Engineering systems are often modeled as networks:

* Fluid networks
* Thermal networks
* Feed systems
* Propulsion systems
* Turbomachinery systems
* Heat exchanger systems

Traditional network solvers are frequently proprietary, difficult to extend, or difficult to integrate into modern Python workflows.

FullFlow allows engineers to construct simulation networks directly in Python using reusable components and shared states.

For example:

```python
from fullflow import *

FeedSystem = Network("Feed System")

SourcePressure = State(5e5)
TankPressure = State(3e5)

FeedLine = DarcyWeisbach(
    "Feed Line",
    FeedSystem,
    upstream_pressure=SourcePressure,
    downstream_pressure=TankPressure,
)

solution = SteadyState(FeedSystem).solve()
```

## Installation

```bash
pip install fullflow
```

or

```bash
uv add fullflow
```

## Running Examples

Example models are provided in the repository.

Clone FullFlow:

```bash
git clone https://github.com/saakethramoju/FullFlow.git
cd FullFlow
```

Install the development environment:

```bash
uv sync
```

Run the pump-fed engine example:

```bash
uv run python examples/pump_fed_engine.py
```

The pump-fed engine example demonstrates a larger FullFlow network with pressurant lines, tanks, runlines, pumps, injector manifolds, a combustion chamber, a nozzle, and balance equations.

## Core Concepts

FullFlow models engineering systems using three primary concepts:

### Network

A `Network` contains the complete engineering system.

Networks manage:

* Components
* States
* Tracking variables
* Solver interactions

```python
NetworkModel = Network("Example System")
```

### State

A `State` represents a simulation variable.

Examples include:

* Pressure
* Temperature
* Mass flow rate
* Enthalpy
* Heat rate
* Rotor speed

```python
Pressure = State(5e5)
Temperature = State(300)
```

### Component

A `Component` represents a physical device, process, or relationship.

Examples include:

* Pipes
* Valves
* Pumps
* Tanks
* Heat transfer elements
* Sensors
* Fluid property models

Components define equations that contribute to the overall network solution.

## Solvers

### SteadyState

The `SteadyState` solver computes a converged operating point for the network.

```python
solution = SteadyState(NetworkModel).solve()
```

### Transient

Transient simulation capabilities are under active development.

## Component Categories

### Branch Components

Branch components model transport processes between nodes.

Examples include:

* Darcy-Weisbach flow elements
* Discharge coefficients
* Pumps
* Turbines
* Regulators
* Compressible flow elements
* Heat transfer elements

### Node Components

Node components model storage and accumulation.

Examples include:

* Volumes
* Tanks
* Junctions
* Combustion chambers
* Solid thermal nodes

### Lookup Components

Lookup components provide thermodynamic and material properties.

Examples include:

* FluidLookup
* IdealGasLookup
* PropellantLookup
* MaterialLookup

### Sensors

Sensor components provide instrumentation-style measurements.

Examples include:

* Thermocouples
* Pressure transducers

## Fluid Systems

FullFlow supports modeling of:

* Compressible flow
* Incompressible flow
* Real fluids
* Ideal gases
* Fluid mixtures
* Liquid rocket propellants
* Flow splitting
* Flow mixing

through integration with ThermoProp.

## Thermal Systems

FullFlow supports:

* Lumped thermal networks
* Conduction
* Convection
* Radiation
* Temperature-dependent material properties
* Coupled fluid-thermal simulation

## Propulsion Systems

FullFlow is designed to support rocket propulsion applications including:

* Feed systems
* Turbopumps
* Turbines
* Combustion chambers
* Nozzles
* Regenerative cooling systems
* Integrated propulsion cycles

## Dependencies

FullFlow builds upon several scientific and aerospace libraries:

* NumPy
* SciPy
* ThermoProp
* RocketCEA
* Rich

## Project Status

FullFlow is currently under active development.

Current development focuses on:

* Fluid networks
* Thermal networks
* Propulsion system modeling
* Heat transfer
* Turbomachinery
* Solver infrastructure

The API may evolve as capabilities continue to expand.

## Roadmap

Planned future capabilities include:

* Improved transient simulation
* Two-phase flow support
* Control system modeling
* Advanced turbomachinery models
* Heat exchanger libraries
* Cycle analysis tools
* Expanded reporting and visualization

## Documentation

Documentation is currently under development.

Examples, source code, and release history are available on GitHub.

## Contributing

Bug reports, feature requests, discussions, and pull requests are welcome.

Please open an issue if you encounter a problem or would like to propose an enhancement.

## Source Code

GitHub:

https://github.com/saakethramoju/FullFlow

## License

FullFlow is released under the GNU General Public License v3.0.

See `LICENSE` for details.
