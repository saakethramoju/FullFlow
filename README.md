# FullFlow

[![PyPI version](https://img.shields.io/pypi/v/fullflow)](https://pypi.org/project/fullflow/)
[![Python](https://img.shields.io/pypi/pyversions/fullflow)](https://pypi.org/project/fullflow/)
[![Downloads](https://img.shields.io/pypi/dm/fullflow)](https://pypi.org/project/fullflow/)
[![License](https://img.shields.io/github/license/saakethramoju/FullFlow)](https://github.com/saakethramoju/FullFlow)
[![Build](https://github.com/saakethramoju/FullFlow/actions/workflows/check.yml/badge.svg)](https://github.com/saakethramoju/FullFlow/actions)
[![Release](https://github.com/saakethramoju/FullFlow/actions/workflows/release.yml/badge.svg)](https://github.com/saakethramoju/FullFlow/actions)

FullFlow is a Python framework for fluid, thermal, and propulsion network simulation.

It provides a component-based architecture for constructing and solving engineering systems composed of interconnected fluid, thermal, combustion, turbomachinery, and heat-transfer elements.

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
* Regenerative cooling systems
* Coupled fluid-thermal systems

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

## 0.1.3 Core Streamline

FullFlow 0.1.3 starts a core cleanup focused on API consistency and speed. The System layer now uses lighter `State`, `Composition`, `Balance`, `Component`, and `Network` internals, cached iteration-variable metadata, simpler export paths, and fewer unnecessary imports. The unused `Exceptions` package has been removed in favor of standard Python exceptions.

The solver and component layers were also streamlined. `SteadyState.py` is now a thin public wrapper over the modular `Solvers/steady_state/` implementation, `Network.solve()` is available for simpler scripts, and common scalar branch calculations now avoid unnecessary NumPy calls. Lookup components cache dynamic attribute proxies and callable signature metadata, reducing overhead in ThermoProp-heavy networks.

## Features

* Component-based network architecture
* Steady-state network solver
* Fluid, thermal, and propulsion modeling
* Real fluids, ideal gases, and propellants
* Temperature-dependent material properties
* Compressible and incompressible flow
* Conduction, convection, and radiation
* Turbomachinery maps
* Model switching and correlation comparison
* Excel and DataFrame result export
* Pure Python implementation

## Example Applications

FullFlow can be used to model:

* Pump-fed liquid rocket engines
* Pressure-fed propulsion systems
* Turbomachinery and pump maps
* Counterflow heat exchangers
* Regenerative cooling systems
* Fluid mixture separation and mixing
* Compressible flow systems
* Coupled fluid-thermal networks
* Material thermal response
* Engineering correlation studies

## Installation

```bash
pip install fullflow
```

or

```bash
uv add fullflow
```

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

### Custom Components

A minimal custom component only needs a constructor, optional explicit state calculations, and residual equations. For simple solve variables, define `_iteration_variable_names` instead of writing boilerplate properties:

```python
class PressureNode(Component):
    _iteration_variable_names = ("pressure",)

    def __init__(self, name, network, pressure, mass_flow_in=None, mass_flow_out=None):
        self.setup()

    @property
    def residuals(self):
        mdot_in, mdot_out = self.values("mass_flow_in", "mass_flow_out")
        return [mdot_in - mdot_out]
```

`self.setup()` still handles FullFlow's normal conversion rules: numbers become `State` objects, existing `State` and `Composition` objects are preserved, and the component is registered with its network.

## Solvers

### SteadyState

The `SteadyState` solver computes a converged operating point for the network.
The existing explicit solver API is still supported:

```python
solution = SteadyState(NetworkModel).solve()
```

For simpler scripts, a network can now call the steady-state solver directly:

```python
solution = NetworkModel.solve()
```

Static evaluation is also available without nonlinear iteration:

```python
solution = NetworkModel.static_evaluate()
```

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
* Pandas
* OpenPyXL
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

* Unit conversions
* Advanced combustors
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

## Examples

### Pump-Fed Rocket Engine

FullFlow can be used to model complete liquid rocket propulsion systems, including pressurization systems, tanks, pumps, injector manifolds, combustion chambers, and nozzles.

Run the example:

```bash
uv run python examples/pump_fed_engine.py
```

```python
from fullflow import *

PumpNetwork = Network("Pumped System")

fuel_shaft_speed = State(25000)
ox_shaft_speed = State(25000)

FuelTankFluid = FluidLookup(...)
OxTankFluid = FluidLookup(...)

FuelPump = PolytropicPump(...)
OxPump = PolytropicPump(...)

MainChamber = MainCombustionChamber(...)
Nozzle = RocketCEAChokedNozzle(...)

solution = SteadyState(PumpNetwork).solve()
```

See `examples/pump_fed_engine.py` for the complete model.

### Mixture Splitter

FullFlow supports fluid mixtures, composition tracking, mixing, and separation.

Run the example:

```bash
uv run python examples/mixture_splitter.py
```

```python
from fullflow import *

SourceFluid = FluidLookup(
    "Source Fluid",
    MixtureNetwork,
    {"gn2": 0.75, "O2": 0.01, "Ar": 0.24},
    pressure=3e5,
    temperature=300,
)

Separator = FlowSplitter(
    "Separator",
    MixtureNetwork,
    pressure=VolumeFluid.pressure,
    composition=VolumeFluid.composition,
    composition_out1=SeparatorOutlet1Composition,
    composition_out2=SeparatorOutlet2Composition,
)

solution = SteadyState(MixtureNetwork).solve()
```

This example demonstrates:

* Multicomponent fluid mixtures
* Composition tracking
* Flow splitting
* Composition balances
* Mixture thermodynamic property evaluation

See `examples/mixture_splitter.py` for the complete model.

### Counterflow Heat Exchanger

FullFlow supports coupled fluid and thermal simulations, allowing heat exchangers to be modeled using fluid networks, thermal networks, material properties, and heat-transfer correlations within a unified framework.

Run the example:

```bash
uv run python examples/heat_exchanger.py
```

```python
from fullflow import *

HeatExchanger = Network("Heat Exchanger")

LiquidSource = FluidLookup(
    "Liquid Source",
    HeatExchanger,
    "rp-1",
    pressure=6e5,
    temperature=1000,
)

CoolantSource = FluidLookup(
    "Coolant Source",
    HeatExchanger,
    "water",
    pressure=20e5,
    temperature=300,
)

TubeNode1 = Solid(...)
TubeNode2 = Solid(...)

TubeConduction = Conduction(...)

Liquid1Solid1Convection = Convection(...)
Coolant2Solid1Convection = Convection(...)

HeatTransferModel = Model(
    "Heat Transfer Correlation",
    HeatExchanger,
    GnielinskiOption,
    DittusBoelterOption,
)

solution = SteadyState(HeatExchanger).solve(
    model="Heat Transfer Correlation"
)
```

This example demonstrates:

* Counterflow heat exchanger modeling
* Coupled fluid and thermal networks
* Temperature-dependent material properties
* Conduction and convection
* Internal and annular flow passages
* Gnielinski and Dittus-Boelter heat-transfer correlations
* Model switching with `Model` and `ModelOption`

See `examples/heat_exchanger.py` for the complete model.

### Model Comparison and Correlation Sweeps

FullFlow supports interchangeable physics models through `Model` and `ModelOption`.

This allows multiple component formulations to be evaluated within the same network without rebuilding the system.

Run the example:

```bash
uv run python examples/model_comparison.py
```

```python
from fullflow import *

PumpModel = Model(
    "Main Pump",
    PumpNetwork,
    ConstantDensityPumpOption,
    PolytropicPumpOption,
)

solution = SteadyState(PumpNetwork).solve(
    model="Main Pump",
    evaluate_all_model_options=True,
)
```

This example demonstrates:

* Model switching
* Alternative component formulations
* Pump-model comparison
* Shared turbomachinery maps
* Correlation sweeps
* Automated evaluation of multiple model options

See `examples/model_comparison.py` for the complete model.

### Running Examples

Clone the repository:

```bash
git clone https://github.com/saakethramoju/FullFlow.git
cd FullFlow
```

Install dependencies:

```bash
uv sync
```

Run an example:

```bash
uv run python examples/pump_fed_engine.py
```

or

```bash
uv run python examples/mixture_splitter.py
```

or

```bash
uv run python examples/heat_exchanger.py
```

or

```bash
uv run python examples/model_comparison.py
```

## Contributing

Bug reports, feature requests, discussions, and pull requests are welcome.

Please open an issue if you encounter a problem or would like to propose an enhancement.

## Source Code

GitHub:

https://github.com/saakethramoju/FullFlow

## License

FullFlow is released under the GNU General Public License v3.0.

See `LICENSE` for details.
