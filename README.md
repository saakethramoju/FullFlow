# FullFlow

[![PyPI version](https://img.shields.io/pypi/v/fullflow)](https://pypi.org/project/fullflow/)
[![Python](https://img.shields.io/pypi/pyversions/fullflow)](https://pypi.org/project/fullflow/)
[![Downloads](https://img.shields.io/pypi/dm/fullflow)](https://pypi.org/project/fullflow/)
[![License](https://img.shields.io/github/license/saakethramoju/FullFlow)](https://github.com/saakethramoju/FullFlow)
[![Build](https://github.com/saakethramoju/FullFlow/actions/workflows/check.yml/badge.svg)](https://github.com/saakethramoju/FullFlow/actions)
[![Release](https://github.com/saakethramoju/FullFlow/actions/workflows/release.yml/badge.svg)](https://github.com/saakethramoju/FullFlow/actions)

FullFlow is a Python-native framework for building and solving fluid, thermal,
propulsion, control, and transient network models.

The package is designed around the way engineering system models are usually
assembled: a network contains components, components share states, and solvers
adjust selected states until component equations, balances, and dynamic residuals
are satisfied.  FullFlow is not a Navier-Stokes CFD solver.  It is a 
lumped/network modeling framework for systems such as feed systems, tanks, valves, 
pipes, pumps,turbines, thrust chambers, heat exchangers, regenerative cooling circuits,
pressurization systems, virtual test stands, and controller-driven startup sequences.

## Installation

```bash
pip3 install fullflow
```

For examples that use ThermoProp property objects directly:

```bash
pip3 install "fullflow[thermo]"
```

For the full example set:

```bash
pip3 install "fullflow[examples]"
```

With `uv`:

```bash
uv add fullflow
uv add "fullflow[examples]"
```

## Required and optional dependencies

Core FullFlow depends on:

- `numpy` for numerical arrays and math utilities.
- `scipy` for nonlinear least-squares solving and interpolation.
- `rich` for optional terminal diagnostics.
- `h5py` for HDF5 result export.
- `fullplot` for trace objects, command traces, sensor traces, and map-generation workflows.

ThermoProp is optional.  FullFlow's core package does not require a specific
fluid-property backend, because `Lookup` can wrap any callable or class.  Many
examples use ThermoProp, so install `fullflow[thermo]` or `fullflow[examples]`
when following those examples.

## Design philosophy

FullFlow follows a small set of rules:

1. A model is a `Network`.
2. A variable is a `State`.
3. A physical device, correlation, controller, sensor, or helper is a `Component`.
4. A steady-state solve drives residuals and derivatives to zero.
5. A transient solve integrates dynamic equations and closes algebraic equations implicitly.
6. Tabular data, property packages, test data, command schedules, and maps should be usable from Python without a special input-file language.

The API is intentionally direct.  A user can build a model in normal Python:

```python
from fullflow import *

PipeNetwork = Network("Pipe Network")

source_pressure = State(5.0e5)
node_pressure = State(3.0e5)
density = State(1000.0)
friction_factor = State(0.02)
mass_flow = State(0.1)

Pipe = DarcyWeisbach(
    "Feed Line",
    PipeNetwork,
    mass_flow=mass_flow,
    upstream_pressure=source_pressure,
    downstream_pressure=node_pressure,
    length=5.0,
    hydraulic_diameter=0.0127,
    density=density,
    cross_sectional_area=1.27e-4,
    friction_factor=friction_factor,
)

solution = SteadyState(PipeNetwork).solve(verbose=True)
```

## Core concepts

### State

`State` is the universal scalar container.  Components read and write
`state.value`.  A state can be initialized with a number, left unassigned with
`None`, bounded for solver use, or derived from a callable expression.

```python
pressure = State(2.0e6)
temperature = State(300.0)
mass_flow = State(0.1, bounds=(0.0, None))
```

Most component constructors also accept scalars.  `Component.setup()` converts
plain values into `State` objects, so these two forms are both valid:

```python
# Explicit state
pressure = State(2.0e6)

# Constant value automatically wrapped as a State by the component
pipe_length = 5.0
```

### Network

`Network` stores components, balances, model options, tracked variables, sensor
events, sequence state, and the simulation time state.

```python
Engine = Network("Engine")
Engine.time.value = 0.0
```

Important network methods:

- `network.track(name, value, ...)` registers variables for export.
- `network.save(filename)` writes a solution record to HDF5.
- `network.solve(...)` is a convenience wrapper around `SteadyState(network).solve(...)`.
- `network.static_evaluate(...)` evaluates components without nonlinear solving.

### Component

A component owns equations.  Every built-in component follows the same contract:

- `evaluate_states()` computes output states, residuals, and derivatives from current input states.
- `balances` returns algebraic residuals as `(iteration_variable, residual)`.
- `dynamics` returns transient equations as `(state, derivative)` or `(iteration_state, stored_state, derivative)`.

This makes custom components straightforward:

```python
class PressureMatch(Component):
    def __init__(self, name, network, variable, predicted, target):
        self.error = State(0.0)
        self.setup()

    def evaluate_states(self):
        self.error.value = self.predicted.value - self.target.value

    @property
    def balances(self):
        return [(self.variable, self.error)]
```

### Balance

`Balance(variable, residual)` is a lightweight user-defined algebraic equation.
Use it when you need one extra closure equation without creating a component.
The solver varies `variable` until `residual` is zero.

### Model options

`Model` lets one network contain multiple mutually exclusive component
configurations.  This is useful for comparing correlations, pump maps, nozzle
models, heat-transfer models, or property backends.

```python
PumpModel = Model("Pump Model", Engine)
PumpModel.add_option("constant density", ConstantDensityPump.model(...))
PumpModel.add_option("polytropic", PolytropicPump.model(...))

SteadyState(Engine).solve(model="Pump Model", evaluate_all_model_options=True)
```

## Solvers

FullFlow has two public solvers: `SteadyState` and `Transient`.

### Static evaluation

`SteadyState(network).static_evaluate(...)` only evaluates component equations.
It does not adjust iteration variables and it does not require residuals to be
zero.  Use it to debug inputs, check property lookups, inspect derived values,
or export an already assigned model state.

```python
SteadyState(network).static_evaluate(verbose=True)
```

### Steady-state solve

`SteadyState(network).solve(...)` adjusts iteration variables until all
component dynamics, component balances, and user balances satisfy the residual
tolerance.

```python
SteadyState(network).solve(
    verbose=True,
    filename="steady_results.h5",
    solver_method="trf",
    jacobian_method="3-point",
    rtol=1e-2,
)
```

Steady-state solver options:

| Option | Meaning |
| --- | --- |
| `model` | Name of a model option group to solve. If omitted, the base network is solved. |
| `evaluate_all_model_options` | Run every option in the selected model and return one result per option. |
| `filename` | Optional HDF5 file. `.h5` is added when no extension is supplied. |
| `return_type` | Currently supports record/dict-style output. |
| `verbose` | Print final solver summary and final network state. |
| `statistics` | Print or export detailed solver evaluation statistics. |
| `static` | Run `static_evaluate` instead of a nonlinear solve. |
| `dt`, `t_final` | When both are supplied, run a quasi-steady time sweep using the transient runtime while forcing dynamic components steady. |
| `save_dt` | Output interval for quasi-steady time sweeps. |
| `exceptions` | Dynamic components excluded from forced-steady treatment during quasi-steady sweeps. |
| `solver_method` | SciPy `least_squares` method. `"trf"` is recommended because it supports bounds. `"lm"` requires an unconstrained system. |
| `jacobian_method` | Finite-difference Jacobian method passed to SciPy, usually `"2-point"` or `"3-point"`. |
| `ftol` | SciPy relative cost-function convergence tolerance. |
| `xtol` | SciPy iteration-variable convergence tolerance. |
| `gtol` | SciPy gradient convergence tolerance. |
| `rtol` | FullFlow final residual acceptance tolerance. The solve is accepted only when the maximum absolute residual is below this value. |
| `state_max_passes` | Maximum repeated component-evaluation passes used to settle derived states during residual evaluation. |
| `state_tolerance` | Derived-state settling tolerance. |
| `ignore_balances` | `None`, `"all"`, or selected user balance names to exclude from the solve. Component balances are not affected. |

### Transient solve

`Transient(network).solve(dt, t_final, ...)` advances `network.time` using an
implicit backward-Euler formulation.  Each accepted step solves dynamic residuals
and algebraic residuals simultaneously.  When a step fails, the solver rolls
back and retries smaller half-steps until the step is accepted or the retry floor
is reached.

```python
Transient(network).solve(
    dt=0.001,
    t_final=0.5,
    save_dt=0.01,
    filename="transient_results.h5",
    verbose=True,
)
```

Transient solver options:

| Option | Meaning |
| --- | --- |
| `dt` | Nominal timestep in seconds. The solver may shorten steps to hit final time, output times, sequence breakpoints, or retry failed steps. |
| `t_final` | Final simulation time. The starting time is `network.time.value`. |
| `model` | Optional model option group to run. |
| `evaluate_all_model_options` | Run every selected model option as an independent transient from the same initial condition. |
| `filename` | Optional HDF5 output file. |
| `return_type` | Output format for saved records. |
| `verbose` | Print final transient summary and final network state. |
| `statistics` | Print accepted-step progression while the solve runs. |
| `solver_method` | SciPy `least_squares` method for each implicit step. `"trf"` is recommended. |
| `jacobian_method` | SciPy finite-difference Jacobian method. |
| `ftol`, `xtol` | SciPy convergence tolerances for each implicit step. |
| `gtol` | SciPy gradient tolerance. The transient default is `None` to avoid false convergence on weak finite-difference gradients. |
| `rtol` | FullFlow accepted-step residual tolerance. The final recomputed step residual must satisfy this value. |
| `state_max_passes` | Maximum derived-state settling passes in each residual call. |
| `state_tolerance` | Derived-state settling tolerance. |
| `max_step_retries` | Number of automatic half-step retries when a timestep does not satisfy `rtol`. |
| `minimum_dt` | Smallest automatic retry step. If omitted, the floor is based on the nominal `dt`. |
| `save_dt` | Output cadence. If omitted, every accepted step is saved. |
| `ignore_balances` | User balances to exclude from the transient solve. |

### Dynamic equations

Components expose transient equations through `dynamics`.

A direct state equation:

```python
@property
def dynamics(self):
    return [(self.position, self.velocity)]
```

means:

```text
d(position)/dt = velocity
```

A conservative equation with a convenient iteration variable:

```python
@property
def dynamics(self):
    return [(self.pressure, self.mass, self.mass_dot)]
```

means:

```text
The solver iterates pressure, but integrates mass using d(mass)/dt = mass_dot.
```

This is important for fluid volumes because pressure is usually a good nonlinear
iteration variable, while mass and energy are the conserved quantities.

## HDF5 output

Passing `filename="results.h5"` to a solver writes records to HDF5.  FullFlow
stores component names, component types, attributes, values, model metadata,
solver metadata, and transient history when available.  HDF5 output is designed
to be inspected with FullPlot or with ordinary `h5py` workflows.

```python
solution = SteadyState(network).solve(filename="solution.h5")
history = Transient(network).solve(dt=0.01, t_final=1.0, filename="run.h5")
```

## Component catalog

### General flow components

| Component | Purpose | Main solver behavior |
| --- | --- | --- |
| `FlowTube` | Finite-length flow path with pressure, friction, gravity, inertia, and optional compressible diagnostics. | Exposes a momentum residual; transient mode can integrate mass flow. |
| `AdiabaticFlow` | Simple adiabatic gas branch using total enthalpy conservation. | Computes mass flow/total enthalpy diagnostics. |
| `DarcyWeisbach` | Incompressible pipe/duct branch using Darcy friction factor. | Algebraic pressure-loss relation or dynamic flow-inertia relation when effective area is supplied. |
| `DischargeCoefficient` | Reversible incompressible CdA/orifice relation. | Computes signed mass flow; optional length enables inertia. |
| `CavitatingVenturi` | Liquid venturi with cavitating and noncavitating regimes. | Computes mass flow and cavitation flag. |
| `SeriesCdA` | Equivalent effective area for series restrictions. | Algebraic effective-area helper. |
| `ParallelCdA` | Equivalent effective area for parallel restrictions. | Algebraic effective-area helper. |
| `RectanglePoiseuille` | Laminar Poiseuille number for rectangular ducts. | Geometry helper. |
| `EllipsePoiseuille` | Laminar Poiseuille number for elliptical ducts. | Geometry helper. |
| `CircularAnnulusPoiseuille` | Laminar Poiseuille number for annular ducts. | Geometry helper. |
| `HydraulicDiameter` | Computes `4A/Pw`. | Geometry helper. |

### Compressible-flow components

| Component | Purpose | Main outputs |
| --- | --- | --- |
| `CompressibleOrifice` | Ideal-gas compressible orifice with choked-flow detection. | `mass_flow`, `choked`, optional total enthalpy. |
| `IsentropicDiffuser` | Ideal-gas area-change relation between two static states. | `mass_flow`, Mach numbers, downstream temperature, total enthalpy. |
| `IsentropicNozzle` | Ideal-gas nozzle with choking and normal-shock diagnostics. | `mass_flow`, exit Mach, exit pressure, exit temperature, exit velocity, shock flags. |

### Nodes and storage components

| Component | Purpose | Main solver behavior |
| --- | --- | --- |
| `Volume` | Lumped fluid control volume with mass and energy conservation. | Steady mass/energy balances or transient integration of mass and energy. |
| `Solid` | Lumped thermal solid node. | Steady heat balance or transient temperature integration. |
| `Composition` | Conservation of arbitrary stream-carried scalar labels. | Steady scalar balance or transient stored-amount integration. |

### Heat-transfer components

| Component | Purpose |
| --- | --- |
| `Conduction` | `kA/L` heat transfer between two temperature states. |
| `Convection` | `hA(T_surface - T_fluid)` convective heat rate. |
| `Radiation` | Diffuse-gray radiation exchange between two surfaces. |
| `AmbientRadiation` | Radiation between a surface and large ambient enclosure. |
| `TemperatureRecoveryFactor` | Laminar or turbulent recovery factor from Prandtl number. |
| `AdiabaticWallTemperature` | Compressible adiabatic wall temperature. |
| `EckertReferenceTemperature` | Eckert reference/film temperature for gas-side properties. |

### Convection coefficient correlations

| Component | Purpose |
| --- | --- |
| `Gnielinski` | Turbulent internal-flow convection coefficient. |
| `Petukhov` | Turbulent forced-convection coefficient. |
| `SiederTate` | Turbulent coefficient with wall/bulk viscosity correction. |
| `DittusBoelter` | Simple turbulent pipe-flow coefficient. |
| `Miropolskii` | Film-boiling/two-phase convection helper. |
| `Bartz` | Rocket thrust-chamber/nozzle gas-side coefficient. |
| `NaturalConvection` | Generic natural-convection coefficient from dimensionless groups. |
| `ChurchillChu` | Churchill-Chu natural-convection correlation. |

### Friction-factor correlations

| Component | Purpose |
| --- | --- |
| `Colebrook` | Colebrook-White Darcy friction factor with laminar fallback. |
| `Churchill` | Continuous all-Reynolds-number Darcy friction factor. |
| `PetukhovFriction` | Petukhov smooth-pipe turbulent friction factor with laminar fallback. |

### Turbomachinery and propulsion components

| Component | Purpose | Main solver behavior |
| --- | --- | --- |
| `Rotor` | Rotor speed dynamic from net torque and inertia. | Steady torque balance or transient speed integration. |
| `GasTurbine` | Simple turbine flow, shaft power, efficiency, and enthalpy change. | Algebraic output component. |
| `ConstantDensityPump` | Pump pressure rise from density and head. | Varies `mass_flow` until predicted discharge pressure matches the connected pressure. |
| `PolytropicPump` | Compressible/polytropic pump pressure-rise relation. | Varies `mass_flow` to close predicted discharge pressure. |
| `SpecificImpulse` | Computes `Isp = thrust / (mdot*g0)`. | Diagnostic output. |
| `IdealCharacteristicVelocity` | Ideal-gas `c*` estimate. | Diagnostic output. |

### Data, lookup, command, and instrumentation components

| Component | Purpose |
| --- | --- |
| `Lookup` | Wraps a callable/class/property object and exposes outputs as state-like attributes. |
| `LookupAttribute` | State-like proxy for one lookup input or output. |
| `Map` | N-dimensional interpolation from in-memory data or HDF5 map groups. |
| `Sensor` | Virtual instrumentation, test-data anchoring, condition traces, redlines, and event detection. |
| `SensorEvent` | Runtime record for a sensor crossing event. |
| `SensorCondition` | FullPlot trace attached to a sensor for event detection. |
| `Sequence` | Time/callable/trace-driven command source and clean abort scheduler. |
| `SequenceCommand` | One command entry managed by a sequence. |
| `SequenceCondition` | One sensor condition required by a sequence command or abort. |
| `SequenceAbort` | One clean transient stop rule. |
| `PID` | Transient-only PID controller. |
| `Actuator` | Command-to-position actuator with optional limits and rate limiting. |

## Lookup examples

`Lookup` is how FullFlow connects to ThermoProp or any other property object.
The wrapped object can be a function, a class, or an already reusable object
with an `update` method.

```python
from fullflow import *
from thermoprop import Fluid

Net = Network("Property Example")

Water = Lookup(
    "Water",
    Net,
    Fluid,
    "Water",
    pressure=State(1.0e5),
    temperature=State(300.0),
)

print(Water.density.value)
```

Lookup outputs can be passed directly to components:

```python
Pipe = DarcyWeisbach(
    "Pipe",
    Net,
    mass_flow=State(0.1),
    upstream_pressure=State(3e5),
    downstream_pressure=State(2e5),
    length=1.0,
    hydraulic_diameter=0.01,
    density=Water.density,
    cross_sectional_area=7.85e-5,
    friction_factor=State(0.02),
)
```

## Map examples

`Map` interpolates tabulated data during a solve:

```python
PumpMap = Map(
    "Pump Map",
    Net,
    inputs={"alpha": State(0.5)},
    axes={"alpha": [0.0, 0.5, 1.0]},
    outputs={"cd": [0.0, 0.2, 0.6]},
)

print(PumpMap.cd.value)
```

HDF5 maps can be read with `Map.from_hdf5(...)` when they follow FullPlot or
FullFlow map conventions.

## Sensors, traces, and test-data anchoring

FullFlow uses FullPlot traces for test-like workflows.  A sensor can simply read
a simulation state, anchor a state to test data, or check redline/blueline/
greenline/yellowline traces.

```python
import fullplot as fplt
from fullflow import *

Net = Network("Anchored Example")
pressure = State(2.0e6)

pressure_trace = fplt.Trace(
    x=[0.0, 1.0, 2.0],
    y=[2.0e6, 2.1e6, 2.05e6],
    name="measured pressure",
    role="data",
)

PressureSensor = Sensor(
    "Chamber Pressure Sensor",
    Net,
    reading=pressure,
    variable=pressure,
    data=pressure_trace,
    extend=False,
)
```

When used in a transient solve, the sensor samples the test data at
`network.time` and exposes a balance between the model variable and the sampled
trace value.  With `extend=False`, the transient stops cleanly when the trace no
longer has data.

## Sequences, controllers, and actuators

`Sequence` applies time-based or trace-based commands.  `PID` computes a control
command during transient solves.  `Actuator` turns the command into a rate-limited
plant position.

```python
Net = Network("Command Example")
valve_command = State(0.0)
valve_position = State(0.0)

OpenValve = Sequence(
    "Open Valve Command",
    Net,
    target=valve_command,
    times=[0.0, 0.5, 1.0],
    values=[0.0, 0.5, 1.0],
)

ValveActuator = Actuator(
    "Valve Actuator",
    Net,
    command=valve_command,
    position=valve_position,
    minimum=0.0,
    maximum=1.0,
    rate_limit=2.0,
)
```

Transient timesteps are shortened to land on command breakpoints where possible.
This avoids stepping past command changes and makes sequence-driven simulations
behave more like test procedures.

## Units

FullFlow does not enforce a global unit system.  Most examples use SI units:

- Pressure: Pa
- Temperature: K
- Mass flow: kg/s
- Density: kg/m^3
- Enthalpy/internal energy: J/kg
- Heat rate/power: W
- Area: m^2
- Length: m
- Time: s
- Rotor speed: rpm in the current turbomachinery components

The user is responsible for consistency.  If a model uses English units, all
states and correlations in that model must use compatible English units.

## Numerical notes and debugging workflow

A good FullFlow debugging workflow is:

1. Build the network with realistic initial guesses.
2. Run `SteadyState(network).static_evaluate(verbose=True)`.
3. Inspect component outputs and missing states.
4. Add balances or component dynamics.
5. Run `SteadyState(network).solve(verbose=True, statistics=True)`.
6. Use `filename="debug.h5"` when you want persistent output.
7. Use transient solves only after the initial state is physically reasonable.

Important solver details:

- The solvers normalize and collect residuals from components and user balances.
- State bounds are passed to SciPy when a state is used as an iteration variable.
- Derived states are settled with repeated evaluation passes before residuals are collected.
- Invalid property-package trial points can be rejected with directional penalty residuals after at least one valid residual exists.
- Transient solves roll back failed steps before retrying smaller steps.

## Examples

The repository includes examples for:

- Simple steady pipe networks
- Compressible-flow branches
- Cavitating venturis
- Heat exchangers
- Flow splitters and mixers
- Turbopump maps
- Equilibrium nozzle workflows
- Blowdown transients
- Pump transients
- Heat-transfer transients
- Water hammer
- Sensor events, commands, and test-data anchoring
- PID throttle examples
- Engine startup-style examples

Examples that import ThermoProp require `fullflow[thermo]` or `fullflow[examples]`.

# Documentation

Full documentation:

https://saakethramoju.github.io/softwares/fullflow/

Source code:

https://github.com/saakethramoju/FullFlow

PyPI:

https://pypi.org/project/fullflow/

## License

FullFlow is released under the GNU General Public License version 3.  See
`LICENSE` and `THIRD_PARTY_LICENSES.md`.
