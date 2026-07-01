# FullFlow System folder overview

The `System` folder contains the core objects that make FullFlow models possible. The classes in this folder are deliberately small. They do not know the details of any one rocket engine, pipe network, thermal circuit, or property package. Instead, they provide the shared language used by all components and solvers.

A FullFlow model is usually built from these ideas:

```text
Network
  ├── Components
  │     ├── inputs are States
  │     ├── outputs are States
  │     ├── optional dynamics
  │     └── optional balances
  ├── Balance objects
  ├── Model objects
  └── tracked States
```

The user usually imports everything from the package:

```python
from fullflow import *
```

That import exposes the important classes from `System`, the solver entry points from `Solvers`, and the HDF5/map helpers from `Exports`.

## State

`State` is the basic value holder used throughout FullFlow. Most component inputs and outputs are `State` objects. FullFlow uses `State` instead of plain floats because a `State` can stay connected to a solver, a component, or another expression.

A `State` can be an independent value:

```python
pressure = State(101325.0)
```

A `State` can also be derived from other States:

```python
density = pressure / (gas_constant * temperature)
```

The derived state does not copy the current value. It stores the relationship. If the solver changes `pressure`, the value of `density` updates when it is read.

A placeholder State can be connected later with `<<=`:

```python
pressure_ratio = State()
pressure_ratio <<= chamber_pressure / injector_pressure
```

This is useful when the object must exist before the equation that defines it is available.

States also store solver information such as bounds:

```python
pressure = State(300000.0, bounds=(0.0, None), keep_feasible=True)
```

Bounds are collected by the solvers and passed to the nonlinear least-squares routine.

## Balance

`Balance` is the user-facing way to add one custom algebraic equation to a network.

A balance has:

```text
variable  -> the State the solver may change
function  -> the residual the solver drives to zero
```

For example:

```python
Balance(
    "Mach Closure",
    network,
    variable=exit_pressure,
    function=throat_mach - 1.0,
)
```

This tells the solver to vary `exit_pressure` until `throat_mach - 1.0 = 0`.

A balance can use a `State` expression or a normal Python function. Use a `State` expression when the equation is simple. Use a function when the equation is easier to read with normal Python code.

Balance objects are intentionally separate from components. They are useful when a user needs one extra closure equation without writing a new component class.

## Component

`Component` is the base class for physical and mathematical building blocks. Components are things like pipes, volumes, pumps, turbines, heat-transfer links, maps, lookups, controllers, and custom user models.

Most component constructors look like this:

```python
class MyComponent(Component):
    def __init__(self, name, network, input1, input2, output=None):
        self.setup()
```

The call to `self.setup()` reads the constructor arguments, stores them on the component, converts simple values to `State` objects where appropriate, and registers the component with the network.

A component usually computes outputs in `evaluate_states()`:

```python
def evaluate_states(self):
    self.output.value = self.input1.value + self.input2.value
```

Components may expose two kinds of equations to the solvers.

### Component balances

A component balance is an algebraic equation owned by a component:

```python
@property
def balances(self):
    return [(self.pressure, self.pressure_error)]
```

In steady state and transient solves, component balances are driven to zero. The variable is adjusted by the solver.

### Component dynamics

A component dynamic is a real storage, inertia, or capacitance equation:

```python
@property
def dynamics(self):
    return [(self.temperature, self.temperature_dot)]
```

In steady state, the solver drives the derivative to zero. In transient, the solver integrates the state through time.

A dynamic can also use a separate solve variable and integrated state:

```python
@property
def dynamics(self):
    return [(self.pressure, self.mass, self.mass_dot)]
```

This form is useful when the physical storage state is mass or energy, but the solver variable is pressure or temperature.

## Network

`Network` is the container for one model. It stores:

```text
component_list
balance_list
model_list
tracked_state_list
time
```

A network does not solve itself. It only holds the objects. The steady-state and transient solvers build their own runtime view from the network.

Typical use:

```python
MyNetwork = Network("Example")

ComponentA(..., MyNetwork, ...)
ComponentB(..., MyNetwork, ...)

Balance(..., MyNetwork, ...)

MyNetwork.track("Pressure [Pa]", pressure)
```

Tracked states are included in verbose output and HDF5 exports. Tracking is usually how a user chooses the values they want to plot later.

## Model and ModelOption

`Model` allows one part of a network to have switchable alternatives. This is useful when several modeling approaches can represent the same physical item.

Examples:

```text
pump model:
  option 1 -> constant-density pump
  option 2 -> polytropic pump

nozzle model:
  option 1 -> shock-free nozzle
  option 2 -> normal-shock nozzle
```

The solver can try the model options in order and use the first one that converges, or it can evaluate all options and save all results.

The important idea is that model options are built only when selected. This keeps the network from containing every possible option at the same time.

## Components subfolder

`System/Components` contains the built-in FullFlow components. These are grouped by topic:

```text
Actuators.py              valve and actuator behavior
Composition.py            scalar/species conservation across streams
CompressibleFlow.py       nozzles, orifices, shock/flow utilities
Controllers.py            control logic components
ConvectionCoefficients.py heat-transfer coefficient correlations
FrictionFactors.py        friction-factor correlations
GeneralFlow.py            general branch/flow components
HeatTransfer.py           conduction, convection, radiation, wall temperature tools
Lookup.py                 wraps functions/classes/property objects
Maps.py                   interpolated tabular maps
Nodes.py                  volumes, tanks, solids, storage nodes
RocketPerformance.py      thrust / rocket-performance utilities
Sequence.py               time or schedule driven inputs
Turbomachinery.py         pumps, turbines, rotors
```

The pattern is the same across all components: inputs and outputs are States, `evaluate_states()` computes values, and optional `balances` or `dynamics` provide equations to the solvers.

## How the System folder fits with the rest of FullFlow

The `System` folder does not directly own the nonlinear solve or the HDF5 export. It defines the objects. The other packages interpret those objects:

```text
System
  Defines Networks, Components, States, Balances, and Models.

Solvers
  Reads the Network and decides which States are variables and which residuals
  must be driven to zero.

Exports
  Saves Network results and generated Maps to HDF5 files.
```

This separation is important. It keeps user models easy to read while allowing the solver and exporter internals to improve independently.
