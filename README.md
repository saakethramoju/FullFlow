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

The built-in component catalog is only a starting point. Any user can create a
custom component by subclassing `Component`, declaring normal Python constructor
arguments, calling `self.setup()`, and exposing equations through
`evaluate_states()`, `balances`, and `dynamics`. Custom components participate in
the same steady-state, transient, model-option, export, and diagnostic workflows
as built-in components.

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

`State` is FullFlow's universal scalar container. Pressures, temperatures, mass
flows, heat rates, commands, component outputs, nonlinear iteration variables,
and integrated storage quantities are all represented with the same object.

```python
from fullflow import State

pressure = State(2.0e6)
temperature = State(300.0)
mass_flow = State(0.1, bounds=(0.0, None))
unassigned_output = State()
```

Use `.value` when a component needs the current numeric value:

```python
print(pressure.value)
pressure.value = 2.5e6
```

An unassigned state raises `UnassignedStateError` when its value is requested
before a component has calculated it. This is useful for optional component
outputs because missing model wiring is detected instead of silently replaced
with zero.

Most component constructors also accept plain Python values. `self.setup()`
wraps those values in `State` objects so component equations can consistently
use `.value`:

```python
Pipe = DarcyWeisbach(
    "Pipe",
    network,
    length=5.0,                 # stored as State(5.0)
    density=density,            # existing State is preserved
    mass_flow=mass_flow,
    upstream_pressure=P1,
    downstream_pressure=P2,
    hydraulic_diameter=0.01,
    cross_sectional_area=7.85e-5,
    friction_factor=0.02,
)
```

#### State math and live derived states

Arithmetic between states returns another `State`. The result is a live derived
expression, not a one-time numeric copy:

```python
area = State(0.0039)
density = State(1000.0)
mass_flow = State(4.0)

velocity = mass_flow / (density * area)
dynamic_pressure = 0.5 * density * velocity**2

print(velocity.value)

mass_flow.value = 5.0
print(velocity.value)  # updates automatically
```

FullFlow supports the common arithmetic operators:

```python
sum_state = a + b
difference = a - b
product = a * b
ratio = a / b
power = a**2
magnitude = abs(a)
negative = -a
```

It also provides state-aware math helpers that preserve the dependency graph:

```python
speed_of_sound = (gamma * gas_constant * temperature).sqrt()
log_pressure = pressure.log()
limited_command = command.clip(0.0, 1.0)
maximum_value = State.maximum(a, b)
minimum_value = State.minimum(a, b)
```

Other helpers include `exp`, `log10`, `sin`, `cos`, `tan`, inverse and
hyperbolic functions, angle conversion, `floor`, `ceil`, `hypot`, and related
operations.

Use `<<=` when a state must be created first and connected to an expression
later:

```python
mixture_ratio = State(2.0)  # startup fallback
mixture_ratio <<= oxidizer_mass_flow / fuel_mass_flow
```

This mutates the existing `mixture_ratio` object so every component that already
references it remains connected. It is equivalent to
`mixture_ratio.derive_from(...)`.

A derived state:

- is recomputed from its source expression,
- cannot be assigned with `state.value = ...`,
- is not a valid independent solver iteration variable,
- may retain its previous stored value as a startup fallback until the expression
  can first be evaluated.

To preserve a live dependency, use states directly in the expression. Writing
`mass_flow.value / density.value` calculates an ordinary float immediately;
writing `mass_flow / density` creates a derived `State`.

#### Bounds and `keep_feasible`

Bounds belong to the state that may become a solver variable:

```python
pressure = State(
    2.0e6,
    bounds=(1.0e5, 20.0e6),
    keep_feasible=True,
)

mass_flow = State(1.0, bounds=(0.0, None))
```

`None` means unbounded on that side. Bounds can also be changed in place:

```python
mass_flow.set_bounds((0.0, 20.0), keep_feasible=True)
```

Important behavior:

- assigning a numeric value outside the bounds raises immediately,
- bounds affect a solve only when the state is collected as an iteration
  variable through `dynamics`, `balances`, or `Balance`,
- bounded solves normally use `solver_method="trf"` or `"dogbox"`,
- `solver_method="lm"` does not support bounded states,
- `keep_feasible=True` is passed to SciPy's bound object as a feasibility hint
  for algorithms that can honor it.

Bounds are physical constraints, not substitutes for good initial guesses. A
state should normally begin inside its valid physical domain and reasonably
close to the expected solution.

#### Transient history

During transient solves, FullFlow stores the last accepted values of dynamic
states:

```python
state.previous
state.second_previous
```

Component authors normally do not call `store_previous()` themselves. The
transient solver advances history only after a timestep has passed the residual
acceptance check.

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
- `network.solve(...)` is a convenience wrapper around
  `SteadyState(network).solve(...)`.
- `network.static_evaluate(...)` evaluates components without nonlinear solving.

### Component

A `Component` is any named piece of physics, empirical behavior, control logic,
instrumentation, or helper logic that belongs to a network. Built-in components
and user-written components follow the same contract:

- `evaluate_states()` calculates outputs and residual quantities from the current
  state values,
- `balances` declares algebraic closure equations,
- `dynamics` declares storage, inertia, capacitance, or other differential
  equations,
- `pre_evaluation()` performs work that must occur before ordinary component
  evaluation,
- `self.setup()` converts constructor arguments into FullFlow attributes and
  registers the component.

The complete custom-component pattern is described in the next section.

### Balance

`Balance` is a lightweight user-defined algebraic equation. Use it when one
extra closure equation is needed but a dedicated component would add no useful
structure.

```python
area = State(1.0e-4, bounds=(1.0e-8, None))

AreaBalance = Balance(
    "Target Mass Flow",
    network,
    variable=area,
    function=branch.mass_flow - target_mass_flow,
)
```

The solver varies `area` until the residual is zero. The variable must be an
assignable, non-derived state-like object. Do not use a `Balance` variable that
is already owned by a component `dynamics` or `balances` equation.

### Model options

`Model` lets one network contain multiple mutually exclusive component
configurations. This is useful for comparing correlations, pump maps, nozzle
models, heat-transfer models, or property backends.

```python
PumpModel = Model("Pump Model", Engine)
PumpModel.add_option("constant density", ConstantDensityPump.model(...))
PumpModel.add_option("polytropic", PolytropicPump.model(...))

SteadyState(Engine).solve(
    model="Pump Model",
    evaluate_all_model_options=True,
)
```

## Writing custom components

FullFlow is intentionally designed so users can create new components without
modifying the solver or learning a separate equation language. A custom
component is an ordinary Python class derived from `Component`.

### Smallest useful component

This restriction has inputs and one calculated output, but no solve equation of
its own:

```python
import math
from fullflow import Component, Network, State


class SquareLawRestriction(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        upstream_pressure: State,
        downstream_pressure: State,
        density: State,
        area: State | float,
        loss_coefficient: State | float,
        mass_flow: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        pressure_drop = self.upstream_pressure.value - self.downstream_pressure.value
        density = self.density.value
        area = self.area.value
        loss_coefficient = self.loss_coefficient.value

        if pressure_drop == 0.0:
            self.mass_flow.value = 0.0
        else:
            self.mass_flow.value = (
                math.copysign(1.0, pressure_drop)
                * area
                * math.sqrt(2.0 * density * abs(pressure_drop) / loss_coefficient)
            )
```

The component can immediately be connected to built-in components:

```python
Restriction = SquareLawRestriction(
    "Feed Restriction",
    network,
    upstream_pressure=source_pressure,
    downstream_pressure=node_pressure,
    density=liquid_density,
    area=1.0e-4,
    loss_coefficient=4.0,
)

Node = Volume(
    "Node",
    network,
    pressure=node_pressure,
    mass_flow_in=Restriction.mass_flow,
)
```

### What `self.setup()` does

Call `self.setup()` directly from the custom component constructor. The
constructor must have `name` and `network` parameters.

```python
class MyComponent(Component):
    def __init__(self, name, network, input_a, coefficient=1.0, output=None):
        self.internal_flag = False
        self.setup()
```

`self.setup()`:

1. stores `self.name` and `self.network`,
2. registers the component with the network,
3. inspects the subclass constructor signature,
4. creates one component attribute for each constructor argument other than
   `self`, `name`, and `network`,
5. preserves existing `State` objects,
6. converts state-like objects such as `LookupAttribute` into connected state
   views through `as_state()`,
7. wraps other values, including strings, booleans, numeric constants, and
   `None`, in `State` objects,
8. attaches diagnostic labels such as `Component Name:attribute`.

Therefore:

```python
coefficient=1.0
```

becomes:

```python
self.coefficient.value == 1.0
```

and:

```python
output=None
```

becomes an unassigned output state that `evaluate_states()` can later fill:

```python
self.output.value = calculated_value
```

Initialize internal flags, caches, or residual placeholders before
`self.setup()` when they are not constructor arguments. Ordinary constructor
arguments should not be manually copied to attributes; `self.setup()` does that
work.

Do not replace a connected output state during evaluation:

```python
# Correct: preserves every connection to this State
self.mass_flow.value = calculated_mass_flow

# Usually wrong: replaces the object and breaks existing connections
self.mass_flow = State(calculated_mass_flow)
```

### `evaluate_states()`

The solver calls `evaluate_states()` repeatedly:

- during equation discovery,
- during static evaluation,
- for every nonlinear residual evaluation,
- several times per residual call when derived outputs need to settle,
- after a solution is accepted so exported values are current.

It should calculate deterministic outputs from the current component inputs. It
should not advance time or permanently commit a timestep. The transient solver
handles timestep acceptance and rollback.

A component may store residual helpers as plain numbers or as states. Both of
these are valid:

```python
self.pressure_error = predicted_pressure - self.target_pressure.value
```

or:

```python
self.pressure_error.value = predicted_pressure - self.target_pressure.value
```

The important requirement is that the object returned through `balances` or
`dynamics` resolves to the current numeric residual.

### Algebraic equations with `balances`

Use `balances` for equations with no physical storage or time derivative. Each
entry is:

```text
(iteration_variable, residual_that_should_equal_zero)
```

Example pump curve:

```python
class QuadraticPump(Component):
    def __init__(
        self,
        name,
        network,
        mass_flow,
        density,
        inlet_pressure,
        discharge_pressure,
    ):
        self.pressure_error = 0.0
        self.setup()

    def evaluate_states(self):
        volumetric_flow = self.mass_flow.value / self.density.value
        head = -2.6e5 * volumetric_flow**2 + 40.0
        predicted_pressure = (
            self.inlet_pressure.value
            + self.density.value * 9.80665 * head
        )
        self.pressure_error = predicted_pressure - self.discharge_pressure.value

    @property
    def balances(self):
        return [(self.mass_flow, self.pressure_error)]
```

At steady state and at every transient timestep, FullFlow varies `mass_flow`
until:

```text
predicted_pressure - discharge_pressure = 0
```

Component balances may return several equations:

```python
@property
def balances(self):
    return [
        (self.variable_1, self.residual_1),
        (self.variable_2, self.residual_2),
    ]
```

### Direct dynamic equations

Use `dynamics` for real storage, inertia, capacitance, thermal mass, rotor
inertia, actuator motion, or another differential equation.

The two-entry form is:

```text
(integrated_state, derivative)
```

Example tank level:

```python
class SimpleTank(Component):
    def __init__(self, name, network, level, area, volume_flow_out):
        self.level_dot = 0.0
        self.setup()

    def evaluate_states(self):
        self.level_dot = -self.volume_flow_out.value / self.area.value

    @property
    def dynamics(self):
        return [(self.level, self.level_dot)]
```

This means:

```text
d(level)/dt = level_dot
```

In a steady-state solve, FullFlow varies `level` until `level_dot = 0`. In a
transient solve, `level` is integrated implicitly with backward Euler.

### Separate iteration and stored states

Thermodynamic systems often have one variable that is convenient for nonlinear
iteration and another quantity that must be conserved. FullFlow supports:

```text
(iteration_state, integrated_stored_state, derivative)
```

For example:

```python
class GasStorage(Component):
    def __init__(
        self,
        name,
        network,
        pressure,
        density,
        volume,
        mass,
        mass_flow_in,
        mass_flow_out,
    ):
        self.mass_dot = 0.0
        self.setup()

    def evaluate_states(self):
        # The property lookup makes density depend on pressure.
        self.mass.value = self.density.value * self.volume.value
        self.mass_dot = self.mass_flow_in.value - self.mass_flow_out.value

    @property
    def dynamics(self):
        return [(self.pressure, self.mass, self.mass_dot)]
```

This tuple means:

- SciPy varies `pressure`,
- the component/property model computes the corresponding stored `mass`,
- FullFlow conserves mass using `d(mass)/dt = mass_dot`.

For steady state, the solver varies `pressure` until:

```text
mass_dot = 0
```

For a transient timestep, it varies the new pressure until the pressure-dependent
new mass satisfies:

```text
mass_new - mass_previous - dt * mass_dot_new = 0
```

This is the standard pattern used when pressure or temperature is numerically
convenient but mass, internal energy, momentum, or another extensive quantity is
the true conserved storage state.

The three members have distinct roles:

| Tuple member | Purpose |
| --- | --- |
| `iteration_state` | Independent unknown changed by SciPy. Must have an initial guess and may have bounds. |
| `integrated_stored_state` | Quantity whose accepted history is stored in `previous` and integrated by backward Euler. |
| `derivative` | Current new-time rate of change of the stored quantity. |

### `pre_evaluation()`

`pre_evaluation()` runs before ordinary residual-state evaluation. Most physical
components do not need it. It is useful for components that must prepare inputs
before the rest of the network evaluates, such as:

- property lookups,
- command schedules,
- sensors and test-data anchoring,
- components that lazily initialize an external object.

```python
class PreparedComponent(Component):
    def __init__(self, name, network, command, output=None):
        self.setup()

    def pre_evaluation(self):
        # Prepare command-dependent input state before evaluate_states passes.
        pass

    def evaluate_states(self):
        self.output.value = self.command.value
```

### Transient-only components and timestep context

A controller, command source, or procedure component can opt out of ordinary
steady-state evaluation:

```python
class MyController(Component):
    TRANSIENT_ONLY = True
```

Such a component remains active in transient solves but is skipped by steady
state.

Components that need the current trial timestep can override
`set_transient_context()`:

```python
def set_transient_context(self, *, dt):
    super().set_transient_context(dt=dt)
    # self._transient_dt now contains the current trial dt
```

This is useful for rate limits and discrete-time behavior. Remember that a trial
step can fail and be retried, so permanent state changes should not be committed
inside a residual evaluation.

### Export control

By default, component attributes are available to diagnostics and result export.
Custom components can suppress large internal objects:

```python
@property
def ignored_export_attributes(self):
    return super().ignored_export_attributes | {"large_table", "cache"}
```

They can also expose additional clean export names without duplicating runtime
states:

```python
@property
def export_attributes(self):
    return {"pressure_ratio": self.outlet_pressure / self.inlet_pressure}
```

### Equation ownership and model consistency

FullFlow constructs the nonlinear unknown vector from:

1. the first state in every active `dynamics` tuple,
2. the first state in every active component `balances` tuple,
3. the variable of every active user `Balance` object.

Important rules:

- each solver variable should represent one independent model degree of freedom,
- a user `Balance` variable cannot also be owned by component dynamics or a
  component balance,
- derived states cannot be iteration variables,
- the solve must have at least as many residuals as variables,
- overdetermined systems are allowed because least squares can minimize more
  residuals than variables,
- an underdetermined system usually means a missing closure equation,
- every iteration variable needs a physically reasonable initial value,
- residuals should be scaled so one equation does not dominate only because its
  units produce a much larger number.

### Custom-component checklist

Before solving a new component, verify:

1. `name` and `network` are the first constructor arguments.
2. `self.setup()` is called directly from `__init__`.
3. constructor inputs and outputs are read through `.value` inside equations.
4. output state objects are updated, not replaced.
5. algebraic equations are in `balances`.
6. actual storage or inertia equations are in `dynamics`.
7. three-entry dynamics conserve the correct stored quantity.
8. iteration variables have initial guesses and useful physical bounds.
9. residual magnitudes are sensibly scaled.
10. `static_evaluate()` works before attempting a nonlinear solve.

## Solver mathematics and numerical behavior

FullFlow has two public nonlinear solvers:

```python
SteadyState(network).solve(...)
Transient(network).solve(dt=..., t_final=...)
```

Both solvers use the same component equations and the same state-bound system.
The difference is how dynamic equations are turned into residuals.

### Unknowns and residuals

A nonlinear solve starts with an unknown vector:

```text
x = [x1, x2, ..., xn]
```

Each element corresponds to an iteration state collected from component
`dynamics`, component `balances`, or user `Balance` objects.

A residual is a numeric equation error that should be zero. For example:

```text
mass-flow residual       = mdot_in - mdot_out
pressure-match residual  = P_predicted - P_target
choking residual          = M_throat - 1
```

After writing a trial vector `x` into the network, FullFlow evaluates all
components and constructs:

```text
r(x) = [r1(x), r2(x), ..., rm(x)]
```

A perfect solution has `r(x) = 0`. In practice, a solve is accepted when SciPy
reports success and FullFlow's final residual test is satisfied.

The number of residuals must be at least the number of unknowns:

```text
m >= n
```

- `m = n` is a square nonlinear system.
- `m > n` is overdetermined and is handled naturally by least squares.
- `m < n` is underdetermined and FullFlow raises a setup error.

### Why FullFlow uses nonlinear least squares

FullFlow calls `scipy.optimize.least_squares` and minimizes the cost:

```text
F(x) = 1/2 * sum(ri(x)^2)
     = 1/2 * ||r(x)||²
```

Least squares works for both square and overdetermined systems, supports bounds
with trust-region methods, and provides robust diagnostics for coupled nonlinear
networks.

Residual scale matters because the objective squares every residual. A pressure
error of `1.0e6 Pa` numerically dominates a dimensionless error of `0.1` unless
the modeler normalizes one or both equations. A common pattern is:

```python
pressure_residual = (predicted_pressure - target_pressure) / pressure_scale
mass_residual = (mass_flow_in - mass_flow_out) / mass_flow_scale
```

Choose scales that make an error of order one represent a physically meaningful
miss. Algebraic residuals are not automatically normalized by FullFlow.

### Jacobian

The Jacobian describes how every residual changes with every unknown:

```text
Jij = partial ri / partial xj
```

In matrix form:

```text
          x1       x2             xn
r1    dr1/dx1  dr1/dx2   ...   dr1/dxn
r2    dr2/dx1  dr2/dx2   ...   dr2/dxn
...      ...      ...             ...
rm    drm/dx1  drm/dx2   ...   drm/dxn
```

The solver uses this local sensitivity information to choose a step in the
unknown vector. FullFlow does not require custom components to provide analytic
Jacobian equations. It asks SciPy to estimate the Jacobian by finite differences.

With `jacobian_method="2-point"`, a Jacobian column is approximately:

```text
J[:, j] ~= [r(x + h ej) - r(x)] / h
```

With `jacobian_method="3-point"`, it is approximately:

```text
J[:, j] ~= [r(x + h ej) - r(x - h ej)] / (2h)
```

where `ej` selects one unknown and SciPy chooses an appropriate perturbation
`h`, adjusted as needed near bounds.

`"3-point"` normally requires more residual evaluations but is more accurate.
`"2-point"` is faster and can be useful for large or expensive models.

FullFlow passes:

```python
x_scale="jac"
```

so SciPy scales trust-region steps using local Jacobian information rather than
assuming every variable has the same numerical scale.

### Derived-state settling inside each residual call

One residual evaluation can require several component passes because components
are connected sequentially:

```text
pressure -> property lookup -> density -> flow branch -> node balance
```

Before solving, FullFlow runs component `pre_evaluation()` hooks and discovers
the active equations. For every trial `x`, it then:

1. writes `x` into the iteration states,
2. calls component `evaluate_states()` methods,
3. repeats component evaluation until non-iteration states stop changing or
   `state_max_passes` is reached,
4. restores iteration states if a component attempted to overwrite them,
5. collects the residual vector.

`state_tolerance` controls the fixed-point settling test. This inner settling
process is different from the outer nonlinear least-squares iteration.

### Static evaluation

Static evaluation performs steps 2 through 4 without adjusting any iteration
variable:

```python
SteadyState(network).static_evaluate(verbose=True)
```

Use it to:

- verify that all required inputs are assigned,
- inspect lookup outputs,
- catch invalid property calls,
- confirm component evaluation order,
- debug a model before nonlinear solving,
- evaluate a pure input-to-output calculator network.

### Steady-state equations

For a component algebraic balance:

```python
(variable, residual)
```

steady state drives:

```text
residual = 0
```

For either dynamic form:

```python
(state, derivative)
```

or:

```python
(iteration_state, integrated_state, derivative)
```

steady state drives:

```text
derivative = 0
```

Therefore steady state is a trim of the dynamic model. A volume reaches zero
mass and energy accumulation, a rotor reaches zero acceleration, and a thermal
solid reaches zero temperature rate.

The steady residual vector is ordered as:

```text
1. component dynamic derivative residuals
2. component algebraic balance residuals
3. user Balance residuals
```

A normal solve is:

```python
solution = SteadyState(network).solve(
    solver_method="trf",
    jacobian_method="3-point",
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    rtol=1e-2,
    state_max_passes=5,
    state_tolerance=1e-10,
    verbose=True,
)
```

The steady solver sequence is:

1. build the selected model configuration,
2. run pre-evaluation and discover equations,
3. collect unknowns, initial values, and bounds,
4. evaluate the initial residual,
5. call SciPy least squares,
6. require SciPy success and `max(abs(sol.fun)) <= rtol`,
7. write the accepted `sol.x` values back to the network,
8. settle derived states again,
9. export and return the final network state.

### Implicit transient equations

FullFlow uses backward Euler. For:

```python
(state, derivative)
```

the physical equation is:

```text
dy/dt = f(y, z, t)
```

and one timestep from `tn` to `tn+1` solves:

```text
y(n+1) - y(n) - dt * f(y(n+1), z(n+1), t(n+1)) = 0
```

The derivative is evaluated at the new time and new state, which makes the
method implicit. Algebraic variables `z(n+1)` are solved simultaneously with the
dynamic states.

For the three-entry form:

```python
(iteration_state, integrated_state, derivative)
```

SciPy varies the new `iteration_state`, the component computes the corresponding
new `integrated_state`, and FullFlow solves:

```text
integrated_state(n+1)
- integrated_state(n)
- dt * derivative(n+1)
= 0
```

This is why a fluid volume can iterate pressure while conserving mass, or
iterate temperature while conserving internal energy.

### Transient residual normalization

FullFlow normalizes only the internally generated backward-Euler integration
residual. For each dynamic equation it uses:

```text
scale = max(
    abs(state_new),
    abs(state_previous),
    abs(dt * derivative_new),
    1
)
```

and returns:

```text
r_dynamic = (
    state_new
    - state_previous
    - dt * derivative_new
) / scale
```

This gives the transient `rtol` a useful relative meaning across quantities with
different units and magnitudes. Component balance residuals and user `Balance`
residuals remain exactly as the component returns them, so those should still be
scaled by the model author when necessary.

### One transient timestep

For each requested timestep, FullFlow:

1. copies the last accepted state forward as the initial new-time guess,
2. sets `network.time` to the trial new time,
3. applies sequence commands and timestep context,
4. evaluates the predictor residual,
5. accepts immediately if that residual already satisfies `rtol`,
6. otherwise solves all dynamic and algebraic residuals with least squares,
7. recomputes the final residual independently of SciPy's cached result,
8. accepts only if `max(abs(residual)) <= rtol`,
9. stores `previous` history only after acceptance,
10. rolls back all mutable states if the step fails.

A normal transient call is:

```python
history = Transient(network).solve(
    dt=0.001,
    t_final=0.5,
    save_dt=0.01,
    solver_method="trf",
    jacobian_method="3-point",
    ftol=1e-12,
    xtol=1e-12,
    gtol=None,
    rtol=1e-8,
    state_max_passes=5,
    state_tolerance=1e-10,
    max_step_retries=8,
    verbose=True,
)
```

The transient solver uses `gtol=None` by default. This disables gradient-only
termination, which can otherwise report convergence on small normalized
residuals before the explicit timestep residual test has been met.

If a timestep fails, FullFlow restores the last accepted state and retries with
half the timestep. Retrying continues up to `max_step_retries` and no smaller
than `minimum_dt`. If `minimum_dt` is omitted, the retry floor is
`dt * 1e-9`.

The nominal timestep may also be shortened to land exactly on:

- `t_final`,
- saved-output times from `save_dt`,
- tabular `Sequence` breakpoints,
- scheduled sequence-abort times.

### Quasi-steady time sweeps with `SteadyState(dt, t_final)`

`SteadyState.solve()` can march through time while forcing dynamic components to
steady state at every time point:

```python
history = SteadyState(network).solve(
    dt=0.01,
    t_final=2.0,
    save_dt=0.05,
)
```

This uses the transient runtime so time-dependent commands, sequences, sensors,
and exports behave like a time history. However, it is not ordinary transient
integration.

For a normal transient dynamic equation, the residual is:

```text
state_new - state_previous - dt * derivative_new = 0
```

For a dynamic component forced steady in the quasi-steady sweep, the residual is
instead:

```text
derivative_new = 0
```

Thus the network is re-trimmed at every time point. It has no accumulation or
inertia for forced-steady components, regardless of `dt`. The timestep controls
when time-dependent inputs are sampled and when results are saved; it does not
restore the discarded dynamic physics.

Selected components can retain their normal transient integration with
`exceptions`:

```python
SteadyState(network).solve(
    dt=0.01,
    t_final=2.0,
    exceptions=[Rotor, WallNode],
)
```

In that example, all other dynamic components satisfy `derivative = 0` at each
step, while `Rotor` and `WallNode` keep their backward-Euler equations. Results
use the normal transient HDF5 time-history layout.

Use a quasi-steady sweep when commands or boundary conditions change slowly and
the neglected storage/inertia is intentionally assumed instantaneous. Use
`Transient` when startup lag, accumulation, water hammer, thermal soak, rotor
acceleration, actuator motion, or another time-dependent effect matters.

### Solver methods

| Method | Characteristics |
| --- | --- |
| `"trf"` | Trust Region Reflective. Recommended default; supports bounds and large sparse-style problems well. |
| `"dogbox"` | Rectangular trust-region method; supports bounds but may be less effective for rank-deficient problems. |
| `"lm"` | Levenberg-Marquardt. Unbounded only and generally requires at least as many residuals as variables. |

### Tolerances and controls

| Option | What it controls | Steady default | Transient default |
| --- | --- | ---: | ---: |
| `ftol` | SciPy termination based on relative cost-function improvement. | `1e-8` | `1e-12` |
| `xtol` | SciPy termination based on the size of the unknown-vector step. | `1e-8` | `1e-12` |
| `gtol` | SciPy termination based on gradient/optimality. | `1e-8` | `None` |
| `rtol` | FullFlow acceptance test on `max(abs(final_residual))`. | `1e-2` | `1e-8` |
| `state_max_passes` | Maximum component passes used to settle derived states inside each residual call. | `5` | `5` |
| `state_tolerance` | Fixed-point tolerance for derived-state settling. | `1e-10` | `1e-10` |
| `max_step_retries` | Half-step retries after a failed timestep. | time sweeps only: `8` | `8` |
| `minimum_dt` | Smallest retry timestep. | time sweeps only | `dt * 1e-9` when omitted |

`ftol`, `xtol`, and `gtol` tell SciPy when to stop iterating. They do not replace
FullFlow's final `rtol` check. A solve can satisfy a SciPy termination condition
and still be rejected because the physical residuals are too large.

### Bounds in both solvers

For every iteration variable, FullFlow collects:

```python
state.lower_bound
state.upper_bound
state.keep_feasible
```

and passes them to SciPy. The same state-bound mechanism is used in steady state,
normal transients, and quasi-steady time sweeps.

### Invalid trial points

Finite-difference Jacobian estimation can temporarily test an invalid property
state. For example, a pressure-enthalpy flash may fail at a trial point even
though the current solution is valid.

After at least one valid residual has been evaluated, FullFlow converts later
invalid trial points into a large directional penalty residual so the
trust-region algorithm can move away from them. The initial model state is not
hidden this way: if the first evaluation is invalid, FullFlow raises the original
configuration/property error so the user can fix the initial condition.

### Practical convergence guidance

When a solve fails:

1. run `static_evaluate()` and fix unassigned or invalid inputs,
2. improve initial guesses,
3. add realistic bounds,
4. inspect the largest residual labels with `verbose=True` or
   `statistics=True`,
5. normalize algebraic residuals with sensible physical scales,
6. check that every unknown has an independent equation,
7. reduce transient `dt` when the physical dynamics are too stiff or commands
   change too abruptly,
8. increase `state_max_passes` only when connected explicit outputs genuinely
   need more settling passes,
9. change Jacobian or solver methods only after the model equations and scaling
   are sound.

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

## Lookup callables, live inputs, and priority switching

`Lookup` connects FullFlow states to normal Python functions, classes, property
packages, and reusable external model objects. The callable is reevaluated from
the current state values during network evaluation, so it remains coupled to the
nonlinear iteration and transient timestep.

### Calling a normal function

A lookup function can return a mapping:

```python
from fullflow import Lookup, Network, State


def ideal_gas_properties(pressure, temperature, gas_constant):
    return {
        "density": pressure / (gas_constant * temperature),
        "enthalpy": 1005.0 * temperature,
    }


GasNetwork = Network("Gas Properties")
pressure = State(2.0e5)
temperature = State(300.0)

Gas = Lookup(
    "Gas",
    GasNetwork,
    ideal_gas_properties,
    pressure=pressure,
    temperature=temperature,
    gas_constant=287.0,
    outputs=("density", "enthalpy"),
)

print(Gas.density.value)
print(Gas.enthalpy.value)
```

`Gas.density` and `Gas.enthalpy` are `LookupAttribute` objects. They behave like
state-like values and can be passed directly to components or used in state
math:

```python
mass = Gas.density * volume
```

### Calling a class

A class constructor is also a callable. This is the usual pattern for
ThermoProp:

```python
from thermoprop import Fluid

Water = Lookup(
    "Water",
    GasNetwork,
    Fluid,
    "water",
    pressure=pressure,
    temperature=temperature,
)

Pipe = DarcyWeisbach(
    "Pipe",
    GasNetwork,
    mass_flow=State(0.1),
    upstream_pressure=State(3.0e5),
    downstream_pressure=State(2.0e5),
    length=1.0,
    hydraulic_diameter=0.01,
    density=Water.density,
    cross_sectional_area=7.85e-5,
    friction_factor=0.02,
)
```

On the first evaluation, `Lookup` constructs the class with the resolved
positional and keyword inputs. If the returned object has an `update(...)`
method and the callable structure has not changed, later evaluations normally
reuse the existing object rather than rebuilding it. If the object also has a
zero-argument `solve()` method, Lookup can call it after updating. This is useful
for property packages that are designed for repeated state changes.

### Inputs update every solver iteration

Lookup inputs may be:

- plain constants,
- `State` objects,
- `LookupAttribute` outputs from another lookup,
- another `Lookup`,
- nested tuples, lists, or dictionaries containing those objects.

Immediately before calling the wrapped function/class, Lookup recursively
resolves the current values. Therefore this remains live:

```python
NodeFluid = Lookup(
    "Node Fluid",
    network,
    Fluid,
    "rp-1",
    pressure=node_pressure,
    temperature=node_temperature,
)
```

When SciPy changes `node_pressure`, the next residual evaluation calls or updates
`Fluid` with that new pressure. Lookup outputs then update before downstream
component residuals are collected.

Lookup objects participate in `pre_evaluation()` and normal repeated state
settling. They defer evaluation when an upstream state is not assigned yet and
retry after other components have produced the missing value.

Inputs can also be changed after construction:

```python
NodeFluid.pressure = another_pressure_state
NodeFluid.update(temperature=350.0)
```

Assigning an accepted input preserves a dynamic state link when one is supplied;
it does not intentionally freeze the current numeric value.

### Output access and state behavior

Lookup returns the wrapped object as:

```python
NodeFluid.value
NodeFluid.obj
NodeFluid()
```

Attributes are exposed lazily:

```python
NodeFluid.pressure
NodeFluid.temperature
NodeFluid.density
NodeFluid.enthalpy
```

A `LookupAttribute` supports:

- `.value`,
- numeric conversion,
- bounds forwarding when backed by a state,
- state math,
- comparisons,
- `.as_state()` for component wiring.

When a lookup attribute is passed into a component constructor,
`Component.setup()` requests its state view. Active lookup inputs remain
connected to their backing input states; outputs remain derived from the current
lookup result.

### Priority input groups

Many property objects accept several valid state pairs. During model setup,
temperature is often the easiest quantity to guess. During the actual solve,
enthalpy may be the conserved quantity that should control the property state.

`priority` lets one lookup begin with one input and automatically switch to a
preferred input when it becomes available:

```python
NodeFluid = Lookup(
    "Node Fluid",
    network,
    Fluid,
    "rp-1",
    pressure=101325.0,
    temperature=300.0,
    priority=("enthalpy", "temperature"),
)
```

The names in one priority tuple are mutually exclusive and ordered from highest
to lowest priority.

For the example above, Lookup behaves as follows:

1. `enthalpy` is preferred, but it is not assigned yet.
2. Lookup falls back to the supplied `temperature=300.0`.
3. The first pressure-temperature property evaluation produces enthalpy.
4. Lookup seeds a backing enthalpy input state from that output.
5. On later evaluations, enthalpy is available and becomes the active input.
6. Temperature is removed from the callable keyword inputs and becomes a normal
   output of the pressure-enthalpy state.

This allows a storage component to use:

```python
Node = Volume(
    "Node",
    network,
    pressure=NodeFluid.pressure,
    enthalpy=NodeFluid.enthalpy,
    temperature=NodeFluid.temperature,
    ...
)
```

without manually constructing a separate startup lookup. The model can start
from an intuitive pressure-temperature guess and then solve through
pressure-enthalpy once the initial state exists.

The first item in each priority group is the solver-preferred input. When its
attribute is requested as a state, Lookup promotes or creates a writable backing
input state while preserving available output guesses and bounds. Lower-priority
items stay derived so they can transition from startup inputs to outputs.

### Multiple independent priority groups

A lookup can have more than one mutually exclusive input group:

```python
Property = Lookup(
    "Property",
    network,
    SomePropertyClass,
    pressure=pressure_guess,
    temperature=temperature_guess,
    density=density_guess,
    priority=[
        ("enthalpy", "temperature"),
        ("specific_volume", "density"),
    ],
)
```

Within each group, Lookup passes only the first currently available candidate.
Names from lower-priority alternatives are removed from the keyword arguments so
the wrapped callable does not accidentally receive an over-specified state such
as pressure, enthalpy, and temperature simultaneously.

Only define priority combinations that are valid for the wrapped callable. The
callable's accepted keyword names are inspected when possible, and a preferred
output can be promoted only if the callable can accept that name as an input.

### Chained lookups

Lookup outputs can feed other lookups directly:

```python
ReactantsLookup = Lookup(
    "Reactants",
    network,
    Reactants,
    fuels=fuel,
    oxidizers=oxidizer,
    mixture_ratio=mixture_ratio,
)

EquilibriumLookup = Lookup(
    "Equilibrium",
    network,
    Equilibrium,
    reactants=ReactantsLookup,
    mode="hp",
    pressure=chamber_pressure,
)
```

When the upstream lookup changes, the downstream lookup sees the current
resolved object during the same repeated state-evaluation process.

### Lookup caching and reuse

Lookup internally avoids unnecessary work by:

- tracking whether dynamic inputs changed,
- caching the most recent resolved input key,
- reusing an existing output object through `update(...)` when possible,
- memoizing recent results,
- deferring until required inputs become available.

These behaviors are automatic in the public constructor. Component authors
should still assume `evaluate_states()` can be called many times and keep
external callables deterministic for the same input values.

### Lookup debugging

Useful inspection helpers include:

```python
NodeFluid.help()              # wrapped signature and docstring
NodeFluid.wrapped_signature
NodeFluid.wrapped_doc
NodeFluid.obj                 # current wrapped object
```

For a priority lookup, the current selection is also included in Lookup's
export/debug metadata. If the wrong state pair is active, check that the
preferred input is assigned, accepted by the callable, and not derived from an
unavailable upstream value.

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
