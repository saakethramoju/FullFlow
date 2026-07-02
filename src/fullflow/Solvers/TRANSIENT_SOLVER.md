# FullFlow transient solver

The transient solver advances a FullFlow network through time. It is used with:

```python
Transient(network).solve(dt=0.01, t_final=1.0)
```

The current transient solver is a fixed-step implicit backward-Euler solver. The user chooses the nominal timestep. The solver may shorten a step to land exactly on final time, output times, sequence breakpoints, or retry steps after a failed nonlinear solve.

## What transient solving means in FullFlow

FullFlow components can expose:

```text
dynamics
    Real storage, inertia, capacitance, or rotor-speed equations.

balances
    Algebraic equations that must be satisfied at each timestep.
```

In transient mode, dynamic equations are integrated. Algebraic equations are still solved at each new time.

For a dynamic equation:

```python
(state, derivative)
```

one backward-Euler timestep solves:

```text
state_new - state_previous - dt * derivative_new = 0
```

The derivative is evaluated at the new time using the current nonlinear solver guess. This is what makes the method implicit.

For the expanded dynamic form:

```python
(variable, integrated_state, derivative)
```

FullFlow varies `variable` but integrates `integrated_state`:

```text
integrated_state_new - integrated_state_previous - dt * derivative_new = 0
```

This is useful when the physical state is mass or energy, but pressure or temperature is a better nonlinear iteration variable.

## Public files in the transient package

The transient implementation is split into small files:

```text
solver.py
    Public Transient class and user-facing solve() method.

runtime.py
    Builds the transient runtime view of the Network.

operations.py
    Performs initialization and one implicit timestep solve.

evaluation.py
    Repeatedly evaluates components until derived States settle.

settings.py
    Stores timestep and retry settings.

models.py
    Handles Model / ModelOption selection and snapshots initial State values.

diagnostics.py
    Prints final transient summaries and failure information.

results.py
    Formats time-history records.
```

`solver.py` is the public entry point. The other files keep the solver internals organized.

## Runtime cache

Before a transient run, the solver builds a `TransientRuntimeCache`. The cache identifies:

```text
which dynamics are integrated
which dynamics are forced steady
which component balances are active
which user Balance objects are active
which States are nonlinear variables
which States need previous values stored
which labels should appear in diagnostics
which bounds should be passed to SciPy
```

The cache is solver-side. The `Network` remains a simple container.

## Dynamic mode

Transient solving now exposes only normal dynamic integration through the public
`Transient.solve()` API:

```text
state_new - state_previous - dt * derivative_new = 0
```

Forced-steady time sweeps live under `SteadyState.solve(dt=..., t_final=...)`.
That API uses the transient runtime internally and stores its HDF5 output under
`transient/runs/...` with `run_mode = "forced_steady_time_sweep"` metadata.

## User balances during transient

User `Balance` objects remain algebraic constraints during transient solves. If a balance is active, its residual is closed at every timestep.

`ignore_balances` can remove user balances from a solve:

```python
Transient(network).solve(..., ignore_balances="all")
Transient(network).solve(..., ignore_balances=["Jet Boundary"])
```

This affects user `Balance` objects only. Component balances are not removed by `ignore_balances`.

## Initialization

At the start of a transient solve, FullFlow evaluates the network and stores the initial state in the history.

The initial condition is important. Transient integration assumes the values already in the States are the starting physical condition. For dynamic States, this is the starting value to integrate from. For algebraic variables, this is the initial guess for the first nonlinear solve.

The solver then advances from the current `network.time` value to `t_final`.

## Timestep selection

The user supplies a nominal timestep:

```python
dt=0.01
```

The solver can shorten a step for four reasons:

```text
1. final time
   The last step lands exactly on t_final.

2. Sequence breakpoints
   If a tabular Sequence has a scheduled time inside the step, the solver lands on it.

3. output times
   If save_dt is used, the solver lands exactly on saved output times.

4. retry after failure
   If a nonlinear step fails, the solver rolls back and retries with half the step.
```

The solver does not automatically grow the timestep. This keeps timestep accuracy under user control.

## One implicit timestep

For one accepted step, the process is:

```text
1. Current time is t_old.
2. The solver chooses t_new and dt_step.
3. Previous values are saved from the accepted old state.
4. network.time is set to t_new.
5. SciPy receives an initial guess vector x0.
6. SciPy proposes a new vector x.
7. FullFlow writes x into the nonlinear variable States.
8. Components are evaluated repeatedly until derived States settle.
9. Dynamic integration residuals are formed.
10. Component balance residuals are formed.
11. User Balance residuals are formed.
12. SciPy iterates until the residual is small enough or it terminates.
13. FullFlow checks the final residual against rtol.
14. If accepted, new values become the current state.
15. If rejected, the old state is restored and a smaller timestep is tried.
```

The key idea is that dynamic integration and algebraic closure happen together at the new time. This avoids the explicit method problem where algebraic variables lag behind the dynamic state.

## Residual order

A transient residual vector generally contains:

```text
1. integration residuals for active dynamics
2. derivative residuals for force-steady dynamics
3. component algebraic balance residuals
4. user Balance residuals
```

The exact labels are collected by the runtime cache and used in failure messages.

## Integration residual scaling

The internally generated integration residuals are normalized by a state/change scale. This helps a temperature state, pressure state, and mass state use the same numerical acceptance check without requiring component authors to write their own scaling code.

Algebraic residuals are used exactly as the component or `Balance` returns them. Therefore, when users write dimensional custom balances, it is often best to write the residual in a physically meaningful scale, such as:

```python
(chamber_pressure - target_pressure) / target_pressure
```

For dimensionless closures like:

```python
throat_mach - 1.0
```

no extra scaling is needed.

## Nonlinear solver settings

Each timestep uses `scipy.optimize.least_squares` through the same basic settings object as steady state.

The transient defaults are stricter than the steady-state defaults:

```text
solver_method   = "trf"
jacobian_method = "3-point"
ftol            = 1e-12
xtol            = 1e-12
gtol            = None
rtol            = 1e-8
state_max_passes = 5
state_tolerance = 1e-10
```

`gtol=None` disables gradient-based termination by default. This avoids false convergence when the residual is small but not yet accepted by FullFlow's per-step residual check.

`rtol` is the final per-timestep residual acceptance tolerance. After SciPy terminates, FullFlow recomputes the residual and requires:

```text
max(abs(residual)) <= rtol
```

## Step retry

If a timestep fails, the solver restores the old accepted state and tries again with half the step.

The retry settings are:

```python
max_step_retries=8
minimum_dt=None
```

If `minimum_dt` is not supplied, the retry floor is:

```text
dt * 1e-9
```

If the solver reaches the retry limit or retry floor, it raises an error showing the largest residuals from the failed step.

## Output history

The transient solver always stores the initial evaluated state. It then stores accepted timesteps.

If `save_dt=None`, every accepted step is saved.

If `save_dt` is provided, the solver still advances with its normal timestep, but output is saved only at multiples of `save_dt` and final time. Steps are shortened when needed so saved output lands exactly on the requested times.

When `filename=...` is provided, transient results are written under:

```text
/<network_name>/transient/runs/base
```

The HDF5 output contains:

```text
time
component histories
tracked histories
table histories
diagnostics
final component values
final tracked values
final table values
metadata
```

## Sequence breakpoints

The transient solver scans components for tabular `Sequence` breakpoints. If a known schedule time lies inside the next step, the solver shortens the step to land exactly on that schedule point.

This prevents a step from jumping over an important discontinuity in a scheduled valve area, heat load, command signal, or other table-driven input.

## Model options

Transient model-option runs are independent runs. Before evaluating model options, FullFlow captures the initial State values. Before each option is run, the snapshot is restored. This prevents one option's transient history from contaminating the next option's initial condition.

The solver can use the first option that succeeds or evaluate every option.

## Verbose and statistics output

`verbose=True` prints the final transient summary and final network state.

`statistics=True` prints accepted timestep progress while the solve is running. This is useful when a long transient is being debugged.

## Difference from steady state

Steady state solves:

```text
derivative = 0
algebraic residual = 0
```

Transient solves:

```text
state_new - state_previous - dt * derivative_new = 0
algebraic residual_new = 0
```

So a component with `dynamics` behaves differently in the two solvers. In steady state, its derivative is trimmed to zero. In transient, its state changes over time according to that derivative.

## Common modeling implications

A thermal wall with heat capacity should usually be run transient. A full steady-state trim would force the wall heat rate to zero, which may not be the desired initial condition.

A gas path with only algebraic components can be solved by `SteadyState` or by the algebraic part of a transient step. If the model has dynamic wall nodes coupled to an algebraic gas path, the physical initial condition is often:

```text
gas path algebraically consistent
wall temperatures user specified
wall heat rates nonzero
```

That is not the same as a full steady thermal state.

## Basic transient solve sequence

A normal transient solve follows this sequence:

```text
1. User calls Transient(network).solve(dt=..., t_final=...).
2. Solver applies model-option selection if requested.
3. TransientRuntimeCache is built.
4. Dynamic mode summary is added to run metadata.
5. Initial network states are evaluated.
6. Initial state is saved to transient history.
7. Solver chooses the next timestep.
8. One implicit backward-Euler step is solved.
9. If the step fails, the network rolls back and retries with a smaller dt.
10. If the step succeeds, the new state is accepted.
11. Accepted state is saved if output is due.
12. Steps repeat until t_final.
13. Final records and diagnostics are written to HDF5 if filename was provided.
14. Verbose output is printed if requested.
15. The time-history records are returned.
```

This is the core transient solver model.
