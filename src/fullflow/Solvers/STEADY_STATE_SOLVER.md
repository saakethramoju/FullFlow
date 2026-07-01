# FullFlow steady-state solver

The steady-state solver is the solver used when a FullFlow network should be brought to an algebraically consistent operating point. It is used with:

```python
SteadyState(network).solve()
```

The steady-state solver does not advance time. It changes selected `State` values until all active residual equations are close to zero.

## What steady state means in FullFlow

FullFlow components can expose two kinds of equations:

```text
dynamics
    Real storage, inertia, capacitance, or rotor-speed equations.
    Example: mass_dot, temperature_dot, rotor_speed_dot.

balances
    Algebraic equations with no physical storage.
    Example: pressure_error, mass_flow_error, throat_mach - 1.
```

In a steady-state solve:

```text
component dynamics -> derivative residuals are driven to zero
component balances -> algebraic residuals are driven to zero
Balance objects    -> user residuals are driven to zero
```

For example, a transient `Solid` may expose:

```python
@property
def dynamics(self):
    return [(self.temperature, self.temperature_dot)]
```

During steady state, the solver varies `temperature` until:

```text
temperature_dot = 0
```

A `Volume` may vary pressure or temperature until mass and energy accumulation rates are zero. A `Rotor` may vary rotor speed until net torque causes zero angular acceleration.

This is similar to the way many network solvers trim a dynamic model: the same component equations are used, but the dynamic derivatives are forced to zero instead of being integrated through time.

## Public files in the steady_state package

The steady-state implementation is split into small files:

```text
solver.py
    Public SteadyState class and user-facing solve() method.

runtime.py
    Builds the solver's runtime view of the Network.

operations.py
    Performs static evaluation and nonlinear least-squares solves.

evaluation.py
    Repeatedly evaluates components until derived States settle.

settings.py
    Small dataclasses for nonlinear solver and state-evaluation settings.

models.py
    Handles optional Model / ModelOption selection and fallback.

diagnostics.py
    Prints Rich tables for verbose output.

results.py
    Formatting helpers for returned records.

statistics.py
    Optional residual/iteration statistics collection.
```

`solver.py` is the only file most users need to know exists. The other files keep the internal behavior organized.

## Runtime cache

The network is intentionally simple. It stores components, balances, model options, tracked states, and time. It does not constantly maintain solver metadata.

Before a solve, the steady-state solver creates a `RuntimeCache`. The cache answers questions like:

```text
Which component dynamics are active?
Which component balances are active?
Which user Balance objects are active?
Which States are iteration variables?
What are their initial values?
What bounds do they have?
What residual labels should appear in diagnostics?
Which component methods should be called during evaluation?
```

The cache is rebuilt when the network version changes. This keeps residual calls fast during SciPy iteration while still allowing the network structure to change between solves.

## Equation discovery

Some components create derivative or balance-error attributes inside `evaluate_states()`. For example, a component may not create `self.mass_dot` until it computes mass conservation.

Because of that, the solver performs equation discovery like this:

```text
1. Run component pre_evaluation() hooks.
2. Try to run component evaluate_states().
3. If a component is waiting on an unassigned upstream State, defer it.
4. Repeat until all components can be evaluated or no progress is possible.
5. Read component.dynamics and component.balances.
```

This lets component constructors stay clean. They can usually just call `self.setup()`.

If equation discovery cannot complete because a needed input is still unassigned, the solver raises an error explaining which component is waiting for which kind of value.

## Iteration variables

The steady-state solver collects iteration variables from three places.

### 1. Component dynamics

For a dynamic equation:

```python
(variable, derivative)
```

steady state uses `variable` as an iteration variable and drives `derivative` to zero.

For the expanded form:

```python
(variable, state, derivative)
```

steady state still uses `variable` as the iteration variable and drives `derivative` to zero. The `state` field matters more in transient solves, where it is the actual integrated storage state.

### 2. Component balances

For a component balance:

```python
(variable, residual)
```

steady state uses `variable` as an iteration variable and drives `residual` to zero.

### 3. User Balance objects

A user balance:

```python
Balance("Name", network, variable=x, function=residual)
```

is normalized internally to the same form as a component balance.

## Variable overlap check

FullFlow rejects a user `Balance` variable that is already owned by a component dynamic or component balance. This avoids ambiguous models where two independent equations both claim the same variable.

For example, if a `Volume` already uses `pressure` as a component solve variable, a user should not also write:

```python
Balance("Extra Pressure Balance", network, variable=pressure, function=...)
```

The usual fix is to choose a different independent variable or remove one of the equations.

## State evaluation

During each residual call, the solver must compute all derived outputs for the current guess.

The process is:

```text
1. SciPy proposes a vector x.
2. RuntimeCache writes x into the iteration States.
3. StateEvaluator runs component evaluate_states() repeatedly.
4. The evaluator stops when non-iteration States stop changing or max passes is reached.
5. RuntimeCache collects residuals.
```

Repeated evaluation is needed because components can depend on other component outputs. A map may use a pressure from a volume. A flow component may use density from a lookup. A heat-transfer component may use the result of another heat-transfer component.

`state_max_passes` and `state_tolerance` control this fixed-point settling process.

## Residual collection

After the network states are evaluated, residuals are collected in a fixed order:

```text
1. component dynamic residuals
2. component balance residuals
3. user Balance residuals
```

The labels printed in diagnostics follow the same order. This is important because the residual vector passed to SciPy must match the diagnostic labels and iteration variables.

## Nonlinear solve

The steady-state solver uses `scipy.optimize.least_squares`.

The default settings are stored in `LeastSquaresSettings`:

```text
solver_method   = "trf"
jacobian_method = "3-point"
ftol            = 1e-8
xtol            = 1e-8
gtol            = 1e-8
rtol            = 1e-2
```

`ftol`, `xtol`, and `gtol` are SciPy convergence tolerances. `rtol` is FullFlow's final residual acceptance tolerance. Even if SciPy says it converged, FullFlow checks the final residual before accepting the solution.

The solver uses:

```text
x_scale="jac"
```

so the trust-region method can scale iteration steps based on the local Jacobian.

## Bounds

Bounds live on `State` objects. The runtime cache collects lower bounds, upper bounds, and `keep_feasible` flags and passes them to SciPy.

The default solver method is `trf` because it supports bounds. The Levenberg-Marquardt method `lm` is only allowed for unbounded problems.

## Invalid nonlinear trial points

Property packages can fail at temporary solver guesses. For example, a thermodynamic flash may fail at a negative pressure during a finite-difference Jacobian evaluation.

When there has already been a valid residual, FullFlow can return a large penalty residual for invalid trial points. This lets the trust-region solver back away instead of immediately crashing the solve.

If the initial point is invalid, FullFlow does not hide the error. It raises the original setup problem because there is no known-good residual to fall back on.

## Static evaluation path

If a network has no iteration variables and no residuals, there is nothing to solve. FullFlow can simply evaluate the components and save/return the results. This is called static evaluation.

Static evaluation is useful for networks that are pure calculators:

```text
inputs -> components -> tracked outputs
```

with no algebraic unknowns.

## Model options

A `Model` contains alternative components or groups of components. The steady-state solver can:

```text
use a specific model option
try options in order and use the first one that works
evaluate all options and save all successful results
```

Before running an option, the solver builds the selected components. Failed options are captured and reported so the user can see why an option was skipped.

## Saving results

When `filename=...` is provided, steady-state results are written to HDF5 under:

```text
/<network_name>/steady_state/runs/base
```

or, for model options:

```text
/<network_name>/steady_state/runs/<model_name>/<option_name>
```

The file stores component attributes, tracked outputs, result tables, metadata, and diagnostics.

## Verbose output

With `verbose=True`, the steady-state solver prints:

```text
solver summary
solution variables
residuals
network solution table
model-option failures if applicable
```

The verbose output is diagnostic. It does not change the solve.

## Basic solve sequence

A normal steady-state solve follows this sequence:

```text
1. User calls SteadyState(network).solve().
2. Solver applies model option selection if requested.
3. RuntimeCache is built from the current Network.
4. Components run pre_evaluation().
5. RuntimeCache is refreshed in case pre_evaluation changed anything.
6. Initial iteration vector x0 is collected.
7. Components are evaluated to settle derived States.
8. Initial residual r0 is collected.
9. If there is nothing to solve, static evaluation runs.
10. If residual count is less than variable count, the solver raises a setup error.
11. SciPy least_squares iterates on x.
12. Each residual call writes x into States and reevaluates the network.
13. Final residual is checked against FullFlow rtol.
14. Accepted solution values are written back into the Network.
15. Final states are evaluated one more time.
16. Results are printed, saved, and returned.
```

This is the core steady-state solver model.
