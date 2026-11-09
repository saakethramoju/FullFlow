# Steady-state solver

`SteadyState` finds an operating point at which the active component derivatives and algebraic balances are zero.

```python
from fullflow import SteadyState

steady = SteadyState(network)
solution = steady.solve(verbose=True)
```

It does not integrate physical states through time. For time-dependent quasi-steady calculations, see [Time sweeps](#time-sweeps).

## Equations and iteration variables

FullFlow collects equations from the active `Component.dynamics`, `Component.balances`, and user-created `Balance` objects. The solver does not distinguish built-in components from custom Python components.

| Source | Equation supplied by component | Steady-state condition |
| --- | --- | --- |
| `dynamics` | `(variable, derivative)` | `derivative = 0` |
| `dynamics` | `(variable, integrated_state, derivative)` | `derivative = 0` |
| `balances` | `(variable, residual)` | `residual = 0` |
| `Balance` | `variable`, `function` | `function = 0` |

The first entry in each tuple supplies an iteration variable. In the three-element dynamic form, `integrated_state` matters during transient integration; steady state still adjusts `variable` to drive its derivative to zero.

Each iteration variable is collected once. Overlapping ownership, such as a `Balance` trying to solve a variable already owned by a component, is rejected. A system must supply at least as many residual equations as unknowns; inconsistent overdetermined systems are not guaranteed an exact root.

For example, a rotor's steady-state speed is the speed at which its angular acceleration is zero. A volume's mass and energy accumulation rates must likewise vanish.

## Automatic numerical sparsity

On large unbounded systems, FullFlow can identify a numerical sparsity pattern from its initial full Jacobian and group independent columns in subsequent Jacobian evaluations. Each grouped result is checked with fresh residual evaluations; if the pattern becomes invalid, the solver returns to dense numerical differentiation. This works with built-in and custom Python components without requiring derivative callbacks or dependency declarations. Bounded and small systems retain their existing differentiation path.

For large unbounded square systems, a sufficiently sparse initial Jacobian can also select SciPy's sparse trust-region linear solver automatically. Small, dense, bounded, and overdetermined problems retain the dense path. When the initial residual is exactly zero and the unknowns are within their bounds, no Jacobian or nonlinear iterations are needed.

These optimizations do not change the equations, solver tolerances, or final scaled-residual acceptance requirement.

## Runtime evaluation

The solver builds a runtime cache from the active network. The cache identifies component evaluation order, iteration variables, bounds, residual sources, and diagnostic labels. It is refreshed when the network structure changes.

Some components produce their equation attributes during evaluation rather than during construction. The solver therefore runs preparation hooks and settles dependent component values before collecting their equations.

At each nonlinear iteration:

1. Write the candidate unknown vector into its `State` objects.
2. Evaluate components until derived values settle or `state_max_passes` is reached.
3. Collect dynamic derivative residuals, component balance residuals, and user `Balance` residuals in a consistent order.
4. Evaluate the scaled residual vector for the nonlinear solver.

`state_max_passes` (default `5`) and `state_tolerance` (default `1e-10`) govern this inner state-evaluation process, **not** the nonlinear iteration count.

If there are no unknowns and no residuals, FullFlow evaluates the network directly without invoking a nonlinear solver.

## Nonlinear solver

Steady state uses `scipy.optimize.least_squares`, normally with the trust-region reflective method. The defaults are:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `solver_method` | `"trf"` | Nonlinear least-squares method |
| `jacobian_method` | `"3-point"` | Numerical Jacobian approximation |
| `ftol` | `1e-8` | SciPy cost-change tolerance |
| `xtol` | `1e-8` | SciPy step-size tolerance |
| `gtol` | `1e-8` | SciPy optimality tolerance |
| `rtol` | `1e-2` | FullFlow maximum scaled-residual acceptance threshold |

SciPy also uses `x_scale="jac"` to scale the unknowns. This is distinct from **equation scaling**, described below. Bounds declared on `State` objects are passed to compatible SciPy methods; `"lm"` cannot be used with bounded unknowns.

The solver accepts a solution only if SciPy reports success **and** the maximum absolute scaled residual does not exceed `rtol`. A successful SciPy termination by itself is not sufficient. Invalid intermediate property evaluations can be represented by a penalty based on an earlier valid trial; invalid initial states cannot be recovered that way.

## Automatic equation scaling

The nonlinear solver automatically scales equations, including arbitrary dimensional component and user `Balance` residuals. Users can write a pressure difference such as `chamber_pressure - ambient_pressure` directly; they do not need to divide it by a pressure reference.

Let `r(x)` be the raw residual vector and `J` its raw Jacobian with respect to the unknowns. With characteristic unknown scales

```math
s_{x,j}=\max(|x_j|,1),
```

FullFlow estimates each row's sensitivity:

```math
q_i=\max_j\left(|J_{ij}|s_{x,j}\right).
```

The row scales are derived from finite, non-negligible sensitivities, bounded below by `1e-10` and above by the lesser of `1e12` and **1,000 times the median usable row sensitivity** (with the implemented safeguards). Rows with unusable sensitivities retain a scale of one. This limit prevents an extreme initial Jacobian row from effectively removing an equation from the least-squares objective.

The nonlinear solver uses

```math
\hat r_i=r_i/s_{r,i},\qquad \hat J_{ij}=J_{ij}/s_{r,i}.
```

The scales stay fixed during an individual solve; they are not recomputed at every trial point. FullFlow reuses its initial numerical Jacobian when constructing the SciPy solve rather than building a separate, redundant Jacobian just for scaling.

Scaling does not change the exact zero of an individual equation. It **can** change the weighted least-squares optimum if no simultaneous zero exists. Automatic numerical scales also cannot encode the physical accuracy requirement for every possible custom equation. Check important physical results, not only the solver's success flag.

## Time sweeps

A quasi-steady time sweep evaluates the active model at successive times while forcing its dynamics to steady state:

```python
SteadyState(network).solve(dt=0.01, t_final=2.0)
```

Time-dependent `Sequence` inputs are evaluated as the simulation time advances. The implementation shares the transient stepping machinery, so it also uses automatic equation scaling and the transient nonlinear solve path. This is **not** the same as integrating every component dynamically.

Selected components can remain transient through `exceptions`:

```python
SteadyState(network).solve(
    dt=0.01,
    t_final=2.0,
    exceptions=[rotor, wall],
)
```

The named components retain backward-Euler integration equations; all other dynamic components use zero-derivative equations. Time-sweep output is stored in the transient HDF5 run layout, with metadata indicating forced-steady operation.

## Results and linearization

`filename=...` saves a static steady-state result under:

```text
/<network_name>/steady_state
```

Pass `group="flight_01"` to `solve()` or `static_evaluate()` to save under
`/<network_name>/flight_01/steady_state`. Nested groups are supported, and
reusing a group replaces only that group's steady-state result. Solver statistics
are stored alongside the result. Time sweeps use the corresponding
`/<network_name>/flight_01/transient` group.


Returned dictionaries and HDF5 contain physical component values and **raw** residuals; summary diagnostics also distinguish raw from scaled residual magnitudes. `verbose=True` prints the solve summary, variables, residuals, and network solution. `statistics=True` enables additional iteration diagnostics.

A successfully solved operating point can be linearized with `steady.linearize(...)`. Linearization independently checks the operating point's equations using scaled consistency criteria; it does not change the nonlinear solver's accepted state. See [LINEARIZATION.md](LINEARIZATION.md).

## Repeated steady-state solves

Use `Network.snapshot()` and `Network.restore(snapshot)` to reset a model before
independent steady-state parameter cases without recreating its Components:

```python
initial = network.snapshot()
steady = SteadyState(network)

for pressure in inlet_pressures:
    network.restore(initial)
    inlet_pressure.value = pressure
    result = steady.solve(filename=None)
```

Capture a snapshot after constructing the network.
The snapshot restores numerical and runtime state while preserving the identity
of each Component and State. It cannot be transferred to another Network or
restored after the Network structure changes. Custom
Components with state outside the Network may need a separate reset.

This reuse pattern avoids reconstructing the model, but does not guarantee
reuse of steady-state solver factorizations or Jacobians between solves. For
quasi-steady time sweeps, `SteadyState.solve(dt=..., t_final=...)` still uses
the transient stepping machinery. See [TRANSIENT_SOLVER.md](TRANSIENT_SOLVER.md)
and [`examples/transient/30monte_carlo.py`](../../../examples/transient/30monte_carlo.py).

## Implementation files

- `steady_state/solver.py`: public API and time-sweep dispatch
- `steady_state/runtime.py`, `evaluation.py`: equation discovery, caching, and component evaluation
- `steady_state/operations.py`: nonlinear solve, equation scaling, and final acceptance
- `steady_state/settings.py`, `statistics.py`, `diagnostics.py`, `results.py`: settings and reporting
- `equation_scaling.py`: shared numerical row-scaling implementation
