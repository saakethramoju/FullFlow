# Transient solver

`Transient` advances a FullFlow network in time using fixed-nominal-step, implicit backward-Euler integration with simultaneous algebraic closure.

```python
from fullflow import Transient

transient = Transient(network)
result = transient.solve(dt=0.01, t_final=1.0)
```

The user chooses the nominal timestep. FullFlow can shorten steps at known schedule times or after a failed step; it does not automatically increase the nominal step.

## Equations

`Component.dynamics` describes physical storage or inertia, while `Component.balances` and user `Balance` objects describe algebraic constraints. All active equations are closed together at the new time.

For a dynamic equation `(state, derivative)`, backward Euler requires

```math
x_{n+1}-x_n-\Delta t\,f_{n+1}=0.
```

For `(variable, integrated_state, derivative)`, the solver adjusts `variable` while integrating `integrated_state`:

```math
s_{n+1}-s_n-\Delta t\,f_{n+1}=0.
```

This supports models in which pressure or temperature is the nonlinear unknown while mass or internal energy is the stored physical state.

At a timestep, the residual vector contains, in order:

1. Backward-Euler integration residuals (or zero-derivative residuals for components forced steady in a time sweep).
2. Component algebraic balances.
3. User-defined `Balance` residuals.

The runtime cache preserves row order, variable ownership, and labels for diagnostics. There must be at least as many residuals as iteration variables. `ignore_balances` can exclude **user** `Balance` objects, but does not disable component balances.

## Timesteps and events

The initial state is the state currently assigned in the network. The solver evaluates and records it before advancing. At each step it:

1. Selects the next time, limited by `t_final`, tabular `Sequence`/command breakpoints, sampled-controller updates, and pending abort times.
2. Saves the last accepted network state and evaluates time-dependent commands.
3. Solves the implicit equations at the new time.
4. Verifies the final scaled residuals, then commits the accepted state.
5. Checks `Sensor` events and operating-mode changes, records the state, and advances.

If a step fails, the solver restores the previous accepted state and halves the attempted timestep. By default it permits `max_step_retries=8` and uses a minimum retry timestep of `dt * 1e-9` unless `minimum_dt` is supplied. Exhausting the retry limit raises `TransientStepError` with diagnostic residuals. This is a **timestep integration safeguard**, not an alternative set of physical equations.

Tabular breakpoints prevent stepping over known discontinuities. Sampled `Sequence` controllers use exact integer-indexed sample times and hold commands between accepted updates without event searches. An event-enabled Sequence can locate discontinuous switching using additional implicit trial solves. Sensor and mode events invalidate cached Jacobians and equation scales as needed. A Sequence may also request an abort, which ends the run and records the stop reason. For controller timing and memory semantics, see [SEQUENCE.md](../System/Components/SEQUENCE.md).

## Nonlinear solution

For eligible unbounded systems, the primary timestep solver is safeguarded **Newton/Broyden** (`transient/newton.py`). It uses numerical Jacobians, line-search backtracking, Broyden updates, and reuse of valid information from preceding accepted steps. Eligibility depends on the current system and bounds, not on whether its components are built-in or custom Python classes.

### Automatic numerical sparsity

For sufficiently large unbounded systems, FullFlow examines a full numerical Jacobian to identify residual dependencies and groups nonoverlapping columns for subsequent finite differences. For three-point differentiation, one color requires two perturbed residual evaluations regardless of how many independent columns it contains. The grouping is learned from actual residual behavior; custom Python components use the same mechanism as built-ins.

Every grouped Jacobian is checked against independent signed full-network perturbations. If the predicted residual changes disagree, FullFlow discards the grouping and computes a dense Jacobian. Dense or small systems continue to use dense differences. Square systems with a sufficiently sparse matrix may use sparse linear algebra automatically; the residual acceptance threshold is unchanged. Discontinuous or poorly scaled problems may require more dense rebuilds, so sparsity is not assumed to be permanent.

For square systems with at least 48 unknowns, a factored Newton matrix can be reused across Broyden rank-one corrections using the Sherman–Morrison formula. Ill-conditioned corrections are rejected and a fresh factorization or Jacobian is used. Small square and overdetermined systems retain their existing dense linear-solve paths.

The existing `scipy.optimize.least_squares` path is retained for bounded or otherwise ineligible systems and for recovery if the primary solver cannot close a step. Its equation-scaling Jacobian is reused at the initial point instead of recomputed, and compatible unbounded sparse systems use independently checked colored differences in subsequent SciPy iterations. The solver may backtrack from nonfinite Newton trial states without accepting an invalid residual. A successful transient run therefore does **not** imply that SciPy was used at every timestep.

The default nonlinear settings are:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `solver_method` | `"trf"` | Method for the SciPy path |
| `jacobian_method` | `"3-point"` | Default numerical Jacobian method |
| `ftol` | `1e-12` | SciPy cost-change tolerance |
| `xtol` | `1e-12` | SciPy step-size tolerance |
| `gtol` | `None` | SciPy gradient stopping disabled by default |
| `rtol` | `1e-8` | Maximum scaled-residual acceptance threshold per timestep |
| `state_max_passes` | `5` | Maximum passes to settle derived component values |
| `state_tolerance` | `1e-10` | Convergence threshold for those evaluation passes |

When a sufficiently large time-commanded operating-mode change is detected, the solver temporarily uses one-sided (`"2-point"`) finite differences instead of the requested symmetric (`"3-point"`) differences. This can avoid derivatives spanning different flow regimes. The recovery window is bounded to sixteen accepted steps after the most recent qualifying change; the requested method is restored afterward. Additional command changes restart the window. The verbose summary distinguishes the requested method from steps evaluated with the temporary method. Residual acceptance tolerances are unchanged.

Successful nonlinear termination alone is insufficient: the computed state is accepted only when its verified scaled residuals meet `rtol`. A failed trial can trigger numerical refinement and, if needed, step rejection. FullFlow reports actual solver statistics and raw residuals; it does not treat an unsuccessful iterate as a converged state.

### Event-probe Jacobian reuse

For `Sequence(event=True)`, multiple trial solves may share the same accepted starting state and pre-switch command. A converged Newton/Broyden trial can supply a Jacobian estimate to a nearby trial when the equation layout, unknown ordering, Jacobian method, and sensor-balance configuration agree. The trial timestep must also be comparable to the previous one.

The solver checks the transferred Jacobian against the actual residual change before using it. A rejected estimate triggers normal Jacobian construction; an accepted estimate is still subject to nonlinear iterations, line searching, and the unchanged final residual test. No Component-specific derivatives or annotations are needed. Each precisely located switch may still require several full-network solves.

### Work diagnostics

Accepted-step solver counts do not include every rejected, speculative, or event-localization evaluation. `transient.work_statistics` reports whole-solve counts for implicit residual calls, Newton attempts and Jacobian builds, SciPy calls, event probes, and post-switch algebraic reconciliation. The counts also distinguish dense and colored Jacobian builds and rejected colorings. The accepted-step diagnostic table retains its own counters separately.

The verbose summary distinguishes these scopes. HDF5 stores the whole-solve counters as `solver_work_*` entries in the run metadata table. SciPy Jacobian counts use the counts reported by completed SciPy solves; the full residual-call count includes numerical differentiation and failed attempts. The `one_sided_jacobian_steps` statistic counts implicit solves performed with the temporary one-sided recovery policy, including retries and event probes.

## Residual scaling

FullFlow handles dynamic integration residuals and user algebraic equations differently.

For an integrated state `s`, its previous value `s_old`, derivative `f`, and attempted timestep `dt`, the integration residual is

```math
R_{\mathrm{BE}}=\frac{s-s_{\mathrm{old}}-dt\,f}
{dt\,\max(|s|,|s_{\mathrm{old}}|,|dt\,f|,1)}.
```

This normalization is built into the transient runtime. These integration rows are **not** scaled a second time.

Algebraic component balances, user `Balance` equations, and forced-steady derivative rows use the shared automatic **Jacobian-based equation scaling**. For unknown scales `s_x,j = max(abs(x_j), 1)`, the raw Jacobian defines row sensitivities

```math
q_i=\max_j(|J_{ij}|s_{x,j}).
```

Usable row scales are bounded between `1e-10` and the lesser of `1e12` and 1,000 times the median usable sensitivity. Invalid or negligible sensitivities retain a row scale of one. The solver works with `r_i / s_r,i`, keeping scales fixed within each nonlinear solve.

When the network, operating mode, controls, timestep, or residual layout changes, prior row scales and Jacobians are invalidated as appropriate. Smooth, compatible steps can reuse them. The nonlinear solve and final acceptance test use scaled residuals, while reports and HDF5 retain the original physical equations and **raw** values.

A dimensional expression such as

```python
chamber_pressure - ambient_pressure
```

is valid as written. FullFlow does not require manual normalization. However, numerical scaling cannot determine a physical error budget automatically, and different weights can change the least-squares optimum when an inconsistent system has no exact root.

## State evaluation and invalid trial points

At each trial unknown vector, FullFlow writes iteration values into `State` objects, applies current commands, and evaluates components until derived values settle. The same rules apply to custom and built-in components.

A property calculation may fail at a temporary trial state. After a valid residual has been established, FullFlow can supply a penalized residual to discourage invalid trial points. It cannot recover from an invalid initial model by inventing physical properties.

## History, output, and linearization

The solver retains the initial state and every accepted timestep in its history. When `filename=...` is specified, HDF5 results are written under:

```text
/<network_name>/transient/runs/base
```

The export includes time, component histories, tracked states, sensors, diagnostics, final values, and metadata. Existing FullPlot data access continues to use these physical values.

Accepted operating-point checkpoints also preserve internal model state for `transient.linearize(...)`. To limit memory growth, unchanged large NumPy arrays in these checkpoints can share stored snapshots; changed arrays are copied. Restoring a checkpoint still creates independent mutable values. An event or mode transition may prevent linearization at the affected point. See [LINEARIZATION.md](LINEARIZATION.md).

`verbose=True` prints the final summary and network state. `statistics=True` prints accepted-step progress and solver diagnostics. Model-option runs start from restored initial conditions, so alternative options are evaluated independently.

## Forced-steady sweeps

`Transient.solve()` performs normal dynamic integration. To advance a time-dependent model while forcing selected dynamics to zero, use:

```python
SteadyState(network).solve(dt=0.01, t_final=2.0)
```

Use `exceptions=[...]` to retain genuine transient integration for selected components. This shares the timestep solver but differs physically from a fully transient model; its HDF5 metadata identifies the forced-steady mode.

## Implementation files

- `transient/solver.py`: time loop, retries, sensor events, and history
- `transient/runtime.py`, `evaluation.py`: equation discovery, residual assembly, and component evaluation
- `transient/operations.py`: timestep solve, numerical scaling, prediction, and acceptance
- `transient/newton.py`: safeguarded Newton/Broyden algorithm
- `transient/models.py`, `results.py`: model-option state and history
- `transient/settings.py`, `diagnostics.py`: step controls and reporting
- `equation_scaling.py`: shared numerical row-scaling implementation
