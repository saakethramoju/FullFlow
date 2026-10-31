# Linearization

FullFlow can calculate a local continuous-time state-space model directly from a successfully solved nonlinear network. The results describe how **small deviations** in selected inputs and physical dynamic states affect dynamic derivatives and outputs near a particular operating point.

This feature is useful for sensitivity analysis, local stability analysis, control-system design, and constructing approximate models of coupled fluid and thermal systems. It does **not** replace nonlinear transient integration, predict switching events, or automatically build a reduced-order simulator.

## Overview

For physical dynamic states `x`, independent inputs `u`, and selected outputs `y`, FullFlow returns

```math
\delta\dot{x}=A\,\delta x+B\,\delta u,\qquad
\delta y=C\,\delta x+D\,\delta u.
```

The deltas are **differences from the solved operating point**, not absolute values. The model also records `x0`, `xdot0`, `u0`, and `y0`. To predict absolute quantities near the operating point, use

```math
\dot{x}\approx\dot{x}_0+A(x-x_0)+B(u-u_0),\qquad
y\approx y_0+C(x-x_0)+D(u-u_0).
```

For a steady-state equilibrium, `xdot0` should be approximately zero. During a transient it can be nonzero. A transient linearization is a local, **frozen-time** approximation; it does not include the explicit time derivative of a scheduled command or changes in the active mode.

The matrices have these shapes:

| Matrix | Shape | Meaning |
| --- | --- | --- |
| `A` | `(n_states, n_states)` | State deviation to state derivative |
| `B` | `(n_states, n_inputs)` | Input deviation to state derivative |
| `C` | `(n_outputs, n_states)` | State deviation to output |
| `D` | `(n_outputs, n_inputs)` | Input deviation to output with physical states held fixed |

For a purely algebraic network without dynamic storage, `A`, `B`, and `C` have empty state dimensions, but `D` can still provide the feed-system input/output sensitivities. A successful `SteadyState.solve()` of a network with no solver unknowns can also be linearized.

## API

Linearization is a method on the existing solver instances, not a separate network type.

### Steady state

```python
steady = SteadyState(network)
steady.solve()

result = steady.linearize(
    inputs=[valve_area, supply_pressure],
    outputs=[chamber_pressure, mass_flow],
    filename="Linearization.h5",
    verbose=False,
)
```

Here the variables in the lists are **existing FullFlow `State` objects** attached to the active network, not strings or numerical values. Input States may be registered through component attributes or through `network.track(...)` when used in a standalone `Balance` expression. The solve must complete successfully before calling `linearize()`.

### Transient: final accepted point

```python
transient = Transient(network)
transient.solve(dt=0.01, t_final=30.0)

result = transient.linearize(
    inputs=[valve_area],
    outputs=[tank_pressure],
)
print(result.time)
```

With `times=None` (the default), FullFlow analyzes the **last accepted operating point**, which is usually `t_final` but may be earlier if the transient stopped early. If the final accepted point is itself a recognized switching event, linearization is rejected.

### Transient: multiple accepted times

```python
transient = Transient(network)
transient.solve(dt=0.01, t_final=30.0)

results = transient.linearize(
    inputs=[valve_area],
    outputs=[tank_pressure],
    times=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    filename="Linearization.h5",
)

A_at_10s = results[10.0].A
```

A `times` call returns a dictionary keyed by the **requested float times**. Requests must be unique and must match accepted solver timesteps (subject to a small floating-point time tolerance). No interpolation is performed.

Checkpoint retention is independent of `save_dt`:

- Every accepted transient timestep receives an in-memory checkpoint, whether or not that timestep is written to output history or HDF5.
- `save_dt` controls output sampling, not which accepted timesteps can be linearized.
- The initial and final operating points are retained as well. Recognized switching-event points are retained but cannot themselves be linearized.
- Requests that do not match an accepted timestep are rejected; exported HDF5 output alone cannot reconstruct a missing operating point.

**Memory cost:** retaining every accepted step means checkpoint memory and capture overhead grow with the number of timesteps and network complexity. For long, finely stepped simulations this can be substantial. The current API does not offer selective or on-disk checkpoint retention.

### Quasi-steady time sweeps

The `SteadyState.solve(dt=..., t_final=..., save_dt=...)` quasi-steady sweep uses the transient solver internally and retains its accepted operating points. `SteadyState.linearize(times=[...])` can analyze its converged equilibria at those accepted times. The usual steady-equilibrium residual requirement still applies; a static evaluation alone is not sufficient.

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `inputs` | `None` | Unique, independent, writable `State` objects to perturb. `None` means no selected inputs. |
| `outputs` | `None` | `State` objects or zero-argument callable outputs evaluated at the operating point. `None` means no selected outputs. |
| `filename` | `None` | Optional HDF5 output filename; `.h5` is added when omitted. |
| `times` | `None` | Accepted timestep sequence for transient or quasi-steady analysis; otherwise use the latest operating point. |
| `verbose` | `False` | Print labeled operating-point variables and matrices to the terminal. |

Do not use solver iteration variables or integrated physical storage states as independent inputs. An input cannot be generated/overwritten by another component during evaluation; it must be a genuinely independent network State. A callable output should be side-effect-free and require no arguments.

## Result object

An individual `LinearizationResult` exposes:

| Attribute | Contents |
| --- | --- |
| `A`, `B`, `C`, `D` | NumPy matrices |
| `x0`, `xdot0`, `u0`, `y0` | NumPy operating-point vectors |
| `states`, `inputs`, `outputs` | Ordered variable labels for matrix rows/columns |
| `time` | Operating-point time in seconds |
| `model_options` | Active model-option names |
| `network_name`, `solver_type` | Metadata for saving |
| `diagnostics` | Residual and numerical conditioning information |
| `jacobian` | Alias for `D`, i.e. the input/output Jacobian at fixed physical states |

The order of `states`, `inputs`, and `outputs` defines the corresponding matrix indices. Matrix values use the model's native units. They are **not automatically normalized** or converted to engineering units.

```python
print(result.states, result.inputs, result.outputs)
print(result.A.shape, result.B.shape, result.C.shape, result.D.shape)
result.print()
result.save("Another_Linearization.h5")
```

The singular or badly conditioned state transformation produces a `LinearizationError`, rather than an arbitrary pseudoinverse result.

## Mathematics

FullFlow networks can be expressed locally as a differential-algebraic system:

```math
\dot{x}=f(w,u),\qquad x=h(w,u),\qquad 0=g(w,u),\qquad y=q(w,u),
```

where `w` contains the solver's iteration variables. These variables need not be the same as the integrated physical states `x`; for instance, a volume can solve for pressure while integrating mass and internal energy.

FullFlow numerically differentiates the equations at the solved operating point and forms

```math
M=\begin{bmatrix} h_w \\ g_w \end{bmatrix}.
```

If `M` is square and nonsingular, the implicit function theorem gives

```math
\begin{bmatrix}w_x&w_u\end{bmatrix}
=M^{-1}\begin{bmatrix}I&-h_u\\0&-g_u\end{bmatrix}.
```

The final matrices are

```math
A=f_w w_x,\quad B=f_u+f_w w_u,\quad
C=q_w w_x,\quad D=q_u+q_w w_u.
```

The implementation uses central finite differences of the network's existing equations, with a second estimate at half the step size as a consistency check. It uses scaled dense linear algebra for the implicit transformation. It does not require analytical derivatives, symbolically differentiate user code, or re-solve the entire nonlinear network for every Jacobian column.

`D` is the **instantaneous** input/output sensitivity at fixed conserved states. It should not be confused with the eventual steady-state gain, which may include the slow state response.

## Operating modes and relief valves

An open relief valve and a closed relief valve can have **different local linearizations**. FullFlow analyzes the mode present at the selected operating point and tries to ensure perturbations do not change it.

- **Away from a switching threshold:** the closed or open valve can be linearized if its active equations are smooth and the local sensitivity system is solvable.
- **At a recognized discrete switching event:** linearization is rejected. Switches triggered by sensors or tracked Boolean/flag changes are not treated as differentiable dynamics.
- **Near a switching threshold:** a model is only valid for perturbations that remain in the same mode. The finite-difference checks cannot discover every arbitrary branch implemented by user code.
- **With hysteresis:** the same pressure may correspond to open or closed modes depending on the stored command/history. Each retained mode can be analyzed separately; a single model does not describe the transition.

FullFlow does **not** compute a hybrid-system transition map, saltation matrix, event-time derivative, or a model that predicts relief-valve opening and closing automatically. Use the nonlinear transient solver to simulate switching, and linearize selected intervals between switches.

## HDF5 output

An individual result is stored at a path resembling:

```text
<Network>/linearization/steady_state/runs/base/
    A, B, C, D
    x0, xdot0, u0, y0
    states, inputs, outputs, time
    model_options/
    diagnostics/
```

For a transient run the path uses `transient` instead of `steady_state`. When exactly one named model option is selected, the run path includes its model/option name instead of `base`.

For multiple operating points, results are written under numbered `points/0`, `points/1`, and so on, with each point's own `time` dataset.

`result.save(filename)` writes or replaces the individual result while preserving any saved `points/` group at the same run path. Multi-time saving replaces the existing `points/` group for the matching run path while preserving the base result. Use separate filenames to retain multiple different base analyses or point selections for the same run.

## Diagnostics and error handling

The returned `diagnostics` currently includes:

- `max_abs_balance_residual`: maximum absolute algebraic residual at the operating point.
- `max_abs_steady_derivative`: maximum absolute dynamic derivative for a steady-state linearization; reported as zero for transient results.
- `sensitivity_condition_number`: condition number of the scaled state transformation system.

`LinearizationError` is available from `fullflow.Solvers.linearization`. Typical failures include stale or unsolved operating points, non-independent inputs, inconsistent equations, invalid bounds, singular transformations, unavailable checkpoint times, and detected mode changes.

## Exact limitations

1. **Local only.** The matrices approximate small continuous deviations around one operating point. Large changes require another operating point or the nonlinear solver.
2. **Continuous-time only.** This is not a discrete-time timestep Jacobian and does not include backward-Euler integration or controller sampling updates.
3. **Fixed active equations and mode.** Switching events, bangs, hysteresis transitions, mode jumps, and explicit event-time sensitivities are not linearized.
4. **Numerical derivatives.** Derivatives can be unreliable for nonsmooth functions, sharp saturation, very small physical scales, property-package discontinuities, or noisy evaluations. Finite-difference consistency checks help but are not a mathematical guarantee.
5. **Locally solvable DAE required.** The physical state and algebraic constraints must provide one locally invertible transformation from iteration variables to physical states. Higher-index and structurally singular DAEs are not generally supported.
6. **No automatically generated model.** A result is a set of matrices and metadata; FullFlow does not turn it into a reduced-order simulator, transfer function, gain-scheduled controller, or lookup table for you.
7. **No automatic state selection or reduction.** All active physical dynamic states are retained. Choose only inputs and outputs; the internal state vector follows the component dynamics.
8. **Accepted-point requirement.** An earlier transient time must match an accepted solver timestep with an in-memory checkpoint from the same solver instance. Output sampling is independent; HDF5 traces alone do not reconstruct internal model/command history.
9. **Snapshot scope.** Checkpointing restores FullFlow State values/history, recognized mutable containers, and nested component attributes to a bounded depth. It cannot guarantee restoration of arbitrary external resources, closures, global variables, or custom side effects.
10. **No guarantee for arbitrary custom branches.** Mode checks recognize exposed discrete flags and equation changes, but cannot exhaustively analyze every user-written conditional.
11. **Special dynamic modes.** The transient `force_steady` configuration is not supported for ordinary transient linearization. Steady-state linearization requires a successful nonlinear equilibrium, except for networks with no solver equations, which can be analyzed after `SteadyState.solve()`. Calling `static=True` does not initialize a linearization checkpoint.
12. **Storage and scaling.** Checkpoints consume memory proportional to the number of retained points and the size of the network state. Dense sensitivity solves scale poorly for very large systems. Absolute residual tolerances and conditioning are affected by equation units/scaling.

## Practical validation

Before relying on a linearized production subsystem, compare the predicted response for small input perturbations against independently re-solved FullFlow operating points or a nonlinear transient. Verify matrix variable ordering, the operating mode, reasonable conditioning, and the range of perturbations for which errors remain acceptable.

The core linearization implementation does not depend on a particular fluid-property package; its reliability still depends on the smoothness and repeatability of the equations provided by your components and property lookup functions.
