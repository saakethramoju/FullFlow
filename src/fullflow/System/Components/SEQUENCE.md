# Sequence controllers and scheduling

`Sequence` supplies commands to a FullFlow network using tables, Python functions,
or FullPlot command traces. A function-based Sequence supports continuously coupled
feedback, discrete switching, and sampled digital control without requiring a
separate controller class.

## Select the controller timing

| Mode | Signature | Evaluation and timing |
| --- | --- | --- |
| Schedule | `times=..., values=...` | Interpolated at solver times; known breakpoints are respected |
| Continuous | `function=..., inputs=...` | Uses current trial State values during implicit residual evaluations |
| Step-driven switch | Function reads its own target as an input | Reads newly accepted measurements after a step; holds the previous command inside each solve |
| Localized switch | `function=..., event=True` | Holds command through trial solves and locates detected changes between timesteps |
| Sampled digital | `function=..., sample_rate=...` | Reads accepted plant States at exact sample boundaries; holds command between samples |

Controller functions remain normal Python callables. FullFlow does not implement
PID or LQR-specific equations. `sample_rate` is measured in updates per second
(Hz), **not** seconds between updates.

## Continuous feedback

For continuous proportional control:

```python
def pressure_control(t, pressure, setpoint):
    return gain * (setpoint - pressure)

Sequence(
    "Pressure Feedback", network,
    target=command, function=pressure_control,
    inputs=[pressure, setpoint],
)
```

The function is evaluated using the nonlinear trial values at time
$t_{n+1}$. In a backward-Euler step, the coupled problem is

$$
x_{n+1}=x_n+\Delta t\,f(x_{n+1},u_{n+1}),\qquad
u_{n+1}=g(x_{n+1}).
$$

There is no imposed one-timestep measurement delay. The feedback function must
be deterministic and side-effect-free because the nonlinear solver may call it
multiple times at a proposed timestep. Hard discontinuities should use the
switching mode instead.

## Bang-bang and localized switching

A controller using hysteresis needs the last accepted command:

```python
def bang_bang(t, pressure, previous_command):
    if pressure > upper_threshold:
        return 0.0
    if pressure < lower_threshold:
        return 1.0
    return previous_command

Sequence(
    "Pressurant Control", network,
    target=valve_command, function=bang_bang,
    inputs=[tank_pressure, valve_command], event=True,
)
```

Without `event=True`, the command updates only at accepted timestep boundaries.
With `event=True`, FullFlow estimates a crossing using inexpensive guard-input
interpolation and checks candidate times using implicit solves. The command
remains fixed during each nonlinear solve so the solver does not iterate across
a discontinuous valve equation. Every localized trial is checked against the
normal residual tolerance; event probes cost additional nonlinear work.

An event search cannot guarantee detection of multiple switches whose endpoints
appear identical within one proposed timestep. Use sufficient temporal
resolution and hysteresis for rapidly switching systems. The event locator is
not a substitute for modeling physical actuator dynamics.

## Sampled PID and other digital controllers

A digital controller updates every $T_s=1/f_s$ seconds. Its first callback is
at the initial simulation time plus $T_s$; the target must have an initial
value. Sample times use an integer index to avoid accumulated clock drift.

```python
def pid(t, pressure, setpoint, *, dt, memory):
    integral, last_error = memory
    error = setpoint - pressure
    integral += error * dt
    derivative = (error - last_error) / dt
    command = max(0.0, min(1.0, Kp*error + Ki*integral + Kd*derivative))
    return command, (integral, error)

Sequence(
    "Digital PID", network,
    target=valve_command, function=pid,
    inputs=[pressure, setpoint], sample_rate=100.0,
    memory=(0.0, 0.0),
)
```

This example uses the backward rectangular integral and backward difference
for the derivative:

$$
I_k=I_{k-1}+e_k T_s,\quad
D_k=\frac{e_k-e_{k-1}}{T_s},\quad
u_k=K_p e_k+K_i I_k+K_d D_k.
$$

The sample callback receives a copy of memory and returns `(command, new_memory)`.
FullFlow commits memory **once per accepted sample**, not during rejected steps,
Jacobian evaluations, or event-search probes. Anti-windup, derivative filtering,
limits, and feedforward belong in the user function. They are illustrated by
`examples/transient/26pid_throttle.py`.

A stateless sampled LQR simply returns its command (no memory is required). A
discrete-time LQR for $x_{k+1}=A_d x_k+B_d u_k$ can obtain its gain from the
discrete algebraic Riccati equation:

$$
K=(R+B_d^\mathsf{T}PB_d)^{-1}B_d^\mathsf{T}P A_d,
\qquad u_k=-Kx_k.
$$

The reference example `examples/transient/29thrust_vectoring_lqr.py` uses an actual
zero-order-held rocket pitch model, an LQR gain from SciPy, and a nonlinear
thrust-vector plant with actuator limits.

## Sampling, integration, and performance

- A sampled command is held between updates (zero-order hold). Sampling does not
  require event-root searches.
- If `sample_rate=100` and the requested plant step is `dt=0.1`, the solver must
  integrate at the intermediate 0.01 s boundaries for genuine new measurements.
  A high-rate controller can therefore make a large network more expensive.
- Multiple digital controllers can run at different frequencies. At coincident
  sample times, they read the same accepted pre-update plant state.
- A sampled Sequence cannot also have `event=True`, but distinct sampled and
  event-enabled Sequences can coexist in one network.
- The command changes at a sample boundary without an additional instantaneous
  algebraic reconciliation solve. Dependent algebraic output datasets may still
  reflect the left limit at that precise time; they update on the next solve.
- State-dependent event localization needs network checkpoints and additional
  solves. Avoid enabling it for smooth feedback or purely digital sample clocks.

## Existing command traces and aborts

`Sequence.command(target, trace, condition=...)` continues to replay FullPlot
traces into State targets. `Sequence.abort(condition=..., delay=...)` remains
available for clean conditional stopping. These workflows are independent of
the sampled controller interface.

## Examples and compatibility

- `examples/transient/26pid_throttle.py` — sampled chamber-pressure PID with a
  throttle map, fuel/oxidizer actuators, and a transient engine network.
- `examples/transient/23closed_loop_control.py` — continuous battery heater feedback.
- `examples/transient/27reaction_control.py` — localized two-axis cold-gas spacecraft RCS switching.
- `examples/transient/29thrust_vectoring_lqr.py` — sampled rocket thrust-vector LQR.
- `examples/transient/5bang_bang_control.py` — detailed fluid-system pressure regulation.

The dedicated `PID` Component was removed. Applications using its continuously
integrated error State or public PID diagnostics must migrate explicitly;
sampled digital PID is not generally mathematically identical to continuous
PID. For continuous controller internal dynamics, use ordinary custom
Components with `dynamics` as before. Other physical Component APIs are unchanged.
