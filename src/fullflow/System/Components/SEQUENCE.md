# Sequence controllers and scheduling

`Sequence` supplies commands to a FullFlow network using tables, Python functions,
or FullPlot command traces. A function-based Sequence supports continuously coupled
feedback, discrete switching, and sampled digital control without requiring a
separate controller class.

## Select the controller timing

| Mode | Signature | Evaluation and timing |
| --- | --- | --- |
| Schedule | `times=..., values=...` | Interpolated at solver times; known breakpoints are respected |
| Accepted-step feedback | `function=..., inputs=...` | Reads previous accepted State values, preserving the existing default behavior |
| Continuous | `function=..., continuous=True` | Uses current trial State values during implicit residual evaluations |
| Step-driven switch | Function reads its own target as an input | Reads newly accepted measurements after a step; holds the previous command inside each solve |
| Localized switch | `function=..., event=True` | Holds command through trial solves and locates detected changes between timesteps |
| Sampled digital | `function=..., sample_rate=...` | Reads accepted plant States at exact sample boundaries; holds command between samples |

Controller functions remain normal Python callables. FullFlow does not implement
PID or LQR-specific equations. `sample_rate` is measured in updates per second
(Hz), **not** seconds between updates.

## Continuous feedback

This example defines the network, shared States, and proportional callback.
Connect `pressure` and `command` to the plant components in your model:

```python
from fullflow import Network, Sequence, State

network = Network("Continuous Pressure Control")
pressure = State(2.0e5)
setpoint = State(2.1e5)
command = State(0.0)
gain = 1.0e-5


def pressure_control(t, pressure, setpoint):
    return gain * (setpoint - pressure)

PressureFeedback = Sequence(
    "Pressure Feedback",
    network,
    target=command,
    function=pressure_control,
    inputs=[pressure, setpoint],
    continuous=True,
)
```

The function is evaluated using the nonlinear trial values at time
$t_{n+1}$. In a backward-Euler step, the coupled problem is

```math
x_{n+1}=x_n+\Delta t\,f(x_{n+1},u_{n+1}),\qquad
u_{n+1}=g(x_{n+1}).
```

There is no imposed one-timestep measurement delay with `continuous=True`.
By default, function-based Sequences instead read the previous accepted States,
which prevents their feedback from changing within a nonlinear trial. Continuous
feedback functions must be deterministic and side-effect-free because the solver
may call them multiple times at a proposed timestep. Hard discontinuities should
use accepted-step feedback or the event-driven switching mode instead.

## Bang-bang and localized switching

A controller using hysteresis needs the last accepted command. Here the
pressure thresholds are in pascals; connect `tank_pressure` and `valve_command`
to the tank and valve components:

```python
from fullflow import Network, Sequence, State

network = Network("Pressurant Control")
tank_pressure = State(2.9e5)
valve_command = State(0.0)
lower_threshold = 2.8e5
upper_threshold = 3.2e5


def bang_bang(t, pressure, previous_command):
    if pressure > upper_threshold:
        return 0.0
    if pressure < lower_threshold:
        return 1.0
    return previous_command

PressurantControl = Sequence(
    "Pressurant Control",
    network,
    target=valve_command,
    function=bang_bang,
    inputs=[tank_pressure, valve_command],
    event=True,
)
```

With the default accepted-step feedback, the command reads the last accepted
measurements and does not switch inside a nonlinear trial. With `event=True`,
FullFlow estimates a crossing using inexpensive guard-input
interpolation and checks candidate times using implicit solves. The command
remains fixed during each nonlinear solve so the solver does not iterate across
a discontinuous valve equation. If a switch is already due at the accepted
time, FullFlow applies all due commands together and reconciles the algebraic
states without integrating over a nearly zero-duration step. For a switch
bracketed extremely close to the accepted boundary, it first verifies the
right-hand command change with the physical States held fixed. A genuine
pressure-threshold crossing still uses implicit event localization; it is not
moved to the boundary merely because its time is close. Failed reconciliation
restores the accepted mode and States. Every localized trial is checked against
the normal residual tolerance; event probes cost additional nonlinear work.

An event search cannot guarantee detection of multiple switches whose endpoints
appear identical within one proposed timestep. Use sufficient temporal
resolution and hysteresis for rapidly switching systems. The event locator is
not a substitute for modeling physical actuator dynamics.

## Sampled PID and other digital controllers

A digital controller updates every $T_s=1/f_s$ seconds. Its first callback is
at the initial simulation time plus $T_s$; the target must have an initial
value. Sample times use an integer index to avoid accumulated clock drift.

This example initializes the gains and controller memory. Connect `pressure`
and `valve_command` to the plant; tune the gains for its units and dynamics:

```python
from fullflow import Network, Sequence, State

network = Network("Sampled Pressure Control")
pressure = State(2.0e5)
setpoint = State(2.1e5)
valve_command = State(0.0)
Kp = 1.0e-5
Ki = 2.0e-6
Kd = 1.0e-7


def pid(t, pressure, setpoint, *, dt, memory):
    integral, last_error = memory
    error = setpoint - pressure
    integral += error * dt
    derivative = (error - last_error) / dt
    command = max(0.0, min(1.0, Kp * error + Ki * integral + Kd * derivative))
    return command, (integral, error)

DigitalPID = Sequence(
    "Digital PID",
    network,
    target=valve_command,
    function=pid,
    inputs=[pressure, setpoint],
    sample_rate=100.0,
    memory=(0.0, 0.0),
)
```

This example uses the backward rectangular integral and backward difference
for the derivative:

```math
I_k=I_{k-1}+e_k T_s,\quad
D_k=\frac{e_k-e_{k-1}}{T_s},\quad
u_k=K_p e_k+K_i I_k+K_d D_k.
```

The sample callback receives a copy of memory and returns `(command, new_memory)`.
FullFlow commits memory **once per accepted sample**, not during rejected steps,
Jacobian evaluations, or event-search probes. Anti-windup, derivative filtering,
limits, and feedforward belong in the user function. They are illustrated by
`examples/transient/27pid_throttle.py`.

A stateless sampled LQR simply returns its command (no memory is required). A
discrete-time LQR for $x_{k+1}=A_d x_k+B_d u_k$ can obtain its gain from the
discrete algebraic Riccati equation:

```math
K=(R+B_d^\mathsf{T}PB_d)^{-1}B_d^\mathsf{T}P A_d,
\qquad u_k=-Kx_k.
```

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
- A sampled Sequence cannot also have `event=True` or `continuous=True`, but
  different controller modes can coexist in one network.
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

- `examples/transient/27pid_throttle.py` — sampled chamber-pressure PID with a
  throttle map, fuel/oxidizer actuators, and a transient engine network.
- `examples/transient/24closed_loop_control.py` — continuous battery heater feedback.
- `examples/transient/28reaction_control.py` — localized two-axis cold-gas spacecraft RCS switching.
- `examples/transient/29thrust_vectoring_lqr.py` — sampled rocket thrust-vector LQR.
- `examples/transient/6bang_bang_control.py` — detailed fluid-system pressure regulation.

The dedicated `PID` Component was removed. Applications using its continuously
integrated error State or public PID diagnostics must migrate explicitly;
sampled digital PID is not generally mathematically identical to continuous
PID. For continuous controller internal dynamics, use ordinary custom
Components with `dynamics` as before. Other physical Component APIs are unchanged.
