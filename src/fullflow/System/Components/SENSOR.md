# Sensor measurements and conditions

`Sensor` represents simulated instrumentation and optionally compares it with
FullPlot traces. Its three main uses are reading a model State, anchoring an
independent model variable to measured data, and detecting excursions across
redline, blueline, greenline, or yellowline traces.

## Reading a simulated channel

```python
from fullflow import Network, Sensor, State

network = Network("Instrumentation")
chamber_pressure = State(2.0e6)
pressure_sensor = Sensor("PC", network, reading=chamber_pressure)
```

The reading can be an expression, such as `chamber_pressure / psi`, so the
trace and condition thresholds use the intended engineering units.

## Anchoring to test data

```python
import fullplot as fplt

measured = fplt.Trace(
    x=[0.0, 1.0, 2.0], y=[2.0e6, 2.1e6, 2.05e6],
    name="Measured pressure", role="data",
)

pressure_sensor = Sensor(
    "PC", network, reading=chamber_pressure,
    variable=chamber_pressure, data=measured,
)
```

When anchoring is active, `Sensor` contributes a nonlinear balance

```math
r(t)=p_\mathrm{model}(t)-p_\mathrm{data}(t)=0.
```

The chosen independent `variable` must be able to satisfy that balance as part
of the overall network. With `extend=True` (default), unavailable data deactivates
the anchor and retains the last accepted value of an otherwise unconstrained
variable. `extend=False` requests a clean transient stop when required data
is unavailable.

## Condition traces and Sequence interaction

```python
limit = fplt.Trace.constant(
    "Ignition permissive", x=[0.0, 1.0],
    y=180.0, role="blueline",
)

sensor = Sensor(
    "Injector Pressure", network,
    reading=injector_pressure / psi,
    conditions=limit,
)
```

A condition is checked against the accepted reading and the corresponding
FullPlot trace value. A crossing is recorded with its direction, time, sampled
value, threshold, and line role. Roles are `redline`, `blueline`, `greenline`,
and `yellowline`; a yellowline is reported as a warning. Conditions fire on
the first qualifying crossing by default; `trigger_all` allows repeated
crossings for specified traces.

Condition events are *observations*, not automatic solver aborts or precise
physical root searches. `Sequence.command(..., condition=...)` may latch a
command after a named condition has triggered, and
`Sequence.abort(condition=...)` can request a clean stop. For accurate
state-dependent bang-bang switching *between* timesteps, see the separate
`Sequence(..., event=True)` mechanism documented in `SEQUENCE.md`.

See `examples/transient/26engine_startup.py` and the sensor examples under
`examples/testing/`.
