from fullflow import *

import numpy as np
import fullplot as fplt


"""
Sensor test-data balance example.

This example demonstrates a Sensor that uses a FullPlot Trace as test data.
When `variable` and `data` are both supplied, the Sensor acts like a Balance:

    residual = sensor reading - FullPlot trace value

The solver samples the FullPlot Trace with the solver time using the last
available test-data point. This means the solver dt does not need to match the
test-data dt.

The trace contains one dropout at t = 2 s. FullPlot windowed traces use the
same convention: missing or outside-window values are NaN while the full time
axis remains intact. When extend=True, FullFlow keeps running through the NaN
and holds the independent variable at its last accepted value. When
extend=False, the run stops cleanly at the last accepted timestep.
"""


SensorBalance = Network("Sensor Test Data Balance")

# This is the independent state the sensor is allowed to perturb. In a real
# model this could be an injector CdA, valve area, source pressure, pump speed,
# regulator setpoint, or any other assignable State.
effective_gain = State(0.0)

# This is the model variable that the virtual sensor reads. Here it is just a
# simple derived state so the example is easy to follow.
model_pressure = 2.0 * effective_gain

# FullPlot Trace object used as external test data.
pressure_data = fplt.Trace(
    x=[0.0, 1.0, 2.0, 3.0, 4.0],
    y=[10.0, 20.0, np.nan, 40.0, 50.0],
    name="CHPT Test Data",
)

CHPT = Sensor(
    "CHPT",
    SensorBalance,
    reading=model_pressure,
    variable=effective_gain,
    data=pressure_data,
    extend=True,
)

Transient(SensorBalance).solve(
    dt=1.0,
    t_final=4.0,
    filename="23sensor_test_data_balance",
    verbose=True,
)
