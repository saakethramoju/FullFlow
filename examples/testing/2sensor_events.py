from pathlib import Path

import numpy as np

from fullflow import *
import fullplot as fplt


"""
Example 2: Sensor condition events.

This example shows how Sensors use FullPlot condition traces during a transient.

A Sensor can own four kinds of condition traces:

    role="greenline"   target / healthy / success line
    role="blueline"    advisory / expected event / permissive line
    role="yellowline"  warning / caution line
    role="redline"     high-severity safety line

All four roles are event lines. None of them stop the transient by themselves.
They are checked only after accepted transient steps. When the Sensor reading
crosses one of the lines, FullFlow:

    - prints a clear event in verbose mode,
    - stores the sparse event in the output HDF5 file,
    - stores the condition trace definition in the output HDF5 file,
    - allows Sequence.command(...) or Sequence.abort(...) to use the condition.

This file only demonstrates the Sensor event side. The next example shows how
Sequence commands and aborts can respond to those events.
"""


HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "sensor_events_result.h5"


# ---------------------------------------------------------------------------
# Network and simple pressure profile
# ---------------------------------------------------------------------------

Test = Network("Sensor Events")

# In a real model this would come from a chamber, tank, line, or turbopump
# component. For this event demonstration, a command trace drives the pressure
# State directly so the line crossings are easy to see.
chamber_pressure = State(14.7)

pc_profile = fplt.Trace(
    x=[0.0, 0.4, 0.9, 1.4, 1.8, 2.3, 3.8, 4.6, 5.2],
    y=[14.7, 70.0, 180.0, 295.0, 370.0, 325.0, 300.0, 95.0, 14.7],
    name="Commanded Chamber Pressure Profile",
    role="command",
)

PressureDriver = Sequence("Pressure Driver", Test)
PressureDriver.command(chamber_pressure, pc_profile)


# ---------------------------------------------------------------------------
# Sensor condition traces
# ---------------------------------------------------------------------------

# All of these traces use the same time base. They could also come from an
# HDF5 file or from any other FullPlot workflow.
line_time = np.array([0.0, 5.2])

greenline = fplt.Trace(
    x=line_time,
    y=np.array([60.0, 60.0]),
    name="Startup Pc Reached",
    role="greenline",
)

blueline = fplt.Trace(
    x=line_time,
    y=np.array([150.0, 150.0]),
    name="Ignition Confirmed",
    role="blueline",
)

yellowline = fplt.Trace(
    x=line_time,
    y=np.array([320.0, 320.0]),
    name="High Pc Warning",
    role="yellowline",
)

redline = fplt.Trace(
    x=line_time,
    y=np.array([360.0, 360.0]),
    name="High Pc Redline",
    role="redline",
)

CHPT = Sensor(
    "CHPT",
    Test,
    reading=chamber_pressure,
    conditions=[greenline, blueline, yellowline, redline],
)

Test.track("CHPT", chamber_pressure)


# ---------------------------------------------------------------------------
# Run the transient
# ---------------------------------------------------------------------------

Transient(Test).solve(
    dt=0.02,
    t_final=5.2,
    verbose=True,
    filename=str(RESULT_FILE),
)


# ---------------------------------------------------------------------------
# Plot the result and condition lines
# ---------------------------------------------------------------------------

result = fplt.open(str(RESULT_FILE)).at("Sensor_Events/transient/runs/base")
#result.tree() # uncomment to see the result file structure

result_time = result.time("time")
pc_result = result.trace(
    y="tracks/CHPT",
    x=result_time,
    name="CHPT Simulation",
    role="data",
)

# The original FullPlot condition traces can be plotted directly with the result
# trace. Their roles determine how FullPlot styles them.
result.plot(
    y=[pc_result, greenline, blueline, yellowline, redline],
    xlabel="Time [s]",
    ylabel="Chamber pressure [psia]",
    title="Sensor Condition Events",
)

fplt.show()
