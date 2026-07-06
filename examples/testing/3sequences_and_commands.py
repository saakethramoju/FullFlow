from pathlib import Path

import h5py

from fullflow import *
import fullplot as fplt


"""
Example 3: Sequence command traces, condition switching, and clean aborts.

This example shows the procedure side of the test-data API.

The intended mental model is simple:

    Sensor owns condition lines.
    Sequence owns command traces.
    Sensor events can activate later Sequence commands.
    Sequence.abort(...) is the explicit way to stop a run cleanly.

The example has one commanded valve and one simple pressure state.

    1. The valve starts closed.
    2. At t = 0.30 s, the valve switches to an opening command trace.
    3. Chamber pressure rises in response to the valve command.
    4. CHPT crosses a redline named "High Pc Abort".
    5. The same redline event activates a closing command trace.
    6. Sequence.abort(...) lets the closing command run for 0.75 seconds.
    7. The transient stops cleanly and the output HDF5 file records that the
       run ended because of a Sequence abort.

The tiny pressure component below is only a demonstration plant. In a real
FullFlow model, the chamber pressure would come from volumes, flow devices,
combustion, nozzles, turbomachinery, etc. The Sequence API is the same.
"""


HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "sequence_commands_abort_result.h5"


# ---------------------------------------------------------------------------
# Small demonstration plant
# ---------------------------------------------------------------------------

class PressurePlant(Component):
    """A tiny first-order pressure response used only for this example."""

    def __init__(
        self,
        name,
        network,
        pressure,
        valve_command,
        ambient_pressure=14.7,
        full_open_pressure=380.0,
        response_time=0.35,
    ):
        self.pressure_dot = 0.0
        self.setup()

    def evaluate_states(self):
        valve = max(0.0, min(1.0, self.valve_command.value))
        target_pressure = self.ambient_pressure.value + self.full_open_pressure.value * valve
        self.pressure_dot = (target_pressure - self.pressure.value) / self.response_time.value

    @property
    def dynamics(self):
        return [(self.pressure, self.pressure_dot)]


# ---------------------------------------------------------------------------
# Network States
# ---------------------------------------------------------------------------

Test = Network("Sequence Commands And Abort")

valve_command = State(0.0)
chamber_pressure = State(14.7)

Plant = PressurePlant(
    "Simple Pressure Plant",
    Test,
    pressure=chamber_pressure,
    valve_command=valve_command,
)


# ---------------------------------------------------------------------------
# Sensor condition
# ---------------------------------------------------------------------------

# This is a redline, but redlines no longer automatically stop the solver.
# They are high-severity Sensor events. The Sequence below decides what to do
# when the event happens.
high_pc_abort = fplt.Trace(
    x=[0.0, 5.0],
    y=[350.0, 350.0],
    name="High Pc Abort",
    role="redline",
)

CHPT = Sensor(
    "CHPT",
    Test,
    reading=chamber_pressure,
    conditions=high_pc_abort,
)


# ---------------------------------------------------------------------------
# Sequence command traces
# ---------------------------------------------------------------------------

hold_closed = fplt.Trace(
    x=[0.0, 0.5],
    y=[0.0, 0.0],
    name="Hold Closed Command",
    role="command",
)

open_valve = fplt.Trace(
    x=[0.0, 0.4, 1.0, 5.0],
    y=[0.0, 1.0, 1.0, 1.0],
    name="Open Valve Command",
    role="command",
)

close_valve = fplt.Trace(
    x=[0.0, 0.20, 0.60, 2.0],
    y=[1.0, 0.25, 0.0, 0.0],
    name="Close Valve Command",
    role="command",
)

Start = Sequence("Start Sequence", Test)

# This command is active immediately at t = 0. It keeps the valve closed
# until another command for the same target becomes active.
Start.command(
    valve_command,
    hold_closed,
)

# A numeric condition is an absolute simulation time. This command starts at
# t = 0.30 s. Its trace time is local to that activation time, so trace x=0
# corresponds to simulation t=0.30.
Start.command(
    valve_command,
    open_valve,
    condition=0.30,
)

# This command becomes active only after CHPT crosses its condition named
# "High Pc Abort". Once active, it is evaluated after the opening command and
# therefore becomes the command that actually drives valve_command.
Start.command(
    valve_command,
    close_valve,
    condition=(CHPT, "High Pc Abort"),
)

# This cleanly stops the transient after the abort command has had time to run.
# HDF5 export still happens and the metadata records the abort.
Start.abort(
    condition=(CHPT, "High Pc Abort"),
    delay=0.75,
    message="High Pc abort sequence complete.",
)


# ---------------------------------------------------------------------------
# Track useful channels
# ---------------------------------------------------------------------------

Test.track("Valve Command", valve_command)
Test.track("Chamber Pressure", chamber_pressure)


# ---------------------------------------------------------------------------
# Run the transient
# ---------------------------------------------------------------------------

Transient(Test).solve(
    dt=0.01,
    t_final=5.0,
    verbose=True,
    filename=str(RESULT_FILE),
)


# ---------------------------------------------------------------------------
# Inspect abort metadata saved in HDF5
# ---------------------------------------------------------------------------

# This is not required for normal plotting. It is included so users can see
# where the clean abort is recorded for automated post-processing.
with h5py.File(RESULT_FILE, "r") as h5:
    metadata = h5["Sequence_Commands_And_Abort/transient/runs/base/metadata"]

    keys = [item.decode() if isinstance(item, bytes) else str(item) for item in metadata["key"][:]]
    values = [item.decode() if isinstance(item, bytes) else str(item) for item in metadata["value"][:]]

    print("\nAbort metadata saved in the result HDF5 file:")
    for key, value in zip(keys, values):
        if "abort" in key or key == "stop_reason":
            print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# Plot the command, pressure, and redline
# ---------------------------------------------------------------------------

result = fplt.open(str(RESULT_FILE)).at("Sequence_Commands_And_Abort/transient/runs/base")
#result.tree() # uncomment to see the result file structure

result_time = result.time("time")

pc_result = result.trace(
    y="tracks/Chamber_Pressure",
    x=result_time,
    name="Chamber Pressure",
    role="data",
)

valve_result = result.trace(
    y="tracks/Valve_Command",
    x=result_time,
    name="Valve Command",
    role="command",
)

result.plot(
    y=[pc_result, high_pc_abort],
    y2=valve_result,
    xlabel="Time [s]",
    ylabel="Chamber pressure [psia]",
    y2label="Valve command [-]",
    title="Sequence Command Switching and Clean Abort",
)

fplt.show()
