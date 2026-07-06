from pathlib import Path
import math

import h5py
import numpy as np

from fullflow import *
import fullplot as fplt


"""
Example 1: importing test data and anchoring a simulation to it.

This example shows the basic FullPlot + FullFlow test-data workflow:

    1. Generate or open a normal HDF5 test-data file.
    2. Build FullPlot Trace objects from the test-data channels.
    3. Give measured data traces role="data".
    4. Attach those data traces to FullFlow Sensor components.
    5. Let the transient solver adjust selected model States so the simulation
       follows the measured data at each solver time.
    6. Save the anchored simulation to HDF5.
    7. Plot the imported test data and the simulation result with FullPlot.

The physical model is intentionally small. It is only a single liquid fuel line
represented by a discharge-coefficient relation:

    tank pressure -> restriction -> injector pressure

The test data provides:

    FTPT       fuel tank pressure, psia
    FIPT       fuel injector pressure, psia
    FUEL_MDOT  fuel mass flow, kg/s

The sensors do three different anchoring jobs:

    FTPT Sensor:
        Adjusts the model upstream pressure until it matches FTPT test data.

    FIPT Sensor:
        Adjusts the model downstream pressure until it matches FIPT test data.

    FFLOW Sensor:
        Adjusts the model discharge coefficient until model mass flow matches
        the FUEL_MDOT test-data trace.

This is the same pattern a user would use to make a model follow real test data
while still solving unknown model quantities such as Cd, valve area, pressure
loss, heat input, etc.
"""


# ---------------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------------

# Keep this True the first time you run the example. After the file exists, set
# it False if you want to keep reusing the same synthetic test-data file.
GENERATE_TEST_DATA = True

HERE = Path(__file__).resolve().parent
TEST_DATA_FILE = HERE / "test_data_anchoring_hotfire.h5"
RESULT_FILE = HERE / "test_data_anchoring_result.h5"

psia_to_pa = 6894.76


# ---------------------------------------------------------------------------
# Synthetic test-data generation
# ---------------------------------------------------------------------------

def smooth_step(t, start, stop):
    """Smooth 0-to-1 ramp used only for creating the synthetic data file."""
    x = np.clip((t - start) / (stop - start), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def write_channel(h5, name, values, units, description):
    """Write one ordinary test-data channel to the HDF5 file."""
    dataset = h5.create_dataset(name, data=np.asarray(values), compression="gzip", shuffle=True)
    dataset.attrs["units"] = units
    dataset.attrs["description"] = description


def generate_test_data(filename):
    """Create a simple HDF5 file that behaves like converted hotfire data."""
    rng = np.random.default_rng(12)

    # Use a pre-start region so the example can demonstrate FullPlot's
    # time-zero shifting. The run starts at t = 1 s in raw test time.
    dt = 0.005
    time = np.arange(-1.0, 8.0 + dt, dt)

    # A simple command-like profile. This is not used by FullFlow in this
    # example; it is just used to make realistic pressure and flow histories.
    open_ramp = smooth_step(time, 0.0, 0.8)
    close_ramp = 1.0 - smooth_step(time, 5.8, 6.5)
    run_level = open_ramp * close_ramp

    # Pressures are in psia because many test systems store pressure channels
    # that way. We convert to SI before giving the traces to FullFlow.
    ftpt = 525.0 - 18.0 * run_level + 1.5 * np.sin(2.0 * np.pi * 0.7 * time) * run_level
    fipt = 310.0 + 16.0 * run_level + 1.0 * np.sin(2.0 * np.pi * 0.9 * time + 0.4) * run_level

    ftpt += rng.normal(0.0, 0.45, len(time))
    fipt += rng.normal(0.0, 0.45, len(time))

    # The synthetic mass-flow channel is generated from the same restriction
    # equation used in the model, but with a time-varying true Cd and a little
    # measurement noise.
    density = 810.0
    area = (math.pi / 4.0) * (0.25 / 39.37) ** 2
    true_cd = 0.18 + 0.54 * run_level
    pressure_drop = np.maximum((ftpt - fipt) * psia_to_pa, 0.0)
    fuel_mdot = true_cd * area * np.sqrt(2.0 * density * pressure_drop)
    fuel_mdot += rng.normal(0.0, 0.006, len(time))
    fuel_mdot = np.maximum(fuel_mdot, 0.0)

    with h5py.File(filename, "w") as h5:
        h5.attrs["description"] = "Synthetic hotfire data for FullFlow Sensor anchoring example"
        h5.attrs["time_units"] = "s"
        h5.attrs["sample_rate_hz"] = 1.0 / dt

        write_channel(h5, "time", time, "s", "Raw test time")
        write_channel(h5, "FTPT", ftpt, "psia", "Fuel tank pressure")
        write_channel(h5, "FIPT", fipt, "psia", "Fuel injector pressure")
        write_channel(h5, "FUEL_MDOT", fuel_mdot, "kg/s", "Fuel mass flow")

    print(f"Wrote synthetic test-data file: {filename}")


if GENERATE_TEST_DATA or not TEST_DATA_FILE.exists():
    generate_test_data(TEST_DATA_FILE)


# ---------------------------------------------------------------------------
# Import the test data with FullPlot
# ---------------------------------------------------------------------------

# FullPlot opens the HDF5 file and gives the user lightweight Trace objects.
# The HDF5 file does not need to be a FullFlow file. It only needs ordinary
# datasets that FullPlot can read.
data = fplt.open(str(TEST_DATA_FILE))

# Use the test-data time channel as the x-axis for each trace. The raw test file
# starts before ignition, so shift the time base so t = 0 means ignition/start.
time = data.time("time")
time.zero_at(0.0)

# Trace roles matter. Sensor.data only accepts role="data" traces.
ftpt_raw = data.trace(y="FTPT", x=time, name="FTPT Test Data", role="data")
fipt_raw = data.trace(y="FIPT", x=time, name="FIPT Test Data", role="data")
fflow = data.trace(y="FUEL_MDOT", x=time, name="Fuel Flow Test Data", role="data")

# A user can process a test-data trace before anchoring to it. The filtered trace
# keeps the same role as the source trace, so it is still valid Sensor.data.
ftpt_filtered = ftpt_raw.filter("moving_average", window=0.04)
fipt_filtered = fipt_raw.filter("moving_average", window=0.04)

# Only use the useful part of the test. Windowing keeps the trace role too.
ftpt = ftpt_filtered.window(start=0.0, stop=6.5) * psia_to_pa
fipt = fipt_filtered.window(start=0.0, stop=6.5) * psia_to_pa
fflow = fflow.window(start=0.0, stop=6.5)

# Conditions are optional in this example. They demonstrate that a Sensor can
# both anchor to data and log line-crossing events during the same run.
fflow_warning = fplt.Trace.constant(
    "Low Fuel Flow Warning",
    x=time,
    y=0.55,
    role="yellowline",
)

fflow_redline = fplt.Trace.constant(
    "Low Fuel Flow Redline",
    x=time,
    y=0.40,
    role="redline",
)

# Plot the imported data before building the model. This is the normal first
# sanity check when working with test data.
data.plot(
    y=[ftpt / psia_to_pa, fipt / psia_to_pa],
    y2=[fflow, fflow_warning, fflow_redline],
    xlabel="Time [s]",
    ylabel="Pressure [psia]",
    y2label="Fuel mass flow [kg/s]",
    title="Imported FullPlot Test Data",
)


# ---------------------------------------------------------------------------
# Build a small FullFlow model to anchor to the data
# ---------------------------------------------------------------------------

Test = Network("Test Data Anchoring")

# These States are the model values that the sensors will adjust.
fuel_tank_pressure = State(ftpt(0.0))
fuel_injector_pressure = State(fipt(0.0))
fuel_cd = State(0.50)

fuel_density = State(810.0)
fuel_line_area = (math.pi / 4.0) * (0.25 / 39.37) ** 2

Line = DischargeCoefficient(
    "Fuel Line",
    Test,
    upstream_pressure=fuel_tank_pressure,
    downstream_pressure=fuel_injector_pressure,
    density=fuel_density,
    discharge_coefficient=fuel_cd,
    cross_sectional_area=fuel_line_area,
    mass_flow=fflow(0.0),
)

# Sensor.data traces must have role="data".
# Sensor.variable is the State the solver is allowed to move.
FTPT = Sensor(
    "FTPT",
    Test,
    reading=Line.upstream_pressure,
    variable=Line.upstream_pressure,
    data=ftpt,
)

FIPT = Sensor(
    "FIPT",
    Test,
    reading=Line.downstream_pressure,
    variable=Line.downstream_pressure,
    data=fipt,
)

FFLOW = Sensor(
    "FFLOW",
    Test,
    reading=Line.mass_flow,
    variable=Line.discharge_coefficient,
    data=fflow,
    conditions=[fflow_warning, fflow_redline],
)

# Track the most important user-facing channels with clean names. These are
# stored under /tracks in the output HDF5 file.
Test.track("Fuel Tank Pressure", Line.upstream_pressure)
Test.track("Fuel Injector Pressure", Line.downstream_pressure)
Test.track("Fuel Mass Flow", Line.mass_flow)
Test.track("Estimated Fuel Cd", Line.discharge_coefficient)


# ---------------------------------------------------------------------------
# Run the anchored transient
# ---------------------------------------------------------------------------

Transient(Test).solve(
    dt=0.02,
    t_final=6.5,
    verbose=True,
    filename=str(RESULT_FILE),
)


# ---------------------------------------------------------------------------
# Plot the anchored simulation result
# ---------------------------------------------------------------------------

result = fplt.open(str(RESULT_FILE)).at("Test_Data_Anchoring/transient/runs/base")
#result.tree() # uncomment to see the result file structure

result_time = result.time("time")

model_flow = result.trace(
    y="tracks/Fuel_Mass_Flow",
    x=result_time,
    name="Model Fuel Flow",
    role="data",
)

sampled_test_flow = result.trace(
    y="sensors/FFLOW/data_value",
    x=result_time,
    name="Sampled Test Fuel Flow",
    role="data",
)

estimated_cd = result.trace(
    y="tracks/Estimated_Fuel_Cd",
    x=result_time,
    name="Estimated Fuel Cd",
    role="data",
)

result.plot(
    y=[model_flow, sampled_test_flow],
    y2=estimated_cd,
    xlabel="Time [s]",
    ylabel="Fuel mass flow [kg/s]",
    y2label="Estimated Cd [-]",
    title="Anchored Model Result",
)

fplt.show()
