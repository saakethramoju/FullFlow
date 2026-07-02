"""
Generate Example HDF5 Plotting Data
===================================

Run this script before running the other plotting examples.

This script creates:

    examples/plotting/plotting_demo.h5

The repository ignores .h5 files, so this file should be generated locally
instead of committed to git.

The generated file is intentionally simple and generic. It is not meant to be a
real fluid network model. It is just smooth, simulation-like data that lets the
plotting examples demonstrate every major FullPlot feature.

The file layout is:

    plotting_demo.h5
    |
    |-- scalars/
    |      initial_pressure        scalar
    |      final_pressure          scalar
    |      final_mass_flow         scalar
    |
    |-- demo_transient/
    |      time                    1D, shape (501,)
    |      source_pressure         1D, shape (501,)
    |      node_pressure           1D, shape (501,)
    |      outlet_pressure         1D, shape (501,)
    |      mass_flow               1D, shape (501,)
    |      valve_area              1D, shape (501,)
    |      reynolds_number         1D, shape (501,)
    |      diagnostics/
    |          time                1D, shape (501,)
    |          max_abs_residual    1D, positive residual-like data
    |          rms_residual        1D, positive residual-like data
    |
    |-- separate_traces/
    |      time                    1D, shape (501,)
    |      station                 1D, shape (6,)
    |      station_1_pressure      1D, shape (501,)
    |      station_2_pressure      1D, shape (501,)
    |      ...
    |
    |-- maps/
    |      time                    1D, shape (501,)
    |      station                 1D, shape (6,)
    |      pressure_map            2D, shape (6, 501), pressure_map[station, time]
    |      temperature_map         2D, shape (6, 501), temperature_map[station, time]
    |      positive_map            2D, positive data for log color-scale examples
    |
    |-- multidimensional/
    |      time                    1D, shape (501,)
    |      station                 1D, shape (6,)
    |      case                    1D, shape (3,)
    |      pressure_3d             3D, shape (3, 6, 501), pressure_3d[case, station, time]
    |      temperature_3d          3D, shape (3, 6, 501), temperature_3d[case, station, time]
    |
    |-- log_data/
           time                    1D, shape (501,)
           frequency               1D, positive log-spaced x data
           gain                    1D, positive y data
           phase_lag               1D, right-axis data
           positive_decay          1D, positive data for log y-axis examples
           positive_growth         1D, positive data for log y-axis examples

The important part is the array shape convention used by later examples:

    pressure_map[station, time]
    pressure_3d[case, station, time]

The last dimension is time, so the plotting examples use axis=-1 when they want
to plot along time.
"""

from pathlib import Path

import h5py
import numpy as np


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"


# ---------------------------------------------------------------------------
# Independent coordinates
# ---------------------------------------------------------------------------
# time is the shared x-axis for most examples.
# station is a simple normalized position from inlet to outlet.
# case represents three synthetic operating cases.
# frequency is positive and log-spaced for log x-axis examples.
# ---------------------------------------------------------------------------

time = np.linspace(0.0, 10.0, 501)
station = np.linspace(0.0, 1.0, 6)
case = np.array([0, 1, 2])
frequency = np.logspace(0.0, 4.0, 300)


# ---------------------------------------------------------------------------
# Simple 1D time histories
# ---------------------------------------------------------------------------
# These behave like ordinary transient solver outputs. Each array has the same
# length as time, so any of them can be plotted against time.
# ---------------------------------------------------------------------------

source_pressure = 350000.0 + 4000.0 * np.sin(2.0 * np.pi * time / 7.0)
node_pressure = 101325.0 + 180000.0 * (1.0 - np.exp(-time / 1.8)) + 8000.0 * np.exp(-time / 4.0) * np.sin(2.0 * np.pi * time / 1.6)
outlet_pressure = 101325.0 + 5000.0 * np.sin(2.0 * np.pi * time / 5.0)
mass_flow = 0.25 * (1.0 - np.exp(-time / 1.2)) + 0.015 * np.exp(-time / 3.0) * np.sin(2.0 * np.pi * time / 0.8)

# Valve area is constant until t = 6 s, then closes linearly over 3 s.
valve_area = 8.0e-5 * np.ones_like(time)
valve_area[time > 6.0] = 8.0e-5 * np.maximum(0.0, 1.0 - (time[time > 6.0] - 6.0) / 3.0)

reynolds_number = 2500.0 + 75000.0 * np.abs(mass_flow) / np.max(np.abs(mass_flow))

# Residual-like data is strictly positive so it can be plotted on a log y-axis.
max_abs_residual = 1.0e-1 * np.exp(-2.2 * time) + 1.0e-8
rms_residual = 2.0e-2 * np.exp(-2.0 * time) + 5.0e-9


# ---------------------------------------------------------------------------
# Related traces at several stations
# ---------------------------------------------------------------------------
# pressure_traces is a list of six 1D arrays.
# pressure_traces[0] is station 0 pressure versus time.
# pressure_traces[1] is station 1 pressure versus time.
# etc.
#
# These traces are written two ways:
#
#   1. As separate 1D datasets under /separate_traces.
#   2. As one 2D dataset under /maps.
#
# That lets the examples show both types of HDF5 layout.
# ---------------------------------------------------------------------------

pressure_traces = []
temperature_traces = []

for x in station:
    pressure = 101325.0 + 210000.0 * (1.0 - np.exp(-time / (1.2 + 0.9 * x)))
    pressure += 18000.0 * np.exp(-time / 5.0) * np.sin(2.0 * np.pi * (time / 2.0 - 0.75 * x))
    pressure -= 35000.0 * x

    temperature = 290.0 + 38.0 * (1.0 - np.exp(-time / 2.5))
    temperature += 4.0 * np.sin(2.0 * np.pi * (time / 8.0 + x))
    temperature -= 10.0 * x

    pressure_traces.append(pressure)
    temperature_traces.append(temperature)

# These are true 2D arrays.
# Axis 0 is station.
# Axis 1 is time.
pressure_map = np.vstack(pressure_traces)
temperature_map = np.vstack(temperature_traces)


# ---------------------------------------------------------------------------
# True 3D arrays
# ---------------------------------------------------------------------------
# These are made by stacking three slightly different 2D maps.
#
# pressure_3d has shape:
#
#     pressure_3d[case, station, time]
#
# Axis 0 is case.
# Axis 1 is station.
# Axis 2 is time.
# ---------------------------------------------------------------------------

pressure_3d = np.stack([
    0.97 * pressure_map,
    pressure_map,
    1.03 * pressure_map,
])

temperature_3d = np.stack([
    temperature_map - 4.0,
    temperature_map,
    temperature_map + 4.0,
])


# ---------------------------------------------------------------------------
# Positive data for log-scale examples
# ---------------------------------------------------------------------------
# Logarithmic axes require positive values. These arrays are positive by
# construction.
# ---------------------------------------------------------------------------

gain = 1.0 / np.sqrt(1.0 + (frequency / 75.0) ** 2)
phase_lag = np.degrees(-np.arctan(frequency / 75.0))
positive_decay = 1.0e2 * np.exp(-time / 1.1) + 1.0e-4
positive_growth = 1.0e-3 * np.exp(time / 1.8)
positive_map = 1.0e-4 + np.abs(pressure_map - np.min(pressure_map)) / np.ptp(pressure_map)


# ---------------------------------------------------------------------------
# Write the HDF5 file
# ---------------------------------------------------------------------------

with h5py.File(filename, "w") as h5:
    h5["description"] = "Synthetic HDF5 data for FullPlot examples."

    scalars = h5.create_group("scalars")
    scalars["initial_pressure"] = float(node_pressure[0])
    scalars["final_pressure"] = float(node_pressure[-1])
    scalars["final_mass_flow"] = float(mass_flow[-1])

    run = h5.create_group("demo_transient")
    run["time"] = time
    run["source_pressure"] = source_pressure
    run["node_pressure"] = node_pressure
    run["outlet_pressure"] = outlet_pressure
    run["mass_flow"] = mass_flow
    run["valve_area"] = valve_area
    run["reynolds_number"] = reynolds_number

    diagnostics = run.create_group("diagnostics")
    diagnostics["time"] = time
    diagnostics["max_abs_residual"] = max_abs_residual
    diagnostics["rms_residual"] = rms_residual

    traces = h5.create_group("separate_traces")
    traces["time"] = time
    traces["station"] = station

    for index, pressure in enumerate(pressure_traces, start=1):
        traces[f"station_{index}_pressure"] = pressure

    for index, temperature in enumerate(temperature_traces, start=1):
        traces[f"station_{index}_temperature"] = temperature

    maps = h5.create_group("maps")
    maps["time"] = time
    maps["station"] = station
    maps["pressure_map"] = pressure_map
    maps["temperature_map"] = temperature_map
    maps["positive_map"] = positive_map

    multidim = h5.create_group("multidimensional")
    multidim["time"] = time
    multidim["station"] = station
    multidim["case"] = case
    multidim["pressure_3d"] = pressure_3d
    multidim["temperature_3d"] = temperature_3d

    log_data = h5.create_group("log_data")
    log_data["time"] = time
    log_data["frequency"] = frequency
    log_data["gain"] = gain
    log_data["phase_lag"] = phase_lag
    log_data["positive_decay"] = positive_decay
    log_data["positive_growth"] = positive_growth


print(f"Wrote {filename}")
print()
print("Important dataset shapes:")
print("  /demo_transient/time              ", time.shape)
print("  /maps/pressure_map                ", pressure_map.shape, "= [station, time]")
print("  /multidimensional/pressure_3d     ", pressure_3d.shape, "= [case, station, time]")
