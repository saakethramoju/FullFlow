"""
Generate Example HDF5 Plotting Data
===================================

FullFlow examples normally write HDF5 results from a solve, but this plotting
folder intentionally does not commit any .h5 files because .h5 files are ignored
by git.

Run this script first. It creates one small HDF5 file used by all of the other
plotting examples:

    plotting_demo.h5

The file contains common plotting data types:

    - scalar values
    - 1D time histories
    - multiple related 1D traces
    - a true 2D map
    - separate 1D datasets that can be stacked into a heat map
    - positive data for log-axis examples
    - a true 3D dataset for slice examples

The data is synthetic, but it is shaped like simple transient simulation output.
"""

from pathlib import Path

import h5py
import numpy as np


example_dir = Path(__file__).resolve().parent
filename = example_dir / "plotting_demo.h5"


# ---------------------------------------------------------------------------
# Synthetic transient data
# ---------------------------------------------------------------------------
# This is not meant to be a detailed physical model. It is just smooth,
# simulation-like data that is easy to plot.

time = np.linspace(0.0, 10.0, 501)
station = np.linspace(0.0, 1.0, 6)
case = np.array([0, 1, 2])
frequency = np.logspace(0.0, 4.0, 300)

source_pressure = 350000.0 + 4000.0 * np.sin(2.0 * np.pi * time / 7.0)
node_pressure = 101325.0 + 180000.0 * (1.0 - np.exp(-time / 1.8)) + 8000.0 * np.exp(-time / 4.0) * np.sin(2.0 * np.pi * time / 1.6)
outlet_pressure = 101325.0 + 5000.0 * np.sin(2.0 * np.pi * time / 5.0)
mass_flow = 0.25 * (1.0 - np.exp(-time / 1.2)) + 0.015 * np.exp(-time / 3.0) * np.sin(2.0 * np.pi * time / 0.8)
valve_area = 8.0e-5 * np.ones_like(time)
valve_area[time > 6.0] = 8.0e-5 * np.maximum(0.0, 1.0 - (time[time > 6.0] - 6.0) / 3.0)
reynolds_number = 2500.0 + 75000.0 * np.abs(mass_flow) / np.max(np.abs(mass_flow))
max_abs_residual = 1.0e-1 * np.exp(-2.2 * time) + 1.0e-8
rms_residual = 2.0e-2 * np.exp(-2.0 * time) + 5.0e-9


# ---------------------------------------------------------------------------
# Related 1D traces that can be plotted together
# ---------------------------------------------------------------------------
# pressure_traces[i, :] is the pressure history at station i.

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

pressure_map = np.vstack(pressure_traces)
temperature_map = np.vstack(temperature_traces)


# ---------------------------------------------------------------------------
# True 3D data
# ---------------------------------------------------------------------------
# pressure_3d[case, station, time]
# temperature_3d[case, station, time]

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
# Positive data for log-axis examples
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
