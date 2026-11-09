# FullFlow Exports folder overview

The `Exports` folder contains the helpers that write and read FullFlow data files. FullFlow primarily uses HDF5 files because HDF5 can store many named arrays, tables, attributes, and groups in one file.

There are two major uses of HDF5 in FullFlow:

```text
1. Solver output files
   Steady-state and transient results from a Network.

2. Generated map files
   Tabulated property or performance maps used by Map.from_hdf5().
```

## Why HDF5

FullFlow models can produce many outputs:

```text
component attributes
tracked values
time histories
diagnostics
map axes
map outputs
map status arrays
metadata
```

A plain CSV file is not a good fit for this because CSV is one table. FullFlow HDF5 files are hierarchical. They can store a complete network result or a complete multidimensional map in one organized file.

## HDF5.py

`HDF5.py` contains the low-level export helpers used by the solvers and map generator.

Important public helpers include:

```text
hdf5_path()
    Normalize a filename and add .h5 when omitted.

hdf5_filename()
    Same idea, but returns a string.

safe_group_name()
    Convert a user object name into a safe HDF5 group or dataset name.

run_group_path()
    Build the standard HDF5 path for steady-state or transient solver runs.

write_solution()
    Write steady-state network results.

write_transient_solution()
    Write transient network history and final state.

write_tables()
    Write simple diagnostic tables.

```

Most users do not call these functions directly. They are called when a solver is run with `filename=...`:

```python
SteadyState(network).solve(filename="my_result")
Transient(network).solve(dt=0.01, t_final=1.0, filename="my_result")
```

## Solver-output layout

The standard steady-state layout is:

```text
/<network_name>
    attrs: kind="network", name="original network name"
    /steady_state
        /metadata
        /components/<component>/<attribute>
        /tracks/<tracked_name>
        /table/<column>
        /diagnostics/<column>
```

The standard transient layout is:

```text
/<network_name>
    /transient
        /metadata
        /time
        /components/<component>/<attribute>
        /tracks/<tracked_name>
        /table/<column>
        /diagnostics/<column>
        /final/components/<component>/<attribute>
        /final/tracks/<tracked_name>
        /final/table/<column>
```

By default, FullFlow stores one result per Network and solver type. Use `group`
to keep multiple simulations in the same file:

```python
SteadyState(network).solve(filename="results", group="flight_01")
Transient(network).solve(dt=0.01, t_final=1.0, filename="results", group="flight_01")
```

These results are stored at `/<network_name>/flight_01/steady_state` and
`/<network_name>/flight_01/transient`. Nested names such as `campaign/flight_01`
are supported. Names follow the same HDF5 normalization as Networks and tracks;
`steady_state`, `transient`, and `linearization` are reserved as the first group
segment. Omitting `group` preserves the default layout above.

Repeating a solve replaces only that group and solver type. Other groups,
Networks, solver types, and maps in the file are preserved.

For parallel simulations, each worker owns its Network and solver. Solve first,
then use `transient.save(filename)` under a shared process lock to serialize only
the file writes. The saved result retains the group passed to `solve()`; pass
`group=...` to `save()` to select a different destination. See
[`30monte_carlo.py`](../../../examples/transient/30monte_carlo.py).

Linearization results are stored under
`/<network_name>/linearization/<solver_type>/`. Multiple operating points use
its `points/<index>/` subgroup.

New exports use HDF5 schema version 7. Existing files keep their old layout until
the corresponding results are regenerated. Update older plotting paths by
removing `/runs/base`.

## HDF5 map files

FullFlow no longer contains plotting utilities or map-generation utilities.
Those live in the independent `fullplot` package.

FullFlow still contains the `Map` component. `Map.from_hdf5(...)` reads a
simple rectangular-grid HDF5 map layout. FullPlot's `generate_map()` is one
convenient way to create that layout, but it is not required. Users can create
compatible map files with any HDF5 writer.

## Map.from_hdf5()

`Map.from_hdf5()` loads a map group with this generic layout:

```text
/<map_group>
    attrs:
        axis_order = '["pressure", "temperature"]'   optional but recommended

    /axes
        /pressure
        /temperature

    /outputs
        /density
        /enthalpy
```

Each axis dataset must be one-dimensional and strictly increasing. Every output
dataset must be a rectangular array whose shape matches the axis lengths in the
same order. For example, with `pressure.shape == (5,)` and
`temperature.shape == (6,)`, every output should have shape `(5, 6)`.

Optional attributes such as `units`, `spacing`, `constants`, `metadata`, and
`status` are useful for inspection and generated-map bookkeeping, but the core
reader only needs the axes and outputs. A `spacing="log"` axis attribute tells
FullFlow to interpolate in `log(axis_value)` while users still supply physical
values.

A minimal compatible file can be written directly with `h5py`:

```python
import json
import h5py
import numpy as np

pressure = np.linspace(1.0e5, 5.0e5, 5)
temperature = np.linspace(250.0, 500.0, 6)

density = np.empty((len(pressure), len(temperature)))
enthalpy = np.empty_like(density)

for i, p in enumerate(pressure):
    for j, t in enumerate(temperature):
        density[i, j] = p / (287.0 * t)
        enthalpy[i, j] = 1005.0 * t

with h5py.File("property_map.h5", "w") as file:
    group = file.create_group("ideal_gas")
    group.attrs["axis_order"] = json.dumps(["pressure", "temperature"])

    axes = group.create_group("axes")
    axes.create_dataset("pressure", data=pressure)
    axes.create_dataset("temperature", data=temperature)

    outputs = group.create_group("outputs")
    outputs.create_dataset("density", data=density)
    outputs.create_dataset("enthalpy", data=enthalpy)
```

Then load it in FullFlow:

```python
GasMap = Map.from_hdf5(
    "Gas Map",
    network,
    filename="property_map.h5",
    group="ideal_gas",
    inputs={
        "pressure": pressure_state,
        "temperature": temperature_state,
    },
)
```
