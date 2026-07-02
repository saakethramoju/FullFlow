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
model-option results
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

write_model_option_results()
    Write model-option comparison results.

write_failures()
    Write model-option failure information.
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
        /runs
            /base
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
        /runs
            /base
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

Model-option runs use deeper paths:

```text
/<network_name>/steady_state/runs/<model_name>/<option_name>
/<network_name>/transient/runs/<model_name>/<option_name>
```

This lets a single file store a base run, several model-option runs, diagnostics, and failure information without overwriting unrelated data.

## Map generation moved to FullPlot

FullFlow no longer contains plotting utilities or map-generation utilities.
Those live in the independent `fullplot` package.

Use FullPlot to generate rectangular HDF5 map files:

```python
import fullplot as fplt

fplt.generate_map(
    "property_map.h5",
    group="properties",
    axes=[
        fplt.Axis.linear("pressure", 100000.0, 500000.0, 5),
        fplt.Axis.linear("temperature", 250.0, 500.0, 6),
    ],
    evaluate=lambda pressure, temperature: {
        "density": pressure / (287.0 * temperature),
    },
)
```

FullFlow still contains the `Map` component. It can load compatible HDF5 maps
with `Map.from_hdf5(...)`, but FullFlow itself does not generate those files.

## Map.from_hdf5()

`Map.from_hdf5()` loads a map file that contains:

```text
/<map_group>/axes/<axis_name>
/<map_group>/outputs/<output_name>
/<map_group>/status/success
/<map_group>/status/message
```

The `fullplot.generate_map()` function writes this layout.
