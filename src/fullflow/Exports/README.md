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

## Maps.py

`Maps.py` contains `Axis`, `MapOutputError`, and `generate_map()`.

`generate_map()` evaluates a Python function on a grid and writes the result to HDF5. The generated map can later be loaded into a network with `Map.from_hdf5()`.

A generated map group has this layout:

```text
/<map_group>
    attrs:
        kind="map"
        name="original map group name"
        map_format="fullflow-map-v3"
        axis_order=[...]
        output_names=[...]
        constants={...}
        metadata={...}
        created_utc="..."
    /axes/<axis_name>
        data: axis values
        attrs: name, units, spacing
    /outputs/<output_name>
        data: N-dimensional output array
    /status/success
        data: Boolean success array
    /status/message
        data: text messages for failed map points
```

## Axis

An `Axis` describes one input direction of a generated map. The axis name is important because it becomes the keyword passed into the `evaluate()` function and the input name expected by `Map.from_hdf5()`.

The three common constructors are:

```python
Axis.linear("temperature", start=250.0, stop=500.0, count=10, units="K")
Axis.log("pressure", start=1e5, stop=1e7, count=20, units="Pa")
Axis.values("mixture_ratio", values=[1.5, 2.0, 2.3, 2.6], units="")
```

Use `Axis.linear()` when equal physical spacing makes sense. Use `Axis.log()` for positive variables that span a large range. Use `Axis.values()` for measured points or hand-picked breakpoints.

## generate_map()

A minimal generated map looks like this:

```python
def evaluate(pressure, temperature):
    density = pressure / (287.0 * temperature)
    return {"density": density}

generate_map(
    filename="gas_map",
    group="air",
    axes=[
        Axis.linear("pressure", 100000.0, 500000.0, 5, units="Pa"),
        Axis.linear("temperature", 250.0, 500.0, 6, units="K"),
    ],
    evaluate=evaluate,
    overwrite=True,
)
```

The map function must return a flat dictionary of scalar numeric outputs. Nested dictionaries, arrays, lists, strings, and booleans are intentionally rejected. If a model produces many values, return each one under a separate scalar key.

## constants vs axes

Use an axis for a value that should vary across the table:

```python
Axis.linear("pressure", ...)
```

Use `constants` for values that should be passed to every evaluation but are fixed for that map:

```python
constants={"gas_constant": 287.0}
```

Constants are stored in the map metadata. They are not stored as axis datasets and are not required as runtime inputs to `Map.from_hdf5()`.

## metadata

The `metadata` argument stores user notes as JSON. It is useful for information like:

```text
source of data
validity range notes
propellant names
map generation date
script version
transport model assumptions
warnings about extrapolation
```

Metadata is not used by the interpolator. It is there so the file can explain itself later.

## resume, overwrite, and failures

`generate_map()` can resume an interrupted map. The status arrays show which points succeeded. With `resume=True`, already successful points are skipped. Failed or missing points are attempted again.

Use `overwrite=True` when the axes, equations, constants, outputs, or assumptions changed and the old map should be deleted before regenerating.

Use `raise_errors=True` while developing a map so mistakes fail immediately. For long production maps, `raise_errors=False` can record failed points in `status/message` and continue.

## Map.from_hdf5()

Generated maps are used inside a network like this:

```python
GasMap = Map.from_hdf5(
    "Gas Map",
    network,
    filename="gas_map",
    group="air",
    inputs={
        "pressure": pressure,
        "temperature": temperature,
    },
)
```

The input names must match the axis names. The output datasets become attributes on the map component.

Outputs can also be selected or renamed:

```python
GasMap = Map.from_hdf5(
    "Gas Map",
    network,
    filename="gas_map",
    group="air",
    inputs={...},
    outputs={
        "rho": "density",
        "h": "enthalpy",
    },
)
```

In that example, the HDF5 dataset `density` becomes `GasMap.rho`, and the dataset `enthalpy` becomes `GasMap.h`.
