# FullFlow Plotting

FullPlot is a small HDF5 plotting helper included with FullFlow. It is meant to
make exported simulation data, test data, and map data easy to inspect without
writing a custom `h5py` and `matplotlib` script every time.

FullPlot does not depend on FullFlow-specific metadata. It reads numeric HDF5
datasets directly.

## Import

```python
from fullflow import fullplot as fplt
```

or:

```python
from fullflow.Plotting import fullplot as fplt
```

## Basic workflow

```python
from fullflow import fullplot as fplt

file = fplt.open("plotting_demo.h5")

file.tree(max_depth=3)
file.list()

run = file.at("/demo_transient")

run.plot(
    x="time",
    y="node_pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Node Pressure",
)
```

## File scoping

Use `file.at(...)` to scope plotting to a group inside the HDF5 file.

```python
file = fplt.open("plotting_demo.h5")
run = file.at("/demo_transient")
```

After scoping, dataset paths are relative to that group:

```python
run.plot(
    x="time",
    y="mass_flow",
)
```

## Inspecting files

Use `tree()` to see the file structure:

```python
file.tree(max_depth=3)
```

Use `list()` to print numeric and non-numeric datasets under the current scope:

```python
run.list()
```

Use `values()` to read scalar datasets:

```python
file.values("/scalars")
```

Use `read()` to get a dataset as a NumPy array:

```python
time = run.read("time")
pressure = run.read("node_pressure")
```

## Single trace

```python
run.plot(
    x="time",
    y="node_pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Node Pressure Time History",
)
```

## Multiple traces

```python
run.plot(
    x="time",
    y=[
        "source_pressure",
        "node_pressure",
        "outlet_pressure",
    ],
    labels=[
        "Source Pressure",
        "Node Pressure",
        "Outlet Pressure",
    ],
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Multiple Pressure Traces",
)
```

## Dual-axis plots

Use `y2` for a right-side y-axis.

```python
run.plot(
    x="time",
    y="node_pressure",
    y2="mass_flow",
    labels="Node Pressure",
    y2labels="Mass Flow",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    y2label="Mass Flow [kg/s]",
    title="Pressure and Mass Flow on Separate Axes",
)
```

## Heat maps

If the HDF5 file already has a 2D dataset, plot it directly:

```python
maps = file.at("/maps")

maps.map(
    z="pressure_map",
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Pressure [Pa]",
    title="Pressure Map",
    cmap="plasma",
)
```

If the file stores each trace as a separate 1D dataset, pass a list to `z`.
FullPlot stacks the selected 1D datasets into a 2D array:

```python
traces = file.at("/separate_traces")

traces.map(
    z=[
        "station_1_pressure",
        "station_2_pressure",
        "station_3_pressure",
        "station_4_pressure",
        "station_5_pressure",
        "station_6_pressure",
    ],
    x="time",
    y="station",
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Pressure [Pa]",
    title="Pressure Heat Map from Separate Traces",
    cmap="plasma",
)
```

For maps, `zlabel` is the colorbar label.

## Multidimensional line plots

A multidimensional dataset can be plotted as multiple line traces. The `axis`
argument selects the dimension used as the x direction. The remaining dimensions
become separate traces.

```python
maps.plot(
    x="time",
    y="pressure_map",
    axis=-1,
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="2D Pressure Dataset as Line Traces",
)
```

Use `slice` to reduce 3D or higher-dimensional data before plotting:

```python
multidim = file.at("/multidimensional")

multidim.plot(
    x="time",
    y="pressure_3d",
    axis=-1,
    slice={0: 1},
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="3D Pressure Dataset, Case 1",
)
```

The same sliced 3D data can also be plotted as a heat map:

```python
multidim.map(
    z="pressure_3d",
    x="time",
    y="station",
    slice={0: 1},
    xlabel="Time [s]",
    ylabel="Station [-]",
    zlabel="Pressure [Pa]",
)
```

## Log axes

Use `xscale`, `yscale`, `y2scale`, or `zscale` with either `"linear"` or `"log"`.

```python
log_data = file.at("/log_data")

log_data.plot(
    x="frequency",
    y="gain",
    xlabel="Frequency [Hz]",
    ylabel="Gain [-]",
    xscale="log",
)
```

```python
log_data.plot(
    x="time",
    y="positive_decay",
    xlabel="Time [s]",
    ylabel="Value [-]",
    yscale="log",
)
```

```python
maps.map(
    z="positive_map",
    x="time",
    y="station",
    zscale="log",
)
```

Log scales only change the displayed axis or color scale. They do not transform
the stored data. Values plotted on a log scale must be positive.

## Themes

FullPlot supports `"dark"` and `"light"` themes. Dark is the default.

```python
run.plot(
    x="time",
    y="node_pressure",
    theme="light",
)
```

## Saving figures

Use `save` to write a figure. The extension controls the output format.

```python
run.plot(
    x="time",
    y="node_pressure",
    save="node_pressure.png",
)
```

Common output formats include `.png`, `.pdf`, and `.svg`.

## Showing multiple figures together

Set `show=False` on each plot, then call `fplt.show()` once at the end.

```python
run.plot(x="time", y="node_pressure", show=False)
run.plot(x="time", y="mass_flow", show=False)
fplt.show()
```

## Module-level helpers

For quick one-off plots, use the module-level functions directly.

```python
fplt.plot(
    "plotting_demo.h5",
    root="/demo_transient",
    x="time",
    y="mass_flow",
)

fplt.map(
    "plotting_demo.h5",
    root="/maps",
    z="temperature_map",
    x="time",
    y="station",
)
```

## Examples

The plotting examples are in `examples/plotting`.

The repository ignores `.h5` files, so generate the example file first:

```bash
python examples/plotting/0generate_plotting_data.py
```

Then run any plotting example:

```bash
python examples/plotting/5heatmap_2d_dataset.py
```
