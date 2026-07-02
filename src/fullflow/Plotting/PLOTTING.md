# FullFlow Plotting

FullPlot is a small HDF5 plotting helper included with FullFlow. It is meant to
make exported simulation data, test data, and map data easy to inspect without
writing a custom `h5py` and `matplotlib` script every time.

FullPlot does not depend on FullFlow-specific metadata. It reads numeric HDF5
datasets and plots them directly.

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

file = fplt.open("water_hammer.h5")

file.tree(max_depth=4)

run = file.at("/Water_Hammer/transient/runs/base")

run.list()

run.plot(
    x="time",
    y="components/Pipe_Node_5/pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Pipe Node 5 Pressure",
)
```

## File scoping

Use `file.at(...)` to scope plotting to a group inside the HDF5 file.

```python
file = fplt.open("water_hammer.h5")
run = file.at("/Water_Hammer/transient/runs/base")
```

After scoping, dataset paths are relative to that group:

```python
run.plot(
    x="time",
    y="components/Pipe_Node_5/pressure",
)
```

## Inspecting files

Use `tree()` to see the file structure:

```python
file.tree(max_depth=4)
```

Use `list()` to print numeric and non-numeric datasets under the current scope:

```python
run.list()
```

Use `read()` to get a dataset as a NumPy array:

```python
time = run.read("time")
pressure = run.read("components/Pipe_Node_5/pressure")
```

Use `values()` to read scalar datasets under the current scope:

```python
run.values()
```

## Single trace

```python
run.plot(
    x="time",
    y="components/Pipe_Node_5/pressure",
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Pipe Node 5 Pressure",
)
```

## Multiple traces

```python
run.plot(
    x="time",
    y=[
        "components/Pipe_Node_1/pressure",
        "components/Pipe_Node_2/pressure",
        "components/Pipe_Node_3/pressure",
        "components/Pipe_Node_4/pressure",
        "components/Pipe_Node_5/pressure",
    ],
    labels=[
        "Pipe Node 1",
        "Pipe Node 2",
        "Pipe Node 3",
        "Pipe Node 4",
        "Pipe Node 5",
    ],
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Water Hammer Pressure Wave",
)
```

## Dual-axis plots

Use `y2` for a right-side y-axis.

```python
run.plot(
    x="time",
    y="components/Outlet_Valve/cross_sectional_area",
    y2="components/Pipe_Node_5/pressure",
    labels="Valve Area",
    y2labels="Pipe Node 5 Pressure",
    xlabel="Time [s]",
    ylabel="Valve Area [m²]",
    y2label="Pressure [Pa]",
    title="Valve Closure and Downstream Pressure Response",
)
```

## Heat maps

If the HDF5 file already has a 2D dataset, plot it directly:

```python
run.map(
    z="pressure_map",
    x="time",
    y="pipe_node",
    xlabel="Time [s]",
    ylabel="Pipe Node",
    zlabel="Pressure [Pa]",
    title="Pressure Heat Map",
)
```

If the file stores each trace as a separate 1D dataset, pass a list to `z`.
FullPlot stacks the selected 1D datasets into a 2D array:

```python
run.map(
    z=[
        "components/Pipe_Node_1/pressure",
        "components/Pipe_Node_2/pressure",
        "components/Pipe_Node_3/pressure",
        "components/Pipe_Node_4/pressure",
        "components/Pipe_Node_5/pressure",
    ],
    x="time",
    y=[1, 2, 3, 4, 5],
    xlabel="Time [s]",
    ylabel="Pipe Node",
    zlabel="Pressure [Pa]",
    title="Water Hammer Pressure Heat Map",
    cmap="plasma",
)
```

For maps, `zlabel` is the colorbar label.

## Multidimensional line plots

A multidimensional dataset can be plotted as multiple line traces. The `axis`
argument selects the dimension used as the x direction. The remaining dimensions
become separate traces.

```python
file.plot(
    x="time",
    y="pressure",
    axis=-1,
    xlabel="Time [s]",
    ylabel="Pressure [Pa]",
    title="Pressure Map Plotted as Line Traces",
)
```

Use `slice` to reduce 3D or higher-dimensional data before plotting:

```python
file.plot(
    x="time",
    y="temperature",
    axis=-1,
    slice={0: 2},
)
```

## Log axes

Use `xscale`, `yscale`, `y2scale`, or `zscale` with either `"linear"` or `"log"`.

```python
run.plot(
    x="diagnostics/time",
    y=[
        "diagnostics/max_abs_residual",
        "diagnostics/rms_residual",
    ],
    yscale="log",
    xlabel="Time [s]",
    ylabel="Residual [-]",
    title="Transient Solver Residuals",
)
```

A log scale changes only how the axis is displayed. It does not transform the
stored data.

Use a log scale for raw positive values that span orders of magnitude:

```python
run.plot(x="time", y="residual", yscale="log")
```

If the data is already stored as `log10(value)`, keep the axis linear and label
it accordingly:

```python
run.plot(
    x="time",
    y="log10_residual",
    yscale="linear",
    ylabel="log10 Residual",
)
```

Log scales require positive plotted values. FullPlot raises `PlotDataError` for
zero or negative values on a requested log scale.

## Themes

FullPlot has two themes:

```python
theme="dark"
theme="light"
```

The default theme is dark.

## Saving figures

Use `save=...` to write the figure to disk. The file extension controls the
format.

```python
run.plot(
    x="time",
    y="components/Pipe_Node_5/pressure",
    save="pipe_node_5_pressure.png",
)
```

Common extensions are `.png`, `.pdf`, and `.svg`.

## Showing multiple figures together

By default, each plot calls `plt.show()`. To create several figures and show
them all at the end, use `show=False` and then call `fplt.show()`.

```python
run.plot(x="time", y="components/Pipe_Node_5/pressure", show=False)
run.plot(x="time", y="components/Pipe_Segment_5/mass_flow", show=False)

fplt.show()
```

## Module-level API

For quick scripts, FullPlot also provides module-level helpers. Use `root=...`
to scope the search path.

```python
fplt.plot(
    "water_hammer.h5",
    root="/Water_Hammer/transient/runs/base",
    x="time",
    y="components/Pipe_Node_5/pressure",
)
```

## Example files

The `examples/plotting` folder demonstrates:

- inspecting an HDF5 file
- single-trace plots
- multiple traces
- dual y-axes
- heat maps from separate 1D datasets
- log axes
- multidimensional datasets
- showing multiple figures at once
- light and dark themes
- module-level plotting helpers
