# FullPlot examples

These examples demonstrate `fullflow.fullplot`, a small HDF5 plotting helper for
simulation output, test data, and map data.

The repository ignores `*.h5` files, so the example file is generated locally.
Run the generator first:

```bash
python examples/plotting/0generate_plotting_data.py
```

Then run any plotting example:

```bash
python examples/plotting/5heatmap_2d_dataset.py
```

The examples intentionally use plain script-style code. They are written the way
a user would typically inspect and plot an HDF5 file.

## Generated file

`0generate_plotting_data.py` creates:

```text
examples/plotting/plotting_demo.h5
```

The file contains:

```text
/scalars
/demo_transient
/separate_traces
/maps
/multidimensional
/log_data
```

The most important datasets are:

```text
/demo_transient/time                 shape (501,)
/demo_transient/node_pressure        shape (501,)
/demo_transient/mass_flow            shape (501,)
/maps/pressure_map                   shape (6, 501)      pressure_map[station, time]
/multidimensional/pressure_3d        shape (3, 6, 501)   pressure_3d[case, station, time]
/log_data/frequency                  shape (300,)
/log_data/gain                       shape (300,)
```

## What each example shows

```text
0generate_plotting_data.py       Creates plotting_demo.h5.
1inspect_hdf5.py                 Shows tree(), list(), values(), and read().
2single_trace.py                 Plots one 1D time history.
3multiple_traces.py              Plots several 1D datasets on one y-axis.
4dual_axis.py                    Uses y for the left axis and y2 for the right axis.
5heatmap_2d_dataset.py           Plots a true 2D dataset with map().
6heatmap_stacked_1d.py           Builds a heat map from separate 1D datasets.
7log_axes.py                     Demonstrates xscale="log" and yscale="log".
8log_color_map.py                Demonstrates zscale="log" for a heat map colorbar.
9multidimensional_line_traces.py Plots pressure_map[station, time] as lines.
10multidimensional_slices.py     Slices pressure_3d[case, station, time].
11show_multiple_figures.py       Uses show=False and fplt.show().
12light_theme.py                 Uses theme="light".
13module_level_api.py            Uses fplt.plot() and fplt.map() directly.
14save_formats.py                Saves figures as PNG and SVG.
```

## Shape conventions used in the examples

The examples use this convention:

```text
pressure_map[station, time]
pressure_3d[case, station, time]
```

For line plots, `axis=-1` means "plot along the last axis." Since the last axis
is time in both of these arrays, `axis=-1` plots pressure versus time. Any other
remaining dimensions become separate traces.

For maps, `z` is the color field. For example:

```python
maps.map(
    z="pressure_map",
    x="time",
    y="station",
)
```

means:

```text
x-axis = time
y-axis = station
color  = pressure_map[station, time]
```

For 3D data, use `slice` to reduce the array before plotting. For example:

```python
slice={0: 1}
```

means select index `1` from axis `0`. If the array is:

```text
pressure_3d[case, station, time]
```

then `slice={0: 1}` selects case 1 and leaves:

```text
pressure_3d[station, time]
```
