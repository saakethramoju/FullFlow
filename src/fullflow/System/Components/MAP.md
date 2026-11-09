# Map interpolation

`Map` supplies properties and performance values from a tabulated dataset without
repeating an expensive calculation during a nonlinear or transient solve. Typical
uses include pump characteristics, discharge coefficients, nozzle data, and
precomputed combustion-product properties.

## In-memory data

```python
from fullflow import Map, Network, State

network = Network("Pump Performance")
speed_fraction = State(0.75)

pump = Map(
    "Pump Map", network,
    inputs={"speed": speed_fraction},
    axes={"speed": [0.0, 0.5, 1.0]},
    outputs={"pressure_rise": [0.0, 2.0e5, 8.0e5]},
)

pressure_rise = pump.pressure_rise
```

The input dictionary keys identify axes, and the output keys become ordinary
FullFlow `State` attributes. For an N-dimensional map, each output array has
shape `(n1, n2, ..., nN)` in the input-axis order. Axis values must be finite,
unique, and contain at least two points. Maps are evaluated from the current
trial inputs during nonlinear solves.

## Interpolation method

For a one-dimensional cell between $x_0$ and $x_1$, linear interpolation is

```math
y(x)=(1-w)y_0+wy_1,\qquad w=\frac{x-x_0}{x_1-x_0}.
```

For two axes, `Map` uses bilinear interpolation; higher-dimensional maps use
multilinear interpolation. Axis dictionaries may specify `"spacing": "log"`;
in that case interpolation uses $\log x$ on that axis, while the supplied
`State` and table-axis values remain in their original units. The default is
linear spacing.

Inputs outside an axis range raise `MapRangeError` by default. Set
`extrapolate=True` only when extrapolation is physically justified; multilinear
extrapolation is not guaranteed to be accurate beyond the tabulated region.

## Loading FullPlot HDF5 maps

```python
products = Map.from_hdf5(
    "Chamber Products", network,
    filename="engine_maps",
    group="products",
    inputs={"chamber_pressure": chamber_pressure, "mixture_ratio": mixture_ratio},
)

temperature = products.temperature
```

The modern HDF5 layout has a group containing `axes/<input_name>` datasets and
`outputs/<output_name>` datasets. `Map.from_hdf5` also supports its documented
legacy group layout. Use `outputs=[...]` to select datasets, or
`outputs={"local_name": "dataset_name"}` to rename outputs.

## Performance and accuracy

`Map` evaluates all outputs together, caches the most recently interpolated
point, and uses a direct interpolation path for one- and two-dimensional maps.
This is useful inside repeated network evaluations, but it does not change the
model's physical or numerical accuracy: the grid density and interpolation
error remain the user's responsibility. Avoid mixing inconsistent units or
extrapolating into unphysical regimes.

Check interpolation against the original property calculation at points between
grid nodes, including the operating-range edges. Repeat with a finer grid and
compare the resulting network outputs. Refine map spacing and transient timesteps
separately: a small nonlinear residual measures neither interpolation error nor
time-integration error.

See `examples/general/8map_manual.py`, `examples/general/9map_hdf5.py`, and
`examples/transient/26engine_startup.py`.
