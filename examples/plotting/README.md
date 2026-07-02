# FullPlot examples

These examples demonstrate `fullflow.fullplot`, a small HDF5 plotting helper for
simulation output, test data, and map data.

The example HDF5 file is not committed because the repository ignores `*.h5`.
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
