from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
import json

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fullflow.Exports.HDF5 import dataset_names, hdf5_filename, safe_group_name
from fullflow.Exceptions import MapLoadError, MapRangeError
from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network




def _decode_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return value


def _read_json_attr(attrs, name: str, default=None):
    if name not in attrs:
        return default

    value = _decode_string(attrs[name])

    if isinstance(value, str):
        return json.loads(value)

    return value


def _axis_values_and_spacing(axis):
    if isinstance(axis, dict):
        if "values" not in axis:
            raise ValueError("Axis dictionaries must contain a 'values' entry.")

        values = axis["values"]
        spacing = axis.get("spacing", "linear")

    elif hasattr(axis, "values") and hasattr(axis, "spacing"):
        values = axis.values
        spacing = axis.spacing

    else:
        values = axis
        spacing = "linear"

    spacing = _decode_string(spacing)

    if spacing is None or spacing == "":
        spacing = "linear"

    spacing = str(spacing).lower()

    if spacing not in {"linear", "values", "log"}:
        raise ValueError(
            f"Unsupported axis spacing '{spacing}'. "
            "Supported spacing values are 'linear', 'values', and 'log'."
        )

    return np.asarray(values, dtype=float), spacing


def _interpolation_axis_values(component_name: str, input_name: str, values, spacing: str):
    if spacing == "log":
        if np.any(values <= 0.0):
            raise ValueError(f"{component_name}: log-spaced axis '{input_name}' requires all values to be positive.")

        return np.log(values)

    return values


def _interpolation_point_value(component_name: str, input_name: str, value: float, spacing: str):
    if spacing == "log":
        if value <= 0.0:
            raise ValueError(f"{component_name}: log-spaced input '{input_name}' must be positive.")

        return np.log(value)

    return value


class Map(Component):
    """Generic N-dimensional interpolation component.

    ``Map`` turns tabulated data into one or more FullFlow output ``State``
    objects. It is useful for property tables, pump/turbine maps, combustion
    products maps, lookup tables, and any scalar output that can be represented
    on a rectangular grid.

    Manual maps
    -----------
    Construct a manual map by passing dictionaries for ``inputs``, ``axes``,
    and ``outputs``::

        Products = Map(
            "Products",
            network,
            inputs={
                "pressure": ChamberPressure,
                "temperature": ChamberTemperature,
            },
            axes={
                "pressure": pressure_values,
                "temperature": temperature_values,
            },
            outputs={
                "density": density_table,
                "enthalpy": enthalpy_table,
            },
        )

    Input names are the keys in ``inputs``. Every input key must have a matching
    key in ``axes``. Output state names are the keys in ``outputs``. The example
    above creates ``Products.density`` and ``Products.enthalpy``.

    HDF5 maps
    ---------
    ``Map.from_hdf5`` loads simple rectangular-grid HDF5 maps. FullPlot's
    ``generate_map`` helper writes one compatible layout, but FullFlow does not
    require FullPlot and the file does not need to have been created by
    FullPlot. The generic expected structure is::

        /<map_group>/axes/<axis_name>
        /<map_group>/outputs/<output_name>

    Every axis is a one-dimensional grid. Every output is a rectangular array
    whose shape matches the axis lengths in axis order. The optional
    ``axis_order`` group attribute records that order. If it is omitted,
    ``Map.from_hdf5`` uses the order of the supplied ``inputs``.

    The required input names come from the HDF5 map axes. For a map with axes
    ``pressure`` and ``temperature``, load it with::

        Products = Map.from_hdf5(
            "Products",
            network,
            "equilibrium_nozzle",
            group="products_tp",
            inputs={
                "pressure": ChamberPressure,
                "temperature": ChamberTemperature,
            },
        )

    Output naming for HDF5 maps is controlled by the ``outputs`` argument:

    ``outputs=None``
        Load every dataset in ``/outputs`` and create state names matching the
        HDF5 dataset names.

    ``outputs=["density", "enthalpy"]``
        Load only the listed datasets and use those same names for the created
        states.

    ``outputs={"rho": "density", "h": "enthalpy"}``
        Create Python-friendly state names from different HDF5 dataset names.
        The dictionary rule is ``created_state_name: hdf5_dataset_name``. This
        example creates ``Products.rho`` from dataset ``density`` and
        ``Products.h`` from dataset ``enthalpy``.

    Axis spacing
    ------------
    Axis values are always stored and supplied in physical units. For an axis
    with ``spacing="log"``, ``Map`` applies ``log`` to both the stored axis
    values and the runtime input value before interpolation. Users should still
    pass the physical value, not its logarithm.

    Extrapolation
    -------------
    If ``extrapolate=False`` then every runtime input must remain inside the
    tabulated range. If an input is outside its range, the map raises an error.
    Set ``extrapolate=True`` to allow SciPy's ``RegularGridInterpolator`` to
    extrapolate beyond the tabulated bounds.
    """

    _reserved_output_names = {
        "name",
        "network",
        "inputs",
        "axes",
        "outputs",
        "extrapolate",
        "input_names",
        "axis_values",
        "axis_sort_indices",
        "axis_spacing",
        "axis_interpolation_values",
        "output_maps",
        "interpolators",
    }

    def __init__(
        self,
        name: str,
        network: Network,
        inputs: dict[str, State],
        axes: dict[str, object],
        outputs: dict[str, object],
        extrapolate: bool = False,
    ):
        self.setup()

        if not isinstance(self.inputs.value, dict):
            raise TypeError(f"{self.name}: inputs must be a dictionary of input States.")

        if not isinstance(self.axes.value, dict):
            raise TypeError(f"{self.name}: axes must be a dictionary of axis values.")

        if not isinstance(self.outputs.value, dict):
            raise TypeError(f"{self.name}: outputs must be a dictionary of output maps.")

        self.input_names = list(self.inputs.value.keys())

        if not self.input_names:
            raise ValueError(f"{self.name}: at least one input is required.")

        if len(self.input_names) != len(set(self.input_names)):
            raise ValueError(f"{self.name}: input names must be unique.")

        for input_name in self.input_names:
            if not isinstance(input_name, str) or not input_name.strip():
                raise ValueError(f"{self.name}: input names must be non-empty strings.")

        missing_axes = [
            input_name
            for input_name in self.input_names
            if input_name not in self.axes.value
        ]

        if missing_axes:
            raise ValueError(f"{self.name}: missing axes for inputs: {missing_axes}")

        extra_axes = [
            axis_name
            for axis_name in self.axes.value
            if axis_name not in self.inputs.value
        ]

        if extra_axes:
            raise ValueError(f"{self.name}: axes were provided for unknown inputs: {extra_axes}")

        self.axis_values = {}
        self.axis_sort_indices = {}
        self.axis_spacing = {}
        self.axis_interpolation_values = {}

        for input_name in self.input_names:
            values, spacing = _axis_values_and_spacing(self.axes.value[input_name])

            if values.ndim != 1:
                raise ValueError(f"{self.name}: axis '{input_name}' must be one-dimensional.")

            if len(values) < 2:
                raise ValueError(f"{self.name}: axis '{input_name}' requires at least two points.")

            if not np.all(np.isfinite(values)):
                raise ValueError(f"{self.name}: axis '{input_name}' values must all be finite.")

            sort_indices = np.argsort(values)
            values = values[sort_indices]

            if np.any(np.diff(values) <= 0.0):
                raise ValueError(f"{self.name}: axis '{input_name}' must be strictly increasing.")

            self.axis_values[input_name] = values
            self.axis_sort_indices[input_name] = sort_indices
            self.axis_spacing[input_name] = spacing
            self.axis_interpolation_values[input_name] = _interpolation_axis_values(
                self.name,
                input_name,
                values,
                spacing,
            )

        expected_shape = tuple(
            len(self.axis_values[input_name])
            for input_name in self.input_names
        )

        sort_index_tuple = np.ix_(
            *[
                self.axis_sort_indices[input_name]
                for input_name in self.input_names
            ]
        )

        self.output_maps = {}
        self.interpolators = {}

        if not self.outputs.value:
            raise ValueError(f"{self.name}: at least one output map is required.")

        for output_name, values in self.outputs.value.items():
            self._validate_output_name(output_name)
            values = np.asarray(values, dtype=float)

            if values.shape != expected_shape:
                raise ValueError(
                    f"{self.name}: output map '{output_name}' must have shape "
                    f"{expected_shape}. Got {values.shape}."
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(f"{self.name}: output map '{output_name}' values must all be finite.")

            values = values[sort_index_tuple]
            self.output_maps[output_name] = values
            self.interpolators[output_name] = RegularGridInterpolator(
                [
                    self.axis_interpolation_values[input_name]
                    for input_name in self.input_names
                ],
                values,
                bounds_error=False,
                fill_value=None,
            )
            setattr(self, output_name, State())

    def _validate_output_name(self, output_name: str) -> None:
        if not isinstance(output_name, str) or not output_name.strip():
            raise ValueError(f"{self.name}: output names must be non-empty strings.")

        if output_name in self._reserved_output_names:
            raise ValueError(f"{self.name}: output name '{output_name}' is reserved.")

        if hasattr(self, output_name):
            raise ValueError(
                f"{self.name}: output name '{output_name}' conflicts with an existing Map attribute."
            )

    @staticmethod
    def _hdf5_output_name_map(component_name: str, outputs, available_outputs):
        available_outputs = [
            str(_decode_string(output_name))
            for output_name in available_outputs
        ]

        if outputs is None:
            return {
                output_name: output_name
                for output_name in available_outputs
            }

        if isinstance(outputs, Mapping):
            dataset_to_output = {}
            used_output_names = set()
            used_dataset_names = set()

            for output_name, dataset_name in outputs.items():
                output_name = str(_decode_string(output_name))
                dataset_name = str(_decode_string(dataset_name))

                if output_name in used_output_names:
                    raise MapLoadError(
                        f"{component_name}: duplicate mapped output name "
                        f"{output_name!r} is not allowed."
                    )

                if dataset_name in used_dataset_names:
                    raise MapLoadError(
                        f"{component_name}: multiple output names cannot point "
                        f"to the same HDF5 dataset {dataset_name!r}."
                    )

                used_output_names.add(output_name)
                used_dataset_names.add(dataset_name)
                dataset_to_output[dataset_name] = output_name

            return dataset_to_output

        return {
            str(_decode_string(output_name)): str(_decode_string(output_name))
            for output_name in outputs
        }

    @staticmethod
    def _validate_hdf5_output_name_map(component_name: str, output_name_map: dict[str, str]) -> None:
        if not output_name_map:
            raise MapLoadError(f"{component_name}: HDF5 map does not contain any output datasets.")

        output_names = list(output_name_map.values())
        duplicate_output_names = sorted(
            {
                output_name
                for output_name in output_names
                if output_names.count(output_name) > 1
            }
        )

        if duplicate_output_names:
            raise MapLoadError(
                f"{component_name}: duplicate mapped output names are not allowed: "
                f"{duplicate_output_names}"
            )

        for dataset_name, output_name in output_name_map.items():
            if not dataset_name:
                raise MapLoadError(f"{component_name}: HDF5 output dataset names must be non-empty.")

            if not output_name:
                raise MapLoadError(f"{component_name}: mapped output names must be non-empty.")

    @classmethod
    def from_hdf5(
        cls,
        name: str,
        network: Network,
        filename: str | Path,
        group: str,
        inputs: dict[str, State],
        outputs: list[str] | tuple[str, ...] | dict[str, str] | None = None,
        extrapolate: bool = False,
    ):
        filename = hdf5_filename(filename)

        with h5py.File(filename, "r") as file:
            if group in file:
                map_group = file[group]
            elif safe_group_name(group) in file:
                map_group = file[safe_group_name(group)]
            else:
                available = [name for name, item in file.items() if isinstance(item, h5py.Group)]
                raise MapLoadError(
                    f"{name}: could not find HDF5 map group {group!r}. "
                    f"Available top-level groups: {available}"
                )

            if "axes" in map_group and isinstance(map_group["axes"], h5py.Group):
                axes, output_maps, inputs = cls._read_v2_group(
                    name,
                    map_group,
                    inputs,
                    outputs,
                )
            else:
                axes, output_maps, inputs = cls._read_legacy_group(
                    name,
                    map_group,
                    inputs,
                    outputs,
                )

        return cls(
            name,
            network,
            inputs=inputs,
            axes=axes,
            outputs=output_maps,
            extrapolate=extrapolate,
        )

    @classmethod
    def _read_v2_group(
        cls,
        name: str,
        map_group: h5py.Group,
        inputs: dict[str, State],
        outputs: list[str] | tuple[str, ...] | dict[str, str] | None,
    ):
        axes_group = map_group["axes"]
        output_group = map_group["outputs"] if "outputs" in map_group else map_group
        axis_order = _read_json_attr(map_group.attrs, "axis_order")

        if axis_order is None:
            axis_order = list(inputs.keys())

        axis_order = [_decode_string(axis_name) for axis_name in axis_order]

        missing_inputs = [axis_name for axis_name in axis_order if axis_name not in inputs]

        if missing_inputs:
            raise MapLoadError(f"{name}: missing inputs for map axes: {missing_inputs}")

        extra_inputs = [input_name for input_name in inputs if input_name not in axis_order]

        if extra_inputs:
            raise MapLoadError(f"{name}: inputs were provided for unknown map axes: {extra_inputs}")

        missing_axes = [axis_name for axis_name in axis_order if axis_name not in axes_group]

        if missing_axes:
            raise MapLoadError(f"{name}: HDF5 map is missing axes: {missing_axes}")

        axes = {}

        for axis_name in axis_order:
            axis_dataset = axes_group[axis_name]
            axes[axis_name] = {
                "values": np.asarray(axis_dataset[()], dtype=float),
                "spacing": _decode_string(axis_dataset.attrs.get("spacing", "linear")),
            }

        ordered_inputs = {
            axis_name: inputs[axis_name]
            for axis_name in axis_order
        }

        available_outputs = dataset_names(output_group, set())
        output_name_map = cls._hdf5_output_name_map(name, outputs, available_outputs)
        cls._validate_hdf5_output_name_map(name, output_name_map)

        missing_outputs = [
            dataset_name
            for dataset_name in output_name_map
            if dataset_name not in output_group
        ]

        if missing_outputs:
            raise MapLoadError(f"{name}: HDF5 map is missing output datasets: {missing_outputs}")

        output_maps = {
            output_name: np.asarray(output_group[dataset_name][()], dtype=float)
            for dataset_name, output_name in output_name_map.items()
        }

        return axes, output_maps, ordered_inputs

    @classmethod
    def _read_legacy_group(
        cls,
        name: str,
        map_group: h5py.Group,
        inputs: dict[str, State],
        outputs: list[str] | tuple[str, ...] | dict[str, str] | None,
    ):
        axes = {}
        axis_order = []
        legacy_axis_datasets = []

        if "x" in map_group:
            x_dataset = map_group["x"]
            x_name = x_dataset.attrs.get("name", "x")
            x_name = _decode_string(x_name)
            axes[x_name] = {
                "values": np.asarray(x_dataset[()], dtype=float),
                "spacing": _decode_string(x_dataset.attrs.get("spacing", "linear")),
            }
            axis_order.append(x_name)
            legacy_axis_datasets.append("x")

        if "y" in map_group:
            y_dataset = map_group["y"]
            y_name = y_dataset.attrs.get("name", "y")
            y_name = _decode_string(y_name)
            axes[y_name] = {
                "values": np.asarray(y_dataset[()], dtype=float),
                "spacing": _decode_string(y_dataset.attrs.get("spacing", "linear")),
            }
            axis_order.append(y_name)
            legacy_axis_datasets.append("y")

        if not axes:
            raise MapLoadError(f"{name}: HDF5 group does not contain map axes.")

        missing_inputs = [axis_name for axis_name in axis_order if axis_name not in inputs]

        if missing_inputs:
            raise MapLoadError(f"{name}: missing inputs for map axes: {missing_inputs}")

        extra_inputs = [input_name for input_name in inputs if input_name not in axis_order]

        if extra_inputs:
            raise MapLoadError(f"{name}: inputs were provided for unknown map axes: {extra_inputs}")

        ordered_inputs = {
            axis_name: inputs[axis_name]
            for axis_name in axis_order
        }

        available_outputs = dataset_names(map_group, set(legacy_axis_datasets))
        output_name_map = cls._hdf5_output_name_map(name, outputs, available_outputs)
        cls._validate_hdf5_output_name_map(name, output_name_map)

        missing_outputs = [
            dataset_name
            for dataset_name in output_name_map
            if dataset_name not in map_group
        ]

        if missing_outputs:
            raise MapLoadError(f"{name}: HDF5 map is missing output datasets: {missing_outputs}")

        output_maps = {}

        for dataset_name, output_name in output_name_map.items():
            values = np.asarray(map_group[dataset_name][()], dtype=float)

            if legacy_axis_datasets == ["x", "y"]:
                values = values.T

            output_maps[output_name] = values

        return axes, output_maps, ordered_inputs

    def _point(self):
        point = []

        for input_name in self.input_names:
            value = float(self.inputs.value[input_name].value)

            if not np.isfinite(value):
                raise MapRangeError(f"{self.name}: input '{input_name}' must be finite.")

            lower = self.axis_values[input_name][0]
            upper = self.axis_values[input_name][-1]

            if not self.extrapolate.value and (value < lower or value > upper):
                raise MapRangeError(
                    f"{self.name}: input '{input_name}'={value} is outside the map range "
                    f"[{lower}, {upper}]. Set extrapolate=True to allow extrapolation."
                )

            point.append(
                _interpolation_point_value(
                    self.name,
                    input_name,
                    value,
                    self.axis_spacing[input_name],
                )
            )

        return np.asarray(point, dtype=float)

    def evaluate_states(self):
        point = self._point()

        for output_name, interpolator in self.interpolators.items():
            getattr(self, output_name).value = float(np.asarray(interpolator(point)).item())

    @property
    def export_attributes(self) -> dict[str, State]:
        """Export live map inputs using their actual input names.

        Manual maps and HDF5-loaded maps both store runtime inputs in
        ``self.inputs``.  Export each input separately so the solution HDF5 has
        user-facing paths such as::

            Chamber_Gas_Map/chamber_pressure
            Chamber_Gas_Map/mixture_ratio

        instead of one generic ``inputs`` dictionary dataset.
        """
        exported: dict[str, State] = {}

        for input_name in self.input_names:
            if input_name in self.output_maps:
                raise ValueError(
                    f"{self.name}: map input name {input_name!r} conflicts with an "
                    "output name. Rename the input or output before exporting."
                )

            exported[input_name] = self.inputs.value[input_name]

        return exported

    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {
            "inputs",
            "axes",
            "outputs",
            "axis_values",
            "axis_sort_indices",
            "axis_spacing",
            "axis_interpolation_values",
            "output_maps",
            "interpolators",
            "input_names",
        }
