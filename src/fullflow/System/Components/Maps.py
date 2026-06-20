from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import json

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fullflow.Exports.HDF5 import dataset_names, hdf5_filename
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


class Map(Component):
    """Generic N-dimensional map lookup.

    ``inputs`` defines the independent variables and their order. ``axes`` gives
    the tabulated grid values for each input. ``outputs`` maps output names to
    N-dimensional scalar arrays whose shapes follow the input order.
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

        for input_name in self.input_names:
            values = np.asarray(self.axes.value[input_name], dtype=float)

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
                    self.axis_values[input_name]
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

    @classmethod
    def from_hdf5(
        cls,
        name: str,
        network: Network,
        filename: str | Path,
        group: str,
        inputs: dict[str, State],
        outputs: list[str] | tuple[str, ...] | None = None,
        extrapolate: bool = False,
    ):
        filename = hdf5_filename(filename)

        with h5py.File(filename, "r") as file:
            map_group = file[group]

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

    @staticmethod
    def _read_v2_group(
        name: str,
        map_group: h5py.Group,
        inputs: dict[str, State],
        outputs: list[str] | tuple[str, ...] | None,
    ):
        axes_group = map_group["axes"]
        output_group = map_group["outputs"] if "outputs" in map_group else map_group
        axis_order = _read_json_attr(map_group.attrs, "axis_order")

        if axis_order is None:
            axis_order = list(inputs.keys())

        axis_order = [_decode_string(axis_name) for axis_name in axis_order]

        missing_inputs = [axis_name for axis_name in axis_order if axis_name not in inputs]

        if missing_inputs:
            raise ValueError(f"{name}: missing inputs for map axes: {missing_inputs}")

        extra_inputs = [input_name for input_name in inputs if input_name not in axis_order]

        if extra_inputs:
            raise ValueError(f"{name}: inputs were provided for unknown map axes: {extra_inputs}")

        missing_axes = [axis_name for axis_name in axis_order if axis_name not in axes_group]

        if missing_axes:
            raise ValueError(f"{name}: HDF5 map is missing axes: {missing_axes}")

        axes = {
            axis_name: np.asarray(axes_group[axis_name][()], dtype=float)
            for axis_name in axis_order
        }

        ordered_inputs = {
            axis_name: inputs[axis_name]
            for axis_name in axis_order
        }

        if outputs is None:
            outputs = dataset_names(output_group, set())

        if not outputs:
            raise ValueError(f"{name}: HDF5 map does not contain any output datasets.")

        output_maps = {
            output_name: np.asarray(output_group[output_name][()], dtype=float)
            for output_name in outputs
        }

        return axes, output_maps, ordered_inputs

    @staticmethod
    def _read_legacy_group(
        name: str,
        map_group: h5py.Group,
        inputs: dict[str, State],
        outputs: list[str] | tuple[str, ...] | None,
    ):
        axes = {}
        axis_order = []
        legacy_axis_datasets = []

        if "x" in map_group:
            x_dataset = map_group["x"]
            x_name = x_dataset.attrs.get("name", "x")
            x_name = _decode_string(x_name)
            axes[x_name] = np.asarray(x_dataset[()], dtype=float)
            axis_order.append(x_name)
            legacy_axis_datasets.append("x")

        if "y" in map_group:
            y_dataset = map_group["y"]
            y_name = y_dataset.attrs.get("name", "y")
            y_name = _decode_string(y_name)
            axes[y_name] = np.asarray(y_dataset[()], dtype=float)
            axis_order.append(y_name)
            legacy_axis_datasets.append("y")

        if not axes:
            raise ValueError(f"{name}: HDF5 group does not contain map axes.")

        missing_inputs = [axis_name for axis_name in axis_order if axis_name not in inputs]

        if missing_inputs:
            raise ValueError(f"{name}: missing inputs for map axes: {missing_inputs}")

        extra_inputs = [input_name for input_name in inputs if input_name not in axis_order]

        if extra_inputs:
            raise ValueError(f"{name}: inputs were provided for unknown map axes: {extra_inputs}")

        ordered_inputs = {
            axis_name: inputs[axis_name]
            for axis_name in axis_order
        }

        if outputs is None:
            outputs = dataset_names(map_group, set(legacy_axis_datasets))

        if not outputs:
            raise ValueError(f"{name}: HDF5 group does not contain any output datasets.")

        output_maps = {}

        for output_name in outputs:
            values = np.asarray(map_group[output_name][()], dtype=float)

            if legacy_axis_datasets == ["x", "y"]:
                values = values.T

            output_maps[output_name] = values

        return axes, output_maps, ordered_inputs

    def _point(self):
        point = np.array(
            [
                self.inputs.value[input_name].value
                for input_name in self.input_names
            ],
            dtype=float,
        )

        if not self.extrapolate.value:
            point = np.array(
                [
                    np.clip(
                        point[index],
                        self.axis_values[input_name][0],
                        self.axis_values[input_name][-1],
                    )
                    for index, input_name in enumerate(self.input_names)
                ],
                dtype=float,
            )

        return point

    def evaluate_states(self):
        point = self._point()

        for output_name, interpolator in self.interpolators.items():
            getattr(self, output_name).value = float(np.asarray(interpolator(point)).item())

    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {
            "axes",
            "outputs",
            "axis_values",
            "axis_sort_indices",
            "output_maps",
            "interpolators",
            "input_names",
        }


class Map1D(Component):
    """Generic one-dimensional map lookup."""

    def __init__(
        self,
        name: str,
        network: Network,
        x_value: State,
        x_map,
        y_maps: dict[str, object],
        extrapolate: bool = False,
    ):
        self.setup()

        self.x_map.value = np.asarray(self.x_map.value, dtype=float)
        self.y_maps.value = {
            output_name: np.asarray(values, dtype=float)
            for output_name, values in self.y_maps.value.items()
        }

        if len(self.x_map.value) < 2:
            raise ValueError(f"{self.name}: x_map requires at least two points.")

        sort_indices = np.argsort(self.x_map.value)
        self.x_map.value = self.x_map.value[sort_indices]

        if np.any(np.diff(self.x_map.value) <= 0.0):
            raise ValueError(f"{self.name}: x_map must be strictly increasing.")

        for output_name, values in self.y_maps.value.items():
            if len(values) != len(self.x_map.value):
                raise ValueError(
                    f"{self.name}: y_map '{output_name}' must have the same length as x_map."
                )

            self.y_maps.value[output_name] = values[sort_indices]
            setattr(self, output_name, State())

    @classmethod
    def from_hdf5(
        cls,
        name: str,
        network: Network,
        filename: str | Path,
        group: str,
        x_value: State,
        x_dataset: str = "x",
        outputs: list[str] | tuple[str, ...] | None = None,
        extrapolate: bool = False,
    ):
        filename = hdf5_filename(filename)

        with h5py.File(filename, "r") as file:
            map_group = file[group]
            x_map = np.asarray(map_group[x_dataset][()], dtype=float)

            if outputs is None:
                outputs = dataset_names(map_group, {x_dataset})

            if not outputs:
                raise ValueError(
                    f"{name}: HDF5 group '{group}' does not contain any output datasets."
                )

            y_maps = {
                output_name: np.asarray(map_group[output_name][()], dtype=float)
                for output_name in outputs
            }

        return cls(
            name,
            network,
            x_value=x_value,
            x_map=x_map,
            y_maps=y_maps,
            extrapolate=extrapolate,
        )

    @staticmethod
    def _interp(x, x_map, values, extrapolate):
        if not extrapolate:
            return float(np.interp(x, x_map, values))

        if x < x_map[0]:
            slope = (values[1] - values[0]) / (x_map[1] - x_map[0])
            return float(values[0] + slope * (x - x_map[0]))

        if x > x_map[-1]:
            slope = (values[-1] - values[-2]) / (x_map[-1] - x_map[-2])
            return float(values[-1] + slope * (x - x_map[-1]))

        return float(np.interp(x, x_map, values))

    def evaluate_states(self):
        x = self.x_value.value
        extrapolate = self.extrapolate.value

        for output_name, values in self.y_maps.value.items():
            getattr(self, output_name).value = self._interp(x, self.x_map.value, values, extrapolate)

    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {"x_map", "y_maps"}


class Map2D(Component):
    """Generic two-dimensional map lookup."""

    def __init__(
        self,
        name: str,
        network: Network,
        x_value: State,
        y_value: State,
        x_map,
        y_map,
        z_maps: dict[str, object],
        extrapolate: bool = False,
    ):
        self.setup()

        self.x_map.value = np.asarray(self.x_map.value, dtype=float)
        self.y_map.value = np.asarray(self.y_map.value, dtype=float)
        self.z_maps.value = {
            output_name: np.asarray(values, dtype=float)
            for output_name, values in self.z_maps.value.items()
        }

        if len(self.x_map.value) < 2:
            raise ValueError(f"{self.name}: x_map requires at least two points.")

        if len(self.y_map.value) < 2:
            raise ValueError(f"{self.name}: y_map requires at least two points.")

        x_sort_indices = np.argsort(self.x_map.value)
        y_sort_indices = np.argsort(self.y_map.value)

        x_count = len(self.x_map.value)
        y_count = len(self.y_map.value)

        self.x_map.value = self.x_map.value[x_sort_indices]
        self.y_map.value = self.y_map.value[y_sort_indices]

        if np.any(np.diff(self.x_map.value) <= 0.0):
            raise ValueError(f"{self.name}: x_map must be strictly increasing.")

        if np.any(np.diff(self.y_map.value) <= 0.0):
            raise ValueError(f"{self.name}: y_map must be strictly increasing.")

        for output_name, values in self.z_maps.value.items():
            if values.shape != (y_count, x_count):
                raise ValueError(
                    f"{self.name}: z_map '{output_name}' must have shape "
                    f"({y_count}, {x_count}). Got {values.shape}."
                )

            self.z_maps.value[output_name] = values[np.ix_(y_sort_indices, x_sort_indices)]
            setattr(self, output_name, State())

    @classmethod
    def from_hdf5(
        cls,
        name: str,
        network: Network,
        filename: str | Path,
        group: str,
        x_value: State,
        y_value: State,
        x_dataset: str = "x",
        y_dataset: str = "y",
        outputs: list[str] | tuple[str, ...] | None = None,
        extrapolate: bool = False,
    ):
        filename = hdf5_filename(filename)

        with h5py.File(filename, "r") as file:
            map_group = file[group]
            x_map = np.asarray(map_group[x_dataset][()], dtype=float)
            y_map = np.asarray(map_group[y_dataset][()], dtype=float)

            if outputs is None:
                outputs = dataset_names(map_group, {x_dataset, y_dataset})

            if not outputs:
                raise ValueError(
                    f"{name}: HDF5 group '{group}' does not contain any output datasets."
                )

            z_maps = {
                output_name: np.asarray(map_group[output_name][()], dtype=float)
                for output_name in outputs
            }

        return cls(
            name,
            network,
            x_value=x_value,
            y_value=y_value,
            x_map=x_map,
            y_map=y_map,
            z_maps=z_maps,
            extrapolate=extrapolate,
        )

    @staticmethod
    def _interp(x, x_map, values, extrapolate):
        if not extrapolate:
            return float(np.interp(x, x_map, values))

        if x < x_map[0]:
            slope = (values[1] - values[0]) / (x_map[1] - x_map[0])
            return float(values[0] + slope * (x - x_map[0]))

        if x > x_map[-1]:
            slope = (values[-1] - values[-2]) / (x_map[-1] - x_map[-2])
            return float(values[-1] + slope * (x - x_map[-1]))

        return float(np.interp(x, x_map, values))

    def evaluate_states(self):
        x = self.x_value.value
        y = self.y_value.value
        extrapolate = self.extrapolate.value

        for output_name, values in self.z_maps.value.items():
            values_at_x = np.array([
                self._interp(x, self.x_map.value, row, extrapolate)
                for row in values
            ])
            getattr(self, output_name).value = self._interp(y, self.y_map.value, values_at_x, extrapolate)

    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {"x_map", "y_map", "z_maps"}
