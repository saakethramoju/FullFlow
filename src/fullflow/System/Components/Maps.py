from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


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