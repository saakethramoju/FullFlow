from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network


class Map1D(Component):
    """
    Generic one-dimensional map lookup.

    `Map1D` interpolates one or more output values from a single independent
    variable. The map input may be a list, tuple, NumPy array, pandas Series,
    or any array-like object accepted by `np.asarray`.

    Multiple output maps can be evaluated simultaneously from the same
    independent variable.

    Parameters
    ----------
    name : str
        Component name
    network : Network
        Network that owns this component
    x_value : State
        Current independent-variable value
    x_map : array-like
        Independent-variable map coordinates
    y_maps : dict[str, array-like]
        Dictionary of dependent-variable maps

    Outputs
    -------
    <map name> : State
        One output State is automatically created for every key in `y_maps`.

    Notes
    -----
    Each output map is evaluated from:

        ``y = interp(x, x_map, y_map)``

    where `interp` is linear interpolation.

    All maps are automatically sorted by `x_map` during initialization.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        x_value: State,
        x_map,
        y_maps: dict[str, object],
    ):
        self.setup()

        self.x_map = np.asarray(self.x_map, dtype=float)
        self.y_maps = {
            output_name: np.asarray(values, dtype=float)
            for output_name, values in self.y_maps.items()
        }

        if len(self.x_map) < 2:
            raise ValueError(f"{self.name}: x_map requires at least two points.")

        sort_indices = np.argsort(self.x_map)
        self.x_map = self.x_map[sort_indices]

        if np.any(np.diff(self.x_map) <= 0.0):
            raise ValueError(f"{self.name}: x_map must be strictly increasing.")

        for output_name, values in self.y_maps.items():
            if len(values) != len(self.x_map):
                raise ValueError(
                    f"{self.name}: y_map '{output_name}' must have the same length as x_map."
                )

            self.y_maps[output_name] = values[sort_indices]
            setattr(self, output_name, State())

    def evaluate_states(self):
        x = self.x_value.value

        for output_name, values in self.y_maps.items():
            getattr(self, output_name).value = float(np.interp(x, self.x_map, values))

    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {"x_map", "y_maps"}









class Map2D(Component):
    """
    Generic two-dimensional map lookup.

    `Map2D` interpolates one or more output maps from two input values. Inputs
    can be lists, tuples, NumPy arrays, pandas Series, or any array-like object
    accepted by `np.asarray`.

    The `z_maps` values must be 2D arrays with shape:

        (len(y_map), len(x_map))

    Example
    -------
    PumpMap = Map2D(
        "Pump Map",
        network,
        x_value=volumetric_flow,
        y_value=rotor_speed,
        x_map=[0.01, 0.02, 0.03],
        y_map=[10000, 20000, 30000],
        z_maps={
            "head_rise": [
                [100, 90, 80],
                [150, 140, 120],
                [200, 180, 160],
            ],
            "torque": [
                [1.0, 1.2, 1.4],
                [2.0, 2.4, 2.8],
                [3.0, 3.6, 4.2],
            ],
        },
    )

    Outputs are created automatically:

        PumpMap.head_rise
        PumpMap.torque
    """

    def __init__(
        self,
        name: str,
        network: Network,
        x_value: State,
        y_value: State,
        x_map,
        y_map,
        z_maps: dict[str, object],
    ):
        self.setup()

        self.x_map = np.asarray(self.x_map, dtype=float)
        self.y_map = np.asarray(self.y_map, dtype=float)
        self.z_maps = {
            output_name: np.asarray(values, dtype=float)
            for output_name, values in self.z_maps.items()
        }

        if len(self.x_map) < 2:
            raise ValueError(f"{self.name}: x_map requires at least two points.")

        if len(self.y_map) < 2:
            raise ValueError(f"{self.name}: y_map requires at least two points.")

        x_sort_indices = np.argsort(self.x_map)
        y_sort_indices = np.argsort(self.y_map)

        x_count = len(self.x_map)
        y_count = len(self.y_map)

        self.x_map = self.x_map[x_sort_indices]
        self.y_map = self.y_map[y_sort_indices]

        if np.any(np.diff(self.x_map) <= 0.0):
            raise ValueError(f"{self.name}: x_map must be strictly increasing.")

        if np.any(np.diff(self.y_map) <= 0.0):
            raise ValueError(f"{self.name}: y_map must be strictly increasing.")

        for output_name, values in self.z_maps.items():
            if values.shape != (y_count, x_count):
                raise ValueError(
                    f"{self.name}: z_map '{output_name}' must have shape "
                    f"({y_count}, {x_count}). Got {values.shape}."
                )

            self.z_maps[output_name] = values[np.ix_(y_sort_indices, x_sort_indices)]
            setattr(self, output_name, State())

    def evaluate_states(self):
        x = self.x_value.value
        y = self.y_value.value

        for output_name, values in self.z_maps.items():
            values_at_x = np.array([np.interp(x, self.x_map, row) for row in values])
            getattr(self, output_name).value = float(np.interp(y, self.y_map, values_at_x))

    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {"x_map", "y_map", "z_maps"}