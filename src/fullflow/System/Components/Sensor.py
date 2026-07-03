from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from fullplot import Trace

from fullflow.Exceptions import SensorDataStop, SolverSetupError
from fullflow.System import Component, State
from fullflow.System.State import is_assignable_state_like, is_state_like

if TYPE_CHECKING:
    from fullflow.System import Network


class Sensor(Component):
    """Virtual instrumentation channel.

    In its simplest form, a Sensor behaves like a named ``track()`` entry. It
    exports the value of ``reading`` under the sensor's name.

    If both ``variable`` and ``data`` are supplied, the Sensor also behaves like
    a Balance. The solver adjusts ``variable`` until ``reading`` matches the
    supplied FullPlot Trace value sampled at the current network time.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        reading: State,
        variable: State | None = None,
        data: Trace | None = None,
        extend: bool = True,
    ) -> None:
        if data is not None and not isinstance(data, Trace):
            raise SolverSetupError(
                f"Sensor {name!r} data must be a fullplot.Trace object. "
                f"Got {type(data).__name__}."
            )

        self.setup()

        # Component.setup wraps every constructor input in a State by default.
        # A FullPlot Trace is not a solver state; it is an external data object.
        # Keep it as the original Trace so time-axis shifts, windows, and NaN
        # masks remain owned by FullPlot.
        self.data = data

        # Export/cache states. These are updated every evaluate_states() pass
        # and during residual evaluation.
        self.data_value = State(math.nan)
        self.error = State(math.nan)
        self.active = State(False)
        self.variable_value = State(math.nan)


    @property
    def value(self):
        return self.reading.value

    @property
    def ignored_export_attributes(self) -> set[str]:
        # Keep external FullPlot Trace objects and configuration metadata out of
        # the component table. The useful sampled values are exported through
        # reading/data_value/error/active/variable_value.
        return {"data", "extend", "error", "variable_value"}

    @staticmethod
    def _resolved(value: Any, default: Any = None) -> Any:
        if is_state_like(value):
            try:
                return value.value
            except Exception:
                return default
        return value

    @property
    def _extend_enabled(self) -> bool:
        return bool(self._resolved(self.extend, default=True))

    @property
    def has_data(self) -> bool:
        return self.data is not None

    @property
    def has_variable(self) -> bool:
        return is_assignable_state_like(self.variable) and self.variable.is_assigned

    @property
    def is_anchor(self) -> bool:
        return self.has_data and self.has_variable

    @staticmethod
    def _finite_number(value: Any) -> float:
        try:
            number = float(value)
        except Exception:
            return math.nan
        return number if math.isfinite(number) else math.nan

    def _trace_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            return np.array([], dtype=float), np.array([], dtype=float)

        if not isinstance(self.data, Trace):
            raise SolverSetupError(
                f"Sensor {self.name!r} data must be a fullplot.Trace object. "
                f"Got {type(self.data).__name__}."
            )

        x_array = np.asarray(self.data.x, dtype=float)
        y_array = np.asarray(self.data.y, dtype=float)

        if x_array.ndim != 1 or y_array.ndim != 1:
            raise SolverSetupError(f"Sensor {self.name!r} FullPlot Trace x and y arrays must be one-dimensional.")

        if x_array.shape != y_array.shape:
            raise SolverSetupError(
                f"Sensor {self.name!r} FullPlot Trace x and y arrays must have the same length. "
                f"Got {len(x_array)} and {len(y_array)}."
            )

        if len(x_array) == 0:
            raise SolverSetupError(f"Sensor {self.name!r} FullPlot Trace is empty.")

        if not np.all(np.isfinite(x_array)):
            raise SolverSetupError(f"Sensor {self.name!r} FullPlot Trace time values must be finite.")

        if len(x_array) > 1 and np.any(np.diff(x_array) <= 0.0):
            raise SolverSetupError(f"Sensor {self.name!r} FullPlot Trace time values must be strictly increasing.")

        return x_array, y_array

    def _sample_trace_previous(self, time_value: float) -> float:
        """Sample the FullPlot Trace using the previous available data point.

        FullFlow solver time takes priority over the test-data sample rate. At
        each solver time, the sensor uses the most recent trace sample whose
        shifted FullPlot time is less than or equal to the current solver time.
        NaN y-values, out-of-range times, and invalid times return NaN.
        """
        if self.data is None or not math.isfinite(time_value):
            return math.nan

        x, y = self._trace_arrays()

        if time_value < x[0] or time_value > x[-1]:
            return math.nan

        index = int(np.searchsorted(x, time_value, side="right") - 1)
        if index < 0 or index >= len(y):
            return math.nan

        return self._finite_number(y[index])

    def target_value(self, time_value: float | None = None) -> float:
        if time_value is None:
            time_value = float(self.network.time.value)
        return self._sample_trace_previous(float(time_value))

    def _variable_numeric_value(self) -> float:
        if not self.has_variable:
            return math.nan
        return self._finite_number(self.variable.numeric_value)

    def _hold_variable_value(self) -> float:
        if not self.has_variable:
            return math.nan

        try:
            return self._finite_number(self.variable.previous)
        except Exception:
            return self._variable_numeric_value()

    def _update_outputs(self, *, stop_on_missing: bool = False) -> float:
        target = self.target_value()
        has_target = math.isfinite(target)
        has_data = self.has_data
        has_variable = self.has_variable

        self.data_value.value = target if has_data else math.nan
        self.variable_value.value = self._variable_numeric_value()
        self.active.value = bool(has_target and has_variable and has_data)

        if has_data and not has_target:
            self.error.value = math.nan
            if stop_on_missing and not self._extend_enabled:
                raise SensorDataStop(
                    f"Sensor {self.name!r} has no finite FullPlot Trace value at "
                    f"time {float(self.network.time.value):.9g}."
                )
            return math.nan

        if has_data and has_target:
            self.error.value = self._finite_number(self.reading.value - target)
        else:
            self.error.value = math.nan

        return target

    def evaluate_states(self) -> None:
        # Pure evaluation/export should never stop a solve. The residual method
        # handles extend=False during active matching.
        self._update_outputs(stop_on_missing=False)

    @property
    def balances(self) -> list[tuple[State, Any]]:
        if not self.is_anchor:
            return []
        return [(self.variable, self.residual)]

    def residual(self) -> float:
        target = self._update_outputs(stop_on_missing=True)

        if math.isfinite(target):
            return float(self.reading.value) - target

        # extend=True: keep marching through missing data. During NaN/dropout
        # regions or outside a windowed trace, the sensor no longer has a data
        # residual. Hold the independent variable at its last accepted value so
        # it is not a free unknown in the nonlinear solve.
        hold_value = self._hold_variable_value()
        if not math.isfinite(hold_value):
            return 0.0
        return float(self.variable.numeric_value) - hold_value
