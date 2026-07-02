from __future__ import annotations

import builtins
import itertools
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import matplotlib.pyplot as plt

from fullflow.Plotting.themes import (
    apply_theme,
    check_theme,
    grid_kwargs,
    style_colorbar,
    style_legend,
    theme_colors,
)


class FullPlotError(Exception):
    """Base exception for FullPlot errors."""


class DatasetNotFoundError(FullPlotError):
    """Raised when a dataset or group cannot be found."""


class AmbiguousDatasetError(FullPlotError):
    """Raised when a short dataset or group name matches more than one object."""


class PlotDataError(FullPlotError):
    """Raised when selected data cannot be plotted."""


@dataclass
class LineSeries:
    data: np.ndarray
    label: str
    length: int


def _is_numeric_dtype(dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.number)
    except TypeError:
        return False


def _shape_string(shape) -> str:
    if shape == ():
        return "scalar"

    return str(shape)


def _clean_root(root: str) -> str:
    root = str(root).strip()

    if root == "":
        root = "/"

    if not root.startswith("/"):
        root = "/" + root

    root = posixpath.normpath(root)

    if root == ".":
        root = "/"

    return root


def _join_h5(root: str, name: str) -> str:
    root = _clean_root(root)
    name = str(name).strip()

    if name.startswith("/"):
        return posixpath.normpath(name)

    if root == "/":
        return posixpath.normpath("/" + name)

    return posixpath.normpath(root + "/" + name)


def _basename(path: str) -> str:
    path = str(path).rstrip("/")

    if path == "":
        return ""

    return path.split("/")[-1]


def _relative_path(path: str, root: str) -> str:
    path = _clean_root(path)
    root = _clean_root(root)

    if root == "/":
        return path.lstrip("/")

    prefix = root.rstrip("/") + "/"

    if path.startswith(prefix):
        return path[len(prefix):]

    return path


def _normalize_name(name: str) -> str:
    name = str(name).lower().strip()

    for char in [" ", "\t", "\n", "\r", "_", "-", "[", "]", "(", ")", "{", "}", ".", ",", "/"]:
        name = name.replace(char, "")

    return name


def _as_list(value) -> list:
    if value is None:
        return []

    if isinstance(value, (builtins.list, tuple)):
        return builtins.list(value)

    return [value]


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")

    if isinstance(value, np.generic):
        return value.item()

    return value


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)

    if ndim <= 0:
        raise PlotDataError("Cannot select an axis from scalar data.")

    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise PlotDataError(f"axis {axis} is out of range for array with {ndim} dimensions.")

    return axis


def _is_integer_index(value) -> bool:
    return isinstance(value, (int, np.integer))


def _normalize_slice_spec(slice_spec, ndim: int) -> dict[int, object]:
    if slice_spec is None:
        return {}

    if not isinstance(slice_spec, dict):
        raise PlotDataError("slice must be a dictionary such as {0: 3} or {2: (0, 10)}.")

    normalized = {}

    for axis, value in slice_spec.items():
        axis = int(axis)

        if axis < 0:
            axis += ndim

        if axis < 0 or axis >= ndim:
            raise PlotDataError(f"slice axis {axis} is out of range for array with {ndim} dimensions.")

        normalized[axis] = value

    return normalized


def _to_numpy_index(value):
    if isinstance(value, builtins.slice):
        return value

    if isinstance(value, tuple) or isinstance(value, builtins.list):
        if len(value) == 2:
            return builtins.slice(value[0], value[1])

        if len(value) == 3:
            return builtins.slice(value[0], value[1], value[2])

        raise PlotDataError("slice tuple/list values must have length 2 or 3.")

    return value


def _apply_slice_for_lines(array: np.ndarray, axis: int, slice_spec) -> tuple[np.ndarray, int]:
    array = np.asarray(array)

    if array.ndim == 0:
        raise PlotDataError("Cannot line-plot scalar data.")

    original_axis = _normalize_axis(axis, array.ndim)
    normalized_slice = _normalize_slice_spec(slice_spec, array.ndim)

    if original_axis in normalized_slice and _is_integer_index(normalized_slice[original_axis]):
        raise PlotDataError("The plotted axis cannot also be removed with an integer slice.")

    indexer = [builtins.slice(None)] * array.ndim

    for slice_axis, value in normalized_slice.items():
        indexer[slice_axis] = _to_numpy_index(value)

    sliced = array[tuple(indexer)]

    removed_before_axis = 0

    for slice_axis, value in normalized_slice.items():
        if slice_axis < original_axis and _is_integer_index(value):
            removed_before_axis += 1

    new_axis = original_axis - removed_before_axis

    return np.asarray(sliced), new_axis


def _apply_slice_for_map(array: np.ndarray, slice_spec) -> np.ndarray:
    array = np.asarray(array)

    if slice_spec is None:
        return array

    normalized_slice = _normalize_slice_spec(slice_spec, array.ndim)
    indexer = [builtins.slice(None)] * array.ndim

    for slice_axis, value in normalized_slice.items():
        indexer[slice_axis] = _to_numpy_index(value)

    return np.asarray(array[tuple(indexer)])


@dataclass
class H5File:
    filename: str | Path
    root: str = "/"

    def __post_init__(self):
        self.filename = str(self.filename)
        self.root = _clean_root(self.root)

    def at(self, group: str) -> "H5File":
        """
        Return a new H5File scoped to a specific HDF5 group.

        Example
        -------

        file = fplt.open("run.h5")
        run = file.at("/Pipe_Network/transient/runs/base")
        run.plot(x="time", y="tracks/Pipe_Mass_Flow_[kg_s]")
        """

        with h5py.File(self.filename, "r") as h5:
            group_path = self._resolve_group_path(h5, group)

        return H5File(self.filename, group_path)

    def tree(self, max_depth: int | None = None, print_output: bool = True) -> str:
        """
        Print the HDF5 tree under the current root.
        """

        with h5py.File(self.filename, "r") as h5:
            if self.root not in h5:
                raise DatasetNotFoundError(f"Root path {self.root!r} was not found in {self.filename!r}.")

            lines = [f"{Path(self.filename).name}:{self.root}"]

            root_object = h5[self.root]

            if isinstance(root_object, h5py.Dataset):
                lines.append(self._dataset_tree_line(root_object, prefix="└── "))
            else:
                self._append_tree_lines(
                    lines=lines,
                    group=root_object,
                    prefix="",
                    depth=0,
                    max_depth=max_depth,
                )

        text = "\n".join(lines)

        if print_output:
            print(text)

        return text

    def list(self, print_output: bool = True) -> str:
        """
        Print numeric and non-numeric datasets under the current root.
        """

        with h5py.File(self.filename, "r") as h5:
            paths = self._dataset_paths(h5)

            one_d = []
            two_d = []
            multi_d = []
            scalars = []
            non_numeric = []

            for path in paths:
                dataset = h5[path]
                rel = _relative_path(path, self.root)
                shape = dataset.shape
                dtype = dataset.dtype

                if not _is_numeric_dtype(dtype):
                    non_numeric.append((rel, shape, dtype))
                    continue

                if shape == ():
                    scalars.append((rel, shape, dtype))
                elif len(shape) == 1:
                    one_d.append((rel, shape, dtype))
                elif len(shape) == 2:
                    two_d.append((rel, shape, dtype))
                else:
                    multi_d.append((rel, shape, dtype))

        lines = [f"{Path(self.filename).name}:{self.root}", ""]

        if one_d:
            lines.append("1D numeric datasets:")
            lines.extend(self._format_dataset_rows(one_d))
            lines.append("")

        if two_d:
            lines.append("2D numeric datasets:")
            lines.extend(self._format_dataset_rows(two_d))
            lines.append("")

        if multi_d:
            lines.append("3D+ numeric datasets:")
            lines.extend(self._format_dataset_rows(multi_d))
            lines.append("")

        if scalars:
            lines.append("Scalar numeric datasets:")
            lines.extend(self._format_dataset_rows(scalars))
            lines.append("")

        if non_numeric:
            lines.append("Non-numeric datasets:")
            lines.extend(self._format_dataset_rows(non_numeric))
            lines.append("")

        text = "\n".join(lines).rstrip()

        if print_output:
            print(text)

        return text

    def values(self, group: str | None = None, print_output: bool = True) -> dict[str, Any]:
        """
        Read scalar datasets under the current root or under a selected group.
        """

        with h5py.File(self.filename, "r") as h5:
            if group is None:
                root = self.root
            else:
                root = self._resolve_group_path(h5, group)

            values = {}

            if isinstance(h5[root], h5py.Dataset):
                dataset = h5[root]

                if dataset.shape == ():
                    values[_basename(root)] = _decode_scalar(dataset[()])

            else:
                paths = self._dataset_paths(h5, root=root)

                for path in paths:
                    dataset = h5[path]

                    if dataset.shape == ():
                        rel = _relative_path(path, root)
                        values[rel] = _decode_scalar(dataset[()])

        if print_output:
            if values:
                width = max(len(name) for name in values)

                for name, value in values.items():
                    print(f"{name:<{width}}  {value}")
            else:
                print("No scalar datasets found.")

        return values

    def read(self, name: str):
        """
        Read a dataset by absolute path, relative path, or unique short name.
        """

        with h5py.File(self.filename, "r") as h5:
            path = self._resolve_dataset_path(h5, name)
            return h5[path][()]

    def plot(
        self,
        y=None,
        x=None,
        y2=None,
        slice=None,
        axis: int = -1,
        labels=None,
        y2labels=None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        y2label: str | None = None,
        title: str | None = None,
        legend: bool = True,
        legend_location: str = "best",
        grid: bool = True,
        theme: str = "dark",
        save: str | Path | None = None,
        dpi: int = 200,
        show: bool = True,
        figsize=(9, 5),
        linewidth: float = 1.6,
    ):
        """
        Plot one or more traces from an HDF5 file.

        Parameters
        ----------
        y:
            Dataset name/path or list of names/paths for the left y-axis.

        x:
            Optional x-axis dataset. If omitted, an integer index is used.

        y2:
            Optional dataset name/path or list of names/paths for the right y-axis.

        slice:
            Optional integer-index slicing for multidimensional arrays.

            Examples:

                slice={0: 3}
                slice={2: (0, 10)}
                slice={1: (0, 20, 2)}

        axis:
            Axis along which to plot multidimensional arrays.
            All remaining dimensions become separate traces.

        save:
            Optional output filename. Extension controls format, such as .png,
            .pdf, or .svg.
        """

        theme = check_theme(theme)

        if y is None and y2 is None:
            raise PlotDataError("plot requires y, y2, or both.")

        with h5py.File(self.filename, "r") as h5:
            x_array = None
            x_name = None

            if x is not None:
                x_path = self._resolve_dataset_path(h5, x)
                x_name = _basename(x_path)
                x_array = np.asarray(h5[x_path][()])

                if x_array.ndim != 1:
                    raise PlotDataError("x must be a 1D dataset.")

                if not _is_numeric_dtype(x_array.dtype):
                    raise PlotDataError("x must be numeric.")

            left_series = []
            right_series = []

            for selector in _as_list(y):
                left_series.extend(
                    self._build_line_series(
                        h5=h5,
                        selector=selector,
                        axis=axis,
                        slice_spec=slice,
                    )
                )

            for selector in _as_list(y2):
                right_series.extend(
                    self._build_line_series(
                        h5=h5,
                        selector=selector,
                        axis=axis,
                        slice_spec=slice,
                    )
                )

        all_series = left_series + right_series

        if not all_series:
            raise PlotDataError("No y data was selected.")

        if x_array is None:
            x_length = all_series[0].length
            x_array = np.arange(x_length)
            x_name = "Index"
        else:
            x_length = len(x_array)

        for series in all_series:
            if series.length != x_length:
                raise PlotDataError(
                    f"Trace {series.label!r} has length {series.length}, "
                    f"but x has length {x_length}."
                )

        self._apply_user_labels(left_series, labels, "labels")
        self._apply_user_labels(right_series, y2labels, "y2labels")

        fig, ax = plt.subplots(figsize=figsize)

        ax2 = None

        if right_series:
            ax2 = ax.twinx()

        apply_theme(fig, [ax, ax2], theme)

        colors = itertools.cycle(theme_colors(theme))

        for series in left_series:
            ax.plot(
                x_array,
                series.data,
                label=series.label,
                color=next(colors),
                linewidth=linewidth,
            )

        if ax2 is not None:
            for series in right_series:
                ax2.plot(
                    x_array,
                    series.data,
                    label=series.label,
                    color=next(colors),
                    linewidth=linewidth,
                )

        if xlabel is None:
            xlabel = x_name

        if ylabel is None:
            if len(left_series) == 1 and not right_series:
                ylabel = left_series[0].label
            else:
                ylabel = "Value"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if ax2 is not None:
            if y2label is None:
                if len(right_series) == 1:
                    y2label = right_series[0].label
                else:
                    y2label = "Value"

            ax2.set_ylabel(y2label)

        if title is not None:
            ax.set_title(title)

        if grid:
            ax.grid(True, **grid_kwargs(theme))
        else:
            ax.grid(False)

        if legend:
            handles, legend_labels = ax.get_legend_handles_labels()

            if ax2 is not None:
                handles2, legend_labels2 = ax2.get_legend_handles_labels()
                handles += handles2
                legend_labels += legend_labels2

            if handles:
                legend_object = ax.legend(handles, legend_labels, loc=legend_location)
                style_legend(legend_object, theme)

        fig.tight_layout()

        self._save_and_show(fig=fig, save=save, dpi=dpi, show=show)

        if ax2 is not None:
            return fig, (ax, ax2)

        return fig, ax

    def map(
        self,
        z,
        x=None,
        y=None,
        slice=None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        title: str | None = None,
        grid: bool = False,
        theme: str = "dark",
        save: str | Path | None = None,
        dpi: int = 200,
        show: bool = True,
        figsize=(8, 6),
        cmap: str = "viridis",
    ):
        """
        Plot a 2D dataset as a map.

        Parameters
        ----------
        z:
            2D dataset name/path, or a multidimensional dataset reduced to 2D
            with slice.

        x:
            Optional 1D x-axis dataset.

        y:
            Optional 1D y-axis dataset.

        slice:
            Optional integer-index slicing for 3D+ arrays.
        """

        theme = check_theme(theme)

        with h5py.File(self.filename, "r") as h5:
            z_path = self._resolve_dataset_path(h5, z)
            z_name = _basename(z_path)
            z_array = np.asarray(h5[z_path][()])

            if not _is_numeric_dtype(z_array.dtype):
                raise PlotDataError("z must be numeric.")

            z_array = _apply_slice_for_map(z_array, slice)

            if z_array.ndim != 2:
                raise PlotDataError(
                    f"map requires 2D data after slicing. "
                    f"Selected dataset has shape {z_array.shape}."
                )

            rows, cols = z_array.shape

            x_array = None
            y_array = None
            x_name = "Column Index"
            y_name = "Row Index"

            if x is not None:
                x_path = self._resolve_dataset_path(h5, x)
                x_name = _basename(x_path)
                x_array = np.asarray(h5[x_path][()])

                if x_array.ndim != 1:
                    raise PlotDataError("x must be a 1D dataset.")

                if len(x_array) not in (cols, cols + 1):
                    raise PlotDataError(
                        f"x has length {len(x_array)}, but z has {cols} columns."
                    )

            if y is not None:
                y_path = self._resolve_dataset_path(h5, y)
                y_name = _basename(y_path)
                y_array = np.asarray(h5[y_path][()])

                if y_array.ndim != 1:
                    raise PlotDataError("y must be a 1D dataset.")

                if len(y_array) not in (rows, rows + 1):
                    raise PlotDataError(
                        f"y has length {len(y_array)}, but z has {rows} rows."
                    )

            if x_array is None:
                x_array = np.arange(cols)

            if y_array is None:
                y_array = np.arange(rows)

        fig, ax = plt.subplots(figsize=figsize)
        apply_theme(fig, ax, theme)

        mesh = ax.pcolormesh(
            x_array,
            y_array,
            z_array,
            shading="auto",
            cmap=cmap,
        )

        colorbar = fig.colorbar(mesh, ax=ax)
        style_colorbar(colorbar, theme)

        if xlabel is None:
            xlabel = x_name

        if ylabel is None:
            ylabel = y_name

        if zlabel is None:
            zlabel = z_name

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        colorbar.set_label(zlabel)

        if title is not None:
            ax.set_title(title)

        if grid:
            ax.grid(True, **grid_kwargs(theme))
        else:
            ax.grid(False)

        fig.tight_layout()

        self._save_and_show(fig=fig, save=save, dpi=dpi, show=show)

        return fig, ax

    def _resolve_dataset_path(self, h5: h5py.File, selector: str) -> str:
        if not isinstance(selector, str):
            raise DatasetNotFoundError(f"Dataset selector must be a string, got {type(selector)}.")

        selector = selector.strip()

        if selector == "":
            raise DatasetNotFoundError("Empty dataset selector.")

        direct_path = _join_h5(self.root, selector)

        if direct_path in h5:
            if isinstance(h5[direct_path], h5py.Dataset):
                return direct_path

            raise DatasetNotFoundError(f"{direct_path!r} exists, but it is not a dataset.")

        paths = self._dataset_paths(h5)

        exact_matches = []

        for path in paths:
            basename = _basename(path)
            relative = _relative_path(path, self.root)

            if selector == basename or selector == relative:
                exact_matches.append(path)

        if len(exact_matches) == 1:
            return exact_matches[0]

        if len(exact_matches) > 1:
            self._raise_ambiguous(selector, exact_matches)

        normalized_selector = _normalize_name(selector)
        normalized_matches = []

        for path in paths:
            basename = _basename(path)
            relative = _relative_path(path, self.root)

            if (
                _normalize_name(basename) == normalized_selector
                or _normalize_name(relative) == normalized_selector
            ):
                normalized_matches.append(path)

        if len(normalized_matches) == 1:
            return normalized_matches[0]

        if len(normalized_matches) > 1:
            self._raise_ambiguous(selector, normalized_matches)

        raise DatasetNotFoundError(
            f"Could not find dataset {selector!r} under {self.root!r} in {self.filename!r}."
        )

    def _resolve_group_path(self, h5: h5py.File, selector: str) -> str:
        selector = str(selector).strip()

        if selector == "":
            raise DatasetNotFoundError("Empty group selector.")

        direct_path = _join_h5(self.root, selector)

        if direct_path in h5:
            if isinstance(h5[direct_path], h5py.Group):
                return direct_path

            raise DatasetNotFoundError(f"{direct_path!r} exists, but it is not a group.")

        groups = self._group_paths(h5)

        matches = []

        for path in groups:
            basename = _basename(path)
            relative = _relative_path(path, self.root)

            if selector == basename or selector == relative:
                matches.append(path)

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            self._raise_ambiguous(selector, matches)

        raise DatasetNotFoundError(
            f"Could not find group {selector!r} under {self.root!r} in {self.filename!r}."
        )

    def _dataset_paths(self, h5: h5py.File, root: str | None = None) -> list[str]:
        if root is None:
            root = self.root

        root = _clean_root(root)

        if root not in h5:
            raise DatasetNotFoundError(f"Root path {root!r} was not found in {self.filename!r}.")

        root_object = h5[root]
        paths = []

        if isinstance(root_object, h5py.Dataset):
            return [root]

        def visit(name, object_):
            if isinstance(object_, h5py.Dataset):
                if root == "/":
                    paths.append("/" + name)
                else:
                    paths.append(root.rstrip("/") + "/" + name)

        root_object.visititems(visit)
        paths.sort()

        return paths

    def _group_paths(self, h5: h5py.File) -> list[str]:
        if self.root not in h5:
            raise DatasetNotFoundError(f"Root path {self.root!r} was not found in {self.filename!r}.")

        root_object = h5[self.root]
        paths = []

        if isinstance(root_object, h5py.Group):
            paths.append(self.root)

            def visit(name, object_):
                if isinstance(object_, h5py.Group):
                    if self.root == "/":
                        paths.append("/" + name)
                    else:
                        paths.append(self.root.rstrip("/") + "/" + name)

            root_object.visititems(visit)

        paths.sort()

        return paths

    def _raise_ambiguous(self, selector: str, matches: list[str]):
        lines = [f"Dataset/group name {selector!r} is ambiguous. Matches:"]

        for path in matches:
            lines.append(f"  - {path}")

        lines.append("Use a full path or scope the file with file.at(...).")

        raise AmbiguousDatasetError("\n".join(lines))

    def _dataset_tree_line(self, dataset: h5py.Dataset, prefix: str) -> str:
        name = _basename(dataset.name)
        shape = _shape_string(dataset.shape)
        dtype = dataset.dtype

        return f"{prefix}{name}  {shape}  {dtype}"

    def _append_tree_lines(
        self,
        lines: list[str],
        group: h5py.Group,
        prefix: str,
        depth: int,
        max_depth: int | None,
    ):
        if max_depth is not None and depth >= max_depth:
            return

        items = sorted(group.items(), key=lambda item: item[0])

        for index, (name, object_) in enumerate(items):
            is_last = index == len(items) - 1
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if isinstance(object_, h5py.Dataset):
                shape = _shape_string(object_.shape)
                dtype = object_.dtype
                lines.append(f"{prefix}{branch}{name}  {shape}  {dtype}")
            else:
                lines.append(f"{prefix}{branch}{name}/")
                self._append_tree_lines(
                    lines=lines,
                    group=object_,
                    prefix=next_prefix,
                    depth=depth + 1,
                    max_depth=max_depth,
                )

    def _format_dataset_rows(self, rows: list[tuple[str, Any, Any]]) -> list[str]:
        if not rows:
            return []

        width = max(len(row[0]) for row in rows)
        formatted = []

        for name, shape, dtype in rows:
            formatted.append(f"  {name:<{width}}  {_shape_string(shape):<16}  {dtype}")

        return formatted

    def _build_line_series(
        self,
        h5: h5py.File,
        selector: str,
        axis: int,
        slice_spec,
    ) -> list[LineSeries]:
        path = self._resolve_dataset_path(h5, selector)
        base_label = _basename(path)

        array = np.asarray(h5[path][()])

        if not _is_numeric_dtype(array.dtype):
            raise PlotDataError(f"Dataset {path!r} is not numeric.")

        array, plot_axis = _apply_slice_for_lines(array, axis=axis, slice_spec=slice_spec)

        if array.ndim == 0:
            raise PlotDataError(f"Dataset {path!r} became scalar after slicing.")

        plot_axis = _normalize_axis(plot_axis, array.ndim)
        length = array.shape[plot_axis]

        moved = np.moveaxis(array, plot_axis, -1)

        if moved.ndim == 1:
            return [
                LineSeries(
                    data=np.asarray(moved),
                    label=base_label,
                    length=length,
                )
            ]

        trace_shape = moved.shape[:-1]
        flattened = moved.reshape((-1, moved.shape[-1]))
        series = []

        for trace_index, trace in enumerate(flattened):
            multi_index = np.unravel_index(trace_index, trace_shape)

            if len(multi_index) == 1:
                label = f"{base_label}[{multi_index[0]}]"
            else:
                label = f"{base_label}{multi_index}"

            series.append(
                LineSeries(
                    data=np.asarray(trace),
                    label=label,
                    length=length,
                )
            )

        return series

    def _apply_user_labels(self, series: list[LineSeries], labels, label_name: str):
        if labels is None:
            return

        labels = _as_list(labels)

        if len(labels) != len(series):
            raise PlotDataError(
                f"{label_name} must contain exactly {len(series)} labels. "
                f"Received {len(labels)}."
            )

        for series_item, label in zip(series, labels):
            series_item.label = str(label)

    def _save_and_show(self, fig, save, dpi: int, show: bool):
        if save is not None:
            fig.savefig(
                save,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

        if show:
            plt.show()


def open(filename: str | Path, root: str = "/") -> H5File:
    return H5File(filename=filename, root=root)


def tree(filename: str | Path, root: str = "/", **kwargs) -> str:
    return open(filename, root=root).tree(**kwargs)


def list_h5(filename: str | Path, root: str = "/", **kwargs) -> str:
    return open(filename, root=root).list(**kwargs)


def read(filename: str | Path, name: str, root: str = "/"):
    return open(filename, root=root).read(name)


def values(filename: str | Path, root: str = "/", group: str | None = None, **kwargs):
    return open(filename, root=root).values(group=group, **kwargs)


def plot(filename: str | Path, y=None, x=None, root: str = "/", **kwargs):
    return open(filename, root=root).plot(y=y, x=x, **kwargs)


def map(filename: str | Path, z, root: str = "/", **kwargs):
    return open(filename, root=root).map(z=z, **kwargs)


list = list_h5


__all__ = [
    "FullPlotError",
    "DatasetNotFoundError",
    "AmbiguousDatasetError",
    "PlotDataError",
    "H5File",
    "open",
    "tree",
    "list",
    "read",
    "values",
    "plot",
    "map",
]