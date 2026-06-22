from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import matplotlib.pyplot as plt


RETRO_COLORS = [
    "#00ffff",
    "#ff00ff",
    "#ffff00",
    "#00ff66",
    "#ff6600",
    "#6699ff",
    "#ff3366",
]


def h5_groups(filename: str | Path) -> list[str]:
    """Return every group path in an HDF5 file."""
    groups = ["/"]

    with h5py.File(filename, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                groups.append("/" + name)

        h5.visititems(visit)

    return groups


def h5_datasets(filename: str | Path) -> list[str]:
    """Return every dataset path in an HDF5 file."""
    paths = []

    with h5py.File(filename, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                paths.append("/" + name)

        h5.visititems(visit)

    return paths


def h5_find(filename: str | Path, text: str) -> list[str]:
    """Find group or dataset paths containing ``text``."""
    text = text.lower()
    paths = h5_groups(filename) + h5_datasets(filename)
    return [path for path in paths if text in path.lower()]


def h5_print(filename: str | Path, contains: str | None = None) -> None:
    """Print a compact overview of a FullFlow HDF5 file."""
    filename = Path(filename)

    with h5py.File(filename, "r") as h5:
        print(filename)
        print("/")
        for key, value in h5.attrs.items():
            print(f"  @{key} = {_decode(value)}")

        print("\nsolutions:")
        for path in h5_solution_groups(filename):
            if contains and contains.lower() not in path.lower():
                continue
            group = h5[path]
            kind = _decode(group.attrs.get("fullflow_kind", ""))
            network = _decode(group.attrs.get("network_name", ""))
            print(f"  {path}  {kind}  {network}")

        print("\nmaps:")
        for path in h5_map_groups(filename):
            if contains and contains.lower() not in path.lower():
                continue
            group = h5[path]
            axes = _decode(group.attrs.get("axis_order", ""))
            outputs = _decode(group.attrs.get("output_names", ""))
            print(f"  {path}  axes={axes} outputs={outputs}")

        print("\ntables:")
        for path in h5_table_groups(filename):
            if contains and contains.lower() not in path.lower():
                continue
            group = h5[path]
            rows = group.attrs.get("row_count", "")
            columns = [_decode(item) for item in group.attrs.get("columns", [])]
            print(f"  {path}  rows={rows} columns={columns}")

        print("\ndatasets:")
        for path in h5_datasets(filename):
            if contains and contains.lower() not in path.lower():
                continue
            dataset = h5[path]
            print(f"  {path:70s} shape={dataset.shape!s:16s} dtype={dataset.dtype}")


def h5_solution_groups(filename: str | Path) -> list[str]:
    """Return groups that look like FullFlow solution groups."""
    groups = []

    with h5py.File(filename, "r") as h5:
        for path in h5_groups(filename):
            if path == "/":
                continue
            group = h5[path]
            kind = str(_decode(group.attrs.get("fullflow_kind", "")))
            if kind.endswith("solution") or kind in {"steady_state_solution", "transient_solution"}:
                groups.append(path)

    return groups


def h5_transient_groups(filename: str | Path) -> list[str]:
    """Return groups that look like FullFlow transient solution groups."""
    groups = []

    with h5py.File(filename, "r") as h5:
        for path in h5_solution_groups(filename):
            kind = str(_decode(h5[path].attrs.get("fullflow_kind", "")))
            if kind == "transient_solution" or "transient" in path.lower():
                groups.append(path)

        if "/transient" in h5 and "/transient" not in groups:
            groups.append("/transient")

    return groups


def h5_map_groups(filename: str | Path) -> list[str]:
    """Return groups that look like FullFlow map groups."""
    groups = []

    with h5py.File(filename, "r") as h5:
        for path in h5_groups(filename):
            if path == "/":
                continue
            group = h5[path]
            if "axes" in group and "outputs" in group:
                groups.append(path)

    return groups


def h5_table_groups(filename: str | Path) -> list[str]:
    """Return groups that look like FullFlow table groups."""
    groups = []

    with h5py.File(filename, "r") as h5:
        for path in h5_groups(filename):
            if path == "/":
                continue
            group = h5[path]
            if "columns" in group.attrs:
                groups.append(path)

    return groups


def h5_attrs(filename: str | Path, path: str) -> dict:
    """Return attributes from one HDF5 group or dataset."""
    path = _resolve_path(filename, path)

    with h5py.File(filename, "r") as h5:
        return {key: _decode(value) for key, value in h5[path].attrs.items()}


def h5_read(filename: str | Path, path: str) -> np.ndarray:
    """Read a raw HDF5 dataset as a NumPy array."""
    path = _resolve_dataset_path(filename, path)

    with h5py.File(filename, "r") as h5:
        return _decode_array(np.asarray(h5[path]))


def h5_arrays(filename: str | Path, paths: Iterable[str]) -> dict[str, np.ndarray]:
    """Read several HDF5 datasets into a dictionary of NumPy arrays."""
    return {path: h5_read(filename, path) for path in paths}


def h5_table(filename: str | Path, group: str) -> dict[str, np.ndarray]:
    """Read a FullFlow table group into a dictionary of NumPy arrays."""
    group = _resolve_group_path(filename, group)

    with h5py.File(filename, "r") as h5:
        table = h5[group]
        columns = [_decode(item) for item in table.attrs.get("columns", list(table.keys()))]
        data = {}

        for column in columns:
            dataset_name = _safe_name(column)
            if dataset_name in table:
                data[column] = _decode_array(np.asarray(table[dataset_name]))

        return data


def h5_latest_solution(filename: str | Path, kind: str | None = None) -> str:
    """Return the last solution group in the file, optionally filtered by kind."""
    groups = h5_solution_groups(filename)

    if kind is not None:
        kind = kind.lower()
        groups = [path for path in groups if kind in path.lower() or kind in str(h5_attrs(filename, path).get("fullflow_kind", "")).lower()]

    if not groups:
        raise KeyError(f"No solution groups found in {filename}.")

    return groups[-1]


def h5_track_names(filename: str | Path, solution: str = "auto") -> list[str]:
    """Return tracked variable names from a transient solution."""
    solution = _resolve_transient_solution(filename, solution)

    with h5py.File(filename, "r") as h5:
        tracks_path = solution.rstrip("/") + "/tracks"
        if tracks_path in h5:
            names = []
            for name, dataset in h5[tracks_path].items():
                if name == "time":
                    continue
                names.append(str(_decode(dataset.attrs.get("label", dataset.attrs.get("long_name", name)))))
            return names

    history = h5_table(filename, solution.rstrip("/") + "/history")
    attributes = np.asarray(history["attribute"]).astype(str)
    return list(dict.fromkeys(attributes))


def h5_track(filename: str | Path, name: str, solution: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Return one tracked transient variable as ``time, values`` arrays."""
    solution = _resolve_transient_solution(filename, solution)

    with h5py.File(filename, "r") as h5:
        tracks_path = solution.rstrip("/") + "/tracks"
        if tracks_path in h5:
            tracks = h5[tracks_path]
            dataset_name = _resolve_track_dataset(tracks, name)
            time = np.asarray(tracks["time"], dtype=float)
            values = np.asarray(tracks[dataset_name], dtype=float)
            return time, values

    history = h5_table(filename, solution.rstrip("/") + "/history")
    time = np.asarray(history["time"], dtype=float)
    attributes = np.asarray(history["attribute"]).astype(str)
    values = np.asarray(history["numeric_value"], dtype=float)
    track_name = _resolve_name(list(dict.fromkeys(attributes)), name, "track")
    mask = attributes == track_name
    return time[mask], values[mask]


def h5_tracks(filename: str | Path, names: Iterable[str] | str, solution: str = "auto") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return multiple transient tracks for custom plotting."""
    if isinstance(names, str):
        names = [names]

    return {name: h5_track(filename, name, solution=solution) for name in names}


def h5_solution(filename: str | Path, solution: str = "auto") -> dict[str, np.ndarray]:
    """Read a steady-state solution records table, or a transient final records table."""
    if solution == "auto":
        solution = h5_latest_solution(filename)
    else:
        solution = _resolve_group_path(filename, solution)

    with h5py.File(filename, "r") as h5:
        if solution.rstrip("/") + "/records" in h5:
            return h5_table(filename, solution.rstrip("/") + "/records")
        if solution.rstrip("/") + "/final/records" in h5:
            return h5_table(filename, solution.rstrip("/") + "/final/records")

    raise KeyError(f"Could not find records table under {solution!r}.")


def h5_solution_value(filename: str | Path, attribute: str, component: str | None = None, solution: str = "auto") -> float:
    """Read one numeric value from a solution table."""
    records = h5_solution(filename, solution=solution)
    attributes = np.asarray(records["attribute"]).astype(str)
    values = np.asarray(records["numeric_value"], dtype=float)
    mask = _partial_match_mask(attributes, attribute)

    if component is not None:
        components = np.asarray(records["component_name"]).astype(str)
        mask = mask & _partial_match_mask(components, component)

    if np.count_nonzero(mask) == 0:
        raise KeyError(f"Could not find solution value matching {attribute!r}.")

    if np.count_nonzero(mask) > 1:
        matches = attributes[mask]
        raise KeyError("Found multiple matching values:\n" + "\n".join(matches))

    return float(values[mask][0])


def h5_map(filename: str | Path, group: str, output: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Return map axes and one output array."""
    group = _resolve_map_group(filename, group)

    with h5py.File(filename, "r") as h5:
        map_group = h5[group]
        axis_order = _decode(map_group.attrs.get("axis_order", ""))
        if axis_order:
            import json
            axis_order = json.loads(axis_order)
        else:
            axis_order = list(map_group["axes"].keys())

        axes = [np.asarray(map_group["axes"][axis], dtype=float) for axis in axis_order]
        output_name = _resolve_name(list(map_group["outputs"].keys()), output, "map output")
        values = np.asarray(map_group["outputs"][output_name], dtype=float)

    return axes, values


def h5_plot(filename: str | Path, y: Iterable[str] | str, solution: str = "auto", grid: bool = True, dark: bool = True, show: bool = True):
    """Plot one or more transient tracks."""
    if isinstance(y, str):
        names = [y]
    else:
        names = list(y)

    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots()

        for i, name in enumerate(names):
            time, values = h5_track(filename, name, solution=solution)
            ax.plot(time, values, label=name, color=RETRO_COLORS[i % len(RETRO_COLORS)])

        ax.set_title(Path(filename).name)
        ax.set_xlabel("Time")
        ax.grid(grid)
        ax.legend()
        fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def h5_plot_steps(filename: str | Path, y: Iterable[str] | str = "max_abs_residual", solution: str = "auto", x: str = "time", grid: bool = True, dark: bool = True, show: bool = True):
    """Plot transient solver diagnostic columns."""
    solution = _resolve_transient_solution(filename, solution)
    steps = h5_table(filename, solution.rstrip("/") + "/steps")

    if isinstance(y, str):
        names = [y]
    else:
        names = list(y)

    x_data = np.asarray(steps[x], dtype=float)

    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots()

        for i, name in enumerate(names):
            ax.plot(x_data, np.asarray(steps[name], dtype=float), label=name, color=RETRO_COLORS[i % len(RETRO_COLORS)])

        ax.set_title(Path(filename).name)
        ax.set_xlabel(x)
        ax.grid(grid)
        ax.legend()
        fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def h5_imshow(filename: str | Path, z: str, title: str | None = None, dark: bool = True, show: bool = True):
    """Quick image plot for a 2D raw dataset or map output."""
    try:
        z_path = _resolve_dataset_path(filename, z)
        z_data = h5_read(filename, z_path)
        label = z_path
    except Exception:
        map_group = h5_map_groups(filename)[0]
        _, z_data = h5_map(filename, map_group, z)
        label = z

    if z_data.ndim != 2:
        raise ValueError(f"{label} is not 2D. Shape is {z_data.shape}.")

    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots()
        image = ax.imshow(z_data, origin="lower", aspect="auto")
        fig.colorbar(image, ax=ax, label=label)
        ax.set_title(title or label)
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")
        fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def _resolve_transient_solution(filename: str | Path, solution: str) -> str:
    if solution == "auto":
        groups = h5_transient_groups(filename)
        if len(groups) == 1:
            return groups[0]
        if len(groups) == 0:
            raise KeyError(f"No transient solutions found in {filename}.")
        raise KeyError("Found multiple transient solutions. Pick one with solution=...:\n" + "\n".join(groups))

    return _resolve_group_path(filename, solution)


def _resolve_map_group(filename: str | Path, group: str) -> str:
    groups = h5_map_groups(filename)
    if group.strip("/") in [item.strip("/") for item in groups]:
        return "/" + group.strip("/")
    if not group.startswith("maps/") and "/maps/" + group.strip("/") in groups:
        return "/maps/" + group.strip("/")
    return _resolve_name(groups, group, "map group")


def _resolve_track_dataset(tracks: h5py.Group, name: str) -> str:
    labels = {}
    for dataset_name, dataset in tracks.items():
        if dataset_name == "time":
            continue
        label = str(_decode(dataset.attrs.get("label", dataset.attrs.get("long_name", dataset_name))))
        labels[dataset_name] = label

    exact = [dataset_name for dataset_name, label in labels.items() if name == label or name == dataset_name]
    if len(exact) == 1:
        return exact[0]

    matches = [dataset_name for dataset_name, label in labels.items() if name.lower() in label.lower() or name.lower() in dataset_name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise KeyError("Could not find tracked variable matching {!r}. Available tracks:\n{}".format(name, "\n".join(labels.values())))
    raise KeyError("Found multiple tracks matching {!r}:\n{}".format(name, "\n".join(labels[item] for item in matches)))


def _resolve_path(filename: str | Path, name: str) -> str:
    paths = h5_groups(filename) + h5_datasets(filename)
    return _resolve_name(paths, name, "path")


def _resolve_dataset_path(filename: str | Path, name: str) -> str:
    return _resolve_name(h5_datasets(filename), name, "dataset")


def _resolve_group_path(filename: str | Path, name: str) -> str:
    return _resolve_name(h5_groups(filename), name, "group")


def _resolve_name(paths: Iterable[str], name: str, kind: str) -> str:
    paths = list(paths)
    exact = "/" + name.strip("/") if str(name).strip("/") else "/"

    if exact in paths:
        return exact

    matches = [path for path in paths if name.lower().strip("/") in path.lower().strip("/")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise KeyError(f"Could not find {kind} matching {name!r}.")
    raise KeyError(f"Found multiple {kind}s matching {name!r}:\n" + "\n".join(matches))


def _partial_match_mask(values: np.ndarray, text: str) -> np.ndarray:
    return np.array([text.lower() in str(value).lower() for value in values])


def _safe_name(name) -> str:
    import re
    text = str(name).strip()
    text = re.sub(r"[\\/\x00]", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unnamed"


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return value


def _decode_array(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind in {"S", "O", "U"}:
        return np.array([_decode(item) for item in array])
    return array


def _dark_style() -> dict:
    return {
        "figure.facecolor": "#111111",
        "axes.facecolor": "#111111",
        "axes.edgecolor": "#dddddd",
        "axes.labelcolor": "#eeeeee",
        "xtick.color": "#eeeeee",
        "ytick.color": "#eeeeee",
        "text.color": "#eeeeee",
        "legend.facecolor": "#111111",
        "legend.edgecolor": "#444444",
        "grid.color": "#444444",
    }
