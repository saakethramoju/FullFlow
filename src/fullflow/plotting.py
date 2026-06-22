"""Tiny temporary HDF5 helpers for testing FullFlow exports.

These are intentionally small and dependency-light.  They are not meant to be a
permanent plotting package; they just make the simple FullFlow HDF5 layout easy
to inspect while developing solvers, maps, and examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import h5py
import numpy as np
import matplotlib.pyplot as plt

from fullflow.Exports.HDF5 import hdf5_path, safe_group_name


RETRO_COLORS = [
    "#00ffff",
    "#ff00ff",
    "#ffff00",
    "#00ff66",
    "#ff6600",
    "#6699ff",
    "#ff3366",
]


def h5_datasets(filename: str | Path) -> list[str]:
    """Return every dataset path in an HDF5 file."""
    paths = []
    with h5py.File(hdf5_path(filename), "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                paths.append("/" + name)
        h5.visititems(visit)
    return paths


def h5_groups(filename: str | Path) -> list[str]:
    """Return every group path in an HDF5 file."""
    paths = ["/"]
    with h5py.File(hdf5_path(filename), "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                paths.append("/" + name)
        h5.visititems(visit)
    return paths


def h5_objects(filename: str | Path, kind: str | None = None) -> list[str]:
    """Return top-level FullFlow objects, optionally filtered by kind."""
    objects = []
    with h5py.File(hdf5_path(filename), "r") as h5:
        for name, item in h5.items():
            if not isinstance(item, h5py.Group):
                continue
            item_kind = _decode(item.attrs.get("kind", ""))
            if kind is not None and item_kind != kind:
                continue
            objects.append("/" + name)
    return objects


def h5_networks(filename: str | Path) -> list[str]:
    """Return top-level network groups."""
    return h5_objects(filename, kind="network")


def h5_maps(filename: str | Path) -> list[str]:
    """Return top-level map groups."""
    return h5_objects(filename, kind="map")


def h5_read(filename: str | Path, path: str):
    """Read one raw HDF5 dataset."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        resolved = _resolve_dataset_path(h5, path)
        data = np.asarray(h5[resolved])
        if data.shape == ():
            return _decode(data.item())
        return _decode_array(data)


def h5_table(filename: str | Path, group: str) -> dict[str, np.ndarray]:
    """Read a table group into a dictionary of NumPy arrays."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        group_path = _resolve_group_path(h5, group)
        table = h5[group_path]
        columns = [_decode(item) for item in table.attrs.get("columns", list(table.keys()))]
        data = {}
        for column in columns:
            dataset_name = safe_group_name(column)
            if dataset_name in table:
                data[column] = _decode_array(np.asarray(table[dataset_name]))
        return data


def h5_print(filename: str | Path, datasets: bool = False) -> None:
    """Print a compact summary of a FullFlow HDF5 file."""
    filename = hdf5_path(filename)
    print(filename.name)

    with h5py.File(filename, "r") as h5:
        if h5.attrs:
            print("/")
            for key, value in h5.attrs.items():
                print(f"  @{key} = {_decode(value)}")

        print("\nobjects:")
        for name, item in h5.items():
            if not isinstance(item, h5py.Group):
                continue
            kind = _decode(item.attrs.get("kind", "group"))
            display_name = _decode(item.attrs.get("name", name))
            print(f"  /{name}  {kind}  {display_name}")

            if kind == "map":
                axes = list(item.get("axes", {}).keys()) if "axes" in item else []
                outputs = list(item.get("outputs", {}).keys()) if "outputs" in item else []
                print(f"      axes: {axes}")
                print(f"      outputs: {outputs}")

            if kind == "network":
                sections = [child for child in item.keys() if isinstance(item[child], h5py.Group)]
                print(f"      sections: {sections}")
                for section in ["steady_state", "transient"]:
                    if section in item:
                        _print_network_section(item[section], indent="      ")

        if datasets:
            print("\ndatasets:")
            for path in h5_datasets(filename):
                dataset = h5[path]
                print(f"  {path:72s} shape={dataset.shape!s:16s} dtype={dataset.dtype}")


def h5_components(filename: str | Path, object_name: str | None = None, section: str = "steady_state") -> list[str]:
    """Return component names in a network solution section."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        network = _resolve_network_group(h5, object_name)
        path = f"{network}/{section}/components"
        if path not in h5:
            raise KeyError(f"Could not find {path}.")
        return [_decode(h5[path][name].attrs.get("name", name)) for name in h5[path]]


def h5_attributes(
    filename: str | Path,
    component: str,
    object_name: str | None = None,
    section: str = "steady_state",
) -> list[str]:
    """Return attributes available for one component in a solution section."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        component_group = _component_group(h5, object_name, section, component)
        return [_decode(dataset.attrs.get("attribute", name)) for name, dataset in component_group.items()]


def h5_solution_value(
    filename: str | Path,
    attribute: str,
    component: str | None = None,
    object_name: str | None = None,
    section: str = "steady_state",
):
    """Read one steady-state or final scalar value by component/attribute."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        network = _resolve_network_group(h5, object_name)
        if component is None:
            table_path = f"{network}/{section}/table"
            table = h5_table(filename, table_path)
            attrs = np.asarray(table["attribute"]).astype(str)
            values = table["numeric_value"] if "numeric_value" in table else table["value"]
            mask = _partial_match_mask(attrs, attribute)
            if np.count_nonzero(mask) != 1:
                raise KeyError(f"Expected one match for {attribute!r}; found {np.count_nonzero(mask)}.")
            return values[mask][0]

        component_group = _component_group(h5, object_name, section, component)
        dataset_path = _resolve_child_dataset(component_group, attribute)
        value = np.asarray(component_group[dataset_path])
        if value.shape == ():
            return _decode(value.item())
        return _decode_array(value)


def h5_history(
    filename: str | Path,
    component: str,
    attribute: str,
    object_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transient time and component-history arrays."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        network = _resolve_network_group(h5, object_name)
        time = np.asarray(h5[f"{network}/transient/time"], dtype=float)
        component_group = _component_group(h5, object_name, "transient", component)
        dataset_name = _resolve_child_dataset(component_group, attribute)
        values = _decode_array(np.asarray(component_group[dataset_name]))
        return time, values


def h5_track_names(filename: str | Path, object_name: str | None = None) -> list[str]:
    """Return tracked transient aliases for a network."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        network = _resolve_network_group(h5, object_name)
        path = f"{network}/transient/tracks"
        if path not in h5:
            return []
        return [_decode(h5[path][name].attrs.get("name", name)) for name in h5[path]]


def h5_track(filename: str | Path, name: str, object_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return one transient tracked alias as time and values."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        network = _resolve_network_group(h5, object_name)
        time = np.asarray(h5[f"{network}/transient/time"], dtype=float)
        tracks = h5[f"{network}/transient/tracks"]
        dataset_name = _resolve_child_dataset(tracks, name)
        values = _decode_array(np.asarray(tracks[dataset_name]))
        return time, values


def h5_tracks(filename: str | Path, names: Iterable[str] | str, object_name: str | None = None):
    """Return multiple transient tracked aliases."""
    if isinstance(names, str):
        names = [names]
    return {name: h5_track(filename, name, object_name=object_name) for name in names}


def h5_map(filename: str | Path, group: str, output: str):
    """Return map axes and one output array."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        map_path = _resolve_object_group(h5, group, kind="map")
        map_group = h5[map_path]
        axis_order = _json_attr(map_group.attrs.get("axis_order", "[]"))
        axes = {axis: np.asarray(map_group["axes"][axis], dtype=float) for axis in axis_order}
        output_name = _resolve_child_dataset(map_group["outputs"], output)
        values = np.asarray(map_group["outputs"][output_name], dtype=float)
        return axes, values


def h5_plot(
    filename: str | Path,
    y: Iterable[str] | str | None = None,
    *,
    component: str | None = None,
    attribute: str | None = None,
    object_name: str | None = None,
    grid: bool = True,
    dark: bool = True,
    title: str | None = None,
    show: bool = True,
):
    """Plot transient tracks or a component attribute history.

    Examples
    --------
    h5_plot("test.h5", "Node Pressure [Pa]")
    h5_plot("test.h5", component="Node", attribute="pressure")
    """
    if dark:
        style = _dark_style()
    else:
        style = {}

    with plt.rc_context(style):
        fig, ax = plt.subplots()

        if component is not None and attribute is not None:
            time, values = h5_history(filename, component, attribute, object_name=object_name)
            ax.plot(time, values, label=f"{component}:{attribute}", color=RETRO_COLORS[0])
        else:
            if y is None:
                raise ValueError("Provide y=... for tracks or component=... and attribute=... for histories.")
            if isinstance(y, str):
                names = [y]
            else:
                names = list(y)
            for i, name in enumerate(names):
                time, values = h5_track(filename, name, object_name=object_name)
                ax.plot(time, values, label=name, color=RETRO_COLORS[i % len(RETRO_COLORS)])

        ax.set_title(title or Path(filename).name)
        ax.set_xlabel("Time")
        ax.grid(grid)
        ax.legend()
        fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def h5_plot_steps(
    filename: str | Path,
    y: Iterable[str] | str = "max_abs_residual",
    object_name: str | None = None,
    x: str = "time",
    grid: bool = True,
    dark: bool = True,
    show: bool = True,
):
    """Plot transient diagnostic columns."""
    with h5py.File(hdf5_path(filename), "r") as h5:
        network = _resolve_network_group(h5, object_name)
        table_path = f"{network}/transient/diagnostics"
    table = h5_table(filename, table_path)
    x_data = np.asarray(table[x], dtype=float)

    if isinstance(y, str):
        y_names = [y]
    else:
        y_names = list(y)

    if dark:
        style = _dark_style()
    else:
        style = {}

    with plt.rc_context(style):
        fig, ax = plt.subplots()
        for i, name in enumerate(y_names):
            ax.plot(x_data, np.asarray(table[name], dtype=float), label=name, color=RETRO_COLORS[i % len(RETRO_COLORS)])
        ax.set_title(Path(filename).name)
        ax.set_xlabel(x)
        ax.grid(grid)
        ax.legend()
        fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


# Backward-compatible aliases for earlier helper names.
def h5_solution(filename: str | Path, group: str | None = None):
    section = "steady_state" if group is None else group.strip("/")
    with h5py.File(hdf5_path(filename), "r") as h5:
        network = _resolve_network_group(h5, None)
        return h5_table(filename, f"{network}/{section}/table")


def h5_transient_groups(filename: str | Path) -> list[str]:
    return [network + "/transient" for network in h5_networks(filename)]


# Internal helpers ---------------------------------------------------------

def _print_network_section(section_group: h5py.Group, indent: str) -> None:
    pieces = []
    if "components" in section_group:
        pieces.append(f"components={len(section_group['components'])}")
    if "tracks" in section_group:
        pieces.append(f"tracks={len(section_group['tracks'])}")
    if "time" in section_group:
        pieces.append(f"time={section_group['time'].shape}")
    if "diagnostics" in section_group:
        rows = section_group["diagnostics"].attrs.get("row_count", "?")
        pieces.append(f"diagnostics_rows={rows}")
    print(f"{indent}{section_group.name}: " + ", ".join(pieces))


def _resolve_network_group(h5: h5py.File, object_name: str | None) -> str:
    if object_name is not None:
        return _resolve_object_group(h5, object_name, kind="network")
    networks = [path for path in h5_objects(h5.filename, kind="network")]
    if len(networks) == 1:
        return networks[0]
    if len(networks) == 0:
        raise KeyError("No network object found in HDF5 file.")
    raise KeyError("Multiple network objects found. Pass object_name=...:\n" + "\n".join(networks))


def _resolve_object_group(h5: h5py.File, name: str, kind: str | None = None) -> str:
    candidates = []
    for candidate in {name.strip("/"), safe_group_name(name)}:
        if candidate in h5:
            path = "/" + candidate
            if kind is None or _decode(h5[path].attrs.get("kind", "")) == kind:
                return path
    for group_name, item in h5.items():
        if not isinstance(item, h5py.Group):
            continue
        if kind is not None and _decode(item.attrs.get("kind", "")) != kind:
            continue
        display_name = _decode(item.attrs.get("name", group_name))
        if name.lower() in display_name.lower() or name.lower() in group_name.lower():
            candidates.append("/" + group_name)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError(f"Could not find object {name!r}.")
    raise KeyError(f"Found multiple objects matching {name!r}:\n" + "\n".join(candidates))


def _component_group(h5: h5py.File, object_name: str | None, section: str, component: str) -> h5py.Group:
    network = _resolve_network_group(h5, object_name)
    components_path = f"{network}/{section}/components"
    if components_path not in h5:
        raise KeyError(f"Could not find {components_path}.")
    components_group = h5[components_path]
    group_name = _resolve_child_group(components_group, component)
    return components_group[group_name]


def _resolve_child_group(group: h5py.Group, name: str) -> str:
    if name in group:
        return name
    safe = safe_group_name(name)
    if safe in group:
        return safe
    matches = []
    for child_name, child in group.items():
        if not isinstance(child, h5py.Group):
            continue
        display_name = _decode(child.attrs.get("name", child_name))
        if name.lower() in display_name.lower() or name.lower() in child_name.lower():
            matches.append(child_name)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Could not find group matching {name!r} in {group.name}.")
    raise KeyError(f"Found multiple groups matching {name!r}:\n" + "\n".join(matches))


def _resolve_child_dataset(group: h5py.Group, name: str) -> str:
    if name in group:
        return name
    safe = safe_group_name(name)
    if safe in group:
        return safe
    matches = []
    for child_name, child in group.items():
        if not isinstance(child, h5py.Dataset):
            continue
        display_name = _decode(child.attrs.get("name", child.attrs.get("attribute", child_name)))
        if name.lower() in display_name.lower() or name.lower() in child_name.lower():
            matches.append(child_name)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Could not find dataset matching {name!r} in {group.name}.")
    raise KeyError(f"Found multiple datasets matching {name!r}:\n" + "\n".join(matches))


def _resolve_dataset_path(h5: h5py.File, path: str) -> str:
    if path in h5 and isinstance(h5[path], h5py.Dataset):
        return path
    if not path.startswith("/") and ("/" + path) in h5 and isinstance(h5["/" + path], h5py.Dataset):
        return "/" + path
    matches = [candidate for candidate in h5_datasets(h5.filename) if path.lower() in candidate.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Could not find dataset matching {path!r}.")
    raise KeyError(f"Found multiple datasets matching {path!r}:\n" + "\n".join(matches))


def _resolve_group_path(h5: h5py.File, path: str) -> str:
    if path in h5 and isinstance(h5[path], h5py.Group):
        return path
    if not path.startswith("/") and ("/" + path) in h5 and isinstance(h5["/" + path], h5py.Group):
        return "/" + path
    matches = [candidate for candidate in h5_groups(h5.filename) if path.lower() in candidate.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Could not find group matching {path!r}.")
    raise KeyError(f"Found multiple groups matching {path!r}:\n" + "\n".join(matches))


def _partial_match_mask(values: np.ndarray, text: str) -> np.ndarray:
    return np.array([text.lower() in str(value).lower() for value in values])


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.decode()
    return value


def _decode_array(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind in {"S", "O", "U"}:
        return np.array([_decode(item) for item in array])
    return array


def _json_attr(value):
    value = _decode(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return value


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
