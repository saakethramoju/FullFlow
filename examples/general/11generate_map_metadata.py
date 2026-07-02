"""
generate_map metadata example.

FullFlow maps are stored in HDF5 files. The file is not just a block of output
numbers. It also stores enough information for users to inspect what was made
and for Map.from_hdf5() to reconstruct the interpolation problem later.

A generated map group stores:

    attrs["kind"]
        Identifies the group as a map.

    attrs["axis_order"]
        The order of the map axes. This is the order used by the output arrays.

    attrs["output_names"]
        The output datasets that were written.

    attrs["constants"]
        Fixed inputs that were passed to every map evaluation.

    attrs["metadata"]
        User notes supplied through the metadata={...} argument.

    /axes/<axis_name>
        Axis values, plus units and spacing attributes.

    /outputs/<output_name>
        One numeric array for each returned scalar output.

    /status/success
        Boolean array showing which map points succeeded.

    /status/message
        Text array with error messages for failed points.

This example generates a small pump-like map, reads back the metadata with
h5py, and then uses only selected outputs in a FullFlow Map.
"""

import json

import h5py

from fullflow import *
from fullplot import Axis, generate_map


# ---------------------------------------------------------------------------
# Map filename
# ---------------------------------------------------------------------------

map_filename = "11generate_map_metadata"


# ---------------------------------------------------------------------------
# Map function
# ---------------------------------------------------------------------------

# This function is intentionally simple. In real examples, evaluate() could call
# a property package, a performance model, a combustion calculation, or any
# other expensive calculation.
def pump_like_map(flow_coefficient, speed_parameter, reference_head):
    head_coefficient = reference_head * speed_parameter**2 * (1.0 - 0.35 * flow_coefficient**2)
    torque_coefficient = 0.20 + 0.08 * speed_parameter + 0.15 * flow_coefficient
    efficiency = 0.78 - 0.20 * (flow_coefficient - 0.6)**2

    return {
        "head_coefficient": head_coefficient,
        "torque_coefficient": torque_coefficient,
        "efficiency": efficiency,
    }


# ---------------------------------------------------------------------------
# Generate the map
# ---------------------------------------------------------------------------

generate_map(
    filename=map_filename,
    group="pump_demo",
    axes=[
        Axis.values("flow_coefficient", values=[0.0, 0.25, 0.50, 0.75, 1.0], units=""),
        Axis.linear("speed_parameter", start=0.75, stop=1.25, count=6, units=""),
    ],
    constants={
        "reference_head": 1.0,
    },
    metadata={
        "description": "Small pump-like map for the general examples folder.",
        "author_note": "metadata can store notes, source information, units notes, or version tags.",
        "valid_for": "demonstration only",
    },
    outputs=[
        "head_coefficient",
        "torque_coefficient",
        "efficiency",
    ],
    evaluate=pump_like_map,
    overwrite=True,
    raise_errors=True,
)


# ---------------------------------------------------------------------------
# Inspect the HDF5 map metadata
# ---------------------------------------------------------------------------

# Users do not need h5py to use maps, but it is useful when checking what a map
# contains. This section prints the same information that Map.from_hdf5() uses.
with h5py.File(map_filename + ".h5", "r") as file:
    map_group = file["pump_demo"]

    axis_order = json.loads(map_group.attrs["axis_order"])
    output_names = json.loads(map_group.attrs["output_names"])
    constants = json.loads(map_group.attrs["constants"])
    metadata = json.loads(map_group.attrs["metadata"])

    print("Map group:", map_group.name)
    print("Map kind:", map_group.attrs["kind"])
    print("Axis order:", axis_order)
    print("Output names:", output_names)
    print("Constants:", constants)
    print("Metadata:", metadata)

    print("\nAxes:")
    for axis_name in axis_order:
        axis_dataset = map_group["axes"][axis_name]
        print(axis_name)
        print("  units:", axis_dataset.attrs["units"])
        print("  spacing:", axis_dataset.attrs["spacing"])
        print("  values:", axis_dataset[()])

    print("\nOutput shapes:")
    for output_name in output_names:
        print(output_name, map_group["outputs"][output_name].shape)

    successful_points = map_group["status"]["success"][()]
    print("\nSuccessful map points:", successful_points.sum(), "out of", successful_points.size)


# ---------------------------------------------------------------------------
# Use selected outputs from the map
# ---------------------------------------------------------------------------

PumpMapNetwork = Network("Pump Metadata Map Example")

flow_coefficient = State(0.55)
speed_parameter = State(1.10)

PumpMap = Map.from_hdf5(
    "Pump Map",
    PumpMapNetwork,
    filename=map_filename,
    group="pump_demo",
    inputs={
        "flow_coefficient": flow_coefficient,
        "speed_parameter": speed_parameter,
    },
    outputs={
        "head": "head_coefficient",
        "efficiency": "efficiency",
    },
)

PumpMap.evaluate_states()

print("\nSelected map outputs:")
print("Flow coefficient:", flow_coefficient.value)
print("Speed parameter:", speed_parameter.value)
print("Head coefficient:", PumpMap.head.value)
print("Efficiency:", PumpMap.efficiency.value)
