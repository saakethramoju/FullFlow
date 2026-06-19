import h5py
import numpy as np

from fullflow import Map1D, Map2D, Network, State


def test_map1d_from_hdf5(tmp_path):
    filename = tmp_path / "maps.h5"

    with h5py.File(filename, "w") as file:
        group = file.create_group("maps/FuelPump")
        group["x"] = np.array([3000.0, 1000.0, 2000.0])
        group["head"] = np.array([30.0, 10.0, 20.0])
        group["efficiency"] = np.array([0.70, 0.40, 0.60])

    network = Network("Map Test")
    speed = State(1500.0)

    pump_map = Map1D.from_hdf5(
        "Fuel Pump Map",
        network,
        filename=tmp_path / "maps",
        group="maps/FuelPump",
        x_value=speed,
    )

    pump_map.evaluate_states()

    assert pump_map.head.value == 15.0
    assert pump_map.efficiency.value == 0.50


def test_map1d_from_hdf5_with_custom_dataset_names(tmp_path):
    filename = tmp_path / "maps.hdf5"

    with h5py.File(filename, "w") as file:
        group = file.create_group("pump_maps/main_fuel_pump")
        group["speed"] = np.array([1000.0, 2000.0, 3000.0])
        group["head"] = np.array([10.0, 20.0, 30.0])
        group["notes"] = "ignore me"

    network = Network("Map Test")
    speed = State(2500.0)

    pump_map = Map1D.from_hdf5(
        "Fuel Pump Map",
        network,
        filename=filename,
        group="pump_maps/main_fuel_pump",
        x_value=speed,
        x_dataset="speed",
        outputs=["head"],
    )

    pump_map.evaluate_states()

    assert pump_map.head.value == 25.0


def test_map2d_from_hdf5(tmp_path):
    filename = tmp_path / "maps.h5"

    with h5py.File(filename, "w") as file:
        group = file.create_group("maps/Compressor")
        group["x"] = np.array([0.6, 0.8, 1.0])
        group["y"] = np.array([1.0, 2.0, 3.0])
        group["pressure_ratio"] = np.array([
            [1.10, 1.20, 1.30],
            [1.25, 1.40, 1.55],
            [1.35, 1.60, 1.85],
        ])

    network = Network("Map Test")
    corrected_speed = State(0.7)
    corrected_mass_flow = State(1.5)

    compressor_map = Map2D.from_hdf5(
        "Compressor Map",
        network,
        filename=tmp_path / "maps",
        group="maps/Compressor",
        x_value=corrected_speed,
        y_value=corrected_mass_flow,
    )

    compressor_map.evaluate_states()

    assert np.isclose(compressor_map.pressure_ratio.value, 1.2375)
