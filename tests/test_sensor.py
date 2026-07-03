import h5py
import numpy as np
from fullplot import Trace

from fullflow import Network, Sensor, State, SteadyState, Transient


def test_sensor_balances_to_previous_trace_point():
    network = Network("Sensor Balance Test")
    variable = State(0.0)
    reading = 2.0 * variable
    data = Trace(x=[0.0, 1.0], y=[10.0, 20.0], name="CHPT")

    Sensor("CHPT", network, reading=reading, variable=variable, data=data)
    network.time.value = 0.5

    SteadyState(network).solve()

    assert np.isclose(variable.value, 5.0)


def test_sensor_extend_holds_variable_through_nan(tmp_path):
    filename = tmp_path / "sensor_extend"
    network = Network("Sensor Extend Test")
    variable = State(0.0)
    reading = 2.0 * variable
    data = Trace(x=[0.0, 1.0, 2.0, 3.0], y=[10.0, 20.0, np.nan, 40.0], name="CHPT")

    Sensor("CHPT", network, reading=reading, variable=variable, data=data, extend=True)

    Transient(network).solve(dt=1.0, t_final=3.0, filename=filename)

    with h5py.File(filename.with_suffix(".h5"), "r") as h5:
        group = h5["Sensor_Extend_Test/transient/runs/base/sensors/CHPT"]
        np.testing.assert_allclose(group["active"][()], [1.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(group["variable_value"][()], [0.0, 10.0, 10.0, 20.0])


def test_sensor_extend_false_stops_cleanly():
    network = Network("Sensor Stop Test")
    variable = State(0.0)
    reading = 2.0 * variable
    data = Trace(x=[0.0, 1.0, 2.0], y=[10.0, 20.0, np.nan], name="CHPT")

    Sensor("CHPT", network, reading=reading, variable=variable, data=data, extend=False)

    Transient(network).solve(dt=1.0, t_final=2.0)

    assert np.isclose(network.time.value, 1.0)
    assert np.isclose(variable.value, 10.0)


def test_sensor_data_rejects_non_data_trace():
    network = Network("Sensor Role Validation Test")
    data = Trace(x=[0.0, 1.0], y=[1.0, 1.0], name="Limit", role="redline")

    try:
        Sensor("T", network, reading=network.time, data=data)
    except Exception as error:
        assert "role='data'" in str(error)
    else:
        raise AssertionError("Sensor accepted a non-data trace as data.")


def test_sensor_conditions_reject_command_trace():
    network = Network("Sensor Command Rejection Test")
    command = Trace(x=[0.0, 1.0], y=[0.0, 1.0], name="Valve Command", role="command")

    try:
        Sensor("T", network, reading=network.time, conditions=[command])
    except Exception as error:
        assert "cannot consume command trace" in str(error)
    else:
        raise AssertionError("Sensor accepted a command trace as a condition.")


def test_sensor_condition_events_export_to_hdf5(tmp_path):
    filename = tmp_path / "sensor_conditions"
    network = Network("Sensor Condition Test")
    yellowline = Trace(x=[0.0, 3.0], y=[1.5, 1.5], name="Time Warning", role="yellowline")

    Sensor("Clock", network, reading=network.time, conditions=[yellowline])

    Transient(network).solve(dt=1.0, t_final=3.0, filename=filename)

    assert len(network.sensor_event_list) == 1
    event = network.sensor_event_list[0]
    assert event.role == "yellowline"
    assert event.action == "warning"
    assert np.isclose(event.time, 2.0)

    with h5py.File(filename.with_suffix(".h5"), "r") as h5:
        run = h5["Sensor_Condition_Test/transient/runs/base"]
        assert "events" in run
        assert run["events"].attrs["row_count"] == 1
        assert run["events/role"][0].decode() == "yellowline"

        trace_group = run["sensors/Clock/conditions/Time_Warning"]
        assert trace_group.attrs["role"] == "yellowline"
        np.testing.assert_allclose(trace_group["value"][()], [1.5, 1.5])


def test_sensor_redline_aborts_and_exports(tmp_path):
    filename = tmp_path / "sensor_redline"
    network = Network("Sensor Redline Test")
    redline = Trace(x=[0.0, 4.0], y=[1.5, 1.5], name="Time Abort", role="redline")

    Sensor("Clock", network, reading=network.time, conditions=[redline])

    Transient(network).solve(dt=1.0, t_final=4.0, filename=filename)

    assert np.isclose(network.time.value, 2.0)
    assert len(network.sensor_event_list) == 1
    assert network.sensor_event_list[0].role == "redline"

    with h5py.File(filename.with_suffix(".h5"), "r") as h5:
        run = h5["Sensor_Redline_Test/transient/runs/base"]
        assert run["events"].attrs["row_count"] == 1
        assert run["events/role"][0].decode() == "redline"
        metadata_keys = [item.decode() for item in run["metadata/key"][()]]
        metadata_values = [item.decode() for item in run["metadata/value"][()]]
        metadata = dict(zip(metadata_keys, metadata_values))
        assert metadata["aborted_by_redline"] == "True"
