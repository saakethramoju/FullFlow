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
