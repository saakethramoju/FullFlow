from fullflow import *
from fullplot import Trace


class _Ramp(Component):
    def __init__(self, name, network, output):
        self.setup()

    def evaluate_states(self):
        self.output.value = float(self.network.time.value)


def test_sensor_condition_command_switches_without_redline_abort():
    network = Network("Condition Command Test")

    reading = State(0.0)
    command = State(0.0)
    _Ramp("Ramp", network, output=reading)

    redline = Trace(
        x=[0.0, 2.0],
        y=[0.5, 0.5],
        name="High Reading",
        role="redline",
    )
    sensor = Sensor("Sensor", network, reading=reading, conditions=redline)

    open_command = Trace(
        x=[0.0, 1.0],
        y=[1.0, 1.0],
        name="Open",
        role="command",
    )
    close_command = Trace(
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        name="Close",
        role="command",
    )

    sequence = Sequence("Sequence", network)
    sequence.command(command, open_command)
    sequence.command(command, close_command, condition=(sensor, "High Reading"))

    Transient(network).solve(dt=0.1, t_final=1.0, verbose=False)

    assert float(network.time.value) == 1.0
    assert command.value == 0.0

    events = network.sensor_event_records()
    assert len(events) == 1
    assert events[0]["role"] == "redline"
    assert events[0]["action"] == "event"


def test_sequence_command_rejects_unknown_sensor_condition():
    network = Network("Bad Condition Test")
    reading = State(0.0)
    command = State(0.0)

    line = Trace(x=[0.0, 1.0], y=[0.5, 0.5], name="Known", role="blueline")
    sensor = Sensor("Sensor", network, reading=reading, conditions=line)
    command_trace = Trace(x=[0.0, 1.0], y=[0.0, 1.0], name="Command", role="command")

    sequence = Sequence("Sequence", network)

    try:
        sequence.command(command, command_trace, condition=(sensor, "Missing"))
    except SolverSetupError:
        pass
    else:
        raise AssertionError("missing Sensor condition should raise SolverSetupError")
