from pathlib import Path

import h5py

from fullflow import Component, Network, State, SteadyState


class StaticComponent(Component):
    def __init__(self, name, network, value=1.0, output=None):
        self.setup()

    def evaluate_states(self):
        self.output.value = 2.0 * self.value.value


class SolveComponent(Component):
    _iteration_variable_names = ("x",)

    def __init__(self, name, network, x=None, target=1.0):
        self.setup()

    @property
    def residuals(self):
        return [self.x.value - self.target.value]


def test_network_save_adds_h5_extension(tmp_path):
    filename = tmp_path / "static_export"
    network = Network("Static Export")
    component = StaticComponent("Static", network, value=2.0)
    network.track("output", component.output)

    SteadyState(network).static_evaluate(filename=str(filename))

    path = filename.with_suffix(".h5")
    assert path.exists()

    with h5py.File(path, "r") as h5:
        assert "solution/current/records" in h5
        assert "numeric_value" in h5["solution/current/records"]


def test_solve_statistics_are_written_to_same_h5_file(tmp_path):
    filename = tmp_path / "solve_export.h5"
    network = Network("Solve Export")
    SolveComponent("Solve", network, x=State(0.0), target=3.0)

    SteadyState(network).solve(
        filename=str(filename),
        statistics=True,
        jacobian_method="2-point",
    )

    with h5py.File(filename, "r") as h5:
        assert "solution/current/records" in h5
        assert "statistics/current/evaluations" in h5
        assert "statistics/current/residuals" in h5
        assert "statistics/current/variables" in h5
