import pytest

from fullflow import Component, Network, State, Transient
import fullflow.Solvers.transient.operations as transient_operations


class ConstantStorage(Component):
    """Storage component whose accepted predictor already satisfies BE."""

    def __init__(self, name, network, value):
        self.rate = 0.0
        self.setup()

    def evaluate_states(self):
        self.rate = 0.0

    @property
    def dynamics(self):
        return [(self.value, self.rate)]


def test_transient_skips_least_squares_when_predictor_residual_is_acceptable(monkeypatch):
    network = Network("Predictor Fast Path")
    ConstantStorage("Constant Storage", network, value=State(3.0))

    def fail_least_squares(*args, **kwargs):
        raise AssertionError("least_squares should not run for an already-satisfied timestep")

    monkeypatch.setattr(transient_operations, "least_squares", fail_least_squares)

    Transient(network).solve(dt=0.01, t_final=0.03, rtol=1.0e-12)

    assert network.time.value == pytest.approx(0.03)
