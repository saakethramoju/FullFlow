from fullflow import Component, Network, State, SteadyState, Transient


class DynamicWithoutConstructorDerivative(Component):
    def __init__(self, name, network, x: State):
        self.setup()

    def evaluate_states(self):
        self.x_dot = self.x.value - 3.0

    @property
    def dynamics(self):
        return [(self.x, self.x_dot)]


class BalanceWithoutConstructorResidual(Component):
    def __init__(self, name, network, x: State):
        self.setup()

    def evaluate_states(self):
        self.error = self.x.value - 4.0

    @property
    def balances(self):
        return [(self.x, self.error)]


def test_dynamic_derivative_can_be_created_in_evaluate_states():
    network = Network("dynamic constructor cleanup")
    x = State(0.0)
    component = DynamicWithoutConstructorDerivative("Dynamic", network, x)

    SteadyState(network).solve(verbose=False)

    assert abs(x.value - 3.0) < 1e-8
    assert abs(component.x_dot) < 1e-8


def test_balance_residual_can_be_created_in_evaluate_states():
    network = Network("balance constructor cleanup")
    x = State(0.0)
    component = BalanceWithoutConstructorResidual("Balance", network, x)

    SteadyState(network).solve(verbose=False)

    assert abs(x.value - 4.0) < 1e-8
    assert abs(component.error) < 1e-8


def test_transient_derivative_can_be_created_in_evaluate_states():
    network = Network("transient constructor cleanup")
    x = State(1.0)
    DynamicWithoutConstructorDerivative("Dynamic", network, x)

    Transient(network).solve(t_final=0.1, dt=0.05, verbose=False)

    assert x.value < 1.0
