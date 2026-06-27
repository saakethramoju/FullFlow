from fullflow import Balance, Component, Network, State, SteadyState, Transient


class HoldState(Component):
    def __init__(self, name, network, x: State):
        self.x_dot = 0.0
        self.setup()

    @property
    def dynamics(self):
        return [(self.x, self.x_dot)]


def test_steady_state_can_ignore_named_balance():
    network = Network("ignore steady balance")
    x = State(0.0)

    Balance("Move X", network, variable=x, function=x - 5.0)

    SteadyState(network).solve(ignore_balances=["Move X"])

    assert x.value == 0.0


def test_steady_state_includes_balance_by_default():
    network = Network("include steady balance")
    x = State(0.0)

    Balance("Move X", network, variable=x, function=x - 5.0)

    SteadyState(network).solve()

    assert abs(x.value - 5.0) < 1e-8


def test_transient_can_ignore_named_balance():
    network = Network("ignore transient balance")
    x = State(1.0)
    y = State(0.0)

    HoldState("Hold X", network, x)
    Balance("Move Y", network, variable=y, function=y - 2.0)

    Transient(network).solve(dt=0.1, t_final=0.1, ignore_balances=["Move Y"])

    assert x.value == 1.0
    assert y.value == 0.0


def test_transient_includes_balance_by_default():
    network = Network("include transient balance")
    x = State(1.0)
    y = State(0.0)

    HoldState("Hold X", network, x)
    Balance("Move Y", network, variable=y, function=y - 2.0)

    Transient(network).solve(dt=0.1, t_final=0.1)

    assert x.value == 1.0
    assert abs(y.value - 2.0) < 1e-8
