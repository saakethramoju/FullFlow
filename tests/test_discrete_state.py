from fullflow import Component, Network, Schedule, State, Step, Transient


class ModeComponent(Component):
    def __init__(self, name, network, signal, output=None, is_on=False):
        self.setup()

    def evaluate_states(self):
        mode = self.is_on.propose(self.signal.value > 0.0)
        self.output.value = 1.0 if mode else 0.0


def test_state_propose_immediate_assignment():
    mode = State(False)

    assert mode.propose(True) is True
    assert mode.value is True

    assert mode.propose(False) is False
    assert mode.value is False


def test_state_propose_freeze_commit_and_reject():
    mode = State(False)

    mode.freeze_discrete()
    assert mode.propose(True) is False
    assert mode.value is False
    mode.commit_discrete()
    assert mode.value is True

    mode.freeze_discrete()
    assert mode.propose(False) is True
    mode.reject_discrete()
    assert mode.value is True


def test_state_propose_hysteresis():
    mode = State(False)

    assert mode.propose(4.0, turn_on=5.0, turn_off=2.0) is False
    assert mode.propose(6.0, turn_on=5.0, turn_off=2.0) is True
    assert mode.propose(3.0, turn_on=5.0, turn_off=2.0) is True
    assert mode.propose(1.0, turn_on=5.0, turn_off=2.0) is False


def test_transient_solver_commits_proposed_modes_after_accepted_step():
    network = Network("discrete mode")
    signal = State(0.0)
    output = State(0.0)

    Schedule("signal schedule", network, target=signal, function=Step(0.05, 0.0, 1.0))
    component = ModeComponent("mode component", network, signal=signal, output=output)

    Transient(network).solve(dt=0.1, t_final=0.1, verbose=False)

    assert component.is_on.value is True
    assert output.value == 1.0
    assert network.time.value == 0.1


def test_cavitating_venturi_uses_proposed_discrete_mode():
    from fullflow import CavitatingVenturi

    network = Network("venturi")
    P1 = State(1.0e6)
    P2 = State(9.9e5)
    rho = State(1000.0)
    Pvap = State(1.0e5)
    mass_flow = State(0.0)

    venturi = CavitatingVenturi(
        "venturi",
        network,
        upstream_pressure=P1,
        downstream_pressure=P2,
        density=rho,
        throat_area=1.0e-4,
        vapor_pressure=Pvap,
        mass_flow=mass_flow,
    )

    venturi.evaluate_states()
    assert venturi.is_cavitating.value is False

    venturi.is_cavitating.freeze_discrete()
    P2.value = 1.0e5
    venturi.evaluate_states()
    assert venturi.is_cavitating.value is False

    venturi.is_cavitating.commit_discrete()
    venturi.evaluate_states()
    assert venturi.is_cavitating.value is True
