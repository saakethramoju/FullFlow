import pytest

from fullflow.System import Component, Model


class DummyNetwork:
    def __init__(self):
        self.components = []
        self.models = []

    def add_component(self, component):
        self.components.append(component)

    def remove_component(self, component):
        self.components.remove(component)

    def add_model(self, model):
        self.models.append(model)


class DummyComponent(Component):
    def __init__(self, name, network, value=1.0):
        self.setup()


def test_template_rejects_network_positional_argument():
    network = DummyNetwork()

    with pytest.raises(TypeError, match="does not accept a Network"):
        DummyComponent.template(network, value=2.0)


def test_template_rejects_network_keyword_argument():
    network = DummyNetwork()

    with pytest.raises(TypeError, match="does not accept network"):
        DummyComponent.template(network=network, value=2.0)


def test_model_uses_option_insertion_order_when_order_is_omitted():
    network = DummyNetwork()
    model = Model("Dummy Slot", network)

    model.option("first", DummyComponent.template(value=1.0))
    model.option("second", DummyComponent.template(value=2.0))

    assert model.order == ["first", "second"]


def test_model_explicit_order_wins_over_insertion_order():
    network = DummyNetwork()
    model = Model("Dummy Slot", network, order=["second", "first"])

    model.option("first", DummyComponent.template(value=1.0))
    model.option("second", DummyComponent.template(value=2.0))

    assert model.order == ["second", "first"]


def test_model_build_supplies_network_to_template():
    network = DummyNetwork()
    model = Model("Dummy Slot", network)
    model.option("first", DummyComponent.template(value=1.0))

    component = model.build()

    assert component.name == "Dummy Slot"
    assert component.network is network
    assert network.components == [component]
