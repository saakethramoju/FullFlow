"""
Continuous feedback control of an orbital spacecraft battery heater.

Physical layout
---------------

    Radiative environment (260 K)
              ^
              | heat loss
      +--------------------+
      | Battery thermal    |<---- heater power command
      | mass, C [J/K]      |          ^
      +--------------------+          |
              |                      |
              +---- temperature ---> P feedback Sequence

The temperature and heater power are solved together by the implicit transient
solver.
"""

import fullplot as fplt
from fullflow import *


# -----------------------------------------------------------------------------
# Run options
# -----------------------------------------------------------------------------

dt = 1.0
t_final = 360.0
filename = '24closed_loop_control'


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

heat_capacity = 1500.0
heat_leak = 6.0
ambient = 260.0
setpoint = 290.0
heater_limit = 360.0


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

Thermal = Network("Spacecraft Battery Heater")
temperature = State(270.0)
heater_power = State(0.0)


class Battery(Component):
    def __init__(self, name, network, temperature, heater, temperature_rate=None):
        self.setup()

    def evaluate_states(self):
        self.temperature_rate.value = (
            self.heater.value - heat_leak * (self.temperature.value - ambient)
        ) / heat_capacity

    @property
    def dynamics(self):
        return [(self.temperature, self.temperature_rate)]


def heater_control(t, measured):
    return max(
        0.0,
        min(
            heater_limit,
            heat_leak * (setpoint - ambient) + 20.0 * (setpoint - measured),
        ),
    )


Sequence(
    "Battery Heater Control",
    Thermal,
    target=heater_power,
    function=heater_control,
    inputs=[temperature],
)

Battery(
    "Battery Thermal Mass",
    Thermal,
    temperature=temperature,
    heater=heater_power,
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

Thermal.track("Battery Temperature [K]", temperature)
Thermal.track("Heater Power [W]", heater_power)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------

Transient(Thermal).solve(
    dt=dt,
    t_final=t_final,
    verbose=True,
    statistics=True,
    filename=filename,
)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

results = fplt.open(filename).at(
    "Spacecraft_Battery_Heater/transient/runs/base/tracks"
)
results.plot(
    y=results.trace(y="Battery Temperature [K]", x="time"),
    y2=results.trace(y="Heater Power [W]", x="time"),
    xlabel="Time [s]",
    title="Continuous Spacecraft Heater Control",
)
fplt.show()
