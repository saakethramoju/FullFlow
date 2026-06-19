import numpy as np

from fullflow import *
from thermoprop import *




class CustomRestriction(Component):

    def __init__(
        self,
        name: str,
        network: Network,
        upstream_pressure: State,
        downstream_pressure: State,
        density: State,
        cross_sectional_area: float,
        loss_coefficient: float,
        mass_flow: State | None = None,
    ):
        self.setup()

    def evaluate_states(self):
        dP = self.upstream_pressure.value - self.downstream_pressure.value
        rho = self.density.value
        A = self.cross_sectional_area.value
        K = self.loss_coefficient.value

        if dP == 0:
            mass_flow = 0.0
        else:
            mass_flow = np.sign(dP) * A * np.sqrt(2.0 * rho * abs(dP) / K)

        self.mass_flow.value = mass_flow





class CustomPump(Component):

    def __init__(self, 
                 name: str,
                 network: Network,
                 mass_flow: State,
                 density: State,
                 upstream_pressure: State,
                 discharge_pressure: State,
                 gravitational_acceleration: float = 9.80665):
        self.setup()

    def evaluate_states(self):
        mdot = self.mass_flow.value
        rho = self.density.value
        P1 = self.upstream_pressure.value
        g = self.gravitational_acceleration.value

        A = -2.6e5
        B = 0.0
        C = 40.0

        Q = mdot / rho
        H = A*Q**2 + B*Q + C

        self.P2_predicted = P1 + rho*g*H

    @property
    def iteration_variables(self):
        return [self.mass_flow]

    @property
    def residuals(self):
        return [self.P2_predicted - self.discharge_pressure.value]
    






LOXPumpSystem = Network("Liquid Oxygen Pump System")

LOX = Lookup(
    "Liquid Oxygen",
    LOXPumpSystem,
    Propellant,
    "lox",
    temperature = 90.17,
    pressure = 50 * 6894.76
)



NodeFluid = Lookup(
    "Node LOX",
    LOXPumpSystem,
    Propellant,
    LOX.composition,
    temperature = 90.17,
    pressure = 45 * 6894.76
)


LOXPumpSystem.track("Source LOX Density", LOX.density)


Fitting = CustomRestriction(
    "Fitting",
    LOXPumpSystem,
    upstream_pressure=LOX.pressure,
    downstream_pressure=NodeFluid.pressure,
    density=LOX.density,
    cross_sectional_area=(np.pi/4) * (1.5 / 39.37)**2,
    loss_coefficient=1.5652,
)


Node = Volume(
    "Pump Inlet",
    LOXPumpSystem,
    pressure=NodeFluid.pressure,
    volume=1,
    mass_flow_in=Fitting.mass_flow,
    mass_flow_out=5
)

LOXPump = CustomPump(
    "LOX Pump",
    LOXPumpSystem,
    mass_flow=Node.mass_flow_out,
    density=NodeFluid.density,
    upstream_pressure=Node.pressure,
    discharge_pressure=80 * 6894.76,
)

LOXPumpSystem.track(
    "Node Fluid Characteristics",
    NodeFluid,
    attributes={
        "Pressure [Pa]": "pressure",
        "Temperature [K]": "temperature",
        "Density [kg/m^3]": "density",
        "Dynamic Viscosity [Pa-s]": "dynamic_viscosity"
    }
)

dP = LOXPump.discharge_pressure - LOXPump.upstream_pressure
H = dP / (LOXPump.density * LOXPump.gravitational_acceleration)
LOXPumpSystem.track("Head Rise [ft]", H*3.28084)
LOXPumpSystem.track("Pressure Rise [psid]", dP/6894.76)


SteadyState(LOXPumpSystem).solve(verbose=True)