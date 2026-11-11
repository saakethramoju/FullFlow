"""
Transient LOX Tank With a Distributed Mass Profile
==================================================

This example integrates one liquid inventory while calculating the mass in
40 axial tank cells. The cell masses are ordinary values inside one State,
not 40 additional solver variables.

The liquid settles toward the aft end of a cylindrical tank. As liquid drains,
the fill boundary moves and the vehicle Layout recalculates total mass and CG.
Both the tank and Layout histories are exported to the same HDF5 file.
"""

import numpy as np
from fullflow import *


filename = "1draining_tank"
dt = 0.5
t_final = 40.0


class PropellantTank(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        liquid_mass: State,
        ullage_mass: State,
        dry_mass: State | float,
        mass_flow_out: State,
        length: float,
        radius: float,
        liquid_density: float,
        cells: int = 40,
        mass_distribution=None,
    ):
        self.setup()

        edges = np.linspace(0.0, self.length.value, int(self.cells.value) + 1)
        self.stations = State(0.5 * (edges[:-1] + edges[1:]))
        self._cell_volume = np.pi * self.radius.value**2 * np.diff(edges)
        self._aft_volume = np.pi * self.radius.value**2 * (edges[-1] - edges[1:])

    def evaluate_states(self):
        liquid_volume = self.liquid_mass.value / self.liquid_density.value
        wet_volume = np.clip(liquid_volume - self._aft_volume, 0.0, self._cell_volume)
        gas_volume = self._cell_volume - wet_volume

        # The calculated cells do not appear in dynamics or balances.
        self.mass_distribution.value = (
            self.dry_mass.value / len(self.stations.value)
            + self.liquid_density.value * wet_volume
            + self.ullage_mass.value * gas_volume / gas_volume.sum()
        )

    @property
    def dynamics(self):
        return [(self.liquid_mass, -self.mass_flow_out.value)]


class Layout(Component):
    def __init__(
        self,
        name: str,
        network: Network,
        tank: PropellantTank,
        mass=None,
        cg=None,
    ):
        self.setup()

    def evaluate_states(self):
        masses = self.tank.mass_distribution.value
        stations = self.tank.stations.value

        self.mass.value = float(np.sum(masses))
        self.cg.value = float(np.dot(masses, stations) / self.mass.value)


# ---------------------------------------------------------------------------
# Network and fluid inventory
# ---------------------------------------------------------------------------

Vehicle = Network("Draining Tank")

liquid_mass = State(30.0)
ullage_mass = State(0.4)
flow_out = State(0.35)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

OxTank = PropellantTank(
    "LOX Tank",
    Vehicle,
    liquid_mass=liquid_mass,
    ullage_mass=ullage_mass,
    dry_mass=8.0,
    mass_flow_out=flow_out,
    length=1.0,
    radius=0.13,
    liquid_density=1100.0,
)

Structure = Layout(
    "Vehicle Layout",
    Vehicle,
    tank=OxTank,
)


# ---------------------------------------------------------------------------
# Outputs and transient solve
# ---------------------------------------------------------------------------

Vehicle.track("Total Mass [kg]", Structure.mass)
Vehicle.track("CG from Forward End [m]", Structure.cg)

Transient(Vehicle).solve(
    dt=dt,
    t_final=t_final,
    filename=filename,
    history="full",
)

print("Final vehicle mass [kg]:", Structure.mass.value)
print("Final CG from forward end [m]:", Structure.cg.value)
