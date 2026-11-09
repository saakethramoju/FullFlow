"""
Pressure-fed rocket Monte Carlo with reusable networks and parallel processes.

Physical layout
---------------

    Helium Bottle --- pressurant flow ---> Propellant Tank
         ^                                    |       ^
         |                                    |       |
         |                              fuel flow    |
         |                                    v       |
         |                               Engine      |
         |                                    |       |
         |                                  thrust    |
         |                                    v       |
         +------------------------------- Vehicle ---+
                                           acceleration

The bottle and tank exchange gas, the tank feeds the engine, and the vehicle
acceleration changes tank outlet head. Each worker restores its own Network
between flights. This is a lumped, vertical-flight model with ideal gases,
a simplified injector/nozzle, and no ground-contact or attitude dynamics.
"""

import math
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from fullflow import *


# -----------------------------------------------------------------------------
# Run options
# -----------------------------------------------------------------------------

samples = 96
workers = 4
seed = 42
dt = 0.15
t_final = 55.0
min_apogee = 4500.0
max_dynamic_pressure = 25000.0


# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------

gravity = 9.80665
sea_level_pressure = 101325.0
sea_level_density = 1.225
scale_height = 8500.0
helium_R = 2077.0
helium_temperature = 300.0
propellant_density = 950.0
tank_volume = 0.020
tank_area = 0.030
bottle_volume = 0.008
chamber_temperature = 2400.0
chamber_R = 350.0
chamber_volume = 0.015
thrust_coefficient = 1.45


# -----------------------------------------------------------------------------
# Custom components
# -----------------------------------------------------------------------------

class PressurantBottle(Component):
    def __init__(
        self,
        name,
        network,
        gas_mass,
        bottle_volume,
        regulator_pressure,
        regulator_coefficient,
    ):
        self.pressure = State(0.0)
        self.mass_flow = State(0.0)
        self.mass_dot = 0.0
        self.setup()

    def evaluate_states(self):
        self.pressure.value = (
            self.gas_mass.value * helium_R * helium_temperature / self.bottle_volume.value
        )
        pressure_drop = max(self.pressure.value - self.tank.pressure.value, 0.0)
        error = max(self.regulator_pressure.value - self.tank.pressure.value, 0.0)
        opening = min(error / 200000.0, 1.0)
        flow = self.regulator_coefficient.value * opening * math.sqrt(pressure_drop)
        self.mass_flow.value = min(flow, max(self.gas_mass.value, 0.0) * 10.0)
        self.mass_dot = -self.mass_flow.value

    @property
    def dynamics(self):
        return [(self.gas_mass, self.mass_dot)]


class PropellantTank(Component):
    def __init__(
        self,
        name,
        network,
        propellant_mass,
        ullage_mass,
        tank_volume,
        tank_area,
    ):
        self.pressure = State(0.0)
        self.outlet_pressure = State(0.0)
        self.mass_dot = 0.0
        self.ullage_dot = 0.0
        self.setup()

    def evaluate_states(self):
        ullage_volume = (
            self.tank_volume.value
            - max(self.propellant_mass.value, 0.0) / propellant_density
        )
        self.pressure.value = (
            self.ullage_mass.value * helium_R * helium_temperature
            / max(ullage_volume, 1e-4)
        )
        effective_g = max(gravity + self.vehicle.acceleration.value, 0.0)
        self.outlet_pressure.value = (
            self.pressure.value
            + self.propellant_mass.value * effective_g / self.tank_area.value
        )
        self.mass_dot = -self.engine.inlet_flow.value
        self.ullage_dot = self.bottle.mass_flow.value

    @property
    def dynamics(self):
        return [
            (self.propellant_mass, self.mass_dot),
            (self.ullage_mass, self.ullage_dot),
        ]


class RocketEngine(Component):
    def __init__(
        self,
        name,
        network,
        tank,
        chamber_pressure,
        injector_cda,
        throat_area,
        cstar,
    ):
        self.inlet_flow = State(0.0)
        self.nozzle_flow = State(0.0)
        self.thrust = State(0.0)
        self.pressure_dot = 0.0
        self.setup()

    def evaluate_states(self):
        ambient = self.vehicle.ambient_pressure.value
        pressure_drop = max(self.tank.outlet_pressure.value - self.chamber_pressure.value, 0.0)
        flow = self.injector_cda.value * math.sqrt(2.0 * propellant_density * pressure_drop)
        self.inlet_flow.value = min(flow, max(self.tank.propellant_mass.value, 0.0) * 8.0)
        self.nozzle_flow.value = (
            max(self.chamber_pressure.value - ambient, 0.0)
            * self.throat_area.value / self.cstar.value
        )
        self.pressure_dot = (
            self.inlet_flow.value - self.nozzle_flow.value
        ) * chamber_R * chamber_temperature / chamber_volume
        self.thrust.value = thrust_coefficient * self.nozzle_flow.value * self.cstar.value

    @property
    def dynamics(self):
        return [(self.chamber_pressure, self.pressure_dot)]


class Vehicle(Component):
    def __init__(
        self,
        name,
        network,
        tank,
        bottle,
        engine,
        altitude,
        velocity,
        dry_mass,
        drag_cda,
        wind,
    ):
        self.ambient_pressure = State(sea_level_pressure)
        self.dynamic_pressure = State(0.0)
        self.acceleration = State(0.0)
        self.altitude_dot = 0.0
        self.velocity_dot = 0.0
        self.setup()

    def evaluate_states(self):
        density = sea_level_density * math.exp(-max(self.altitude.value, 0.0) / scale_height)
        self.ambient_pressure.value = sea_level_pressure * math.exp(
            -max(self.altitude.value, 0.0) / scale_height
        )
        relative_velocity = self.velocity.value - self.wind.value
        self.dynamic_pressure.value = 0.5 * density * relative_velocity ** 2
        drag = 0.5 * density * self.drag_cda.value * relative_velocity * abs(relative_velocity)
        total_mass = (
            self.dry_mass.value
            + max(self.tank.propellant_mass.value, 0.0)
            + max(self.bottle.gas_mass.value, 0.0)
        )
        self.acceleration.value = (self.engine.thrust.value - drag) / total_mass - gravity
        self.altitude_dot = self.velocity.value
        self.velocity_dot = self.acceleration.value

    @property
    def dynamics(self):
        return [
            (self.altitude, self.altitude_dot),
            (self.velocity, self.velocity_dot),
        ]


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

Flight = Network("Pressure-Fed Rocket")

Bottle = PressurantBottle(
    "Helium Bottle",
    Flight,
    gas_mass=State(0.14),
    bottle_volume=bottle_volume,
    regulator_pressure=3.0e6,
    regulator_coefficient=1.0e-5,
)

Tank = PropellantTank(
    "Propellant Tank",
    Flight,
    propellant_mass=State(12.0),
    ullage_mass=State(0.039),
    tank_volume=tank_volume,
    tank_area=tank_area,
)

Engine = RocketEngine(
    "Engine",
    Flight,
    tank=Tank,
    chamber_pressure=State(sea_level_pressure),
    injector_cda=1.0e-5,
    throat_area=4.5e-4,
    cstar=1600.0,
)

Rocket = Vehicle(
    "Vehicle",
    Flight,
    tank=Tank,
    bottle=Bottle,
    engine=Engine,
    altitude=State(0.0),
    velocity=State(0.0),
    dry_mass=32.0,
    drag_cda=0.035,
    wind=0.0,
)

Bottle.tank = Tank
Tank.bottle = Bottle
Tank.engine = Engine
Tank.vehicle = Rocket
Engine.vehicle = Rocket


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

Flight.track("Altitude [m]", Rocket.altitude)
Flight.track("Velocity [m/s]", Rocket.velocity)
Flight.track("Dynamic Pressure [Pa]", Rocket.dynamic_pressure)
Flight.track("Chamber Pressure [Pa]", Engine.chamber_pressure)
Flight.track("Tank Pressure [Pa]", Tank.pressure)
Flight.track("Propellant Mass [kg]", Tank.propellant_mass)
Flight.track("Thrust [N]", Engine.thrust)


# -----------------------------------------------------------------------------
# Reusable solver
# -----------------------------------------------------------------------------

initial = Flight.snapshot()
solver = Transient(Flight)


def run_case(case):
    Flight.restore(initial)

    Tank.propellant_mass.value = case["propellant_mass"]
    ullage_volume = tank_volume - Tank.propellant_mass.value / propellant_density
    Tank.ullage_mass.value = (
        case["tank_pressure"] * ullage_volume / (helium_R * helium_temperature)
    )
    Bottle.gas_mass.value = (
        case["bottle_pressure"] * bottle_volume / (helium_R * helium_temperature)
    )
    Bottle.regulator_pressure.value = case["tank_pressure"]
    Engine.injector_cda.value = case["injector_cda"]
    Engine.throat_area.value = case["throat_area"]
    Engine.cstar.value = case["cstar"]
    Rocket.dry_mass.value = case["dry_mass"]
    Rocket.drag_cda.value = case["drag_cda"]
    Rocket.wind.value = case["wind"]

    try:
        result = solver.solve(
            dt=dt,
            t_final=t_final,
            history="summary",
            filename=None,
        )
        tracked = result["tracked"]
        return {
            "case": case["case"],
            "apogee": tracked["Altitude [m]"]["max"],
            "time_to_apogee": tracked["Altitude [m]"]["time_max"],
            "max_q": tracked["Dynamic Pressure [Pa]"]["max"],
            "max_pc": tracked["Chamber Pressure [Pa]"]["max"],
            "remaining_propellant": tracked["Propellant Mass [kg]"]["final"],
            "success": result["solver"]["success"],
        }
    except Exception as exc:
        return {"case": case["case"], "success": False, "error": str(exc)}


# -----------------------------------------------------------------------------
# Parallel simulations
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(seed)
    cases = []

    for i in range(samples):
        cases.append(
            {
                "case": i + 1,
                "tank_pressure": 3.0e6 * rng.lognormal(0.0, 0.08),
                "bottle_pressure": 11.0e6 * rng.lognormal(0.0, 0.08),
                "propellant_mass": rng.normal(12.0, 0.4),
                "injector_cda": 1.0e-5 * rng.lognormal(0.0, 0.07),
                "throat_area": 4.5e-4 * rng.lognormal(0.0, 0.025),
                "cstar": 1600.0 * rng.lognormal(0.0, 0.04),
                "dry_mass": rng.normal(32.0, 0.8),
                "drag_cda": 0.035 * rng.lognormal(0.0, 0.16),
                "wind": rng.normal(0.0, 3.0),
            }
        )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_case, cases, chunksize=4))

    # -------------------------------------------------------------------------
    # Results and statistics
    # -------------------------------------------------------------------------

    valid = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    if failed:
        print(f"Failed simulations: {len(failed)}")
        for row in failed[:5]:
            print(f"  Case {row['case']}: {row.get('error', 'solver unsuccessful')}")
    if not valid:
        raise RuntimeError("No simulations completed successfully")

    apogees = np.array([r["apogee"] for r in valid])
    pressures = np.array([r["max_pc"] for r in valid])
    max_q = np.array([r["max_q"] for r in valid])
    percentiles = np.percentile(apogees, [5, 50, 95])
    requirement = (apogees >= min_apogee) & (max_q <= max_dynamic_pressure)

    print(f"\nCompleted: {len(valid)}/{samples}")
    print(f"Apogee, mean:    {apogees.mean():9.1f} m")
    print(f"Apogee, P05:     {percentiles[0]:9.1f} m")
    print(f"Apogee, median:  {percentiles[1]:9.1f} m")
    print(f"Apogee, P95:     {percentiles[2]:9.1f} m")
    print(f"Maximum q, P95:  {np.percentile(max_q, 95):9.0f} Pa")
    print(f"Peak Pc, P95:    {np.percentile(pressures, 95) / 1e5:9.2f} bar")
    passed = requirement.mean()
    z = 1.96
    n = len(valid)
    center = (passed + z * z / (2 * n)) / (1 + z * z / n)
    margin = (
        z * math.sqrt(passed * (1 - passed) / n + z * z / (4 * n * n))
        / (1 + z * z / n)
    )
    print(f"Meets both limits: {passed * 100:7.1f}% of completed flights")
    print(f"Pass-rate 95% CI: {100 * (center - margin):5.1f}% to {100 * (center + margin):5.1f}%")

    unfinished = sum(r["time_to_apogee"] >= t_final - dt for r in valid)
    if unfinished:
        print(f"Warning: {unfinished} flights may not have reached apogee; increase t_final")

    print("\nCorrelation with apogee:")
    successful_cases = [cases[r["case"] - 1] for r in valid]
    for key in (
        "tank_pressure",
        "bottle_pressure",
        "propellant_mass",
        "injector_cda",
        "throat_area",
        "cstar",
        "dry_mass",
        "drag_cda",
        "wind",
    ):
        values = np.array([case[key] for case in successful_cases])
        correlation = np.corrcoef(values, apogees)[0, 1]
        print(f"  {key:19s} {correlation:+.3f}")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(apogees, bins=16, edgecolor="black")
    axes[0].axvline(min_apogee, linestyle="--", label="Apogee requirement")
    axes[0].set(xlabel="Apogee [m]", ylabel="Flights", title="Apogee distribution")
    axes[0].legend()
    axes[1].scatter(apogees, max_q, s=15, alpha=0.7)
    axes[1].axhline(max_dynamic_pressure, linestyle="--", label="Maximum q limit")
    axes[1].set(xlabel="Apogee [m]", ylabel="Maximum q [Pa]", title="Flight performance")
    axes[1].legend()

    fig.tight_layout()
    plt.show()
