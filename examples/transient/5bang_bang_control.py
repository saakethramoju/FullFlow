"""
Bang-bang pressurized methane tank transient example.

Physical layout
---------------

    High-pressure GN2 COPV
    5500 psia, 300 K
            |
            |  Bang-bang pressurization valve
            |  CdA opens/closes to regulate tank pressure
            v
    +---------------------------------------+
    |               Ullage GN2              |
    |        pressure controlled near       |
    |              450 psia                 |
    |                                       |----> Relief valve to ambient
    |  P_ullage = P_liquid_surface          |
    |  V_ullage = V_tank - V_liquid         |
    |                                       |
    |---------------------------------------|  <-- liquid/gas interface
    |              Liquid CH4               |
    |                                       |
    |  liquid height = V_liquid / A_tank    |
    |  bottom pressure = surface pressure   |
    |                  + rho*g*h            |
    |                                       |
    +-------------------+-------------------+
                        |
                        |  tank outlet / main line
                        v
              Liquid manifold node
                        |
                        |  main valve ramps open after
                        |  tank reaches set pressure
                        v
                  Ambient discharge


Model notes
-----------

This example uses a moving-interface tank model.

The total tank volume is enforced directly by defining

    ullage_volume = tank_volume - liquid_volume

so no separate total-volume balance is needed.

The single Balance in this model enforces liquid/gas pressure compatibility
at the interface:

    liquid_surface_pressure = ullage_pressure

During the initial SteadyState solve, that balance is ignored so the steady
initializer does not move the initial liquid inventory. During the transient
solve, the balance is active, so liquid volume is free to evolve from mass
conservation.
"""

import numpy as np

from fullflow import *
from thermoprop import *


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

BangBangSim = Network("Bang Bang Simulation")


# ---------------------------------------------------------------------------
# Constants and initial conditions
# ---------------------------------------------------------------------------

psi_to_pa = 6894.76

initial_copv_pressure = 5500 * psi_to_pa
initial_copv_temperature = 300

tank_set_pressure = 450 * psi_to_pa

# Relief valve command settings.
# The relief valve is independent of the bang-bang pressurization valve.
# It opens only if ullage pressure rises above the relief band.
relief_valve_open_pressure = 475 * psi_to_pa
relief_valve_close_pressure = 465 * psi_to_pa
relief_valve_area = 0.005 / 1550
relief_valve_open_cd = 0.6

tank_volume = 125 / 1000
initial_liquid_volume = 110 / 1000

tank_cross_sectional_area = (np.pi / 4) * (12 / 39.37)**2

g = 9.80665


# ---------------------------------------------------------------------------
# Iteration / dynamic states
# ---------------------------------------------------------------------------

# Bang-bang valve coefficient. This is changed by the bang-bang Sequence.
bang_bang_cda = State(0.0)

# Relief valve coefficient. This is changed by the relief valve Sequence.
relief_valve_cd = State(0.0)

# Relief valve mass flow from the ullage to ambient.
relief_valve_mass_flow = State(0.0)

# Ullage pressure is the pressure of the gaseous nitrogen above the liquid.
ullage_pressure = State(101325)

# Liquid surface pressure is the pressure of the methane at the gas/liquid
# interface. The Balance below forces this to match the ullage pressure during
# the transient solve.
liquid_surface_pressure = State(101325)

# Liquid volume is the moving-interface coordinate for the tank.
# The ullage volume is derived from the fixed total tank volume.
liquid_volume = State(initial_liquid_volume)
ullage_volume = tank_volume - liquid_volume


# ---------------------------------------------------------------------------
# Fluid property lookups
# ---------------------------------------------------------------------------

# Fluid is used instead of IdealGas for the pressurant because PYroMat and CEA
# gaseous species usually have minimum temperatures around 200 K, while
# isentropic nitrogen blowdown can cool below that.
Pressurant = Lookup(
    "Pressurant Gas",
    BangBangSim,
    Fluid,
    "gn2",
    pressure=initial_copv_pressure,
    temperature=initial_copv_temperature,
)

UllageGas = Lookup(
    "Ullage Gas",
    BangBangSim,
    Fluid,
    "gn2",
    temperature=300,
    pressure=ullage_pressure,
)

# Methane properties at the liquid free surface.
# This pressure is forced to match the ullage pressure during transient.
Fuel = Lookup(
    "Fuel",
    BangBangSim,
    Fluid,
    "ch4",
    pressure=liquid_surface_pressure,
    temperature=100,
)

# Liquid manifold properties downstream of the tank outlet line.
NodeFluid = Lookup(
    "Node Fluid",
    BangBangSim,
    Fluid,
    Fuel.fluid,
    pressure=101325,
    temperature=100,
)


# ---------------------------------------------------------------------------
# High-pressure COPV
# ---------------------------------------------------------------------------

COPV = Volume(
    "COPV",
    BangBangSim,
    volume=20 / 1000,
    pressure=Pressurant.pressure,
    temperature=Pressurant.temperature,
    density=Pressurant.density,
    internal_energy=Pressurant.internal_energy,
    enthalpy=Pressurant.enthalpy,
    energy_variable="T",
)


# ---------------------------------------------------------------------------
# Bang-bang tank pressure controller
# ---------------------------------------------------------------------------

def bang_bang_condition(t, pressure):
    """
    Simple bang-bang controller for the pressurization valve.

    If the tank pressure is above the upper deadband, the valve closes.
    If the tank pressure is below the lower deadband, the valve opens.
    Inside the deadband, the valve keeps its previous value.
    """

    if pressure > tank_set_pressure + 5 * psi_to_pa:
        return 0.0

    if pressure < tank_set_pressure - 5 * psi_to_pa:
        return 1.0

    return bang_bang_cda.value


BangBangSequence = Sequence(
    "Bang Bang Cd",
    BangBangSim,
    target=bang_bang_cda,
    function=bang_bang_condition,
    inputs=[ullage_pressure],
)


# ---------------------------------------------------------------------------
# Pressurization valve
# ---------------------------------------------------------------------------

BangBangValve = CompressibleOrifice(
    "Bang Bang Valve",
    BangBangSim,
    upstream_total_pressure=COPV.pressure,
    upstream_total_temperature=Pressurant.temperature,
    downstream_pressure=UllageGas.pressure,
    discharge_coefficient=bang_bang_cda,
    cross_sectional_area=0.01 / 1550,
    gas_constant=Pressurant.gas_constant,
    specific_heat_ratio=Pressurant.gamma,
    upstream_static_enthalpy=Pressurant.enthalpy,
    upstream_static_temperature=Pressurant.temperature,
    mass_flow=COPV.mass_flow_out,
)


# ---------------------------------------------------------------------------
# Relief valve schedule
# ---------------------------------------------------------------------------

def relief_valve_condition(t, pressure):
    """
    Simple relief valve schedule.

    If the ullage pressure rises above the opening pressure, the relief valve
    opens. If the ullage pressure falls below the closing pressure, it closes.
    Inside the band, it keeps its previous value.
    """

    if pressure > relief_valve_open_pressure:
        return relief_valve_open_cd

    if pressure < relief_valve_close_pressure:
        return 0.0

    return relief_valve_cd.value


ReliefValveSequence = Sequence(
    "Relief Valve Cd",
    BangBangSim,
    target=relief_valve_cd,
    function=relief_valve_condition,
    inputs=[ullage_pressure],
)


# ---------------------------------------------------------------------------
# Relief valve from ullage to ambient
# ---------------------------------------------------------------------------

ReliefValve = CompressibleOrifice(
    "Relief Valve",
    BangBangSim,
    upstream_total_pressure=UllageGas.pressure,
    upstream_total_temperature=UllageGas.temperature,
    downstream_pressure=101325,
    discharge_coefficient=relief_valve_cd,
    cross_sectional_area=relief_valve_area,
    gas_constant=UllageGas.gas_constant,
    specific_heat_ratio=UllageGas.gamma,
    upstream_static_enthalpy=UllageGas.enthalpy,
    upstream_static_temperature=UllageGas.temperature,
    mass_flow=relief_valve_mass_flow,
)


# ---------------------------------------------------------------------------
# Ullage gas control volume
# ---------------------------------------------------------------------------

# The ullage volume is derived from the fixed tank volume and the moving liquid
# volume. As the liquid drains, ullage_volume increases.
Ullage = Volume(
    "Ullage",
    BangBangSim,
    volume=ullage_volume,
    pressure=UllageGas.pressure,
    temperature=UllageGas.temperature,
    density=UllageGas.density,
    enthalpy=UllageGas.enthalpy,
    internal_energy=UllageGas.internal_energy,
    energy_variable="T",
    mass_flow_in=BangBangValve.mass_flow,
    mass_flow_out=ReliefValve.mass_flow,
    total_enthalpy_in=BangBangValve.total_enthalpy,
)


# ---------------------------------------------------------------------------
# Liquid methane tank inventory
# ---------------------------------------------------------------------------

# The liquid volume is allowed to change during transient. Its pressure is the
# liquid surface pressure, not the bottom pressure. The bottom pressure is
# calculated below by adding hydrostatic head.
TankLiquid = Volume(
    "Tank Liquid",
    BangBangSim,
    volume=liquid_volume,
    pressure=Fuel.pressure,
    density=Fuel.density,
    mass_flow_out=0,
)


# This balance enforces pressure compatibility at the liquid/gas interface:
#
#     P_liquid_surface = P_ullage
#
# It is ignored during the initial SteadyState solve so that the steady
# initializer does not move the initial liquid inventory. It is active during
# the transient solve, which lets liquid_volume evolve naturally.
LiquidSurfacePressureBalance = Balance(
    "Liquid Surface Pressure Balance",
    BangBangSim,
    variable=liquid_volume,
    function=Fuel.pressure - UllageGas.pressure,
)


# ---------------------------------------------------------------------------
# Hydrostatic head from liquid surface to tank outlet
# ---------------------------------------------------------------------------

# For a cylindrical tank, liquid height is liquid volume divided by tank area.
liquid_height = TankLiquid.volume / tank_cross_sectional_area

# Hydrostatic pressure contribution at the tank outlet.
head_pressure = TankLiquid.density * g * liquid_height


# Methane properties at the bottom of the tank / tank outlet.
FuelwithHeadPressure = Lookup(
    "Fuel with Head Pressure",
    BangBangSim,
    Fluid,
    Fuel.fluid,
    pressure=Fuel.pressure + head_pressure,
    temperature=100,
)


# ---------------------------------------------------------------------------
# Main liquid feed line from tank outlet to manifold
# ---------------------------------------------------------------------------

MainLine = DarcyWeisbach(
    "Main Line",
    BangBangSim,
    mass_flow=TankLiquid.mass_flow_out,
    upstream_pressure=FuelwithHeadPressure.pressure,
    downstream_pressure=NodeFluid.pressure,
    length=3,
    hydraulic_diameter=0.5 / 39.37,
    cross_sectional_area=(np.pi / 4) * (0.5 / 39.37)**2,
    density=FuelwithHeadPressure.density,
    friction_factor=0.02,
    height_change=-3,
)


# ---------------------------------------------------------------------------
# Liquid manifold
# ---------------------------------------------------------------------------

Node = Volume(
    "Liquid Manifold",
    BangBangSim,
    volume=(np.pi / 4) * (0.075 / 39.37)**2,
    pressure=NodeFluid.pressure,
    density=NodeFluid.density,
    mass_flow_in=MainLine.mass_flow,
)


# ---------------------------------------------------------------------------
# Main valve opening schedule
# ---------------------------------------------------------------------------

main_valve_cd = State(0.0)

main_valve_ramp_time = 0.5


def make_main_valve_sequence():
    """
    Create a stateful valve-opening schedule.

    The main valve remains closed until tank pressure reaches the set pressure.
    Once triggered, it ramps from Cd = 0 to Cd = 0.6 over
    main_valve_ramp_time seconds.
    """

    opened = False
    open_time = None

    def main_valve_sequence(t, tank_pressure):
        nonlocal opened, open_time

        if not opened and tank_pressure >= tank_set_pressure:
            opened = True
            open_time = t

        if not opened:
            return 0.0

        ramp_fraction = (t - open_time) / main_valve_ramp_time

        if ramp_fraction >= 1.0:
            return 0.6

        return 0.6 * max(0.0, ramp_fraction)

    return main_valve_sequence


MainValveSequence = Sequence(
    "Main Valve Cd",
    BangBangSim,
    target=main_valve_cd,
    function=make_main_valve_sequence(),
    inputs=[ullage_pressure],
)


# ---------------------------------------------------------------------------
# Main valve discharge to ambient
# ---------------------------------------------------------------------------

MainValve = DischargeCoefficient(
    "Main Valve",
    BangBangSim,
    upstream_pressure=Node.pressure,
    downstream_pressure=101325,
    density=Node.density,
    discharge_coefficient=main_valve_cd,
    cross_sectional_area=MainLine.cross_sectional_area,
    mass_flow=Node.mass_flow_out,
)


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

BangBangSim.track("Ullage Pressure [Pa]", UllageGas.pressure)
BangBangSim.track("Bang Bang Valve Cd [-]", bang_bang_cda)
BangBangSim.track("Relief Valve Cd [-]", relief_valve_cd)
BangBangSim.track("Relief Valve Mass Flow [kg/s]", ReliefValve.mass_flow)
BangBangSim.track("COPV Pressure [Pa]", COPV.pressure)
BangBangSim.track("Main Valve Cd [-]", main_valve_cd)
BangBangSim.track("Main Valve Mass Flow [kg/s]", MainValve.mass_flow)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

filename = "BangBang"

# The interface pressure balance is ignored only during steady initialization.
# This keeps liquid_volume at its user-provided initial fill level instead of
# letting the steady solver move the tank inventory.
SteadyState(BangBangSim).solve(
    verbose=True,
    ignore_balances=["Liquid Surface Pressure Balance"],
    filename=filename,
)

# During the transient solve, all balances are active. The liquid surface
# pressure is forced to match the ullage pressure, and liquid_volume evolves
# from the transient mass conservation equations.
Transient(BangBangSim).solve(
    dt=0.01,
    t_final=22.5,
    filename=filename,
)