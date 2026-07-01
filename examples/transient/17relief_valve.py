"""
Simple gas tank filling example with a relief valve.

Physical layout
---------------

    High-pressure gas source
    fixed P, fixed T
            |
            |  Fill orifice
            v
    +-----------------------------+
    |                             |
    |          Gas tank            |
    |    pressure and temperature  |
    |       solved dynamically     |
    |                             |
    +-------------+---------------+
                  |
                  |  Relief valve
                  |  opens when tank pressure is too high
                  v
              Ambient

This example keeps the model intentionally simple:

1. The source pressure and temperature are fixed.
2. The tank is a single gas control volume.
3. The tank pressure and temperature are dynamic states.
4. The fill valve is always open.
5. The relief valve opens when tank pressure exceeds the relief open pressure.
6. The relief valve closes when tank pressure falls below the relief close pressure.
7. No custom components are used.

The relief valve uses a Sequence that reads TankPressure and changes the
discharge coefficient of a CompressibleOrifice.
"""

from fullflow import *


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

psi_to_pa = 6894.76

ambient_pressure = 14.7 * psi_to_pa

source_pressure = 1000.0 * psi_to_pa
source_temperature = 300.0

initial_tank_pressure = 14.7 * psi_to_pa
initial_tank_temperature = 300.0

# Use a low relief pressure so the valve opens during this short example.
relief_open_pressure = 100.0 * psi_to_pa
relief_close_pressure = 90.0 * psi_to_pa

tank_volume = 0.05

gas_constant = 296.8
specific_heat_ratio = 1.4
specific_heat_cv = gas_constant / (specific_heat_ratio - 1.0)
specific_heat_cp = specific_heat_ratio * specific_heat_cv

fill_valve_cd = 0.8
fill_valve_area = 1.0e-6

# Make the relief valve larger than the fill valve so it can actually reduce
# tank pressure after it opens.
relief_valve_area = 1.5e-5


# -----------------------------------------------------------------------------
# Relief valve schedule
# -----------------------------------------------------------------------------

def make_relief_valve_schedule():
    """
    Create a simple hysteresis relief-valve schedule.

    The valve opens when tank pressure rises above relief_open_pressure.
    The valve closes when tank pressure falls below relief_close_pressure.

    The hysteresis prevents rapid open/closed chatter.
    """

    relief_is_open = False

    def relief_valve_schedule(t, tank_pressure):
        nonlocal relief_is_open

        if tank_pressure >= relief_open_pressure:
            relief_is_open = True

        if tank_pressure <= relief_close_pressure:
            relief_is_open = False

        if relief_is_open:
            return 1.0

        return 0.0

    return relief_valve_schedule


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

GasTankNetwork = Network("Gas Tank With Relief Valve")


# Fixed boundaries.
SourcePressure = State(source_pressure)
SourceTemperature = State(source_temperature)
AmbientPressure = State(ambient_pressure)


# Tank dynamic states.
TankPressure = State(initial_tank_pressure)
TankTemperature = State(initial_tank_temperature)


# Ideal-gas properties using State math.
SourceEnthalpy = specific_heat_cp * SourceTemperature

TankDensity = TankPressure / (gas_constant * TankTemperature)
TankInternalEnergy = specific_heat_cv * TankTemperature
TankEnthalpy = specific_heat_cp * TankTemperature


# Mass-flow states.
FillMassFlow = State(0.0)
ReliefMassFlow = State(0.0)


# Relief valve command.
ReliefValveCd = State(0.0)

ReliefValveSequence = Sequence(
    "Relief Valve Cd Schedule",
    GasTankNetwork,
    target=ReliefValveCd,
    function=make_relief_valve_schedule(),
    inputs=[TankPressure],
)


# Fill orifice from source to tank.
FillValve = CompressibleOrifice(
    "Fill Valve",
    GasTankNetwork,
    upstream_total_pressure=SourcePressure,
    upstream_total_temperature=SourceTemperature,
    downstream_pressure=TankPressure,
    discharge_coefficient=fill_valve_cd,
    cross_sectional_area=fill_valve_area,
    gas_constant=gas_constant,
    specific_heat_ratio=specific_heat_ratio,
    upstream_static_enthalpy=SourceEnthalpy,
    upstream_static_temperature=SourceTemperature,
    mass_flow=FillMassFlow,
)


# Relief valve from tank to ambient.
ReliefValve = CompressibleOrifice(
    "Relief Valve",
    GasTankNetwork,
    upstream_total_pressure=TankPressure,
    upstream_total_temperature=TankTemperature,
    downstream_pressure=AmbientPressure,
    discharge_coefficient=ReliefValveCd,
    cross_sectional_area=relief_valve_area,
    gas_constant=gas_constant,
    specific_heat_ratio=specific_heat_ratio,
    upstream_static_enthalpy=TankEnthalpy,
    upstream_static_temperature=TankTemperature,
    mass_flow=ReliefMassFlow,
)


# Tank control volume.
Tank = Volume(
    "Tank",
    GasTankNetwork,
    volume=tank_volume,
    pressure=TankPressure,
    temperature=TankTemperature,
    density=TankDensity,
    internal_energy=TankInternalEnergy,
    enthalpy=TankEnthalpy,
    energy_variable="temperature",
    mass_flow_in=FillValve.mass_flow,
    total_enthalpy_in=FillValve.total_enthalpy,
    mass_flow_out=ReliefValve.mass_flow,
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

GasTankNetwork.track("Tank Pressure [Pa]", TankPressure)
GasTankNetwork.track("Tank Pressure [psia]", TankPressure / psi_to_pa)
GasTankNetwork.track("Tank Temperature [K]", TankTemperature)
GasTankNetwork.track("Tank Density [kg/m3]", TankDensity)
GasTankNetwork.track("Fill Mass Flow [kg/s]", FillValve.mass_flow)
GasTankNetwork.track("Relief Mass Flow [kg/s]", ReliefValve.mass_flow)
GasTankNetwork.track("Relief Valve Cd [-]", ReliefValveCd)


# -----------------------------------------------------------------------------
# Transient solve
# -----------------------------------------------------------------------------

Transient(GasTankNetwork).solve(
    dt=0.01,
    t_final=25.0,
    save_dt=0.01,
    filename="17relief_valve",
    verbose=True,
)