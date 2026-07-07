import numpy as np

from fullflow import *
from thermoprop import *
import fullplot as fplt


"""
Sudden Valve Opening Into a Water Column With Trapped Air
========================================================

This example models pressure oscillations caused by suddenly opening a valve
upstream of a water-filled pipe section with an entrapped air pocket at the
closed downstream end.

The purpose is to show a GFSSP-style node/branch transient model in FullFlow:

    - pipe branches carry flow inertia
    - pipe nodes store liquid mass
    - the terminal air pocket stores mass and energy
    - the water/air interface moves by changing the terminal water volume
    - the trapped air pocket behaves like a compressible spring
    - the sudden valve opening excites pressure and flow oscillations

Physical layout
---------------

    High-pressure water reservoir
    PR = 7 relative to initial line pressure
              |
              | opening valve
              v
        Valve outlet junction
              |
              v
    Pipe Segment 1
              |
              v
    Pipe Node 1
              |
              v
        ...
              |
              v
    Pipe Segment N
              |
              v
    Terminal Water Volume
              |
              | moving water/air interface
              v
    Terminal Air Volume
    closed end / trapped air pocket

Oscillation mechanism
---------------------

    1. The upstream valve opens quickly.
    2. Reservoir pressure accelerates the water column.
    3. The moving water column compresses the trapped air pocket.
    4. The trapped air pressure rises.
    5. The rising air pressure decelerates the water column.
    6. Liquid inertia and trapped-air compliance produce oscillation.

Modeling notes
--------------

This is a lumped finite-volume model, not a method-of-characteristics model.

The model is distributed into several pipe branches and liquid storage nodes,
but it does not explicitly enforce an acoustic Courant condition. The timestep
used here is chosen for practical transient resolution of the observed pressure
oscillation, not for exact wave-speed propagation.

The terminal air volume includes energy storage and boundary work:

    boundary_work_rate = -P * dV_air/dt

When the water volume grows, the air volume shrinks. Since dV_air/dt is
negative during compression, boundary work is positive and energy is added to
the trapped air.
"""


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

ValveOpeningSim = Network("Sudden Valve Opening")


# -----------------------------------------------------------------------------
# Unit conversions
# -----------------------------------------------------------------------------

psi_to_pa = 6894.757293168
ft_to_m = 0.3048
inch_to_m = 0.0254


# -----------------------------------------------------------------------------
# Boundary and initial conditions
# -----------------------------------------------------------------------------

# The pipe initially sits near ambient pressure. The upstream reservoir is set
# to seven times ambient pressure to create a strong sudden-opening transient.
ambient_pressure = 101325.0
source_pressure = 7.0 * ambient_pressure

# Water temperature is held fixed in the liquid property lookups. The water
# Volumes in this example only solve mass storage, not liquid energy storage.
water_temperature = 300.0

# The trapped air volume solves both mass and energy storage, so its temperature
# can change during compression and expansion.
initial_air_temperature = 300.0


# -----------------------------------------------------------------------------
# Pipe geometry
# -----------------------------------------------------------------------------

total_line_length = 20.0 * ft_to_m
pipe_diameter = 0.50 * inch_to_m
pipe_area = (np.pi / 4.0) * pipe_diameter**2
line_volume = pipe_area * total_line_length

# Initial trapped-air fraction.
#
# The pipe starts with a finite liquid column and a finite trapped air pocket.
# This is the physical condition needed for trapped-air oscillation.
initial_air_length_fraction = 0.45

initial_air_length = initial_air_length_fraction * total_line_length
initial_water_length = total_line_length - initial_air_length

initial_air_volume = pipe_area * initial_air_length
initial_water_volume = pipe_area * initial_water_length


# -----------------------------------------------------------------------------
# Pipe discretization
# -----------------------------------------------------------------------------

# Number of inertial pipe branches in the initial water column.
#
# The last branch flows into the terminal water volume, which is adjacent to the
# trapped air pocket. The first water_node_count - 1 water nodes are fixed-volume
# liquid storage nodes.
water_node_count = 5
fixed_node_count = water_node_count - 1

# Each pipe segment uses the initial water length divided uniformly.
dx = initial_water_length / water_node_count
node_volume = pipe_area * dx

# The terminal control volume contains:
#
#     initial terminal water volume = one pipe-node volume
#     initial terminal air volume   = trapped air volume
#
# During the transient, the terminal water volume changes and the terminal air
# volume is derived from the fixed terminal total volume.
terminal_total_volume = node_volume + initial_air_volume
initial_terminal_water_volume = node_volume
initial_terminal_air_volume = terminal_total_volume - initial_terminal_water_volume


# -----------------------------------------------------------------------------
# Valve opening sequence
# -----------------------------------------------------------------------------

# The valve area is commanded by a Sequence during the transient solve.
# It starts closed, then ramps to the full pipe area.
valve_area = State(0.0)

valve_open_area = pipe_area
valve_open_cd = 0.6

valve_start_time = 0.15
valve_open_time = 0.25


def valve_area_sequence(t):
    """
    Sudden but finite valve opening.

    The valve stays closed until valve_start_time, then ramps linearly to the
    full valve area over valve_open_time seconds.
    """

    if t < valve_start_time:
        return 0.0

    ramp = (t - valve_start_time) / valve_open_time

    if ramp >= 1.0:
        return valve_open_area

    return valve_open_area * max(0.0, ramp)


ValveAreaSequence = Sequence(
    "Valve Area Sequence",
    ValveOpeningSim,
    target=valve_area,
    function=valve_area_sequence,
)


# -----------------------------------------------------------------------------
# Dynamic and solved states
# -----------------------------------------------------------------------------

# Algebraic valve mass flow.
valve_mass_flow = State(0.0)

# Junction pressure between the opening valve and the first inertial pipe
# segment.
valve_outlet_pressure = State(ambient_pressure)

# Moving-interface coordinate.
#
# The terminal water volume is the balance variable used to enforce pressure
# compatibility between the terminal liquid and trapped air.
terminal_water_volume = State(initial_terminal_water_volume)

# The terminal air volume is derived from the fixed terminal volume.
terminal_air_volume = terminal_total_volume - terminal_water_volume

# Terminal pressure and air-temperature states.
terminal_water_pressure = State(ambient_pressure)
terminal_air_pressure = State(ambient_pressure)
terminal_air_temperature = State(initial_air_temperature)

# Useful derived outputs.
#
# The fixed water nodes always contain water. The moving terminal water volume
# adds to that fixed liquid volume to give the total liquid fill fraction and
# interface position.
fill_fraction = (
    fixed_node_count * node_volume + terminal_water_volume
) / line_volume

interface_position = (
    fixed_node_count * node_volume + terminal_water_volume
) / pipe_area


# -----------------------------------------------------------------------------
# Fluid property lookups
# -----------------------------------------------------------------------------

# Source water is a fixed-pressure, fixed-temperature boundary fluid.
SourceWater = Lookup(
    "Source Water",
    ValveOpeningSim,
    Fluid,
    "Water",
    pressure=source_pressure,
    temperature=water_temperature,
)

# Internal pipe-node pressure states and water property lookups.
node_pressures = []
node_fluids = []

for i in range(fixed_node_count):
    node_pressure = State(ambient_pressure)
    node_pressures.append(node_pressure)

    node_fluid = Lookup(
        f"Pipe Node {i + 1} Water",
        ValveOpeningSim,
        Fluid,
        "Water",
        pressure=node_pressure,
        temperature=water_temperature,
    )

    node_fluids.append(node_fluid)


# Terminal water pressure is solved independently, then forced to match the
# terminal air pressure by the interface balance.
TerminalWaterFluid = Lookup(
    "Terminal Water",
    ValveOpeningSim,
    Fluid,
    "Water",
    pressure=terminal_water_pressure,
    temperature=water_temperature,
)

# Terminal air has dynamic pressure and temperature because the air volume
# solves mass and energy storage.
TerminalAirFluid = Lookup(
    "Terminal Air",
    ValveOpeningSim,
    Fluid,
    "Air",
    pressure=terminal_air_pressure,
    temperature=terminal_air_temperature,
)


# -----------------------------------------------------------------------------
# Opening valve
# -----------------------------------------------------------------------------

# The opening valve is algebraic. It provides a pressure-flow relation between
# the high-pressure source and the valve outlet junction.
OpeningValve = DischargeCoefficient(
    "Opening Valve",
    ValveOpeningSim,
    upstream_pressure=SourceWater.pressure,
    downstream_pressure=valve_outlet_pressure,
    density=SourceWater.density,
    discharge_coefficient=valve_open_cd,
    cross_sectional_area=valve_area,
    mass_flow=valve_mass_flow,
)


# -----------------------------------------------------------------------------
# Pipe inertia branches
# -----------------------------------------------------------------------------

# Each pipe segment gets a mass-flow State. The DarcyWeisbach component uses
# that State dynamically, so each segment contributes flow inertia.
pipe_flows = []
pipe_segments = []

for i in range(water_node_count):
    pipe_flows.append(State(0.0))


for i in range(water_node_count):

    if i == 0:
        # First pipe segment:
        #
        #     valve outlet junction -> pipe node 1
        upstream_pressure = valve_outlet_pressure
        downstream_pressure = node_pressures[0]
        density = SourceWater.density

    elif i < fixed_node_count:
        # Internal pipe segment:
        #
        #     pipe node i -> pipe node i + 1
        upstream_pressure = node_pressures[i - 1]
        downstream_pressure = node_pressures[i]
        density = node_fluids[i - 1].density

    else:
        # Last pipe segment:
        #
        #     final fixed pipe node -> terminal water volume
        upstream_pressure = node_pressures[-1]
        downstream_pressure = TerminalWaterFluid.pressure
        density = node_fluids[-1].density

    segment = DarcyWeisbach(
        f"Pipe Segment {i + 1}",
        ValveOpeningSim,
        mass_flow=pipe_flows[i],
        upstream_pressure=upstream_pressure,
        downstream_pressure=downstream_pressure,
        length=dx,
        hydraulic_diameter=pipe_diameter,
        cross_sectional_area=pipe_area,
        density=density,
        friction_factor=0.02,
    )

    pipe_segments.append(segment)


# -----------------------------------------------------------------------------
# Valve outlet junction
# -----------------------------------------------------------------------------

# Algebraic zero-volume junction:
#
#     valve_mass_flow = pipe_flows[0]
#
# Because no volume/density is provided, this Volume acts as an algebraic flow
# balance rather than a storage volume.
ValveOutletJunction = Volume(
    "Valve Outlet Junction",
    ValveOpeningSim,
    pressure=valve_outlet_pressure,
    mass_flow_in=OpeningValve.mass_flow,
    mass_flow_out=pipe_flows[0],
)


# -----------------------------------------------------------------------------
# Fixed-volume water pipe nodes
# -----------------------------------------------------------------------------

# Each fixed pipe node stores liquid mass:
#
#     d(mass_i)/dt = mdot_in_i - mdot_out_i
#
# The pressure at each node is solved so that liquid density and stored mass are
# consistent with the transient mass balance.
pipe_nodes = []

for i in range(fixed_node_count):

    node = Volume(
        f"Pipe Node {i + 1}",
        ValveOpeningSim,
        pressure=node_pressures[i],
        volume=node_volume,
        density=node_fluids[i].density,
        mass_flow_in=pipe_flows[i],
        mass_flow_out=pipe_flows[i + 1],
    )

    pipe_nodes.append(node)


# -----------------------------------------------------------------------------
# Terminal water volume
# -----------------------------------------------------------------------------

# This is the water side of the moving water/air interface.
#
# The terminal water volume is allowed to change. The incoming liquid flow from
# the last pipe segment changes the stored liquid mass, and the interface
# pressure balance moves terminal_water_volume until:
#
#     terminal water pressure = terminal air pressure
TerminalWater = Volume(
    "Terminal Water Volume",
    ValveOpeningSim,
    volume=terminal_water_volume,
    pressure=TerminalWaterFluid.pressure,
    density=TerminalWaterFluid.density,
    mass_flow_in=pipe_flows[-1],
    mass_flow_out=0.0,
)


# -----------------------------------------------------------------------------
# Terminal trapped air volume
# -----------------------------------------------------------------------------

# The trapped air has no inlet or outlet. Its mass is fixed.
#
# Its volume changes because:
#
#     terminal_air_volume = terminal_total_volume - terminal_water_volume
#
# The air energy equation includes boundary work. When the water pushes into the
# air pocket, terminal_air_volume decreases and the air pressure/temperature
# rise.
TerminalAir = Volume(
    "Terminal Air Volume",
    ValveOpeningSim,
    volume=terminal_air_volume,
    pressure=TerminalAirFluid.pressure,
    temperature=TerminalAirFluid.temperature,
    density=TerminalAirFluid.density,
    enthalpy=TerminalAirFluid.enthalpy,
    internal_energy=TerminalAirFluid.internal_energy,
    energy_variable="T",
    mass_flow_in=0.0,
    mass_flow_out=0.0,
    work_pressure=TerminalAirFluid.pressure,
)


# -----------------------------------------------------------------------------
# Interface pressure compatibility
# -----------------------------------------------------------------------------

# The terminal water and trapped air are adjacent at a moving interface.
#
# This Balance uses terminal_water_volume as the moving-interface coordinate.
# During transient, FullFlow adjusts terminal_water_volume until:
#
#     P_terminal_water - P_terminal_air = 0
InterfacePressureBalance = Balance(
    "Interface Pressure Balance",
    ValveOpeningSim,
    variable=terminal_water_volume,
    function=TerminalWaterFluid.pressure - TerminalAirFluid.pressure,
)


# -----------------------------------------------------------------------------
# Tracks
# -----------------------------------------------------------------------------

ValveOpeningSim.track("Valve Area [m2]", valve_area)
ValveOpeningSim.track("Valve Cd [-]", valve_open_cd)
ValveOpeningSim.track("Valve Mass Flow [kg/s]", OpeningValve.mass_flow)

for i in range(fixed_node_count):
    ValveOpeningSim.track(
        f"Pipe Node {i + 1} Pressure [psia]",
        node_pressures[i] / psi_to_pa,
    )

ValveOpeningSim.track("Terminal Water Pressure [psia]", TerminalWaterFluid.pressure / psi_to_pa)
ValveOpeningSim.track("Terminal Air Pressure [psia]", TerminalAirFluid.pressure / psi_to_pa)
ValveOpeningSim.track("Terminal Air Temperature [K]", TerminalAirFluid.temperature)

ValveOpeningSim.track("Terminal Water Volume [m3]", terminal_water_volume)
ValveOpeningSim.track("Terminal Air Volume [m3]", terminal_air_volume)
ValveOpeningSim.track("Fill Fraction [-]", fill_fraction)
ValveOpeningSim.track("Interface Position [m]", interface_position)

for i in range(water_node_count):
    ValveOpeningSim.track(
        f"Pipe Segment {i + 1} Mass Flow [kg/s]",
        pipe_flows[i],
    )

ValveOpeningSim.track("Air Boundary Work Rate [W]", TerminalAir.boundary_work_rate)


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------

filename = "sudden_valve_opening"

# This timestep is chosen to resolve the oscillation in this example at a
# practical runtime. This is not intended to enforce an exact MOC-style Courant
# number.
dt = 0.01
t_final = 2.0

'''
# Steady initialization:
#
#     - valve closed
#     - pipe flow states initialized at zero
#     - pipe and trapped air pressures initialized at ambient
#     - trapped-air volume initialized from the user-defined air fraction
#
# The interface balance is ignored during steady solve so the initializer does
# not move the prescribed initial interface location.
SteadyState(ValveOpeningSim).solve(
    verbose=True,
    ignore_balances=["Interface Pressure Balance"],
    filename=filename,
)


# Transient solve:
#
#     - valve opens rapidly
#     - pipe flow states accelerate
#     - fixed water nodes store mass
#     - terminal water volume moves
#     - trapped air compresses and expands
#
# A modest rtol is used because this is a demonstration example and because the
# timestep is intentionally coarser than a detailed wave-propagation study.
Transient(ValveOpeningSim).solve(
    dt=dt,
    t_final=t_final,
    filename=filename,
    verbose=False,
    statistics=True,
    rtol=1.0e-5,
)

'''
# -----------------------------------------------------------------------------
# Plot with FullPlot
# -----------------------------------------------------------------------------

result = fplt.open(filename).at(
    "Sudden_Valve_Opening/transient/runs/base/tracks"
)


# -----------------------------------------------------------------------------
# Valve opening
# -----------------------------------------------------------------------------

valve_area_trace = result.trace(
    y="Valve Area [m2]",
    x="time",
    name="Valve Area",
    role="command",
)

result.plot(
    y=[valve_area_trace],
    xlabel="Time [s]",
    ylabel="Valve Area [m2]",
    title="Sudden Opening: Valve Area",
)


# -----------------------------------------------------------------------------
# Valve mass flow
# -----------------------------------------------------------------------------

valve_mass_flow_trace = result.trace(
    y="Valve Mass Flow [kg/s]",
    x="time",
    name="Valve Mass Flow",
    role="data",
)

result.plot(
    y=[valve_mass_flow_trace],
    xlabel="Time [s]",
    ylabel="Mass Flow [kg/s]",
    title="Sudden Opening: Valve Mass Flow",
)


# -----------------------------------------------------------------------------
# Terminal trapped-air pressure oscillation
# -----------------------------------------------------------------------------

terminal_water_pressure_trace = result.trace(
    y="Terminal Water Pressure [psia]",
    x="time",
    name="Terminal Water Pressure",
    role="data",
)

terminal_air_pressure_trace = result.trace(
    y="Terminal Air Pressure [psia]",
    x="time",
    name="Terminal Air Pressure",
    role="data",
)

result.plot(
    y=[terminal_water_pressure_trace, terminal_air_pressure_trace],
    xlabel="Time [s]",
    ylabel="Pressure [psia]",
    title="Sudden Opening: Trapped-Air End Pressure",
)


# -----------------------------------------------------------------------------
# Interface motion
# -----------------------------------------------------------------------------

terminal_water_volume_trace = result.trace(
    y="Terminal Water Volume [m3]",
    x="time",
    name="Terminal Water Volume",
    role="data",
)

terminal_air_volume_trace = result.trace(
    y="Terminal Air Volume [m3]",
    x="time",
    name="Terminal Air Volume",
    role="data",
)

result.plot(
    y=[terminal_water_volume_trace, terminal_air_volume_trace],
    xlabel="Time [s]",
    ylabel="Volume [m3]",
    title="Sudden Opening: Interface Volume Motion",
)


# -----------------------------------------------------------------------------
# Distributed pipe pressures
# -----------------------------------------------------------------------------

pipe_pressure_traces = []

for i in range(fixed_node_count):
    pipe_pressure_traces.append(
        result.trace(
            y=f"Pipe Node {i + 1} Pressure [psia]",
            x="time",
            name=f"Pipe Node {i + 1}",
            role="data",
        )
    )

pipe_pressure_traces.append(terminal_air_pressure_trace)

result.plot(
    y=pipe_pressure_traces,
    xlabel="Time [s]",
    ylabel="Pressure [psia]",
    title="Sudden Opening: Distributed Pipe Pressures",
)


fplt.show()