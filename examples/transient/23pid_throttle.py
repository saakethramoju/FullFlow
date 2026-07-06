from fullflow import *
import fullplot as fplt
from thermoprop import *
import math


"""
Closed-loop RP-1 / LOX throttle-control example using a Pc PID controller.

Physical layout
---------------

    Fuel Tank / Boundary                         Ox Tank / Boundary
    RP-1, 450 psia                               LOX, 400 psia
          |                                            |
          v                                            v

    +-------------------+                    +-------------------+
    | Fuel Main Line    |                    | Ox Main Line      |
    | Cd from actuator  |                    | Cd from actuator  |
    +-------------------+                    +-------------------+
          |                                            |
          v                                            v

    +-------------------+                    +-------------------+
    | Fuel Manifold     |                    | Ox Manifold       |
    | finite volume     |                    | finite volume     |
    +-------------------+                    +-------------------+
          |                                            |
          v                                            v

    +-------------------+                    +-------------------+
    | Fuel Injector     |                    | Ox Injector       |
    | fixed CdA         |                    | fixed CdA         |
    +-------------------+                    +-------------------+
          |                                            |
          +----------------------+---------------------+
                                 |
                                 v

                         +---------------+
                         | Chamber       |
                         | finite volume |
                         +---------------+
                                 |
                                 v

                         +---------------+
                         | Nozzle        |
                         +---------------+


Control architecture
--------------------

The controller uses a single throttle coordinate, alpha:

    alpha = 0.0  -> low-power operating point
    alpha = 1.0  -> high-power operating point

A precomputed throttle map is stored in the same HDF5 file as the solution:

    group="throttle"

with one input axis:

    alpha

and three outputs:

    chamber_pressure
    fuel_main_cd
    ox_main_cd

The transient controller is:

    Pc setpoint
        |
        v
    feed-forward alpha trim
        |
        v
    PID correction on alpha
        |
        v
    alpha command
        |
        v
    throttle map
        |
        +--------------------+
        |                    |
        v                    v
    fuel Cd command      ox Cd command
        |                    |
        v                    v
    fuel actuator        ox actuator
        |                    |
        v                    v
    fuel Cd position     ox Cd position
        |                    |
        v                    v
    fuel main line       ox main line


Important modeling notes
------------------------

1. The PID does not directly command a valve.
   It commands the throttle coordinate alpha.

2. The throttle map converts alpha into coordinated fuel and oxidizer valve
   discharge coefficients.

3. The actuators add finite valve motion rate limits.

4. The steady-state solve initializes the plant at the initial setpoint.

5. The transient solve applies a Pc setpoint step and lets the PID, map,
   actuators, line inertia, manifolds, chamber, and nozzle evolve together.
"""


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

psia_to_pa = 6894.76
in_to_m = 1 / 39.37
in2_to_m2 = 1 / 1550
in3_to_m3 = in_to_m**3
lbf_to_n = 4.44822


# ---------------------------------------------------------------------------
# File / map generation switches
# ---------------------------------------------------------------------------

# FullPlot writes maps and FullFlow solver results into this HDF5 file.
filename = "23pid_throttle"

# Set this True only when the combustion-products map needs to be regenerated.
# This can take longer than running the transient example.
generate_combustion_map = False

# Set this True only when the throttle calibration map needs to be regenerated.
# The throttle map solves a steady-state calibration problem at each alpha point.
generate_throttle_map = False


# ---------------------------------------------------------------------------
# Optional combustion-products map generation
# ---------------------------------------------------------------------------

if generate_combustion_map:
    map_fuel = Propellant("rp-1", temperature=300)
    map_ox = Propellant("lox", temperature=90)

    def rp1_lox_products(chamber_pressure, mixture_ratio):
        """Return equilibrium gas properties for the chamber map.

        The transient model later uses this map instead of calling the
        equilibrium solver directly at every timestep.
        """
        reactants = Reactants(
            fuels=map_fuel,
            oxidizers=map_ox,
            mixture_ratio=mixture_ratio,
        )

        gas = Equilibrium(
            reactants=reactants,
            pressure=chamber_pressure,
        )

        return {
            "temperature": gas.temperature,
            "gamma": gas.gamma,
            "gas_constant": gas.gas_constant,
            "density": gas.density,
        }

    fplt.generate_map(
        filename,
        group="products",
        axes=[
            fplt.Axis.linear(
                "chamber_pressure",
                start=250 * psia_to_pa,
                stop=420 * psia_to_pa,
                count=30,
                units="Pa",
            ),
            fplt.Axis.linear(
                "mixture_ratio",
                start=1.5,
                stop=3.5,
                count=30,
            ),
        ],
        evaluate=rp1_lox_products,
        overwrite=True,
        raise_errors=True,
    )


# ---------------------------------------------------------------------------
# Optional throttle calibration map generation
# ---------------------------------------------------------------------------

if generate_throttle_map:

    # The throttle map is generated along a nominal constant-MR operating line.
    #
    # For each alpha:
    #
    #     Pc_target = Pc_min + alpha * (Pc_max - Pc_min)
    #
    # FullFlow solves for the two valve discharge coefficients that produce:
    #
    #     chamber_pressure = Pc_target
    #     mixture_ratio    = MR_target
    #
    # The resulting HDF5 group is intentionally simple:
    #
    #     throttle/alpha
    #     throttle/chamber_pressure
    #     throttle/fuel_main_cd
    #     throttle/ox_main_cd

    MR_target = 2.0

    Pc_min = 250.0 * psia_to_pa
    Pc_max = 310.0 * psia_to_pa

    # Residual scaling for the calibration balances.
    # These make the Pc and MR residuals numerically comparable.
    pc_scale = 5.0 * psia_to_pa
    mr_scale = 0.01

    # Main-line geometry used by the calibration model.
    runline_length = 5.0
    runline_area = 0.5e-4

    # Initial valve Cd guesses based on nominal CdA values.
    fuel_throttle_cda = 0.5e-4
    ox_throttle_cda = 1.0e-4

    fuel_main_cd = fuel_throttle_cda / runline_area
    ox_main_cd = ox_throttle_cda / runline_area

    # Warm-start guesses. Each successful alpha point seeds the next point.
    throttle_guess = {
        "fuel_main_cd": fuel_main_cd,
        "ox_main_cd": ox_main_cd,
    }


    def throttle_pc_target(alpha):
        """Map throttle coordinate alpha to the desired Pc calibration target."""
        return Pc_min + float(alpha) * (Pc_max - Pc_min)


    def solve_throttle_alpha(alpha):
        """Solve one steady-state throttle calibration point.

        The unknowns are fuel_main_cd and ox_main_cd.

        The residuals are chamber-pressure error and mixture-ratio error.
        Although each Balance has one variable, the steady-state solver solves
        the full coupled system, so both valves affect both residuals.
        """
        alpha = float(alpha)
        Pc_target = throttle_pc_target(alpha)

        EngineMap = Network("Throttle Map Engine")

        fuel_map = Fluid("RP1", pressure=450 * psia_to_pa, temperature=300)
        ox_map = Fluid("LOX", pressure=400 * psia_to_pa, quality=0.0)

        InjFuelMap = Lookup(
            "Injector Manifold Fuel",
            EngineMap,
            Propellant,
            "rp-1",
            pressure=350 * psia_to_pa,
            temperature=300,
        )

        InjOxMap = Lookup(
            "Injector Manifold Oxidizer",
            EngineMap,
            Propellant,
            "lox",
            pressure=350 * psia_to_pa,
            temperature=90,
        )

        chamber_pressure_map = State(Pc_target)
        mixture_ratio_map = State(MR_target)

        ChamberGasMap = Map.from_hdf5(
            "Chamber Gas Map",
            EngineMap,
            filename,
            group="products",
            inputs={
                "chamber_pressure": chamber_pressure_map,
                "mixture_ratio": mixture_ratio_map,
            },
        )

        # These two states are the calibration unknowns.
        fuel_main_cd_map = State(throttle_guess["fuel_main_cd"])
        ox_main_cd_map = State(throttle_guess["ox_main_cd"])

        fuel_main_mdot_map = State(1.0)

        FuelMainMap = DischargeCoefficient(
            "Fuel Main Line",
            EngineMap,
            upstream_pressure=fuel_map.pressure,
            downstream_pressure=InjFuelMap.pressure,
            density=fuel_map.density,
            discharge_coefficient=fuel_main_cd_map,
            cross_sectional_area=runline_area,
            length=runline_length,
            mass_flow=fuel_main_mdot_map,
        )

        ox_main_mdot_map = State(2.0)

        OxMainMap = DischargeCoefficient(
            "Oxidizer Main Line",
            EngineMap,
            upstream_pressure=ox_map.pressure,
            downstream_pressure=InjOxMap.pressure,
            density=ox_map.density,
            discharge_coefficient=ox_main_cd_map,
            cross_sectional_area=runline_area,
            length=runline_length,
            mass_flow=ox_main_mdot_map,
        )

        FuelManifoldMap = Volume(
            "Injector Fuel Manifold",
            EngineMap,
            volume=0.1287,
            pressure=InjFuelMap.pressure,
            density=InjFuelMap.density,
            mass_flow_in=FuelMainMap.mass_flow,
        )

        OxManifoldMap = Volume(
            "Injector Oxidizer Manifold",
            EngineMap,
            volume=0.1287,
            pressure=InjOxMap.pressure,
            density=InjOxMap.density,
            mass_flow_in=OxMainMap.mass_flow,
        )

        FuelOrfMap = DischargeCoefficient(
            "Injector Fuel Orifices",
            EngineMap,
            upstream_pressure=InjFuelMap.pressure,
            downstream_pressure=chamber_pressure_map,
            density=InjFuelMap.density,
            discharge_coefficient=1.0,
            cross_sectional_area=0.5e-4,
            mass_flow=FuelManifoldMap.mass_flow_out,
        )

        OxOrfMap = DischargeCoefficient(
            "Injector Ox Orifices",
            EngineMap,
            upstream_pressure=InjOxMap.pressure,
            downstream_pressure=chamber_pressure_map,
            density=InjOxMap.density,
            discharge_coefficient=1.0,
            cross_sectional_area=1.0e-4,
            mass_flow=OxManifoldMap.mass_flow_out,
        )

        mixture_ratio_map <<= OxOrfMap.mass_flow / FuelOrfMap.mass_flow

        ChamberMap = Volume(
            "Combustion Chamber",
            EngineMap,
            volume=6.0e-2,
            pressure=chamber_pressure_map,
            density=ChamberGasMap.density,
            mass_flow_in=FuelOrfMap.mass_flow + OxOrfMap.mass_flow,
        )

        NozzleMap = IsentropicNozzle(
            "Nozzle",
            EngineMap,
            upstream_total_pressure=ChamberMap.pressure,
            upstream_total_temperature=ChamberGasMap.temperature,
            ambient_pressure=14.67 * psia_to_pa,
            specific_heat_ratio=ChamberGasMap.gamma,
            gas_constant=ChamberGasMap.gas_constant,
            throat_area=6.05 * in2_to_m2,
            expansion_ratio=4.7,
            mass_flow=ChamberMap.mass_flow_out,
        )

        thrust_map = NozzleMap.mass_flow * NozzleMap.exit_velocity + (NozzleMap.exit_static_pressure - NozzleMap.ambient_pressure) * NozzleMap.expansion_ratio * NozzleMap.throat_area

        # Coupled calibration balances.
        #
        # The variable assignment does not imply fuel controls Pc and ox controls
        # MR independently. It only adds both valve states to the solver variable
        # vector. The steady-state solver solves both residual equations together.
        PcBalance = Balance(
            "Throttle Pc Balance",
            EngineMap,
            variable=fuel_main_cd_map,
            function=(chamber_pressure_map - Pc_target) / pc_scale,
        )

        MRBalance = Balance(
            "Throttle MR Balance",
            EngineMap,
            variable=ox_main_cd_map,
            function=(mixture_ratio_map - MR_target) / mr_scale,
        )

        SteadyState(EngineMap).solve(
            verbose=False,
            statistics=False,
        )

        throttle_guess["fuel_main_cd"] = float(fuel_main_cd_map.value)
        throttle_guess["ox_main_cd"] = float(ox_main_cd_map.value)

        return {
            "chamber_pressure": float(chamber_pressure_map.value),
            "fuel_main_cd": float(fuel_main_cd_map.value),
            "ox_main_cd": float(ox_main_cd_map.value),
        }


    fplt.generate_map(
        filename,
        group="throttle",
        axes=[
            fplt.Axis.linear(
                "alpha",
                start=0.0,
                stop=1.0,
                count=31,
            ),
        ],
        evaluate=solve_throttle_alpha,
        overwrite=True,
        raise_errors=True,
    )















# ---------------------------------------------------------------------------
# Transient engine model
# ---------------------------------------------------------------------------

Engine = Network("Engine")


# ---------------------------------------------------------------------------
# Shared geometry and nominal valve data
# ---------------------------------------------------------------------------

runline_length = 5.0
runline_area = 0.5e-4

fuel_throttle_cda = 0.5e-4
ox_throttle_cda = 1.0e-4

fuel_main_cd = fuel_throttle_cda / runline_area
ox_main_cd = ox_throttle_cda / runline_area


# ---------------------------------------------------------------------------
# Propellant boundary conditions
# ---------------------------------------------------------------------------

fuel = Fluid("RP1", pressure=450 * psia_to_pa, temperature=300)
ox = Fluid("LOX", pressure=400 * psia_to_pa, quality=0.0)


# ---------------------------------------------------------------------------
# Injector manifold thermodynamic states
# ---------------------------------------------------------------------------

InjFuel = Lookup(
    "Injector Manifold Fuel",
    Engine,
    Propellant,
    "rp-1",
    pressure=350 * psia_to_pa,
    temperature=300,
)

InjOx = Lookup(
    "Injector Manifold Oxidizer",
    Engine,
    Propellant,
    "lox",
    pressure=350 * psia_to_pa,
    temperature=90,
)


# ---------------------------------------------------------------------------
# Chamber gas lookup
# ---------------------------------------------------------------------------

chamber_pressure = State(300 * psia_to_pa)
mixture_ratio = State(2.0)

ChamberGas = Map.from_hdf5(
    "Chamber Gas Map",
    Engine,
    filename,
    group="products",
    inputs={
        "chamber_pressure": chamber_pressure,
        "mixture_ratio": mixture_ratio,
    },
)


# ---------------------------------------------------------------------------
# Pc throttle controller
# ---------------------------------------------------------------------------

Pc_min_psia = 250.0
Pc_max_psia = 310.0

pc_setpoint = State(290.0)

def pc_setpoint_step(t):
    """Pc command sequence.

    The engine starts at 290 psia, then steps to 300 psia at t = 0.5 s.
    The PID and feed-forward throttle map then drive the engine toward the new
    operating point.
    """
    if t < 0.5:
        return 290.0

    return 300.0

PcSetpointSequence = Sequence(
    "Pc Setpoint Step",
    Engine,
    target=pc_setpoint,
    function=pc_setpoint_step,
)

# Feed-forward alpha estimate.
#
# Since the throttle map was generated with alpha=0 at Pc_min and alpha=1 at
# Pc_max, the desired Pc can be converted into a nominal alpha trim.
alpha_initial = (float(pc_setpoint.value) - Pc_min_psia) / (Pc_max_psia - Pc_min_psia)

alpha_trim = (pc_setpoint - Pc_min_psia) / (Pc_max_psia - Pc_min_psia)
alpha_cmd = State(alpha_initial)

# The PID uses actual Pc as feedback and desired Pc as setpoint.
#
# Its command is alpha, and its trim is the feed-forward alpha. Therefore:
#
#     alpha_cmd = alpha_trim + PID correction
#
# The PID output is limited to the valid throttle-map range [0, 1].
PcController = PID(
    "Pc Controller",
    Engine,
    feedback=chamber_pressure / psia_to_pa,
    setpoint=pc_setpoint,
    command=alpha_cmd,
    trim=alpha_trim,
    proportional_gain=0.001,
    integral_gain=0.001,
    derivative_gain=0.0,
    minimum=0.0,
    maximum=1.0,
)


# ---------------------------------------------------------------------------
# Throttle map
# ---------------------------------------------------------------------------

# The throttle map converts alpha into the two commanded valve discharge
# coefficients. These are command values, not actual valve positions.
ThrottleMap = Map.from_hdf5(
    "Throttle Map",
    Engine,
    filename,
    group="throttle",
    inputs={
        "alpha": alpha_cmd,
    },
)


# ---------------------------------------------------------------------------
# Valve actuators
# ---------------------------------------------------------------------------

# The actuators convert commanded valve Cd values into actual valve positions.
# These actual positions are what the plant uses.
#
# The fuel and oxidizer valves can have different rate limits because the mapped
# fuel and oxidizer Cd changes do not necessarily have the same magnitude.
fuel_cd_position = State()
ox_cd_position = State()

FuelActuator = Actuator(
    "Fuel Actuator",
    Engine,
    command=ThrottleMap.fuel_main_cd,
    position=fuel_cd_position,
    minimum=0.0,
    maximum=3.0,
    rate_limit=0.5,
)

OxActuator = Actuator(
    "Ox Actuator",
    Engine,
    command=ThrottleMap.ox_main_cd,
    position=ox_cd_position,
    minimum=0.0,
    maximum=3.0,
    rate_limit=1.0,
)


# ---------------------------------------------------------------------------
# Feed lines
# ---------------------------------------------------------------------------

fuel_main_mdot = State(1.0)

FuelMain = DischargeCoefficient(
    "Fuel Main Line",
    Engine,
    upstream_pressure=fuel.pressure,
    downstream_pressure=InjFuel.pressure,
    density=fuel.density,
    discharge_coefficient=fuel_cd_position,
    cross_sectional_area=runline_area,
    length=runline_length,
    mass_flow=fuel_main_mdot,
)


ox_main_mdot = State(2.0)

OxMain = DischargeCoefficient(
    "Oxidizer Main Line",
    Engine,
    upstream_pressure=ox.pressure,
    downstream_pressure=InjOx.pressure,
    density=ox.density,
    discharge_coefficient=ox_cd_position,
    cross_sectional_area=runline_area,
    length=runline_length,
    mass_flow=ox_main_mdot,
)


# ---------------------------------------------------------------------------
# Injector manifolds
# ---------------------------------------------------------------------------

FuelManifold = Volume(
    "Injector Fuel Manifold",
    Engine,
    volume=0.1287,
    pressure=InjFuel.pressure,
    density=InjFuel.density,
    mass_flow_in=FuelMain.mass_flow,
)

OxManifold = Volume(
    "Injector Oxidizer Manifold",
    Engine,
    volume=0.1287,
    pressure=InjOx.pressure,
    density=InjOx.density,
    mass_flow_in=OxMain.mass_flow,
)


# ---------------------------------------------------------------------------
# Injector orifices
# ---------------------------------------------------------------------------

FuelOrf = DischargeCoefficient(
    "Injector Fuel Orifices",
    Engine,
    upstream_pressure=InjFuel.pressure,
    downstream_pressure=chamber_pressure,
    density=InjFuel.density,
    discharge_coefficient=1.0,
    cross_sectional_area=0.5e-4,
    mass_flow=FuelManifold.mass_flow_out,
)

OxOrf = DischargeCoefficient(
    "Injector Ox Orifices",
    Engine,
    upstream_pressure=InjOx.pressure,
    downstream_pressure=chamber_pressure,
    density=InjOx.density,
    discharge_coefficient=1.0,
    cross_sectional_area=1.0e-4,
    mass_flow=OxManifold.mass_flow_out,
)

# The chamber mixture ratio is derived from the injector mass flows.
mixture_ratio <<= OxOrf.mass_flow / FuelOrf.mass_flow


# ---------------------------------------------------------------------------
# Combustion chamber
# ---------------------------------------------------------------------------

Chamber = Volume(
    "Combustion Chamber",
    Engine,
    volume=6.0e-2,
    pressure=chamber_pressure,
    density=ChamberGas.density,
    mass_flow_in=FuelOrf.mass_flow + OxOrf.mass_flow,
)


# ---------------------------------------------------------------------------
# Nozzle
# ---------------------------------------------------------------------------

Nozzle = IsentropicNozzle(
    "Nozzle",
    Engine,
    upstream_total_pressure=Chamber.pressure,
    upstream_total_temperature=ChamberGas.temperature,
    ambient_pressure=14.67 * psia_to_pa,
    specific_heat_ratio=ChamberGas.gamma,
    gas_constant=ChamberGas.gas_constant,
    throat_area=6.05 * in2_to_m2,
    expansion_ratio=4.7,
    mass_flow=Chamber.mass_flow_out,
)


# ---------------------------------------------------------------------------
# Derived thrust estimate
# ---------------------------------------------------------------------------

F = Nozzle.mass_flow * Nozzle.exit_velocity + (Nozzle.exit_static_pressure - Nozzle.ambient_pressure) * Nozzle.expansion_ratio * Nozzle.throat_area


# ---------------------------------------------------------------------------
# Tracked outputs
# ---------------------------------------------------------------------------

Engine.track("Fuel Injector Pressure [psia]", InjFuel.pressure / psia_to_pa)
Engine.track("Ox Injector Pressure [psia]", InjOx.pressure / psia_to_pa)
Engine.track("Chamber Pressure [psia]", Chamber.pressure / psia_to_pa)

Engine.track("Pc Setpoint [psia]", pc_setpoint)

Engine.track("Alpha Trim", alpha_trim)
Engine.track("Alpha Command", alpha_cmd)

Engine.track("Throttle Map Pc [psia]", ThrottleMap.chamber_pressure / psia_to_pa)

Engine.track("Fuel Cd Command", ThrottleMap.fuel_main_cd)
Engine.track("Ox Cd Command", ThrottleMap.ox_main_cd)

Engine.track("Fuel Cd Position", fuel_cd_position)
Engine.track("Ox Cd Position", ox_cd_position)

Engine.track("Mixture Ratio", mixture_ratio)
Engine.track("Fuel Mass Flow [kg/s]", FuelOrf.mass_flow)
Engine.track("Ox Mass Flow [kg/s]", OxOrf.mass_flow)

Engine.track("Thrust [lbf]", F / lbf_to_n)


# ---------------------------------------------------------------------------
# Steady-state initialization
# ---------------------------------------------------------------------------

# The steady-state solve initializes the engine at the initial setpoint before
# the transient controller is allowed to move the valves.
SteadyState(Engine).solve(
    verbose=True,
    statistics=True,
    filename=filename,
)


# ---------------------------------------------------------------------------
# Transient solve
# ---------------------------------------------------------------------------

# The transient solve activates the sequence, PID controller, actuators, line
# inertia, manifold volumes, chamber volume, and nozzle response.
Transient(Engine).solve(
    dt=0.01,
    t_final=5.0,
    verbose=True,
    statistics=True,
    filename=filename,
)


# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------

result = fplt.open(filename).at("Engine/transient/runs/base/tracks")
# result.tree()

mr = result.trace(y="Mixture_Ratio", x="time", name="Mixture Ratio")

alpha_trim = result.trace(y="Alpha Trim", x="time", name="Alpha Trim")
alpha_cmd = result.trace(y="Alpha Command", x="time", name="Alpha Command", role="command")

fuel_cd_cmd = result.trace(y="Fuel Cd Command", x="time", name="Fuel Cd Command", role="command")
ox_cd_cmd = result.trace(y="Ox Cd Command", x="time", name="Ox Cd Command", role="command")

fuel_cd_position = result.trace(y="Fuel Cd Position", x="time", name="Fuel Cd Position")
ox_cd_position = result.trace(y="Ox Cd Position", x="time", name="Ox Cd Position")

pc = result.trace(y="Chamber Pressure [psia]", x="time", name="Chamber Pressure")
pc_sp = result.trace(y="Pc Setpoint [psia]", x="time", name="Pc Setpoint", role="command")

fipt = result.trace(y="Fuel Injector Pressure [psia]", x="time", name="Fuel Inj Pressure")
oipt = result.trace(y="Ox Injector Pressure [psia]", x="time", name="Ox Inj Pressure")


# Plot the throttle command against the pressure response.
result.plot(
    y=[alpha_cmd],
    y2=[pc_sp, pc],
    xlabel="Time [s]",
    ylabel="Alpha",
    y2label="Pressure [psia]",
    title="Pc PID Throttle Control",
)


# Plot commanded and actual valve Cd values. The gap between command and
# position shows actuator rate limiting.
result.plot(
    y=[fuel_cd_cmd, fuel_cd_position, ox_cd_cmd, ox_cd_position],
    y2=mr,
    xlabel="Time [s]",
    ylabel="Valve Discharge Coefficient",
    y2label="Mixture Ratio",
    title="Throttle Valve Commands and Positions",
)


fplt.show()