import numpy as np

from fullflow import *

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

PSIA_TO_PA = 6894.757293168
PA_TO_PSIA = 1.0 / PSIA_TO_PA

IN_TO_M = 0.0254
IN2_TO_M2 = IN_TO_M**2
IN3_TO_M3 = IN_TO_M**3

FT_TO_M = 0.3048
LBM_TO_KG = 0.45359237
LBM_FT3_TO_KG_M3 = LBM_TO_KG / FT_TO_M**3


def main() -> None:
    """Run a pump-fed rocket engine network example."""

    # -----------------------------------------------------------------------
    # Network Definition
    # -----------------------------------------------------------------------

    PumpNetwork = Network("Pumped System")

    fuel_shaft_speed = State(25000.0)
    ox_shaft_speed = State(25000.0)

    chamber_pressure = State(285.0 * PSIA_TO_PA)
    atmospheric_pressure = 14.67 * PSIA_TO_PA

    # -----------------------------------------------------------------------
    # Fluids
    # -----------------------------------------------------------------------

    pressurant = "gn2"
    fuel = "rp-1"
    oxidizer = "lox"

    PressurantSupply = FluidLookup(
        "COPV Gas",
        PumpNetwork,
        pressurant,
        pressure=6000.0 * PSIA_TO_PA,
        temperature=300.0,
    )

    FuelPressVolumeFluid = FluidLookup(
        "Fuel Press Volume Gas",
        PumpNetwork,
        pressurant,
        pressure=5000.0 * PSIA_TO_PA,
        temperature=300.0,
        flash_values=("pressure", "enthalpy"),
    )

    OxPressVolumeFluid = FluidLookup(
        "Oxygen Press Volume Gas",
        PumpNetwork,
        pressurant,
        pressure=5000.0 * PSIA_TO_PA,
        temperature=300.0,
        flash_values=("pressure", "enthalpy"),
    )

    FuelUllageGas = IdealGasLookup(
        "Fuel Tank Ullage Gas",
        PumpNetwork,
        pressurant,
        pressure=80.0 * PSIA_TO_PA,
        temperature=300.0,
    )

    OxUllageGas = IdealGasLookup(
        "Ox Tank Ullage Gas",
        PumpNetwork,
        pressurant,
        pressure=80.0 * PSIA_TO_PA,
        temperature=300.0,
    )

    FuelTankFluid = FluidLookup(
        "Tank Fuel Liquid",
        PumpNetwork,
        fuel,
        pressure=FuelUllageGas.pressure,
        temperature=300.0,
    )

    OxTankFluid = FluidLookup(
        "Tank Oxidizer Liquid",
        PumpNetwork,
        oxidizer,
        pressure=OxUllageGas.pressure,
        temperature=90.0,
    )

    FuelPumpInletFluid = FluidLookup(
        "Fuel Pump Inlet Fluid",
        PumpNetwork,
        fuel,
        pressure=State(75.0 * PSIA_TO_PA),
        temperature=293.15,
        flash_values=("pressure", "enthalpy"),
    )

    OxPumpInletFluid = FluidLookup(
        "Ox Pump Inlet Fluid",
        PumpNetwork,
        oxidizer,
        pressure=State(75.0 * PSIA_TO_PA),
        temperature=90.0,
        flash_values=("pressure", "enthalpy"),
    )

    FuelPumpDischargeFluid = FluidLookup(
        "Fuel Pump Outlet Fluid",
        PumpNetwork,
        fuel,
        pressure=State(430.0 * PSIA_TO_PA),
        temperature=293.15,
        flash_values=("pressure", "enthalpy"),
    )

    OxPumpDischargeFluid = FluidLookup(
        "Ox Pump Outlet Fluid",
        PumpNetwork,
        oxidizer,
        pressure=State(335.0 * PSIA_TO_PA),
        temperature=90.0,
        flash_values=("pressure", "enthalpy"),
    )

    FuelManifoldFluid = FluidLookup(
        "Fuel Manifold Liquid",
        PumpNetwork,
        fuel,
        pressure=State(410.0 * PSIA_TO_PA),
        temperature=293.15,
        flash_values=("pressure", "enthalpy"),
    )

    OxManifoldFluid = FluidLookup(
        "Ox Manifold Liquid",
        PumpNetwork,
        oxidizer,
        pressure=State(320.0 * PSIA_TO_PA),
        temperature=90.0,
        flash_values=("pressure", "enthalpy"),
    )

    # -----------------------------------------------------------------------
    # Pump Design Conditions
    # -----------------------------------------------------------------------

    FuelDesignCoefficients = TurboDesignCoefficients(
        "Fuel Pump Design Coefficients",
        PumpNetwork,
        rotor_speed=23000.0,
        volumetric_flow=2.7 / 1000.0,
        head_rise=323.4,
        torque=6.31,
        density=50.8 * LBM_FT3_TO_KG_M3,
        flow_geometric_parameter=1.56 * IN_TO_M,
        head_geometric_parameter=1.56 * IN_TO_M,
        torque_geometric_parameter=1.56 * IN_TO_M,
    )

    OxidizerDesignCoefficients = TurboDesignCoefficients(
        "Oxidizer Pump Design Coefficients",
        PumpNetwork,
        rotor_speed=23000.0,
        volumetric_flow=4.4 / 1000.0,
        head_rise=159.8,
        torque=4.77,
        density=1104.0,
        flow_geometric_parameter=1.84 * IN_TO_M,
        head_geometric_parameter=1.84 * IN_TO_M,
        torque_geometric_parameter=1.84 * IN_TO_M,
    )

    # -----------------------------------------------------------------------
    # Pump Map Data
    # -----------------------------------------------------------------------

    normalized_flow_coefficient_map = [
        0.00,
        0.40,
        0.70,
        1.00,
        1.15,
        1.30,
        1.45,
    ]

    normalized_head_coefficient_map = [
        1.25,
        1.20,
        1.12,
        1.00,
        0.84,
        0.62,
        0.35,
    ]

    normalized_torque_coefficient_map = [
        0.15,
        0.38,
        0.66,
        1.00,
        1.17,
        1.33,
        1.50,
    ]

    # -----------------------------------------------------------------------
    # Components
    # -----------------------------------------------------------------------

    COPV = Boundary(
        "COPV",
        PumpNetwork,
        PressurantSupply.pressure,
        PressurantSupply.temperature,
    )

    FuelPressurantLine = CompressibleFlowTube(
        "Fuel Press Line",
        PumpNetwork,
        mass_flow=0.03,
        upstream_static_pressure=COPV.pressure,
        upstream_static_temperature=COPV.temperature,
        upstream_density=PressurantSupply.density,
        downstream_static_pressure=FuelPressVolumeFluid.pressure,
        downstream_static_temperature=FuelPressVolumeFluid.temperature,
        downstream_density=FuelPressVolumeFluid.density,
        friction_factor=0.003,
        length=5.0 * IN_TO_M,
        inner_diameter=0.25 * IN_TO_M,
        upstream_static_enthalpy=PressurantSupply.enthalpy,
        upstream_speed_of_sound=PressurantSupply.speed_of_sound,
        downstream_speed_of_sound=FuelPressVolumeFluid.speed_of_sound,
        specific_heat_ratio=FuelUllageGas.specific_heat_ratio,
    )

    FuelPressLineFriction = Churchill(
        "Fuel Press Line Friction",
        PumpNetwork,
        mass_flow=FuelPressurantLine.mass_flow,
        friction_factor=FuelPressurantLine.friction_factor,
        hydraulic_diameter=FuelPressurantLine.inner_diameter,
        dynamic_viscosity=PressurantSupply.dynamic_viscosity,
        cross_sectional_area=(np.pi / 4.0) * FuelPressurantLine.inner_diameter**2,
        roughness=1e-6,
    )

    FuelPressVolume = Volume(
        "Fuel Pressurant Volume",
        PumpNetwork,
        pressure=FuelPressVolumeFluid.pressure,
        enthalpy=FuelPressVolumeFluid.enthalpy,
        volume=1.0,
        temperature=FuelPressVolumeFluid.temperature,
        total_enthalpy_in=FuelPressurantLine.total_enthalpy,
        mass_flow_in=FuelPressurantLine.mass_flow,
    )

    OxPressurantLine = CompressibleFlowTube(
        "Oxygen Press Line",
        PumpNetwork,
        mass_flow=0.03,
        upstream_static_pressure=COPV.pressure,
        upstream_static_temperature=COPV.temperature,
        upstream_density=PressurantSupply.density,
        downstream_static_pressure=OxPressVolumeFluid.pressure,
        downstream_static_temperature=OxPressVolumeFluid.temperature,
        downstream_density=OxPressVolumeFluid.density,
        friction_factor=0.003,
        length=10.0 * IN_TO_M,
        inner_diameter=0.25 * IN_TO_M,
        upstream_static_enthalpy=PressurantSupply.enthalpy,
        upstream_speed_of_sound=PressurantSupply.speed_of_sound,
        downstream_speed_of_sound=OxPressVolumeFluid.speed_of_sound,
        specific_heat_ratio=OxUllageGas.specific_heat_ratio,
    )

    OxPressLineFriction = Churchill(
        "Ox Press Line Friction",
        PumpNetwork,
        mass_flow=OxPressurantLine.mass_flow,
        friction_factor=OxPressurantLine.friction_factor,
        hydraulic_diameter=OxPressurantLine.inner_diameter,
        dynamic_viscosity=PressurantSupply.dynamic_viscosity,
        cross_sectional_area=(np.pi / 4.0) * OxPressurantLine.inner_diameter**2,
        roughness=1e-6,
    )

    OxPressVolume = Volume(
        "Ox Pressurant Volume",
        PumpNetwork,
        pressure=OxPressVolumeFluid.pressure,
        enthalpy=OxPressVolumeFluid.enthalpy,
        volume=1.0,
        temperature=OxPressVolumeFluid.temperature,
        total_enthalpy_in=OxPressurantLine.total_enthalpy,
        mass_flow_in=OxPressurantLine.mass_flow,
    )

    FuelBangBang = IsentropicGasRegulator(
        "FBB",
        PumpNetwork,
        upstream_total_pressure=FuelPressurantLine.downstream_total_pressure,
        upstream_total_temperature=FuelPressurantLine.downstream_total_temperature,
        set_pressure=FuelUllageGas.pressure,
        discharge_coefficient=1.0,
        cross_sectional_area=np.pi / 4.0 * (0.25 * IN_TO_M) ** 2,
        specific_gas_constant=FuelUllageGas.gas_constant,
        specific_heat_ratio=FuelUllageGas.specific_heat_ratio,
        total_enthalpy=FuelPressVolume.total_enthalpy_out,
        mass_flow=FuelPressVolume.mass_flow_out,
    )

    OxBangBang = IsentropicGasRegulator(
        "OBB",
        PumpNetwork,
        upstream_total_pressure=OxPressurantLine.downstream_total_pressure,
        upstream_total_temperature=OxPressurantLine.downstream_total_temperature,
        set_pressure=OxUllageGas.pressure,
        discharge_coefficient=1.0,
        cross_sectional_area=np.pi / 4.0 * (0.25 * IN_TO_M) ** 2,
        specific_gas_constant=OxUllageGas.gas_constant,
        specific_heat_ratio=OxUllageGas.specific_heat_ratio,
        total_enthalpy=OxPressVolume.total_enthalpy_out,
        mass_flow=OxPressVolume.mass_flow_out,
    )

    FuelTank = PressurizedTank(
        "Fuel Tank",
        PumpNetwork,
        pressure=FuelTankFluid.pressure,
        pressurant_density=FuelUllageGas.density,
        liquid_density=FuelTankFluid.density,
        collapse_factor=1.0,
        ullage_temperature=FuelUllageGas.temperature,
        liquid_temperature=FuelTankFluid.temperature,
        mass_flow_in=FuelBangBang.mass_flow,
        mass_flow_out=State((2.7 / 1000.0) * (50.8 * LBM_FT3_TO_KG_M3)),
    )

    OxTank = PressurizedTank(
        "Ox Tank",
        PumpNetwork,
        pressure=OxTankFluid.pressure,
        pressurant_density=OxUllageGas.density,
        liquid_density=OxTankFluid.density,
        collapse_factor=1.4,
        ullage_temperature=OxUllageGas.temperature,
        liquid_temperature=OxTankFluid.temperature,
        mass_flow_in=OxBangBang.mass_flow,
        mass_flow_out=State(2.2 * (2.7 / 1000.0) * (50.8 * LBM_FT3_TO_KG_M3)),
    )

    OxGravitydP = GravityPressureChange(
        "Ox Runline Gravity dP",
        PumpNetwork,
        upstream_pressure=OxTank.pressure,
        density=OxTank.liquid_density,
        elevation_change=-0.5,
        mass_flow=OxTank.mass_flow_out,
    )

    FuelRunline = DarcyWeisbach(
        "Fuel Main Line",
        PumpNetwork,
        upstream_pressure=State(),
        downstream_pressure=FuelPumpInletFluid.pressure,
        length=0.5,
        cross_sectional_area=np.pi / 4.0 * (0.5 * IN_TO_M) ** 2,
        hydraulic_diameter=0.5 * IN_TO_M,
        density=FuelTank.liquid_density,
        mass_flow=FuelTank.mass_flow_out,
        friction_factor=State(0.02),
    )

    FuelGravitydP = GravityPressureChange(
        "Fuel Runline Gravity dP",
        PumpNetwork,
        upstream_pressure=FuelTank.pressure,
        density=FuelTank.liquid_density,
        elevation_change=-0.5,
        mass_flow=FuelTank.mass_flow_out,
        downstream_pressure=FuelRunline.upstream_pressure,
    )

    FuelRunlineFriction = Churchill(
        "Fuel Main Line Friction",
        PumpNetwork,
        mass_flow=FuelRunline.mass_flow,
        friction_factor=FuelRunline.friction_factor,
        hydraulic_diameter=FuelRunline.hydraulic_diameter,
        dynamic_viscosity=FuelTankFluid.dynamic_viscosity,
        cross_sectional_area=FuelRunline.cross_sectional_area,
        roughness=1.5e-6,
    )

    OxRunline = DarcyWeisbach(
        "Ox Main Line",
        PumpNetwork,
        upstream_pressure=OxGravitydP.downstream_pressure,
        downstream_pressure=OxPumpInletFluid.pressure,
        length=0.5,
        cross_sectional_area=np.pi / 4.0 * (0.5 * IN_TO_M) ** 2,
        hydraulic_diameter=0.5 * IN_TO_M,
        density=OxTank.liquid_density,
        mass_flow=OxTank.mass_flow_out,
        friction_factor=State(0.02),
    )

    OxRunlineFriction = Colebrook(
        "Ox Main Line Friction",
        PumpNetwork,
        mass_flow=OxRunline.mass_flow,
        hydraulic_diameter=OxRunline.hydraulic_diameter,
        dynamic_viscosity=OxTankFluid.dynamic_viscosity,
        cross_sectional_area=OxRunline.cross_sectional_area,
        roughness=1.5e-6,
        friction_factor=OxRunline.friction_factor,
    )

    FuelPumpInlet = Volume(
        "Fuel Suction Inlet",
        PumpNetwork,
        pressure=FuelPumpInletFluid.pressure,
        enthalpy=FuelPumpInletFluid.enthalpy,
        volume=1.0,
        density=FuelPumpInletFluid.density,
        total_enthalpy_in=FuelTankFluid.enthalpy,
        mass_flow_in=FuelRunline.mass_flow,
        mass_flow_out=State((2.7 / 1000.0) * (50.8 * LBM_FT3_TO_KG_M3)),
    )

    OxPumpInlet = SimpleVolume(
        "Ox Suction Inlet",
        PumpNetwork,
        pressure=OxPumpInletFluid.pressure,
        temperature=OxPumpInletFluid.temperature,
        density=OxPumpInletFluid.density,
        enthalpy=OxPumpInletFluid.enthalpy,
        mass_flow_in=OxRunline.mass_flow,
        mass_flow_out=State(2.2 * (2.7 / 1000.0) * (50.8 * LBM_FT3_TO_KG_M3)),
        volume=0.2 * IN3_TO_M3,
    )

    FuelEPumpMap = TurboMap(
        "Fuel E-Pump Map",
        PumpNetwork,
        rotor_speed=fuel_shaft_speed,
        volumetric_flow=FuelPumpInlet.mass_flow_out / FuelPumpDischargeFluid.density,
        density=FuelPumpInletFluid.density,
        flow_geometric_parameter=1.56 * IN_TO_M,
        head_geometric_parameter=1.56 * IN_TO_M,
        torque_geometric_parameter=1.56 * IN_TO_M,
        design_flow_coefficient=FuelDesignCoefficients.flow_coefficient,
        design_head_coefficient=FuelDesignCoefficients.head_coefficient,
        design_torque_coefficient=FuelDesignCoefficients.torque_coefficient,
        normalized_flow_coefficient_map=normalized_flow_coefficient_map,
        normalized_head_coefficient_map=normalized_head_coefficient_map,
        normalized_torque_coefficient_map=normalized_torque_coefficient_map,
    )

    OxEPumpMap = TurboMap(
        "Ox E-Pump Map",
        PumpNetwork,
        rotor_speed=ox_shaft_speed,
        volumetric_flow=OxPumpInlet.mass_flow_out / OxPumpDischargeFluid.density,
        density=OxPumpInletFluid.density,
        flow_geometric_parameter=1.84 * IN_TO_M,
        head_geometric_parameter=1.84 * IN_TO_M,
        torque_geometric_parameter=1.84 * IN_TO_M,
        design_flow_coefficient=OxidizerDesignCoefficients.flow_coefficient,
        design_head_coefficient=OxidizerDesignCoefficients.head_coefficient,
        design_torque_coefficient=OxidizerDesignCoefficients.torque_coefficient,
        normalized_flow_coefficient_map=normalized_flow_coefficient_map,
        normalized_head_coefficient_map=normalized_head_coefficient_map,
        normalized_torque_coefficient_map=normalized_torque_coefficient_map,
    )

    FuelEPump = PolytropicPump(
        "Fuel Electric Pump",
        PumpNetwork,
        rotor_speed=fuel_shaft_speed,
        head_rise=FuelEPumpMap.head_rise,
        mass_flow=FuelPumpInlet.mass_flow_out,
        torque=FuelEPumpMap.torque,
        upstream_density=FuelPumpInlet.density,
        downstream_density=FuelPumpDischargeFluid.density,
        upstream_total_pressure=FuelPumpInlet.pressure,
        discharge_total_pressure=FuelPumpDischargeFluid.pressure,
        upstream_total_enthalpy=FuelPumpInlet.enthalpy,
    )

    OxEPump = PolytropicPump(
        "Ox Electric Pump",
        PumpNetwork,
        rotor_speed=ox_shaft_speed,
        head_rise=OxEPumpMap.head_rise,
        mass_flow=OxPumpInlet.mass_flow_out,
        torque=OxEPumpMap.torque,
        upstream_density=OxPumpInlet.density,
        downstream_density=OxPumpDischargeFluid.density,
        upstream_total_pressure=OxPumpInlet.pressure,
        discharge_total_pressure=OxPumpDischargeFluid.pressure,
        upstream_total_enthalpy=OxPumpInlet.enthalpy,
    )

    FuelPumpOutlet = Volume(
        "Fuel Discharge Outlet",
        PumpNetwork,
        pressure=FuelPumpDischargeFluid.pressure,
        enthalpy=FuelPumpDischargeFluid.enthalpy,
        volume=0.2 * IN3_TO_M3,
        temperature=FuelPumpDischargeFluid.temperature,
        density=FuelPumpDischargeFluid.density,
        mass_flow_in=FuelEPump.mass_flow,
        mass_flow_out=State((2.7 / 1000.0) * (50.8 * LBM_FT3_TO_KG_M3)),
        total_enthalpy_in=FuelEPump.discharge_total_enthalpy,
    )

    OxPumpOutlet = Volume(
        "Ox Discharge Outlet",
        PumpNetwork,
        pressure=OxPumpDischargeFluid.pressure,
        enthalpy=OxPumpDischargeFluid.enthalpy,
        volume=0.2 * IN3_TO_M3,
        temperature=OxPumpDischargeFluid.temperature,
        density=OxPumpDischargeFluid.density,
        mass_flow_in=OxEPump.mass_flow,
        mass_flow_out=State(2.2 * (2.7 / 1000.0) * (50.8 * LBM_FT3_TO_KG_M3)),
        total_enthalpy_in=OxEPump.discharge_total_enthalpy,
    )

    FuelInjectorInletLine = DarcyWeisbach(
        "Fuel Injector Inlet",
        PumpNetwork,
        upstream_pressure=FuelPumpOutlet.pressure,
        downstream_pressure=FuelManifoldFluid.pressure,
        length=0.25,
        cross_sectional_area=np.pi / 4.0 * (0.5 * IN_TO_M) ** 2,
        hydraulic_diameter=0.5 * IN_TO_M,
        density=FuelPumpOutlet.density,
        mass_flow=FuelPumpOutlet.mass_flow_out,
        friction_factor=State(0.02),
    )

    FuelInjectorInletFriction = Colebrook(
        "Fuel Injector Inlet Friction",
        PumpNetwork,
        mass_flow=FuelInjectorInletLine.mass_flow,
        hydraulic_diameter=FuelInjectorInletLine.hydraulic_diameter,
        dynamic_viscosity=FuelPumpDischargeFluid.dynamic_viscosity,
        cross_sectional_area=FuelInjectorInletLine.cross_sectional_area,
        roughness=1.5e-6,
        friction_factor=FuelInjectorInletLine.friction_factor,
    )

    OxInjectorInletLine = DarcyWeisbach(
        "Ox Injector Inlet",
        PumpNetwork,
        upstream_pressure=OxPumpOutlet.pressure,
        downstream_pressure=OxManifoldFluid.pressure,
        length=0.25,
        cross_sectional_area=np.pi / 4.0 * (0.5 * IN_TO_M) ** 2,
        hydraulic_diameter=0.5 * IN_TO_M,
        density=OxPumpOutlet.density,
        mass_flow=OxPumpOutlet.mass_flow_out,
        friction_factor=State(0.02),
    )

    OxInjectorInletFriction = Colebrook(
        "Ox Injector Inlet Friction",
        PumpNetwork,
        mass_flow=OxInjectorInletLine.mass_flow,
        hydraulic_diameter=OxInjectorInletLine.hydraulic_diameter,
        dynamic_viscosity=OxPumpDischargeFluid.dynamic_viscosity,
        cross_sectional_area=OxInjectorInletLine.cross_sectional_area,
        roughness=1.5e-6,
        friction_factor=OxInjectorInletLine.friction_factor,
    )

    FuelManifold = Volume(
        "Fuel Injector Manifold",
        PumpNetwork,
        pressure=FuelManifoldFluid.pressure,
        enthalpy=FuelManifoldFluid.enthalpy,
        volume=685.0 * IN3_TO_M3,
        temperature=FuelManifoldFluid.temperature,
        density=FuelManifoldFluid.density,
        mass_flow_in=FuelInjectorInletLine.mass_flow,
        total_enthalpy_in=FuelPumpOutlet.enthalpy,
    )

    OxManifold = Volume(
        "Ox Injector Manifold",
        PumpNetwork,
        pressure=OxManifoldFluid.pressure,
        enthalpy=OxManifoldFluid.enthalpy,
        volume=685.0 * IN3_TO_M3,
        temperature=OxManifoldFluid.temperature,
        density=OxManifoldFluid.density,
        mass_flow_in=OxInjectorInletLine.mass_flow,
        total_enthalpy_in=OxPumpOutlet.enthalpy,
    )

    FuelInjector = DischargeCoefficient(
        "Fuel Injector Orifices",
        PumpNetwork,
        upstream_pressure=FuelManifold.pressure,
        downstream_pressure=chamber_pressure,
        density=FuelManifold.density,
        discharge_coefficient=1.0,
        cross_sectional_area=0.56e-4,
        mass_flow=FuelManifold.mass_flow_out,
    )

    OxInjector = DischargeCoefficient(
        "Ox Injector Orifices",
        PumpNetwork,
        upstream_pressure=OxManifold.pressure,
        downstream_pressure=chamber_pressure,
        density=OxManifold.density,
        discharge_coefficient=1.0,
        cross_sectional_area=1.25e-4,
        mass_flow=OxManifold.mass_flow_out,
    )

    MainChamber = MainCombustionChamber(
        "MCC",
        PumpNetwork,
        chamber_pressure=chamber_pressure,
        oxidizer_mass_flow=OxInjector.mass_flow,
        fuel_mass_flow=FuelInjector.mass_flow,
    )

    Nozzle = RocketCEAChokedNozzle(
        "Nozzle",
        PumpNetwork,
        fuel=fuel,
        oxidizer=oxidizer,
        chamber_pressure=MainChamber.chamber_pressure,
        throat_area=10.0 * IN2_TO_M2,
        expansion_ratio=4.0,
        mixture_ratio=MainChamber.oxidizer_mass_flow / MainChamber.fuel_mass_flow,
        ambient_pressure=atmospheric_pressure,
        characterstic_velocity_efficiency=1.0,
        thrust_coefficient_efficiency=1.0,
        mass_flow=MainChamber.nozzle_mass_flow,
    )

    # -----------------------------------------------------------------------
    # Balances
    # -----------------------------------------------------------------------

    ChamberPressureBalance = Balance(
        "Chamber Pressure Balance",
        PumpNetwork,
        variable=FuelBangBang.discharge_coefficient,
        function=(chamber_pressure - 285.0 * PSIA_TO_PA),
    )

    MixtureRatioBalance = Balance(
        "Mixture Ratio Balance",
        PumpNetwork,
        variable=OxBangBang.discharge_coefficient,
        function=(OxInjector.mass_flow / FuelInjector.mass_flow - 2.2),
    )

    FuelPumpShaftSpeedBalance = Balance(
        "FPump Shaft Speed Balance",
        PumpNetwork,
        variable=FuelEPump.rotor_speed,
        function=FuelTank.pressure - 80.0 * PSIA_TO_PA,
    )

    OxPumpShaftSpeedBalance = Balance(
        "OPump Shaft Speed Balance",
        PumpNetwork,
        variable=OxEPump.rotor_speed,
        function=OxTank.pressure - 80.0 * PSIA_TO_PA,
    )

    # -----------------------------------------------------------------------
    # Solver
    # -----------------------------------------------------------------------

    solution = SteadyState(PumpNetwork).solve(
        return_type="dataframe",
        verbose=True,
        static=False,
        print_solution=True,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 50)
    print("              SYSTEM PRESSURES")
    print("=" * 50)

    print("\n[Tank Pressures]")
    print(f"  Fuel Tank Pressure              : {FuelTank.pressure.value * PA_TO_PSIA:10.3f} psia")
    print(f"  Ox Tank Pressure                : {OxTank.pressure.value * PA_TO_PSIA:10.3f} psia")

    print("\n[Pump Suction]")
    print(f"  Fuel Pump Inlet Pressure        : {FuelPumpInlet.pressure.value * PA_TO_PSIA:10.3f} psia")
    print(f"  Ox Pump Inlet Pressure          : {OxPumpInlet.pressure.value * PA_TO_PSIA:10.3f} psia")

    print("\n[Pump Discharge]")
    print(f"  Fuel Pump Outlet Pressure       : {FuelPumpOutlet.pressure.value * PA_TO_PSIA:10.3f} psia")
    print(f"  Ox Pump Outlet Pressure         : {OxPumpOutlet.pressure.value * PA_TO_PSIA:10.3f} psia")

    print("\n[Injector Manifolds]")
    print(f"  Fuel Manifold Pressure          : {FuelManifold.pressure.value * PA_TO_PSIA:10.3f} psia")
    print(f"  Ox Manifold Pressure            : {OxManifold.pressure.value * PA_TO_PSIA:10.3f} psia")

    print("\n[Chamber]")
    print(f"  Chamber Pressure                : {MainChamber.chamber_pressure.value * PA_TO_PSIA:10.3f} psia")

    print("\n[Pressure Drops]")
    print(f"  Fuel Main Line dP               : {(FuelRunline.upstream_pressure.value - FuelRunline.downstream_pressure.value) * PA_TO_PSIA:10.3f} psid")
    print(f"  Ox Main Line dP                 : {(OxRunline.upstream_pressure.value - OxRunline.downstream_pressure.value) * PA_TO_PSIA:10.3f} psid")
    print(f"  Fuel Pump dP                    : {(FuelPumpOutlet.pressure.value - FuelPumpInlet.pressure.value) * PA_TO_PSIA:10.3f} psid")
    print(f"  Ox Pump dP                      : {(OxPumpOutlet.pressure.value - OxPumpInlet.pressure.value) * PA_TO_PSIA:10.3f} psid")
    print(f"  Fuel Injector dP                : {(FuelManifold.pressure.value - chamber_pressure.value) * PA_TO_PSIA:10.3f} psid")
    print(f"  Ox Injector dP                  : {(OxManifold.pressure.value - chamber_pressure.value) * PA_TO_PSIA:10.3f} psid")

    print("\n[Injector Stiffness]")
    fuel_injector_dp = FuelManifold.pressure.value - chamber_pressure.value
    ox_injector_dp = OxManifold.pressure.value - chamber_pressure.value

    fuel_injector_stiffness = 100.0 * fuel_injector_dp / chamber_pressure.value
    ox_injector_stiffness = 100.0 * ox_injector_dp / chamber_pressure.value

    print(f"  Fuel Injector Stiffness         : {fuel_injector_stiffness:10.3f} %")
    print(f"  Ox Injector Stiffness           : {ox_injector_stiffness:10.3f} %")

    print("\n[Pump Speeds]")
    print(f"  Fuel Shaft Speed                : {fuel_shaft_speed.value:10.3f} rpm")
    print(f"  Ox Shaft Speed                  : {ox_shaft_speed.value:10.3f} rpm")
    print("=" * 50)


if __name__ == "__main__":
    main()