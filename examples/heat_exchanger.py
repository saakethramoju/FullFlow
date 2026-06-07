"""
Counterflow heat exchanger example.

Demonstrates:

- Coupled fluid and thermal networks
- Counterflow heat exchanger modeling
- Temperature-dependent material properties
- Conduction through solid structures
- Convective heat transfer
- Internal and annular flow passages
- Friction-factor and heat-transfer correlations
- Model switching with Model and ModelOption
- Gnielinski and Dittus-Boelter convection models

Run with:

    uv run python examples/heat_exchanger.py
"""

import numpy as np

from fullflow import *

IN_TO_M = 0.0254
IN3_TO_M3 = IN_TO_M**3


def harmonic_mean(a, b):
    """Return the harmonic mean of two State-like or numeric values."""
    return 2.0 * a * b / (a + b)


def main() -> None:
    HeatExchanger = Network("Heat Exchanger")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    L_total = 10.0 * IN_TO_M
    Nseg = 2
    L = L_total / Nseg

    D_inner = 0.75 * IN_TO_M
    wall_thickness = 0.035 * IN_TO_M
    D_tube_outer = D_inner + 2.0 * wall_thickness
    D_annulus_outer = 1.5 * IN_TO_M

    Aw = (np.pi / 4.0) * D_inner**2
    Aox = (np.pi / 4.0) * (D_annulus_outer**2 - D_tube_outer**2)

    Dh_hot = D_inner
    Pw_annulus = np.pi * (D_annulus_outer + D_tube_outer)

    A_hot_wet = np.pi * D_inner * L
    A_coolant_wet = np.pi * D_tube_outer * L

    ri = D_inner / 2.0
    ro = D_tube_outer / 2.0

    dx_cond = ro - ri
    A_cond = 2.0 * np.pi * L * (ro - ri) / np.log(ro / ri)

    hot_fluid = "rp-1"
    coolant = "water"
    metal = "c101"

    # ------------------------------------------------------------------
    # Fluid sources / pressure boundaries
    # ------------------------------------------------------------------

    LiquidSource = FluidLookup(
        "Liquid Source",
        HeatExchanger,
        hot_fluid,
        pressure=6e5,
        temperature=1000.0,
    )

    LiquidNode1Fluid = FluidLookup(
        "Liquid Node 1 Fluid",
        HeatExchanger,
        hot_fluid,
        pressure=5e5,
        temperature=490.0,
        flash_values=("pressure", "enthalpy"),
    )

    LiquidNode2Fluid = FluidLookup(
        "Liquid Node 2 Fluid",
        HeatExchanger,
        hot_fluid,
        pressure=4e5,
        temperature=480.0,
        flash_values=("pressure", "enthalpy"),
    )

    LiquidDrain = PressureBoundary(
        "Liquid Drain",
        HeatExchanger,
        pressure=3e5,
    )

    CoolantSource = FluidLookup(
        "Coolant Source",
        HeatExchanger,
        coolant,
        pressure=20e5,
        temperature=300.0,
    )

    CoolantNode1Fluid = FluidLookup(
        "Coolant Node 1 Fluid",
        HeatExchanger,
        coolant,
        pressure=16e5,
        temperature=305.0,
        flash_values=("pressure", "enthalpy"),
    )

    CoolantNode2Fluid = FluidLookup(
        "Coolant Node 2 Fluid",
        HeatExchanger,
        coolant,
        pressure=12e5,
        temperature=310.0,
        flash_values=("pressure", "enthalpy"),
    )

    CoolantDrain = PressureBoundary(
        "Coolant Drain",
        HeatExchanger,
        pressure=8e5,
    )

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    SolidNode1Material = MaterialLookup(
        "Solid Node 1 Material",
        HeatExchanger,
        metal,
    )

    SolidNode2Material = MaterialLookup(
        "Solid Node 2 Material",
        HeatExchanger,
        metal,
    )

    # ------------------------------------------------------------------
    # Hot fluid side
    # ------------------------------------------------------------------

    LiquidTube1 = DarcyWeisbach(
        "Liquid Tube 1",
        HeatExchanger,
        mass_flow=20.0,
        upstream_pressure=LiquidSource.pressure,
        downstream_pressure=LiquidNode1Fluid.pressure,
        length=L,
        cross_sectional_area=Aw,
        hydraulic_diameter=Dh_hot,
        density=LiquidSource.density,
        friction_factor=2e-5,
    )

    LiquidTube1Friction = Colebrook(
        "Liquid Tube 1 Friction",
        HeatExchanger,
        mass_flow=LiquidTube1.mass_flow,
        friction_factor=LiquidTube1.friction_factor,
        hydraulic_diameter=LiquidTube1.hydraulic_diameter,
        dynamic_viscosity=LiquidSource.dynamic_viscosity,
        cross_sectional_area=LiquidTube1.cross_sectional_area,
        roughness=1e-9,
    )

    LiquidNode1 = Volume(
        "Liquid Node 1",
        HeatExchanger,
        pressure=LiquidNode1Fluid.pressure,
        enthalpy=LiquidNode1Fluid.enthalpy,
        volume=Aw * L,
        total_enthalpy_in=LiquidSource.enthalpy,
        mass_flow_in=LiquidTube1.mass_flow,
        mass_flow_out=20.0,
    )

    LiquidTube2 = DarcyWeisbach(
        "Liquid Tube 2",
        HeatExchanger,
        mass_flow=LiquidNode1.mass_flow_out,
        upstream_pressure=LiquidNode1.pressure,
        downstream_pressure=LiquidNode2Fluid.pressure,
        length=L,
        cross_sectional_area=Aw,
        hydraulic_diameter=Dh_hot,
        density=LiquidNode1Fluid.density,
        friction_factor=2e-5,
    )

    LiquidTube2Friction = Colebrook(
        "Liquid Tube 2 Friction",
        HeatExchanger,
        mass_flow=LiquidTube2.mass_flow,
        friction_factor=LiquidTube2.friction_factor,
        hydraulic_diameter=LiquidTube2.hydraulic_diameter,
        dynamic_viscosity=LiquidNode1Fluid.dynamic_viscosity,
        cross_sectional_area=LiquidTube2.cross_sectional_area,
        roughness=1e-9,
    )

    LiquidNode2 = Volume(
        "Liquid Node 2",
        HeatExchanger,
        pressure=LiquidNode2Fluid.pressure,
        enthalpy=LiquidNode2Fluid.enthalpy,
        volume=Aw * L,
        total_enthalpy_in=LiquidNode1Fluid.enthalpy,
        mass_flow_in=LiquidTube2.mass_flow,
        mass_flow_out=20.0,
    )

    LiquidTube3 = DarcyWeisbach(
        "Liquid Tube 3",
        HeatExchanger,
        mass_flow=LiquidNode2.mass_flow_out,
        upstream_pressure=LiquidNode2.pressure,
        downstream_pressure=LiquidDrain.pressure,
        length=L,
        cross_sectional_area=Aw,
        hydraulic_diameter=Dh_hot,
        density=LiquidNode2Fluid.density,
        friction_factor=2e-5,
    )

    LiquidTube3Friction = Colebrook(
        "Liquid Tube 3 Friction",
        HeatExchanger,
        mass_flow=LiquidTube3.mass_flow,
        friction_factor=LiquidTube3.friction_factor,
        hydraulic_diameter=LiquidTube3.hydraulic_diameter,
        dynamic_viscosity=LiquidNode2Fluid.dynamic_viscosity,
        cross_sectional_area=LiquidTube3.cross_sectional_area,
        roughness=1e-9,
    )

    # ------------------------------------------------------------------
    # Coolant side
    # ------------------------------------------------------------------

    AnnulusPoiseuille = CircularAnnulus(
        "Circular Annulus Poiseuille",
        HeatExchanger,
        inner_diameter=D_tube_outer,
        outer_diameter=D_annulus_outer,
    )

    AnnulusHydraulicDiameter = HydraulicDiameter(
        "Annulus Hydraulic Diameter",
        HeatExchanger,
        cross_sectional_area=Aox,
        wetted_perimeter=Pw_annulus,
    )

    CoolantTube1 = DarcyWeisbach(
        "Coolant Tube 1",
        HeatExchanger,
        mass_flow=10.0,
        upstream_pressure=CoolantSource.pressure,
        downstream_pressure=CoolantNode1Fluid.pressure,
        length=L,
        cross_sectional_area=Aox,
        hydraulic_diameter=AnnulusHydraulicDiameter.hydraulic_diameter,
        density=CoolantSource.density,
        friction_factor=2e-5,
    )

    CoolantTube1Friction = Colebrook(
        "Coolant Tube 1 Friction",
        HeatExchanger,
        mass_flow=CoolantTube1.mass_flow,
        friction_factor=CoolantTube1.friction_factor,
        hydraulic_diameter=CoolantTube1.hydraulic_diameter,
        dynamic_viscosity=CoolantSource.dynamic_viscosity,
        cross_sectional_area=CoolantTube1.cross_sectional_area,
        roughness=1e-9,
        poiseuille_number=AnnulusPoiseuille.poiseuille_number,
    )

    CoolantNode1 = Volume(
        "Coolant Node 1",
        HeatExchanger,
        pressure=CoolantNode1Fluid.pressure,
        enthalpy=CoolantNode1Fluid.enthalpy,
        volume=Aox * L,
        total_enthalpy_in=CoolantSource.enthalpy,
        mass_flow_in=CoolantTube1.mass_flow,
        mass_flow_out=10.0,
    )

    CoolantTube2 = DarcyWeisbach(
        "Coolant Tube 2",
        HeatExchanger,
        mass_flow=CoolantNode1.mass_flow_out,
        upstream_pressure=CoolantNode1.pressure,
        downstream_pressure=CoolantNode2Fluid.pressure,
        length=L,
        cross_sectional_area=Aox,
        hydraulic_diameter=AnnulusHydraulicDiameter.hydraulic_diameter,
        density=CoolantNode1Fluid.density,
        friction_factor=2e-5,
    )

    CoolantTube2Friction = Colebrook(
        "Coolant Tube 2 Friction",
        HeatExchanger,
        mass_flow=CoolantTube2.mass_flow,
        friction_factor=CoolantTube2.friction_factor,
        hydraulic_diameter=CoolantTube2.hydraulic_diameter,
        dynamic_viscosity=CoolantNode1Fluid.dynamic_viscosity,
        cross_sectional_area=CoolantTube2.cross_sectional_area,
        roughness=1e-9,
        poiseuille_number=AnnulusPoiseuille.poiseuille_number,
    )

    CoolantNode2 = Volume(
        "Coolant Node 2",
        HeatExchanger,
        pressure=CoolantNode2Fluid.pressure,
        enthalpy=CoolantNode2Fluid.enthalpy,
        volume=Aox * L,
        total_enthalpy_in=CoolantNode1Fluid.enthalpy,
        mass_flow_in=CoolantTube2.mass_flow,
        mass_flow_out=10.0,
    )

    CoolantTube3 = DarcyWeisbach(
        "Coolant Tube 3",
        HeatExchanger,
        mass_flow=CoolantNode2.mass_flow_out,
        upstream_pressure=CoolantNode2.pressure,
        downstream_pressure=CoolantDrain.pressure,
        length=L,
        cross_sectional_area=Aox,
        hydraulic_diameter=AnnulusHydraulicDiameter.hydraulic_diameter,
        density=CoolantNode2Fluid.density,
        friction_factor=2e-5,
    )

    CoolantTube3Friction = Colebrook(
        "Coolant Tube 3 Friction",
        HeatExchanger,
        mass_flow=CoolantTube3.mass_flow,
        friction_factor=CoolantTube3.friction_factor,
        hydraulic_diameter=CoolantTube3.hydraulic_diameter,
        dynamic_viscosity=CoolantNode2Fluid.dynamic_viscosity,
        cross_sectional_area=CoolantTube3.cross_sectional_area,
        roughness=1e-9,
        poiseuille_number=AnnulusPoiseuille.poiseuille_number,
    )

    # ------------------------------------------------------------------
    # Heat transfer
    # ------------------------------------------------------------------

    TubeNode1 = Solid(
        "Tube Node 1",
        HeatExchanger,
        temperature=SolidNode1Material.temperature,
    )

    TubeNode2 = Solid(
        "Tube Node 2",
        HeatExchanger,
        temperature=SolidNode2Material.temperature,
    )

    k = harmonic_mean(
        SolidNode1Material.thermal_conductivity,
        SolidNode2Material.thermal_conductivity,
    )

    TubeConduction = Conduction(
        "Inter-Tube Conduction",
        HeatExchanger,
        temperature1=TubeNode1.temperature,
        temperature2=TubeNode2.temperature,
        thermal_conductivity=k,
        conductive_area=A_cond,
        length=dx_cond,
    )

    Liquid1Solid1Convection = Convection(
        "Liquid Node 1 to Solid Node 1 Convection",
        HeatExchanger,
        surface_temperature=TubeNode1.temperature,
        fluid_temperature=LiquidNode1Fluid.temperature,
        convective_area=A_hot_wet,
        convection_coefficient=25.0,
    )

    Coolant2Solid1Convection = Convection(
        "Coolant Node 2 to Solid Node 1 Convection",
        HeatExchanger,
        surface_temperature=TubeNode1.temperature,
        fluid_temperature=CoolantNode2Fluid.temperature,
        convective_area=A_coolant_wet,
        convection_coefficient=25.0,
    )

    Liquid2Solid2Convection = Convection(
        "Liquid Node 2 to Solid Node 2 Convection",
        HeatExchanger,
        surface_temperature=TubeNode2.temperature,
        fluid_temperature=LiquidNode2Fluid.temperature,
        convective_area=A_hot_wet,
        convection_coefficient=25.0,
    )

    Coolant1Solid2Convection = Convection(
        "Coolant Node 1 to Solid Node 2 Convection",
        HeatExchanger,
        surface_temperature=TubeNode2.temperature,
        fluid_temperature=CoolantNode1Fluid.temperature,
        convective_area=A_coolant_wet,
        convection_coefficient=25.0,
    )

    # ------------------------------------------------------------------
    # Heat transfer coefficient model options
    # ------------------------------------------------------------------

    GnielinskiOption = ModelOption(
        "gnielinski",
        components=[
            Gnielinski.model(
                "Liquid Node 1 to Solid Node 1 Gnielinski",
                hydraulic_diameter=LiquidTube1.hydraulic_diameter,
                friction_factor=LiquidTube1.friction_factor,
                fluid_conductivity=LiquidNode1Fluid.conductivity,
                fluid_specific_heat=LiquidNode1Fluid.specific_heat_cp,
                fluid_dynamic_viscosity=LiquidNode1Fluid.dynamic_viscosity,
                cross_sectional_area=LiquidTube1.cross_sectional_area,
                mass_flow=LiquidTube1.mass_flow,
                convection_coefficient=Liquid1Solid1Convection.convection_coefficient,
            ),
            Gnielinski.model(
                "Liquid Node 2 to Solid Node 2 Gnielinski",
                hydraulic_diameter=LiquidTube2.hydraulic_diameter,
                friction_factor=LiquidTube2.friction_factor,
                fluid_conductivity=LiquidNode2Fluid.conductivity,
                fluid_specific_heat=LiquidNode2Fluid.specific_heat_cp,
                fluid_dynamic_viscosity=LiquidNode2Fluid.dynamic_viscosity,
                cross_sectional_area=LiquidTube2.cross_sectional_area,
                mass_flow=LiquidTube2.mass_flow,
                convection_coefficient=Liquid2Solid2Convection.convection_coefficient,
            ),
            Gnielinski.model(
                "Coolant Node 1 to Solid Node 2 Gnielinski",
                hydraulic_diameter=CoolantTube1.hydraulic_diameter,
                friction_factor=CoolantTube1.friction_factor,
                fluid_conductivity=CoolantNode1Fluid.conductivity,
                fluid_specific_heat=CoolantNode1Fluid.specific_heat_cp,
                fluid_dynamic_viscosity=CoolantNode1Fluid.dynamic_viscosity,
                cross_sectional_area=CoolantTube1.cross_sectional_area,
                mass_flow=CoolantTube1.mass_flow,
                convection_coefficient=Coolant1Solid2Convection.convection_coefficient,
            ),
            Gnielinski.model(
                "Coolant Node 2 to Solid Node 1 Gnielinski",
                hydraulic_diameter=CoolantTube2.hydraulic_diameter,
                friction_factor=CoolantTube2.friction_factor,
                fluid_conductivity=CoolantNode2Fluid.conductivity,
                fluid_specific_heat=CoolantNode2Fluid.specific_heat_cp,
                fluid_dynamic_viscosity=CoolantNode2Fluid.dynamic_viscosity,
                cross_sectional_area=CoolantTube2.cross_sectional_area,
                mass_flow=CoolantTube2.mass_flow,
                convection_coefficient=Coolant2Solid1Convection.convection_coefficient,
            ),
        ],
    )

    DittusBoelterOption = ModelOption(
        "dittus_boelter",
        components=[
            DittusBoelter.model(
                "Liquid Node 1 to Solid Node 1 Dittus-Boelter",
                hydraulic_diameter=LiquidTube1.hydraulic_diameter,
                fluid_conductivity=LiquidNode1Fluid.conductivity,
                fluid_specific_heat=LiquidNode1Fluid.specific_heat_cp,
                fluid_dynamic_viscosity=LiquidNode1Fluid.dynamic_viscosity,
                cross_sectional_area=LiquidTube1.cross_sectional_area,
                mass_flow=LiquidTube1.mass_flow,
                convection_coefficient=Liquid1Solid1Convection.convection_coefficient,
            ),
            DittusBoelter.model(
                "Liquid Node 2 to Solid Node 2 Dittus-Boelter",
                hydraulic_diameter=LiquidTube2.hydraulic_diameter,
                fluid_conductivity=LiquidNode2Fluid.conductivity,
                fluid_specific_heat=LiquidNode2Fluid.specific_heat_cp,
                fluid_dynamic_viscosity=LiquidNode2Fluid.dynamic_viscosity,
                cross_sectional_area=LiquidTube2.cross_sectional_area,
                mass_flow=LiquidTube2.mass_flow,
                convection_coefficient=Liquid2Solid2Convection.convection_coefficient,
            ),
            DittusBoelter.model(
                "Coolant Node 1 to Solid Node 2 Dittus-Boelter",
                hydraulic_diameter=CoolantTube1.hydraulic_diameter,
                fluid_conductivity=CoolantNode1Fluid.conductivity,
                fluid_specific_heat=CoolantNode1Fluid.specific_heat_cp,
                fluid_dynamic_viscosity=CoolantNode1Fluid.dynamic_viscosity,
                cross_sectional_area=CoolantTube1.cross_sectional_area,
                mass_flow=CoolantTube1.mass_flow,
                convection_coefficient=Coolant1Solid2Convection.convection_coefficient,
            ),
            DittusBoelter.model(
                "Coolant Node 2 to Solid Node 1 Dittus-Boelter",
                hydraulic_diameter=CoolantTube2.hydraulic_diameter,
                fluid_conductivity=CoolantNode2Fluid.conductivity,
                fluid_specific_heat=CoolantNode2Fluid.specific_heat_cp,
                fluid_dynamic_viscosity=CoolantNode2Fluid.dynamic_viscosity,
                cross_sectional_area=CoolantTube2.cross_sectional_area,
                mass_flow=CoolantTube2.mass_flow,
                convection_coefficient=Coolant2Solid1Convection.convection_coefficient,
            ),
        ],
    )

    HeatTransferModel = Model(
        "Heat Transfer Correlation",
        HeatExchanger,
        GnielinskiOption,
        DittusBoelterOption,
        order=[
            "gnielinski",
            "dittus_boelter",
        ],
    )

    # ------------------------------------------------------------------
    # Heat-rate connections
    # ------------------------------------------------------------------

    TubeNode1.heat_rate = (
        TubeConduction.heat_rate
        + Liquid1Solid1Convection.heat_rate
        + Coolant2Solid1Convection.heat_rate
    )

    TubeNode2.heat_rate = (
        -TubeConduction.heat_rate
        + Liquid2Solid2Convection.heat_rate
        + Coolant1Solid2Convection.heat_rate
    )

    LiquidNode1.heat_rate = -Liquid1Solid1Convection.heat_rate
    LiquidNode2.heat_rate = -Liquid2Solid2Convection.heat_rate

    CoolantNode1.heat_rate = -Coolant1Solid2Convection.heat_rate
    CoolantNode2.heat_rate = -Coolant2Solid1Convection.heat_rate

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    HeatExchanger.track("Liquid Inlet Temperature", LiquidSource.temperature)
    HeatExchanger.track("Liquid Node 1 Temperature", LiquidNode1Fluid.temperature)
    HeatExchanger.track("Liquid Outlet Temperature", LiquidNode2Fluid.temperature)

    HeatExchanger.track("Coolant Inlet Temperature", CoolantSource.temperature)
    HeatExchanger.track("Coolant Node 1 Temperature", CoolantNode1Fluid.temperature)
    HeatExchanger.track("Coolant Outlet Temperature", CoolantNode2Fluid.temperature)

    HeatExchanger.track("Solid Node 1 Temperature", TubeNode1.temperature)
    HeatExchanger.track("Solid Node 2 Temperature", TubeNode2.temperature)

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    solution = SteadyState(HeatExchanger).solve(
        model="Heat Transfer Correlation",
        evaluate_all_model_options=True,
        verbose=True,
        print_solution=True,
        filename="heat_exchanger_results.xlsx",
    )

    print("\n" + "=" * 60)
    print("                 HEAT EXCHANGER SUMMARY")
    print("=" * 60)
    print(f"  Liquid Inlet Temperature        : {LiquidSource.temperature.value:12.3f} K")
    print(f"  Liquid Node 1 Temperature       : {LiquidNode1Fluid.temperature.value:12.3f} K")
    print(f"  Liquid Outlet Temperature       : {LiquidNode2Fluid.temperature.value:12.3f} K")
    print(f"  Coolant Inlet Temperature       : {CoolantSource.temperature.value:12.3f} K")
    print(f"  Coolant Node 1 Temperature      : {CoolantNode1Fluid.temperature.value:12.3f} K")
    print(f"  Coolant Outlet Temperature      : {CoolantNode2Fluid.temperature.value:12.3f} K")
    print(f"  Solid Node 1 Temperature        : {TubeNode1.temperature.value:12.3f} K")
    print(f"  Solid Node 2 Temperature        : {TubeNode2.temperature.value:12.3f} K")
    print("=" * 60)


if __name__ == "__main__":
    main()