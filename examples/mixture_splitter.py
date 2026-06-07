"""
Mixture splitter example.

This example demonstrates a simple FullFlow mixture network with:

- A mixed GN2/O2/Ar source fluid
- A separator volume
- Two outlet streams
- A balance equation that adjusts one outlet discharge coefficient
  until argon is removed from the second outlet stream

Run from the repository root with:

    uv run python examples/mixture_splitter.py
"""

import numpy as np

from fullflow import *

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

IN_TO_M = 0.0254


def main() -> None:
    """Run the mixture splitter example."""

    MixtureNetwork = Network("Mixture Flow")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    D = 3.0 * IN_TO_M
    A = (np.pi / 4.0) * D**2

    # ------------------------------------------------------------------
    # Shared outlet compositions
    # ------------------------------------------------------------------

    SeparatorOutlet1Composition = Composition({"Ar": 1.0})
    SeparatorOutlet2Composition = Composition()

    # ------------------------------------------------------------------
    # Fluid lookups
    # ------------------------------------------------------------------

    SourceFluid = FluidLookup(
        "Source Fluid",
        MixtureNetwork,
        {"gn2": 0.75, "O2": 0.01, "Ar": 0.24},
        pressure=3e5,
        temperature=300.0,
    )

    VolumeFluid = FluidLookup(
        "Volume Fluid",
        MixtureNetwork,
        SourceFluid.composition,
        pressure=2e5,
        temperature=300.0,
        flash_values=("pressure", "enthalpy"),
    )

    SeparatorOutlet1Fluid = FluidLookup(
        "Separator Outlet 1 Fluid",
        MixtureNetwork,
        SeparatorOutlet1Composition,
        pressure=VolumeFluid.pressure,
        temperature=VolumeFluid.temperature,
    )

    SeparatorOutlet2Fluid = FluidLookup(
        "Separator Outlet 2 Fluid",
        MixtureNetwork,
        SeparatorOutlet2Composition,
        pressure=VolumeFluid.pressure,
        temperature=VolumeFluid.temperature,
    )

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    Inlet = DischargeCoefficient(
        "Inlet",
        MixtureNetwork,
        upstream_pressure=SourceFluid.pressure,
        downstream_pressure=VolumeFluid.pressure,
        density=SourceFluid.density,
        discharge_coefficient=1.0,
        cross_sectional_area=A,
    )

    Separator = FlowSplitter(
        "Separator",
        MixtureNetwork,
        pressure=VolumeFluid.pressure,
        volume=1.0,
        mass_flow_in=Inlet.mass_flow,
        composition=VolumeFluid.composition,
        composition_out1=SeparatorOutlet1Composition,
        composition_out2=SeparatorOutlet2Composition,
        total_enthalpy_in=SourceFluid.enthalpy,
        enthalpy=VolumeFluid.enthalpy,
        total_enthalpy_out1=SeparatorOutlet1Fluid.enthalpy,
        total_enthalpy_out2=SeparatorOutlet2Fluid.enthalpy,
    )

    Outlet1 = DischargeCoefficient(
        "Outlet 1",
        MixtureNetwork,
        upstream_pressure=Separator.pressure,
        downstream_pressure=101325.0,
        density=SeparatorOutlet1Fluid.density,
        discharge_coefficient=1.0,
        cross_sectional_area=A / 4.0,
        mass_flow=Separator.mass_flow_out1,
    )

    Outlet2 = DischargeCoefficient(
        "Outlet 2",
        MixtureNetwork,
        upstream_pressure=Separator.pressure,
        downstream_pressure=101325.0,
        density=SeparatorOutlet2Fluid.density,
        discharge_coefficient=1.0,
        cross_sectional_area=A,
        mass_flow=Separator.mass_flow_out2,
    )

    ArgonBalance = Balance(
        "Argon Balance",
        MixtureNetwork,
        variable=Outlet1.discharge_coefficient,
        function=Separator.composition_out2["Argon"],
    )

    total_outlet_discharge_coefficient = (
        Outlet1.discharge_coefficient + Outlet2.discharge_coefficient
    )

    MixtureNetwork.track(
        "Total Outlet Discharge Coefficient",
        total_outlet_discharge_coefficient,
    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    solution = SteadyState(MixtureNetwork).solve(
        return_type="dataframe",
        verbose=True,
        static=False,
        print_solution=True,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("                 MIXTURE SPLITTER SUMMARY")
    print("=" * 60)

    print("\n[Mass Flow]")
    print(f"  Inlet Mass Flow                 : {Inlet.mass_flow.value:12.6f} kg/s")
    print(f"  Outlet 1 Mass Flow              : {Outlet1.mass_flow.value:12.6f} kg/s")
    print(f"  Outlet 2 Mass Flow              : {Outlet2.mass_flow.value:12.6f} kg/s")

    print("\n[Discharge Coefficients]")
    print(f"  Outlet 1 Cd                     : {Outlet1.discharge_coefficient.value:12.6f}")
    print(f"  Outlet 2 Cd                     : {Outlet2.discharge_coefficient.value:12.6f}")
    print(
        "  Total Outlet Cd                 : "
        f"{total_outlet_discharge_coefficient.value:12.6f}"
    )

    print("\n[Outlet Compositions]")
    print(f"  Outlet 1 Composition            : {Separator.composition_out1}")
    print(f"  Outlet 2 Composition            : {Separator.composition_out2}")

    print("=" * 60)


if __name__ == "__main__":
    main()