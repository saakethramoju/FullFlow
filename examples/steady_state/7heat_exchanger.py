import numpy as np

from fullflow import *
from thermoprop import *


# -----------------------------------------------------------------------------
# Synthetic hot gasoline-like liquid petroleum property data
#
# This is not real gasoline data. It is a smooth synthetic liquid-property map
# intended for heat-exchanger examples over a wider hot-liquid temperature range.
# -----------------------------------------------------------------------------

pressure_axis = np.array([100000.0, 200000.0, 500000.0, 1000000.0, 2000000.0])
temperature_axis = np.array([260.0, 300.0, 340.0, 380.0, 450.0, 525.0, 600.0, 700.0])

P, T = np.meshgrid(
    pressure_axis,
    temperature_axis,
    indexing="ij",
)

T_ref = 288.15
P_ref = 101325.0

rho_ref = 745.0
beta = 6.5e-4
bulk_modulus = 1.2e9

mu_ref = 5.5e-4
B_mu = 900.0
alpha_mu = 7.0e-9
mu_floor = 6.0e-5

cp_ref = 2200.0
cp_slope = 0.85

gasoline_density = (
    rho_ref
    * (1.0 - beta * (T - T_ref))
    * (1.0 + (P - P_ref) / bulk_modulus)
)

gasoline_dynamic_viscosity = (
    mu_ref
    * np.exp(B_mu * (1.0 / T - 1.0 / T_ref))
    * np.exp(alpha_mu * (P - P_ref))
    + mu_floor
)

gasoline_enthalpy = (
    cp_ref * (T - T_ref)
    + 0.5 * cp_slope * (T - T_ref) ** 2
    + (P - P_ref) / gasoline_density
)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

IN_TO_M = 0.0254

L_total = 40.0 * IN_TO_M
Nseg = 8
L = L_total / Nseg

D_inner = 0.75 * IN_TO_M

wall_thickness = 0.035 * IN_TO_M
D_tube_outer = D_inner + 2.0 * wall_thickness

D_annulus_outer = 1.50 * IN_TO_M

ri = D_inner / 2.0
ro = D_tube_outer / 2.0

Aw = np.pi * D_inner**2 / 4.0

Aox = np.pi * (
    D_annulus_outer**2 -
    D_tube_outer**2
) / 4.0

Dh_hot = D_inner
Dh_coolant = D_annulus_outer - D_tube_outer

Pw_hot = np.pi * D_inner
Pw_coolant = np.pi * (
    D_annulus_outer +
    D_tube_outer
)

A_hot_wet = Pw_hot * L
A_coolant_wet = np.pi * D_tube_outer * L

dx_cond = ro - ri


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

HeatExchanger = Network("Heat Exchanger")


hot_fluid_source_pressure = State(6.00e5)
hot_fluid_source_temperature = State(380.0)

SourceGasolineMap = Map(
    "Source Gasoline Map",
    HeatExchanger,
    inputs={
        "pressure": hot_fluid_source_pressure,
        "temperature": hot_fluid_source_temperature,
    },
    axes={
        "pressure": pressure_axis,
        "temperature": temperature_axis,
    },
    outputs={
        "density": gasoline_density,
        "dynamic_viscosity": gasoline_dynamic_viscosity,
        "enthalpy": gasoline_enthalpy,
    },
)


hot_fluid_node1_pressure = State(5.80e5)
hot_fluid_node1_temperature = State(380.0)

GasolineMap1 = Map(
    "Gasoline Map 1",
    HeatExchanger,
    inputs={
        "pressure": hot_fluid_node1_pressure,
        "temperature": hot_fluid_node1_temperature,
    },
    axes={
        "pressure": pressure_axis,
        "temperature": temperature_axis,
    },
    outputs={
        "density": gasoline_density,
        "dynamic_viscosity": gasoline_dynamic_viscosity,
        "enthalpy": gasoline_enthalpy,
    },
)



hot_fluid_node2_pressure = State(5.60e5)
hot_fluid_node2_temperature = State(380.0)

GasolineMap2 = Map(
    "Gasoline Map 2",
    HeatExchanger,
    inputs={
        "pressure": hot_fluid_node2_pressure,
        "temperature": hot_fluid_node2_temperature,
    },
    axes={
        "pressure": pressure_axis,
        "temperature": temperature_axis,
    },
    outputs={
        "density": gasoline_density,
        "dynamic_viscosity": gasoline_dynamic_viscosity,
        "enthalpy": gasoline_enthalpy,
    },
)




HotFluidTube1 = DarcyWeisbach(
    "Hot Fluid Tube 1",
    HeatExchanger,
    mass_flow=20.0,
    upstream_pressure=hot_fluid_source_pressure,
    downstream_pressure=hot_fluid_node1_pressure,
    length=L,
    cross_sectional_area=Aw,
    hydraulic_diameter=Dh_hot,
    density=SourceGasolineMap.density,
    friction_factor=0.02,
)

HotFluidFriction1 = Colebrook(
    "Hot Fluid Friction 1",
    HeatExchanger,
    mass_flow=HotFluidTube1.mass_flow,
    friction_factor=HotFluidTube1.friction_factor,
    hydraulic_diameter=HotFluidTube1.hydraulic_diameter,
    dynamic_viscosity=SourceGasolineMap.dynamic_viscosity,
    cross_sectional_area=HotFluidTube1.cross_sectional_area,
    roughness=1e-9,
)


HotFluidNode1 = Volume(
    "Hot Fluid Node 1",
    HeatExchanger,
    volume=1,
    pressure=hot_fluid_node1_pressure,
    temperature=hot_fluid_node1_temperature,
    enthalpy=GasolineMap1.enthalpy,
    energy_variable="T",
    total_enthalpy_in=SourceGasolineMap.enthalpy,
    mass_flow_in=HotFluidTube1.mass_flow,
    mass_flow_out=20
)


HotFluidTube2 = DarcyWeisbach(
    "Hot Fluid Tube 2",
    HeatExchanger,
    mass_flow=HotFluidNode1.mass_flow_out,
    upstream_pressure=hot_fluid_node1_pressure,
    downstream_pressure=hot_fluid_node2_pressure,
    length=L,
    cross_sectional_area=Aw,
    hydraulic_diameter=Dh_hot,
    density=GasolineMap1.density,
    friction_factor=0.02,
)

HotFluidFriction2 = Colebrook(
    "Hot Fluid Friction 2",
    HeatExchanger,
    mass_flow=HotFluidTube2.mass_flow,
    friction_factor=HotFluidTube2.friction_factor,
    hydraulic_diameter=HotFluidTube2.hydraulic_diameter,
    dynamic_viscosity=GasolineMap1.dynamic_viscosity,
    cross_sectional_area=HotFluidTube2.cross_sectional_area,
    roughness=1e-9,
)



HotFluidNode2 = Volume(
    "Hot Fluid Node 2",
    HeatExchanger,
    volume=1,
    pressure=hot_fluid_node2_pressure,
    temperature=hot_fluid_node2_temperature,
    enthalpy=GasolineMap2.enthalpy,
    energy_variable="T",
    total_enthalpy_in=GasolineMap1.enthalpy,
    mass_flow_in=HotFluidTube2.mass_flow,
    mass_flow_out=20
)


HotFluidTube3 = DarcyWeisbach(
    "Hot Fluid Tube 3",
    HeatExchanger,
    mass_flow=HotFluidNode2.mass_flow_out,
    upstream_pressure=hot_fluid_node2_pressure,
    downstream_pressure=3e5,
    length=L,
    cross_sectional_area=Aw,
    hydraulic_diameter=Dh_hot,
    density=GasolineMap2.density,
    friction_factor=0.02,
)

HotFluidFriction3 = Colebrook(
    "Hot Fluid Friction 3",
    HeatExchanger,
    mass_flow=HotFluidTube3.mass_flow,
    friction_factor=HotFluidTube3.friction_factor,
    hydraulic_diameter=HotFluidTube3.hydraulic_diameter,
    dynamic_viscosity=GasolineMap2.dynamic_viscosity,
    cross_sectional_area=HotFluidTube3.cross_sectional_area,
    roughness=1e-9,
)





CoolantSource = Lookup(
    "Coolant Source",
    HeatExchanger,
    Fluid,
    "water",
    pressure = 3e5,
    temperature = 280
)


CoolantNode1Fluid = Lookup(
    "Coolant Node 1 Fluid",
    HeatExchanger,
    Fluid,
    CoolantSource.fluid,
    pressure =  2.5e5,
    temperature = 290
)


CoolantNode2Fluid = Lookup(
    "Coolant Node 2 Fluid",
    HeatExchanger,
    Fluid,
    CoolantSource.fluid,
    pressure =  2e5,
    temperature = 300
)



AnnulusPoiseuille = CircularAnnulusPoiseuille(
    "Circular Annulus Poiseuille",
    HeatExchanger,
    inner_diameter=D_tube_outer,
    outer_diameter=D_annulus_outer,
)

AnnulusHydraulicDiameter = HydraulicDiameter(
    "Annulus Hydraulic Diameter",
    HeatExchanger,
    cross_sectional_area=Aox,
    wetted_perimeter=Pw_coolant,
)



CoolantTube1 = DarcyWeisbach(
    "Coolant Tube 1",
    HeatExchanger,
    mass_flow=10,
    upstream_pressure=CoolantSource.pressure,
    downstream_pressure=CoolantNode1Fluid.pressure,
    length=L,
    cross_sectional_area=Aox,
    hydraulic_diameter=Dh_coolant,
    density=CoolantSource.density,
    friction_factor=0.02,
)

CoolantFriction1 = Colebrook(
    "Coolant Tube 1 Friction",
    HeatExchanger,
    mass_flow=CoolantTube1.mass_flow,
    friction_factor=CoolantTube1.friction_factor,
    hydraulic_diameter=CoolantTube1.hydraulic_diameter,
    dynamic_viscosity=CoolantSource.dynamic_viscosity,
    cross_sectional_area=CoolantTube1.cross_sectional_area,
    roughness=1e-9,
)


CoolantNode1 = Volume(
    "Coolant Node 1",
    HeatExchanger,
    volume=1,
    pressure=CoolantNode1Fluid.pressure,
    temperature=CoolantNode1Fluid.temperature,
    enthalpy=CoolantNode1Fluid.enthalpy,
    energy_variable="T",
    total_enthalpy_in=CoolantSource.enthalpy,
    mass_flow_in=CoolantTube1.mass_flow,
    mass_flow_out=10
)



CoolantTube2 = DarcyWeisbach(
    "Coolant Tube 2",
    HeatExchanger,
    mass_flow=10,
    upstream_pressure=CoolantNode1Fluid.pressure,
    downstream_pressure=CoolantNode2Fluid.pressure,
    length=L,
    cross_sectional_area=Aox,
    hydraulic_diameter=Dh_coolant,
    density=CoolantNode1Fluid.density,
    friction_factor=0.02,
)

CoolantFriction2 = Colebrook(
    "Coolant Tube 2 Friction",
    HeatExchanger,
    mass_flow=CoolantTube2.mass_flow,
    friction_factor=CoolantTube2.friction_factor,
    hydraulic_diameter=CoolantTube2.hydraulic_diameter,
    dynamic_viscosity=CoolantNode1Fluid.dynamic_viscosity,
    cross_sectional_area=CoolantTube2.cross_sectional_area,
    roughness=1e-9,
)


CoolantNode2 = Volume(
    "Coolant Node 2",
    HeatExchanger,
    volume=1,
    pressure=CoolantNode2Fluid.pressure,
    temperature=CoolantNode2Fluid.temperature,
    enthalpy=CoolantNode2Fluid.enthalpy,
    energy_variable="T",
    total_enthalpy_in=CoolantNode1Fluid.enthalpy,
    mass_flow_in=CoolantTube2.mass_flow,
    mass_flow_out=10
)

CoolantTube3 = DarcyWeisbach(
    "Coolant Tube 3",
    HeatExchanger,
    mass_flow=10,
    upstream_pressure=CoolantNode2Fluid.pressure,
    downstream_pressure=101325,
    length=L,
    cross_sectional_area=Aox,
    hydraulic_diameter=Dh_coolant,
    density=CoolantNode2Fluid.density,
    friction_factor=0.02,
)

CoolantFriction3 = Colebrook(
    "Coolant Tube 3 Friction",
    HeatExchanger,
    mass_flow=CoolantTube3.mass_flow,
    friction_factor=CoolantTube3.friction_factor,
    hydraulic_diameter=CoolantTube3.hydraulic_diameter,
    dynamic_viscosity=CoolantNode2Fluid.dynamic_viscosity,
    cross_sectional_area=CoolantTube3.cross_sectional_area,
    roughness=1e-9,
)





SolidNode1Material = Lookup(
    "Solid Node 1 Metal",
    HeatExchanger,
    Material,
    "c101"
)

SolidNode2Material = Lookup(
    "Solid Node 2 Metal",
    HeatExchanger,
    Material,
    "c101"
)


print(SolidNode1Material.material)

TubeNode1 = Solid(
    "Tube Node 1",
    HeatExchanger,
    temperature=SolidNode1Material.temperature,
)

SteadyState(HeatExchanger).solve(verbose=True)