"""
Steady-state concentric-tube heat exchanger example.

This example builds a small counterflow heat exchanger using FullFlow. The hot
liquid flows through the inner tube, while water coolant flows through the
annulus around the tube. The tube wall is represented by two solid thermal nodes,
which exchange heat with the hot liquid and coolant through convection and with
each other through axial conduction.

The example demonstrates several FullFlow features together:

    1. Tabulated property maps for a synthetic hot gasoline-like liquid.
    2. ThermoProp property lookups for the coolant and solid wall material.
    3. Darcy-Weisbach pressure drop with Colebrook friction updates.
    4. Lumped fluid volumes with mass and energy balances.
    5. Solid wall thermal nodes.
    6. Convection, conduction, and Gnielinski heat-transfer coefficient updates.
    7. Network tracking for pressures and temperatures.

The hot-fluid property map is synthetic and is only intended to provide smooth,
reasonable liquid-like properties for this example. It should not be used as real
gasoline or RP-1 data.

Physical layout
---------------

The modeled system is a three-segment counterflow heat exchanger.

Hot liquid path:

    Hot Fluid Source
        |
        v
    Hot Fluid Tube 1
        |
        v
    Hot Fluid Node 1  ---- heat ---->  Tube/Solid Node 1
        |
        v
    Hot Fluid Tube 2
        |
        v
    Hot Fluid Node 2  ---- heat ---->  Tube/Solid Node 2
        |
        v
    Hot Fluid Tube 3
        |
        v
    Hot Fluid Outlet


Coolant path:

    Coolant Source
        |
        v
    Coolant Tube 1
        |
        v
    Coolant Node 1  ---- heat ---->  Tube/Solid Node 2
        |
        v
    Coolant Tube 2
        |
        v
    Coolant Node 2  ---- heat ---->  Tube/Solid Node 1
        |
        v
    Coolant Tube 3
        |
        v
    Coolant Outlet


Thermal wall model:

    Hot Fluid Node 1  <---- convection ---->  Tube/Solid Node 1
                                                   |
                                                   | conduction
                                                   v
    Hot Fluid Node 2  <---- convection ---->  Tube/Solid Node 2

    Coolant Node 2    <---- convection ---->  Tube/Solid Node 1
    Coolant Node 1    <---- convection ---->  Tube/Solid Node 2


Approximate cross section:

        +---------------------------------------------+
        |                                             |
        |        Coolant annulus flow area            |
        |                                             |
        |      +-------------------------------+      |
        |      |        tube wall / solid      |      |
        |      |   +-----------------------+   |      |
        |      |   |                       |   |      |
        |      |   |   hot liquid tube     |   |      |
        |      |   |                       |   |      |
        |      |   +-----------------------+   |      |
        |      +-------------------------------+      |
        |                                             |
        +---------------------------------------------+

"""

import numpy as np

from fullflow import *
from thermoprop import *


def harmonic_mean(a, b):
    """
    Return the harmonic mean of two State-like or numeric values.

    The harmonic mean is useful for combining two thermal conductivities at an
    interface because it weights the smaller conductivity more strongly.

    Parameters
    ----------
    a : State-like or float
        First value.
    b : State-like or float
        Second value.

    Returns
    -------
    State-like or float
        Harmonic mean of ``a`` and ``b``.
    """
    return 2.0 * a * b / (a + b)


# -----------------------------------------------------------------------------
# Synthetic hot gasoline-like liquid petroleum property data
#
# This is not real gasoline data. It is a smooth synthetic liquid-property map
# intended for heat-exchanger examples over a wider hot-liquid temperature range.
#
# The hot fluid is handled through a FullFlow Map rather than ThermoProp. This
# makes the example self-contained on the hot side and demonstrates how users can
# bring in arbitrary tabulated property data.
# -----------------------------------------------------------------------------

# Independent axes for the two-dimensional property table.
# The Map components below interpolate properties from pressure and temperature.
pressure_axis = np.array([100000.0, 200000.0, 500000.0, 1000000.0, 2000000.0])
temperature_axis = np.array([260.0, 300.0, 340.0, 380.0, 450.0, 525.0, 600.0, 700.0])

# Create pressure and temperature mesh arrays with pressure as axis 0 and
# temperature as axis 1. This matches the order of the Map axes below.
P, T = np.meshgrid(
    pressure_axis,
    temperature_axis,
    indexing="ij",
)

# Reference state for the synthetic correlations.
T_ref = 288.15
P_ref = 101325.0

# Density model constants.
rho_ref = 745.0
beta = 6.5e-4
bulk_modulus = 1.2e9

# Dynamic-viscosity model constants.
mu_ref = 5.5e-4
B_mu = 900.0
alpha_mu = 7.0e-9
mu_floor = 6.0e-5

# Specific-heat and enthalpy model constants.
cp_ref = 2200.0
cp_slope = 0.85

# Synthetic liquid density. Density decreases with temperature and increases
# weakly with pressure through a bulk-modulus correction.
gasoline_density = (
    rho_ref
    * (1.0 - beta * (T - T_ref))
    * (1.0 + (P - P_ref) / bulk_modulus)
)

# Synthetic dynamic viscosity. Viscosity decreases with temperature and increases
# weakly with pressure. A small floor prevents unrealistically low values.
gasoline_dynamic_viscosity = (
    mu_ref
    * np.exp(B_mu * (1.0 / T - 1.0 / T_ref))
    * np.exp(alpha_mu * (P - P_ref))
    + mu_floor
)

# Synthetic specific enthalpy. This integrates the linear specific-heat model
# and includes a small pressure/density term.
gasoline_enthalpy = (
    cp_ref * (T - T_ref)
    + 0.5 * cp_slope * (T - T_ref) ** 2
    + (P - P_ref) / gasoline_density
)

# Thermal-conductivity model constants.
k_ref = 0.13
k_slope = -1.2e-4
k_pressure_slope = 1.0e-11
k_floor = 0.06

# Synthetic thermal conductivity. It decreases mildly with temperature and is
# clipped to a small positive floor.
gasoline_conductivity = np.maximum(
    k_floor,
    k_ref
    + k_slope * (T - T_ref)
    + k_pressure_slope * (P - P_ref)
)

# Synthetic constant-pressure specific heat.
gasoline_specific_heat_cp = (
    cp_ref
    + cp_slope * (T - T_ref)
)


# -----------------------------------------------------------------------------
# Geometry
#
# The geometry is a concentric-tube heat exchanger:
#
#   - Hot fluid flows inside the inner tube.
#   - Coolant flows through the annulus outside the tube.
#   - The tube wall separates the two fluids.
#
# The current network uses three hydraulic tube elements per side, so Nseg is 3.
# -----------------------------------------------------------------------------

IN_TO_M = 0.0254

# Total heat-exchanger length and length of each model segment.
L_total = 40.0 * IN_TO_M
Nseg = 3
L = L_total / Nseg

# Inner-tube inside diameter. This is the hot-fluid flow diameter.
D_inner = 0.50 * IN_TO_M

# Tube wall thickness and resulting tube outside diameter.
wall_thickness = 0.035 * IN_TO_M
D_tube_outer = D_inner + 2.0 * wall_thickness

# Outer diameter of the annulus. Coolant flows between D_tube_outer and this ID.
D_annulus_outer = 0.875 * IN_TO_M

# Basic geometry check to avoid a negative annular area.
if D_tube_outer >= D_annulus_outer:
    raise ValueError("D_tube_outer must be smaller than D_annulus_outer.")

# Inner and outer radii of the tube wall.
ri = D_inner / 2.0
ro = D_tube_outer / 2.0

# Hot-side flow area.
Aw = np.pi * D_inner**2 / 4.0

# Coolant annulus flow area.
Aox = np.pi * (
    D_annulus_outer**2 -
    D_tube_outer**2
) / 4.0

# Hydraulic diameters for the hot tube and coolant annulus.
Dh_hot = D_inner
Dh_coolant = D_annulus_outer - D_tube_outer

# Wetted perimeters. The coolant annulus wetted perimeter includes both the
# inner tube outside surface and the annulus outer wall.
Pw_hot = np.pi * D_inner
Pw_coolant = np.pi * (
    D_annulus_outer +
    D_tube_outer
)

# Heat-transfer areas for one segment.
A_hot_wet = Pw_hot * L
A_coolant_wet = np.pi * D_tube_outer * L

# Metal wall cross-sectional area for axial conduction between the two solid
# wall nodes.
A_wall_cross_section = np.pi * (ro**2 - ri**2)

# Radial wall thickness. Kept here because it is often useful when switching
# from axial wall conduction to radial wall conduction.
dx_cond = ro - ri


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

HeatExchanger = Network("Heat Exchanger")


# -----------------------------------------------------------------------------
# Hot-fluid source and property map
# -----------------------------------------------------------------------------

# Source pressure and temperature are fixed input states for the hot liquid.
hot_fluid_source_pressure = State(6.00e5)
hot_fluid_source_temperature = State(380.0)

# Source map evaluates hot-fluid properties at the inlet/source state.
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
        "conductivity": gasoline_conductivity,
        "thermal_conductivity": gasoline_conductivity,
        "specific_heat_cp": gasoline_specific_heat_cp,
        "specific_heat": gasoline_specific_heat_cp,
    },
)


# -----------------------------------------------------------------------------
# Hot-fluid node 1 and property map
# -----------------------------------------------------------------------------

# Node 1 pressure and temperature are solve states used by the node volume and
# property map.
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
        "conductivity": gasoline_conductivity,
        "thermal_conductivity": gasoline_conductivity,
        "specific_heat_cp": gasoline_specific_heat_cp,
        "specific_heat": gasoline_specific_heat_cp,
    },
)



# -----------------------------------------------------------------------------
# Hot-fluid node 2 and property map
# -----------------------------------------------------------------------------

# Node 2 pressure and temperature are solve states used by the node volume and
# property map.
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
        "conductivity": gasoline_conductivity,
        "thermal_conductivity": gasoline_conductivity,
        "specific_heat_cp": gasoline_specific_heat_cp,
        "specific_heat": gasoline_specific_heat_cp,
    },
)




# -----------------------------------------------------------------------------
# Hot-fluid hydraulic segment 1:
#
#   Hot source -> Hot node 1
# -----------------------------------------------------------------------------

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

# Colebrook updates the Darcy friction factor used by HotFluidTube1.
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


# Hot node 1 is a real lumped storage volume. In steady state, FullFlow drives
# mass_dot and total_internal_energy_dot to zero. Because energy_variable="T",
# temperature is the thermal solve variable while enthalpy comes from the map.
HotFluidNode1 = Volume(
    "Hot Fluid Node 1",
    HeatExchanger,
    volume=1,
    pressure=hot_fluid_node1_pressure,
    density=GasolineMap1.density,
    temperature=hot_fluid_node1_temperature,
    enthalpy=GasolineMap1.enthalpy,
    internal_energy=GasolineMap1.enthalpy - hot_fluid_node1_pressure / GasolineMap1.density,
    energy_variable="T",
    total_enthalpy_in=SourceGasolineMap.enthalpy,
    mass_flow_in=HotFluidTube1.mass_flow,
    mass_flow_out=20
)


# -----------------------------------------------------------------------------
# Hot-fluid hydraulic segment 2:
#
#   Hot node 1 -> Hot node 2
# -----------------------------------------------------------------------------

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

# Colebrook updates the Darcy friction factor used by HotFluidTube2.
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



# Hot node 2 is another lumped storage volume. Steady state drives its mass and
# energy derivatives to zero.
HotFluidNode2 = Volume(
    "Hot Fluid Node 2",
    HeatExchanger,
    volume=1,
    pressure=hot_fluid_node2_pressure,
    density=GasolineMap2.density,
    temperature=hot_fluid_node2_temperature,
    enthalpy=GasolineMap2.enthalpy,
    internal_energy=GasolineMap2.enthalpy - hot_fluid_node2_pressure / GasolineMap2.density,
    energy_variable="T",
    total_enthalpy_in=GasolineMap1.enthalpy,
    mass_flow_in=HotFluidTube2.mass_flow,
    mass_flow_out=20
)


# -----------------------------------------------------------------------------
# Hot-fluid hydraulic segment 3:
#
#   Hot node 2 -> Hot outlet
# -----------------------------------------------------------------------------

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

# Colebrook updates the Darcy friction factor used by HotFluidTube3.
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





# -----------------------------------------------------------------------------
# Coolant property lookups
#
# The coolant side uses ThermoProp Fluid objects through FullFlow Lookup
# components. Each lookup exposes pressure, temperature, enthalpy, density,
# viscosity, conductivity, and specific heat as State-like attributes.
# -----------------------------------------------------------------------------

# Coolant source state.
CoolantSource = Lookup(
    "Coolant Source",
    HeatExchanger,
    Fluid,
    "water",
    pressure = 3e5,
    temperature = 280
)


# Coolant node 1 state. The first positional argument reuses the same fluid name
# as the coolant source.
CoolantNode1Fluid = Lookup(
    "Coolant Node 1 Fluid",
    HeatExchanger,
    Fluid,
    CoolantSource.fluid,
    pressure =  2.5e5,
    temperature = 290
)


# Coolant node 2 state.
CoolantNode2Fluid = Lookup(
    "Coolant Node 2 Fluid",
    HeatExchanger,
    Fluid,
    CoolantSource.fluid,
    pressure =  2e5,
    temperature = 300
)



# -----------------------------------------------------------------------------
# Annulus helper components
#
# These components compute auxiliary annulus geometry values used by the coolant
# friction model.
# -----------------------------------------------------------------------------

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



# -----------------------------------------------------------------------------
# Coolant hydraulic segment 1:
#
#   Coolant source -> Coolant node 1
# -----------------------------------------------------------------------------

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

# Colebrook updates the Darcy friction factor used by CoolantTube1. The annulus
# Poiseuille number gives the laminar correction appropriate for this annulus.
CoolantFriction1 = Colebrook(
    "Coolant Tube 1 Friction",
    HeatExchanger,
    mass_flow=CoolantTube1.mass_flow,
    friction_factor=CoolantTube1.friction_factor,
    hydraulic_diameter=CoolantTube1.hydraulic_diameter,
    dynamic_viscosity=CoolantSource.dynamic_viscosity,
    cross_sectional_area=CoolantTube1.cross_sectional_area,
    poiseuille_number=AnnulusPoiseuille.poiseuille_number,
    roughness=1e-9,
)


# Coolant node 1 is a lumped storage volume. Steady state drives its mass and
# energy derivatives to zero.
CoolantNode1 = Volume(
    "Coolant Node 1",
    HeatExchanger,
    volume=1,
    pressure=CoolantNode1Fluid.pressure,
    density=CoolantNode1Fluid.density,
    temperature=CoolantNode1Fluid.temperature,
    enthalpy=CoolantNode1Fluid.enthalpy,
    internal_energy=CoolantNode1Fluid.internal_energy,
    energy_variable="T",
    total_enthalpy_in=CoolantSource.enthalpy,
    mass_flow_in=CoolantTube1.mass_flow,
    mass_flow_out=10
)



# -----------------------------------------------------------------------------
# Coolant hydraulic segment 2:
#
#   Coolant node 1 -> Coolant node 2
# -----------------------------------------------------------------------------

CoolantTube2 = DarcyWeisbach(
    "Coolant Tube 2",
    HeatExchanger,
    mass_flow=CoolantNode1.mass_flow_out,
    upstream_pressure=CoolantNode1Fluid.pressure,
    downstream_pressure=CoolantNode2Fluid.pressure,
    length=L,
    cross_sectional_area=Aox,
    hydraulic_diameter=Dh_coolant,
    density=CoolantNode1Fluid.density,
    friction_factor=0.02,
)

# Colebrook updates the Darcy friction factor used by CoolantTube2.
CoolantFriction2 = Colebrook(
    "Coolant Tube 2 Friction",
    HeatExchanger,
    mass_flow=CoolantTube2.mass_flow,
    friction_factor=CoolantTube2.friction_factor,
    hydraulic_diameter=CoolantTube2.hydraulic_diameter,
    dynamic_viscosity=CoolantNode1Fluid.dynamic_viscosity,
    cross_sectional_area=CoolantTube2.cross_sectional_area,
    poiseuille_number=AnnulusPoiseuille.poiseuille_number,
    roughness=1e-9,
)


# Coolant node 2 is a lumped storage volume. Steady state drives its mass and
# energy derivatives to zero.
CoolantNode2 = Volume(
    "Coolant Node 2",
    HeatExchanger,
    volume=1,
    pressure=CoolantNode2Fluid.pressure,
    density=CoolantNode2Fluid.density,
    temperature=CoolantNode2Fluid.temperature,
    enthalpy=CoolantNode2Fluid.enthalpy,
    internal_energy=CoolantNode2Fluid.internal_energy,
    energy_variable="T",
    total_enthalpy_in=CoolantNode1Fluid.enthalpy,
    mass_flow_in=CoolantTube2.mass_flow,
    mass_flow_out=10
)

# -----------------------------------------------------------------------------
# Coolant hydraulic segment 3:
#
#   Coolant node 2 -> Coolant outlet
# -----------------------------------------------------------------------------

CoolantTube3 = DarcyWeisbach(
    "Coolant Tube 3",
    HeatExchanger,
    mass_flow=CoolantNode2.mass_flow_out,
    upstream_pressure=CoolantNode2Fluid.pressure,
    downstream_pressure=101325,
    length=L,
    cross_sectional_area=Aox,
    hydraulic_diameter=Dh_coolant,
    density=CoolantNode2Fluid.density,
    friction_factor=0.02,
)

# Colebrook updates the Darcy friction factor used by CoolantTube3.
CoolantFriction3 = Colebrook(
    "Coolant Tube 3 Friction",
    HeatExchanger,
    mass_flow=CoolantTube3.mass_flow,
    friction_factor=CoolantTube3.friction_factor,
    hydraulic_diameter=CoolantTube3.hydraulic_diameter,
    dynamic_viscosity=CoolantNode2Fluid.dynamic_viscosity,
    cross_sectional_area=CoolantTube3.cross_sectional_area,
    poiseuille_number=AnnulusPoiseuille.poiseuille_number,
    roughness=1e-9,
)





# -----------------------------------------------------------------------------
# Solid wall material lookups
#
# Each solid node uses a material lookup for Copper C101. The lookup temperature
# is the same State used by the corresponding Solid component, so the material
# properties update as the wall temperature changes.
# -----------------------------------------------------------------------------

SolidNode1Material = Lookup(
    "Solid Node 1 Metal",
    HeatExchanger,
    Material,
    "c101",
    temperature = 298
)

SolidNode2Material = Lookup(
    "Solid Node 2 Metal",
    HeatExchanger,
    Material,
    "c101",
    temperature = 298
)



# Solid thermal nodes. These provide energy balances for the tube wall.
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

# Effective tube-wall conductivity between the two solid nodes.
k = harmonic_mean(
    SolidNode1Material.thermal_conductivity,
    SolidNode2Material.thermal_conductivity,
)

# Axial conduction through the wall between the two solid nodes.
TubeConduction = Conduction(
    "Inter-Tube Conduction",
    HeatExchanger,
    temperature1=TubeNode1.temperature,
    temperature2=TubeNode2.temperature,
    thermal_conductivity=k,
    conductive_area=A_wall_cross_section,
    length=L,
)



# -----------------------------------------------------------------------------
# Convection components
#
# Each Convection component calculates heat transfer between one fluid node and
# one solid wall node. The convection coefficients are initialized to 25 W/m^2-K
# and then updated by the Gnielinski correlations below.
# -----------------------------------------------------------------------------

HotFluid1Solid1Convection = Convection(
    "Hot Fluid Node 1 to Solid Node 1 Convection",
    HeatExchanger,
    surface_temperature=TubeNode1.temperature,
    fluid_temperature=hot_fluid_node1_temperature,
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

HotFluid2Solid2Convection = Convection(
    "Hot Fluid Node 2 to Solid Node 2 Convection",
    HeatExchanger,
    surface_temperature=TubeNode2.temperature,
    fluid_temperature=hot_fluid_node2_temperature,
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


# -----------------------------------------------------------------------------
# Gnielinski heat-transfer coefficient updates
#
# These components compute convection coefficients from hydraulic diameter,
# friction factor, fluid conductivity, fluid specific heat, viscosity, area, and
# mass flow. The resulting coefficient is written into the corresponding
# Convection component.
# -----------------------------------------------------------------------------

HotFluid1Solid1Gnielinksi = Gnielinski(
    "Hot Fluid Node 1 to Solid Node 1 Gnielinski",
    HeatExchanger,
    hydraulic_diameter=HotFluidTube1.hydraulic_diameter,
    friction_factor=HotFluidTube1.friction_factor,
    fluid_conductivity=GasolineMap1.conductivity,
    fluid_specific_heat=GasolineMap1.specific_heat_cp,
    fluid_dynamic_viscosity=GasolineMap1.dynamic_viscosity,
    cross_sectional_area=HotFluidTube1.cross_sectional_area,
    mass_flow=HotFluidTube1.mass_flow,
    convection_coefficient=HotFluid1Solid1Convection.convection_coefficient,
)

HotFluid2Solid2Gnielinksi = Gnielinski(
    "Hot Fluid Node 2 to Solid Node 2 Gnielinski",
    HeatExchanger,
    hydraulic_diameter=HotFluidTube2.hydraulic_diameter,
    friction_factor=HotFluidTube2.friction_factor,
    fluid_conductivity=GasolineMap2.conductivity,
    fluid_specific_heat=GasolineMap2.specific_heat_cp,
    fluid_dynamic_viscosity=GasolineMap2.dynamic_viscosity,
    cross_sectional_area=HotFluidTube2.cross_sectional_area,
    mass_flow=HotFluidTube2.mass_flow,
    convection_coefficient=HotFluid2Solid2Convection.convection_coefficient,
)


Coolant1Solid2Gnielinksi = Gnielinski(
    "Coolant Node 1 to Solid Node 2 Gnielinski",
    HeatExchanger,
    hydraulic_diameter=CoolantTube1.hydraulic_diameter,
    friction_factor=CoolantTube1.friction_factor,
    fluid_conductivity=CoolantNode1Fluid.conductivity,
    fluid_specific_heat=CoolantNode1Fluid.specific_heat_cp,
    fluid_dynamic_viscosity=CoolantNode1Fluid.dynamic_viscosity,
    cross_sectional_area=CoolantTube1.cross_sectional_area,
    mass_flow=CoolantTube1.mass_flow,
    convection_coefficient=Coolant1Solid2Convection.convection_coefficient,
)


Coolant2Solid1Gnielinksi = Gnielinski(
    "Coolant Node 2 to Solid Node 1 Gnielinski",
    HeatExchanger,
    hydraulic_diameter=CoolantTube2.hydraulic_diameter,
    friction_factor=CoolantTube2.friction_factor,
    fluid_conductivity=CoolantNode2Fluid.conductivity,
    fluid_specific_heat=CoolantNode2Fluid.specific_heat_cp,
    fluid_dynamic_viscosity=CoolantNode2Fluid.dynamic_viscosity,
    cross_sectional_area=CoolantTube2.cross_sectional_area,
    mass_flow=CoolantTube2.mass_flow,
    convection_coefficient=Coolant2Solid1Convection.convection_coefficient,
)


# -----------------------------------------------------------------------------
# Heat-rate coupling
#
# FullFlow components expose heat rates as State-like values. Here the heat rates
# are connected into the energy balances of the two solid wall nodes and the four
# neighboring fluid nodes.
#
# Sign convention:
#
#   - Positive heat_rate adds energy to a Volume or Solid.
#   - A fluid node receives the negative of the convection heat rate when heat
#     leaves that fluid and enters the wall.
# -----------------------------------------------------------------------------

TubeNode1.heat_rate = (
    TubeConduction.heat_rate
    + HotFluid1Solid1Convection.heat_rate
    + Coolant2Solid1Convection.heat_rate
)

TubeNode2.heat_rate = (
    -TubeConduction.heat_rate
    + HotFluid2Solid2Convection.heat_rate
    + Coolant1Solid2Convection.heat_rate
)

HotFluidNode1.heat_rate = -HotFluid1Solid1Convection.heat_rate
HotFluidNode2.heat_rate = -HotFluid2Solid2Convection.heat_rate

CoolantNode1.heat_rate = -Coolant1Solid2Convection.heat_rate
CoolantNode2.heat_rate = -Coolant2Solid1Convection.heat_rate





# -----------------------------------------------------------------------------
# Tracks
#
# These values are printed or stored by the solver output so users can quickly
# inspect the final pressure and temperature distribution through the heat
# exchanger.
# -----------------------------------------------------------------------------

PA_TO_PSIA = 1.0 / 6894.757293

HeatExchanger.track("Hot Fluid Source Pressure [psia]", hot_fluid_source_pressure * PA_TO_PSIA)
HeatExchanger.track("Hot Fluid Node 1 Pressure [psia]", hot_fluid_node1_pressure * PA_TO_PSIA)
HeatExchanger.track("Hot Fluid Node 2 Pressure [psia]", hot_fluid_node2_pressure * PA_TO_PSIA)

HeatExchanger.track("Coolant Source Pressure [psia]", CoolantSource.pressure * PA_TO_PSIA)
HeatExchanger.track("Coolant Node 1 Pressure [psia]", CoolantNode1Fluid.pressure * PA_TO_PSIA)
HeatExchanger.track("Coolant Node 2 Pressure [psia]", CoolantNode2Fluid.pressure * PA_TO_PSIA)

HeatExchanger.track("Hot Fluid Source Temperature [K]", hot_fluid_source_temperature)
HeatExchanger.track("Hot Fluid Node 1 Temperature [K]", hot_fluid_node1_temperature)
HeatExchanger.track("Hot Fluid Node 2 Temperature [K]", hot_fluid_node2_temperature)

HeatExchanger.track("Coolant Source Temperature [K]", CoolantSource.temperature)
HeatExchanger.track("Coolant Node 1 Temperature [K]", CoolantNode1Fluid.temperature)
HeatExchanger.track("Coolant Node 2 Temperature [K]", CoolantNode2Fluid.temperature)

HeatExchanger.track("Solid Node 1 Temperature [K]", TubeNode1.temperature)
HeatExchanger.track("Solid Node 2 Temperature [K]", TubeNode2.temperature)




# -----------------------------------------------------------------------------
# Solve and export
# -----------------------------------------------------------------------------

SteadyState(HeatExchanger).solve(verbose=True)
