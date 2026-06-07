from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from fullflow.System import Component, State

if TYPE_CHECKING:
    from fullflow.System import Network
    import pandas as pd



class TurboMap(Component):
    """
    Generic one-dimensional turbomachinery performance map.

    Uses separate geometric parameters for each coefficient:

        flow_coefficient =
            volumetric_flow / (omega * flow_geometric_parameter**3)

        head_coefficient =
            head_rise / (omega**2 * head_geometric_parameter**2 / g)

        torque_coefficient =
            torque / (density * omega**2 * torque_geometric_parameter**5)
    """

    def __init__(self,
                 name: str,
                 network: Network,

                 rotor_speed: State,
                 volumetric_flow: State,
                 density: State,

                 flow_geometric_parameter: State,
                 head_geometric_parameter: State,
                 torque_geometric_parameter: State,

                 design_flow_coefficient: float,
                 design_head_coefficient: float,
                 design_torque_coefficient: float,

                 normalized_flow_coefficient_map: list | tuple | np.ndarray | pd.Series,
                 normalized_head_coefficient_map: list | tuple | np.ndarray | pd.Series,
                 normalized_torque_coefficient_map: list | tuple | np.ndarray | pd.Series,

                 gravitational_acceleration: float = 9.80665,

                 normalized_flow_coefficient: State | None = None,
                 normalized_head_coefficient: State | None = None,
                 normalized_torque_coefficient: State | None = None,

                 flow_coefficient: State | None = None,
                 head_coefficient: State | None = None,
                 torque_coefficient: State | None = None,

                 head_rise: State | None = None,
                 torque: State | None = None):

        self.setup()

        normalized_flow = np.asarray(normalized_flow_coefficient_map, dtype=float)
        normalized_head = np.asarray(normalized_head_coefficient_map, dtype=float)
        normalized_torque = np.asarray(normalized_torque_coefficient_map, dtype=float)

        if not (
            len(normalized_flow)
            == len(normalized_head)
            == len(normalized_torque)
        ):
            raise ValueError(
                f"{self.name}: normalized map arrays must have the same length."
            )

        if len(normalized_flow) < 2:
            raise ValueError(
                f"{self.name}: at least two map points are required."
            )

        sort_indices = np.argsort(normalized_flow)

        self._normalized_flow_map = normalized_flow[sort_indices]
        self._normalized_head_map = normalized_head[sort_indices]
        self._normalized_torque_map = normalized_torque[sort_indices]

        if np.any(np.diff(self._normalized_flow_map) <= 0.0):
            raise ValueError(
                f"{self.name}: normalized_flow_coefficient_map "
                "must be strictly increasing."
            )

    def pre_evaluation(self):
        self.evaluate_states()

    def evaluate_states(self):
        Q = self.volumetric_flow.value
        N = self.rotor_speed.value
        rho = self.density.value
        g = self.gravitational_acceleration.value

        D_flow = self.flow_geometric_parameter.value
        D_head = self.head_geometric_parameter.value
        D_torque = self.torque_geometric_parameter.value

        design_flow_coefficient = self.design_flow_coefficient.value
        design_head_coefficient = self.design_head_coefficient.value
        design_torque_coefficient = self.design_torque_coefficient.value

        omega = np.pi / 30.0 * N

        if abs(omega) < 1e-12:
            raise ValueError(f"{self.name}: rotor_speed is too close to zero.")

        if abs(rho) < 1e-12:
            raise ValueError(f"{self.name}: density is too close to zero.")

        if abs(g) < 1e-12:
            raise ValueError(f"{self.name}: gravitational_acceleration is too close to zero.")

        if abs(D_flow) < 1e-12:
            raise ValueError(f"{self.name}: flow_geometric_parameter is too close to zero.")

        if abs(D_head) < 1e-12:
            raise ValueError(f"{self.name}: head_geometric_parameter is too close to zero.")

        if abs(D_torque) < 1e-12:
            raise ValueError(f"{self.name}: torque_geometric_parameter is too close to zero.")

        if abs(design_flow_coefficient) < 1e-12:
            raise ValueError(f"{self.name}: design_flow_coefficient is too close to zero.")

        flow_coefficient = Q / (omega * D_flow**3)
        normalized_flow_coefficient = flow_coefficient / design_flow_coefficient

        normalized_head_coefficient = float(
            np.interp(
                normalized_flow_coefficient,
                self._normalized_flow_map,
                self._normalized_head_map,
            )
        )

        normalized_torque_coefficient = float(
            np.interp(
                normalized_flow_coefficient,
                self._normalized_flow_map,
                self._normalized_torque_map,
            )
        )

        head_coefficient = normalized_head_coefficient * design_head_coefficient
        torque_coefficient = normalized_torque_coefficient * design_torque_coefficient

        head_rise = head_coefficient * omega**2 * D_head**2 / g
        torque = torque_coefficient * rho * omega**2 * D_torque**5

        self.flow_coefficient.value = flow_coefficient
        self.normalized_flow_coefficient.value = normalized_flow_coefficient

        self.head_coefficient.value = head_coefficient
        self.normalized_head_coefficient.value = normalized_head_coefficient

        self.torque_coefficient.value = torque_coefficient
        self.normalized_torque_coefficient.value = normalized_torque_coefficient

        self.head_rise.value = head_rise
        self.torque.value = torque


    @property
    def ignored_export_attributes(self) -> set[str]:
        return super().ignored_export_attributes | {
            "property_states",
            "normalized_flow_coefficient_map",
            "normalized_head_coefficient_map",
            "normalized_torque_coefficient_map",
        }
    





class TurboDesignCoefficients(Component):
    """
    Computes turbomachinery design-point coefficients from
    dimensional design conditions.

    Equations
    ---------
    omega = rotor_speed * pi / 30

    flow_coefficient = volumetric_flow / (omega * flow_geometric_parameter**3)

    head_coefficient = head_rise / (omega**2 * head_geometric_parameter**2 / g)

    torque_coefficient = torque / (density * omega**2 * torque_geometric_parameter**5)
    """

    def __init__(self,
                 name: str,
                 network: Network,

                 rotor_speed: State,
                 volumetric_flow: State,
                 head_rise: State,
                 torque: State,
                 density: State,

                 flow_geometric_parameter: State,
                 head_geometric_parameter: State,
                 torque_geometric_parameter: State,

                 gravitational_acceleration: float = 9.80665,

                 flow_coefficient: State | None = None,
                 head_coefficient: State | None = None,
                 torque_coefficient: State | None = None):

        self.setup()

    def pre_evaluation(self):
        self.evaluate_states()

    def evaluate_states(self):
        N = self.rotor_speed.value
        Q = self.volumetric_flow.value
        H = self.head_rise.value
        T = self.torque.value
        rho = self.density.value
        g = self.gravitational_acceleration.value

        D_flow = self.flow_geometric_parameter.value
        D_head = self.head_geometric_parameter.value
        D_torque = self.torque_geometric_parameter.value

        omega = np.pi / 30.0 * N

        if abs(omega) < 1e-12:
            raise ValueError(f"{self.name}: rotor_speed is too close to zero.")

        if abs(rho) < 1e-12:
            raise ValueError(f"{self.name}: density is too close to zero.")

        if abs(g) < 1e-12:
            raise ValueError(f"{self.name}: gravitational_acceleration is too close to zero.")

        if abs(D_flow) < 1e-12:
            raise ValueError(f"{self.name}: flow_geometric_parameter is too close to zero.")

        if abs(D_head) < 1e-12:
            raise ValueError(f"{self.name}: head_geometric_parameter is too close to zero.")

        if abs(D_torque) < 1e-12:
            raise ValueError(f"{self.name}: torque_geometric_parameter is too close to zero.")

        flow_coefficient = Q / (omega * D_flow**3)

        head_coefficient = H / (omega**2 * D_head**2 / g)

        torque_coefficient = T / (rho * omega**2 * D_torque**5)

        self.flow_coefficient.value = flow_coefficient
        self.head_coefficient.value = head_coefficient
        self.torque_coefficient.value = torque_coefficient


    
class Rotor(Component):

    def __init__(self,
                 name: str,
                 network: Network,
                 rotor_speed: State,
                 net_torque: State | float,
                 polar_moment_of_inertia: float):
        self.setup()

    