# Network.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fullflow.System import Component, State, Balance


class Network:
    """
    Container for FullFlow components, balances, models, and shared states.

    `Network` defines a complete system of equations to be evaluated or solved.
    Components register their physics equations with the network, balances add
    user-defined algebraic constraints, and solvers operate on the resulting
    collection of iteration variables and residual equations.

    A Network is typically the top-level object in a FullFlow model:

    ``network = Network("Propellant Feed System")``

    Components and balances automatically register themselves with the network
    when constructed.

    Tracked states may be added using:

    ``network.track("Chamber Pressure", chamber_pressure)``

    to include additional values in exported results without introducing
    residual equations or iteration variables.

    Parameters
    ----------
    name : str
        User-defined network name

    Notes
    -----
    Network is intentionally lightweight. It is responsible for:

    * Component registration
    * Balance registration
    * Model registration
    * Iteration variable collection
    * Residual collection
    * State evaluation
    * Solution export

    Solver algorithms are implemented separately by FullFlow solver classes.

    Components may contribute residual equations and iteration variables,
    while balances provide user-defined solve targets.

    A `State` cannot simultaneously appear as both a component iteration
    variable and a balance iteration variable. `Network` validates this
    condition before solving and raises an error if overlap is detected.
    """

    def __init__(self, name: str):
        self.name = name
        self.component_list = []
        self.balance_list = []
        self.tracked_state_list = []
        self.model_list = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_component(self, component: Component) -> None:
        """Register a component with the network."""
        self.component_list.append(component)


    def remove_component(self, component: Component) -> None:
        """
        Remove a component from the network.

        This is mainly used by Model when replacing one active component
        implementation with another.
        """

        if component not in self.component_list:
            raise ValueError(
                f"Component {component.name!r} is not registered "
                f"with network {self.name!r}."
            )

        self.component_list.remove(component)


    def add_balance(self, balance: Balance) -> None:
        """Register an algebraic balance with the network."""
        self.balance_list.append(balance)


    def add_model(self, model) -> None:
        """
        Register a model with the network.

        Models are not evaluated directly by Network. They are used by solvers
        to build and replace optional component implementations.
        """
        self.model_list.append(model)


    def track(self, name: str, state: State) -> State:
        """
        Track an extra State or derived State in printed/exported results.

        This does not add residuals or iteration variables. It only exports the
        value after the network has been evaluated/solved.
        """
        self.tracked_state_list.append((name, state))
        return state

    @property
    def components(self) -> list[str]:
        """Return component names."""
        return [c.name for c in self.component_list]

    @property
    def balances(self) -> list[str]:
        """Return balance names."""
        return [b.name for b in self.balance_list]
        
    @property
    def models(self) -> list[str]:
        """Return model names."""
        return [m.name for m in self.model_list]

    # ------------------------------------------------------------------
    # Iteration variable metadata
    # ------------------------------------------------------------------

    @property
    def lower_bounds(self) -> list[float]:
        """Lower bounds for all component and balance iteration variables."""
        return [var.lower_bound for var in self.collect_all_iteration_variables()]

    @property
    def upper_bounds(self) -> list[float]:
        """Upper bounds for all component and balance iteration variables."""
        return [var.upper_bound for var in self.collect_all_iteration_variables()]

    @property
    def has_bounds(self) -> bool:
        """True if any iteration variable has finite bounds."""
        return any([var.has_bounds for var in self.collect_all_iteration_variables()])

    @property
    def keep_feasible(self) -> list[float]:
        """Per-variable keep_feasible flags for scipy Bounds."""
        return [var.keep_feasible for var in self.collect_all_iteration_variables()]

    @property
    def iteration_variables(self) -> str:
        """
        Return readable names for all iteration variables.

        This is used for debugging/printing, not for solving.
        """
        names = []

        def find_name(obj, target):
            for k, v in obj.__dict__.items():
                if v is target:
                    return f"{obj.name}:{k}"
            return f"{obj.name}:<unknown>"

        for comp in self.component_list:
            for var in comp.iteration_variables:
                names.append(find_name(comp, var))

        for bal in self.balance_list:
            for var in bal.iteration_variables:
                names.append(find_name(bal, var))

        return "\n".join(names)

    @property
    def iteration_values(self) -> list[float]:
        """
        Current numeric values of all iteration variables.

        Also checks that a Balance is not trying to solve for a State that is
        already owned by a component residual equation.
        """
        self._validate_no_iteration_overlap()
        return [var.value for var in self.collect_all_iteration_variables()]

    # ------------------------------------------------------------------
    # Residual collection
    # ------------------------------------------------------------------

    @property
    def residuals(self) -> list[float]:
        """
        Collect all component residuals and balance residuals.

        Components contribute physics residuals.
        Balances contribute user-requested algebraic targets.
        """
        residuals = []

        # Component residuals.
        for comp in self.component_list:
            try:
                comp_residuals = comp.residuals
            except Exception as e:
                original_msg = str(e).splitlines()[0]

                raise RuntimeError(
                    f"Failed while evaluating residuals for component `{comp.name}` "
                    f"of type `{type(comp).__name__}`.\n\n"
                    "A State used inside this component's residual equations is probably "
                    "unassigned.\n\n"
                    "Likely fixes:\n"
                    "  - Give the missing State an initial value\n"
                    "  - Connect it to another component that computes it\n"
                    "  - Make it an iteration variable\n"
                    "  - If this is a static/transient-only quantity, do not use it in "
                    "steady-state residuals\n\n"
                    f"Original error: {original_msg}"
                ) from None

            if isinstance(comp_residuals, (list, tuple)):
                residuals.extend(comp_residuals)
            else:
                residuals.append(comp_residuals)

        # Balance residuals.
        balance_residuals = [
            r for balance in self.balance_list for r in balance.residuals
        ]

        residuals.extend(balance_residuals)

        return residuals

    # ------------------------------------------------------------------
    # Basic network evaluation
    # ------------------------------------------------------------------

    def pre_evaluation(self) -> None:
        """
        Run one-time component setup before solving/evaluation.

        This stays simple and generic. The solver decides how many times
        evaluate_states() should be called.
        """
        for c in self.component_list:
            c.pre_evaluation()

    def evaluate_states(self) -> None:
        """
        Evaluate each component once in user-defined order.

        This is intentionally simple. Order-independent repeated settling is
        handled by SteadyState, not by Network.
        """
        for c in self.component_list:
            c.evaluate_states()

    # ------------------------------------------------------------------
    # Iteration variable collection and assignment
    # ------------------------------------------------------------------

    def collect_iteration_variables(self) -> list[State]:
        """Collect iteration variables owned by components."""
        iter_vars = []

        for comp in self.component_list:
            iter_vars.extend(comp.iteration_variables)

        return iter_vars

    def collect_balance_iteration_variables(self) -> list[State]:
        """Collect iteration variables owned by balances."""
        iter_vars = []

        for bal in self.balance_list:
            iter_vars.extend(bal.iteration_variables)

        return iter_vars

    def collect_all_iteration_variables(self) -> list[State]:
        """Collect component iteration variables followed by balance variables."""
        return (
            self.collect_iteration_variables()
            + self.collect_balance_iteration_variables()
        )

    def assign_iteration_values(self, iteration_values: list[float]) -> None:
        """
        Assign solver vector values back into State objects.

        The solver works with a numeric vector x. The network/components work
        with State objects. This method maps x -> State.value.
        """
        iter_var_list = self.collect_all_iteration_variables()

        if len(iteration_values) != len(iter_var_list):
            raise ValueError(
                f"Length mismatch: got {len(iteration_values)} iteration values "
                f"but expected {len(iter_var_list)}"
            )

        for val, var in zip(iteration_values, iter_var_list):
            var.value = val

    # ------------------------------------------------------------------
    # Iteration variable validation
    # ------------------------------------------------------------------

    def _validate_no_iteration_overlap(self) -> None:
        """
        Ensure balance variables do not duplicate component iteration variables.

        A State cannot be solved by both a component equation and a Balance.
        """
        comp_ids = {id(v) for v in self.collect_iteration_variables()}
        bal_ids = {id(v) for v in self.collect_balance_iteration_variables()}
        overlap_ids = comp_ids & bal_ids

        if overlap_ids:
            raise ValueError(self._format_iteration_overlap_error(overlap_ids))

    def _format_iteration_overlap_error(self, overlap_ids: set[int]) -> str:
        """Build a readable error message for duplicated iteration variables."""
        component_names = []
        balance_names = []

        for comp in self.component_list:
            comp_iter_ids = {id(v) for v in comp.iteration_variables}
            shared_ids = comp_iter_ids & overlap_ids

            if not shared_ids:
                continue

            for attr_name, attr_value in comp.__dict__.items():
                if id(attr_value) in shared_ids:
                    component_names.append(f"{comp.name}:{attr_name}")

        for bal in self.balance_list:
            bal_iter_ids = {id(v) for v in bal.iteration_variables}
            shared_ids = bal_iter_ids & overlap_ids

            if not shared_ids:
                continue

            for attr_name, attr_value in bal.__dict__.items():
                if id(attr_value) in shared_ids:
                    balance_names.append(f"{bal.name}:{attr_name}")

        component_names = sorted(set(component_names))
        balance_names = sorted(set(balance_names))

        lines = [
            "Iteration variable overlap detected.",
            "",
            "Balance iteration variables cannot be the same as component iteration variables.",
            "A State used as a Balance solve variable must not also appear in a Component iteration_variables list.",
            "",
            "Overlapping component variables:",
            *[f"  - {name}" for name in component_names],
            "",
            "Conflicting balance variables:",
            *[f"  - {name}" for name in balance_names],
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a simple text summary of the network."""
        lines = []

        lines.append(f"Network: {self.name}")
        lines.append(f"Components ({len(self.component_list)}):")

        for comp in self.component_list:
            lines.append(
                f"  ├─ [{comp.__class__.__name__}]: {comp.name}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Solution exporting
    # ------------------------------------------------------------------

    def save(
        self,
        filename: str | None = None,
        return_type: str = "dict",
    ):
        """
        Export component and balance state values.

        Supports:
            return_type="dict"
            return_type="dataframe"

        File output supports:
            .csv
            .json
            .xlsx / .xls
        """
        return_type = return_type.lower()

        records = []

        # --------------------------------------------------------------
        # Export component attributes.
        # --------------------------------------------------------------
        for comp in self.component_list:
            ignored_attributes = {"name", "network"} | comp.ignored_export_attributes

            for attr_name, attr_value in comp.__dict__.items():
                if attr_name in ignored_attributes or attr_name.startswith("_"):
                    continue

                # Composition-like attribute.
                if hasattr(attr_value, "fraction"):
                    if attr_value.is_assigned:
                        for species, state in attr_value:
                            try:
                                value = state.value
                            except Exception:
                                value = "<unavailable>"

                            records.append({
                                "component_name": comp.name,
                                "component_type": comp.__class__.__name__,
                                "attribute": f"{attr_name}.{species}",
                                "value": value,
                            })
                    else:
                        records.append({
                            "component_name": comp.name,
                            "component_type": comp.__class__.__name__,
                            "attribute": attr_name,
                            "value": "<uninitialized>",
                        })

                    continue

                # State-like attribute.
                if hasattr(attr_value, "is_assigned"):
                    if attr_value.is_assigned:
                        try:
                            value = attr_value.value
                        except Exception:
                            value = "<unavailable>"
                    else:
                        value = "<uninitialized>"

                # Plain Python attribute.
                else:
                    value = attr_value

                records.append({
                    "component_name": comp.name,
                    "component_type": comp.__class__.__name__,
                    "attribute": attr_name,
                    "value": value,
                })

        # --------------------------------------------------------------
        # Export balance attributes.
        # --------------------------------------------------------------
        for bal in self.balance_list:
            ignored_attributes = {"name", "network"}

            for attr_name, attr_value in bal.__dict__.items():
                if attr_name in ignored_attributes or attr_name.startswith("_"):
                    continue

                # Composition-like attribute.
                if hasattr(attr_value, "fraction"):
                    if attr_value.is_assigned:
                        for species, state in attr_value:
                            records.append({
                                "component_name": bal.name,
                                "component_type": bal.__class__.__name__,
                                "attribute": f"{attr_name}.{species}",
                                "value": state.value,
                            })
                    else:
                        records.append({
                            "component_name": bal.name,
                            "component_type": bal.__class__.__name__,
                            "attribute": attr_name,
                            "value": "<uninitialized>",
                        })

                    continue

                # State-like attribute.
                if hasattr(attr_value, "is_assigned"):
                    if attr_value.is_assigned:
                        try:
                            value = attr_value.value
                        except Exception:
                            value = "<unavailable>"
                    else:
                        value = "<uninitialized>"

                # Plain Python attribute.
                else:
                    value = attr_value

                records.append({
                    "component_name": bal.name,
                    "component_type": bal.__class__.__name__,
                    "attribute": attr_name,
                    "value": value,
                })

        # --------------------------------------------------------------
        # Export tracked states.
        # --------------------------------------------------------------
        for name, state in self.tracked_state_list:
            if state.is_assigned:
                try:
                    value = state.value
                except Exception:
                    value = "<unavailable>"
            else:
                value = "<uninitialized>"

            records.append({
                "component_name": self.name,
                "component_type": "TrackedState",
                "attribute": name,
                "value": value,
            })

            
        # --------------------------------------------------------------
        # Return object.
        # --------------------------------------------------------------
        if return_type == "dict":
            result = records

        elif return_type == "dataframe":
            import pandas as pd
            result = pd.DataFrame(records)

        else:
            raise ValueError("return_type must be 'dict' or 'dataframe'")

        # --------------------------------------------------------------
        # Optional file export.
        # --------------------------------------------------------------
        if filename is not None:
            ext = filename.split(".")[-1].lower()

            import pandas as pd
            df = pd.DataFrame(records)

            if ext == "csv":
                df.to_csv(filename, index=False)

            elif ext == "json":
                import json
                with open(filename, "w") as f:
                    json.dump(records, f, indent=4)

            elif ext in {"xlsx", "xls"}:
                df.to_excel(filename, index=False)

            else:
                raise ValueError(
                    "Unsupported file extension. Use .csv, .json, or .xlsx"
                )

        return result