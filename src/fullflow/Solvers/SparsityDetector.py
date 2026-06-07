from __future__ import annotations

import numpy as np


class SparsityDetector:
    """
    Detect the Jacobian sparsity structure of a network residual system.

    This class empirically determines which residual equations depend on
    which iteration variables by perturbing each variable independently
    using symmetric 3-point finite differencing and observing which
    residuals change.

    The resulting sparsity matrix can later be supplied to
    scipy.optimize.least_squares via `jac_sparsity` to accelerate
    finite-difference Jacobian estimation and sparse linear solves.

    Detection Method
    ----------------
    For each iteration variable x_j:

        1) Evaluate residuals at:
               x + h e_j
               x - h e_j

        2) Compare both perturbed residual vectors against the baseline.

        3) Mark residual r_i as depending on x_j if either perturbation
           changes r_i by more than `residual_tolerance`.

    Notes
    -----
    - The detected sparsity is conservative:
      dependencies are only added, never removed.

    - Slightly overestimating sparsity is acceptable and safe.

    - Underestimating sparsity can break Jacobian estimation and solver
      convergence.

    Parameters
    ----------
    relative_step : float
        Relative perturbation size used for finite differencing.

        Perturbation magnitude is computed approximately as:

            h = relative_step * max(abs(x_j), 1.0)

        Larger values improve robustness to numerical noise but may
        incorrectly detect weak nonlinear couplings.

    absolute_step : float
        Minimum absolute perturbation size.

        Prevents extremely small perturbations when x_j is near zero.

    residual_tolerance : float
        Threshold used to determine whether a residual changed after a
        perturbation.

        Residual changes smaller than this value are treated as numerical
        noise and ignored.
    """
    def __init__(self, network):
        self.network = network

    def residual(self, x: np.ndarray) -> np.ndarray:
        self.network.assign_iteration_values(list(x))
        self.network.evaluate_states()
        return np.array(self.network.residuals, dtype=float)

    def detect(
        self,
        relative_step: float = 1e-6,
        absolute_step: float = 1e-8,
        residual_tolerance: float = 1e-10,
    ) -> np.ndarray:

        x0 = np.array(
            self.network.iteration_values,
            dtype=float,
        )

        self.network.evaluate_states()

        r0 = np.array(
            self.network.residuals,
            dtype=float,
        )

        n_residuals = len(r0)
        n_variables = len(x0)

        sparsity = np.zeros(
            (n_residuals, n_variables),
            dtype=bool,
        )

        for j in range(n_variables):

            h = relative_step * max(abs(x0[j]), 1.0)
            h = max(h, absolute_step)

            x_plus = x0.copy()
            x_minus = x0.copy()

            x_plus[j] += h
            x_minus[j] -= h

            r_plus = self.residual(x_plus)
            r_minus = self.residual(x_minus)

            changed_plus = np.abs(r_plus - r0) > residual_tolerance
            changed_minus = np.abs(r_minus - r0) > residual_tolerance

            sparsity[:, j] = changed_plus | changed_minus

        self.network.assign_iteration_values(list(x0))
        self.network.evaluate_states()

        return sparsity

    def _iteration_variable_labels(self) -> list[str]:

        labels = []

        def find_label(target):
            for component in self.network.component_list:
                for attr_name, attr_value in component.__dict__.items():
                    if attr_value is target:
                        return f"{component.name}.{attr_name}"

            for balance in self.network.balance_list:
                for attr_name, attr_value in balance.__dict__.items():
                    if attr_value is target:
                        return f"{balance.name}.{attr_name}"

            return str(target)

        for var in self.network.collect_all_iteration_variables():
            labels.append(find_label(var))

        return labels


    def _residual_labels(self) -> list[str]:

        labels = []

        for component in self.network.component_list:
            component_residuals = component.residuals

            if isinstance(component_residuals, (list, tuple)):
                for i in range(len(component_residuals)):
                    labels.append(f"{component.name}.residual[{i}]")
            else:
                labels.append(f"{component.name}.residual")

        for balance in self.network.balance_list:
            balance_residuals = balance.residuals

            if isinstance(balance_residuals, (list, tuple)):
                for i in range(len(balance_residuals)):
                    labels.append(f"{balance.name}.residual[{i}]")
            else:
                labels.append(f"{balance.name}.residual")

        return labels

    def plot(
        self,
        sparsity: np.ndarray,
        figsize: tuple = (14, 10),
        filename: str = "jacobian_sparsity.png",
    ) -> None:

        import matplotlib.pyplot as plt
        import numpy as np

        variable_labels = self._iteration_variable_labels()
        residual_labels = self._residual_labels()

        fig, ax = plt.subplots(figsize=figsize)

        ax.spy(
            sparsity,
            markersize=8,
            color="#D84135",
        )

        ax.set_title(
            "Jacobian Sparsity Pattern",
            fontsize=16,
            pad=20,
        )

        ax.set_xlabel("Iteration Variables")
        ax.set_ylabel("Residuals")

        ax.set_xticks(np.arange(len(variable_labels)))
        ax.set_yticks(np.arange(len(residual_labels)))

        ax.set_xticklabels(
            [f"x[{i}]" for i in range(len(variable_labels))],
            rotation=90,
            fontsize=8,
        )

        ax.set_yticklabels(
            [
                f"r[{i}] {label}"
                for i, label in enumerate(residual_labels)
            ],
            fontsize=8,
        )

        # Gridlines centered on each matrix entry.
        ax.set_xticks(
            np.arange(-0.5, len(variable_labels), 1),
            minor=True,
        )

        ax.set_yticks(
            np.arange(-0.5, len(residual_labels), 1),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="#2a2a40",
            linestyle="-",
            linewidth=0.5,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        legend_text = "\n".join(
            [
                f"x[{i}] = {label}"
                for i, label in enumerate(variable_labels)
            ]
        )

        fig.text(
            1.02,
            0.5,
            legend_text,
            fontsize=8,
            va="center",
            family="monospace",
        )

        plt.tight_layout()

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
