"""Internal, fixed-per-solve scaling of nonlinear residual equations.

The physical residuals remain unchanged. Scaling is a numerical preconditioner
based on the Jacobian with respect to magnitude-scaled iteration variables.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize._numdiff import approx_derivative


class EquationScaling:
    """Scale unnormalized equation rows without requiring component metadata."""

    def __init__(self, mask: np.ndarray, scales: np.ndarray | None = None):
        self.mask = np.asarray(mask, dtype=bool)
        self.scales: np.ndarray | None = None
        self.reused = False
        if scales is not None:
            values = np.asarray(scales, dtype=float)
            if (values.shape == self.mask.shape and np.all(np.isfinite(values))
                    and np.all(values > 0.0)):
                self.scales = values.copy()
                self.reused = True

    @classmethod
    def transient(cls, cache, residual_count: int, scales=None):
        """Exclude already-normalized backward-Euler integration equations."""
        mask = np.ones(residual_count, dtype=bool)
        dynamic_rows = tuple(cache.transient_items)
        if len(dynamic_rows) > residual_count:
            raise ValueError("Inconsistent transient residual layout.")
        for i, item in enumerate(dynamic_rows):
            mask[i] = bool(item.force_steady)
        return cls(mask, scales)

    def fit(self, jacobian_scaled_variables: np.ndarray) -> np.ndarray:
        """Fit once to the raw Jacobian, then keep row scales fixed."""
        matrix = np.asarray(jacobian_scaled_variables, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.mask):
            raise ValueError("Jacobian dimensions do not match equation rows.")
        if self.scales is None:
            row_size = np.max(np.abs(matrix), axis=1, initial=0.0)
            scales = np.ones(len(self.mask), dtype=float)
            valid = self.mask & np.isfinite(row_size) & (row_size >= 1.0e-10)
            typical = float(np.median(row_size[valid])) if np.any(valid) else 1.0
            largest = min(1.0e12, max(1.0e-10, typical * 1000.0))
            scales[valid] = np.clip(row_size[valid], 1.0e-10, largest)
            self.scales = scales
        return matrix / self.scales[:, None]

    def residual(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=float)
        return values if self.scales is None else values / self.scales

    def raw(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=float)
        return values if self.scales is None else values * self.scales

    def initialize(self, fun, x: np.ndarray, method: str, *, bounds=None):
        """Build scales only if the fast solver did not already build them.

        Return the first raw Jacobian so SciPy can reuse it at its initial point.
        """
        if self.scales is not None:
            return None
        x = np.asarray(x, dtype=float)
        if not len(x):
            self.scales = np.ones(len(self.mask))
            return None
        kw = {}
        if bounds is not None:
            kw['bounds'] = bounds
        raw_jac = np.atleast_2d(np.asarray(approx_derivative(fun, x, method=method, **kw), dtype=float))
        self.fit(raw_jac * np.maximum(np.abs(x), 1.0)[None, :])
        return raw_jac
