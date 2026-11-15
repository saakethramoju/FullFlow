"""Verified finite-difference Jacobian coloring for generic nonlinear systems."""

from __future__ import annotations

import numpy as np


class JacobianStructure:
    """Validated numerical sparsity learned from full finite differences."""

    def __init__(self):
        self.groups = ()
        self.supports = ()
        self.shape = None
        self.dense_builds = 0
        self.colored_builds = 0
        self.rejections = 0

    def observe(self, jac):
        rows, columns = jac.shape
        magnitude = np.max(np.abs(jac), axis=0)
        support = np.abs(jac) > (np.maximum(magnitude, 1e-15) * 1e-9)[None, :]
        groups = []
        occupied = []
        for column in sorted(range(columns), key=lambda j: -int(np.count_nonzero(support[:, j]))):
            col_support = support[:, column]
            for group, used in zip(groups, occupied):
                if not np.any(used & col_support):
                    group.append(column)
                    used |= col_support
                    break
            else:
                groups.append([column])
                occupied.append(col_support.copy())
        probes = max(2, (columns - 1).bit_length()) + 1
        if not np.any(support) or len(groups) * 2 + probes >= columns * 2:
            self.groups, self.supports = (), ()
        else:
            self.groups = tuple(np.asarray(group, dtype=np.intp) for group in groups)
            self.supports = tuple(np.flatnonzero(support[:, column]) for column in range(columns))
        self.shape = jac.shape

    def build(self, current_y, current_f, evaluate, eps, method):
        if not self.groups or self.shape != (len(current_f), len(current_y)):
            return None
        probe_count = max(2, (len(current_y) - 1).bit_length()) + 1
        multiplier = 2 if method == "3-point" else 1
        if multiplier * len(self.groups) + probe_count >= multiplier * len(current_y):
            return None
        jac = np.zeros(self.shape)
        step = eps * np.maximum(np.abs(current_y), 1.0)
        for group in self.groups:
            positive = current_y.copy()
            positive[group] += step[group]
            plus = evaluate(positive)
            if method == "3-point":
                negative = current_y.copy()
                negative[group] -= step[group]
                difference = (plus - evaluate(negative)) * 0.5
            else:
                difference = plus - current_f
            for column in group:
                rows = self.supports[column]
                jac[rows, column] = difference[rows] / step[column]
        indices = np.arange(len(current_y), dtype=np.intp)
        for probe in range(probe_count):
            if probe == probe_count - 1:
                direction = np.random.default_rng(17).choice((-1.0, 1.0), len(current_y))
            else:
                direction = np.where((indices >> probe) & 1, 1.0, -1.0)
            delta = step * direction
            actual = evaluate(current_y + delta) - current_f
            expected = jac @ delta
            error = float(np.linalg.norm(actual - expected))
            reference = max(float(np.linalg.norm(actual)), float(np.linalg.norm(expected)))
            if not np.isfinite(error) or error > 1e-3 * reference + 1e-13 * max(1.0, float(np.linalg.norm(current_f))):
                self.rejections += 1
                self.groups = ()
                return None
        self.colored_builds += 1
        return jac
