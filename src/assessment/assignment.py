"""Minimum-cost target association used by ArmorTracker."""
from __future__ import annotations
import numpy as np


def hungarian_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost rectangular assignment, no SciPy dependency, O(rows^2*cols)."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("cost must be a finite matrix")
    rows, cols = cost.shape
    if not rows or not cols:
        return []
    if rows > cols:
        return [(j, i) for i, j in hungarian_assignment(cost.T)]
    u, v = np.zeros(rows + 1), np.zeros(cols + 1)
    p, way = np.zeros(cols + 1, dtype=int), np.zeros(cols + 1, dtype=int)
    for i in range(1, rows + 1):
        p[0] = i
        minv, used = np.full(cols + 1, np.inf), np.zeros(cols + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(int(p[j] - 1), j - 1) for j in range(1, cols + 1) if p[j]]
