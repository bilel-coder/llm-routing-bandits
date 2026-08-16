"""B0 — Cheapest router. Always selects the arm with lowest mean training cost."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Router


class CheapestRouter(Router):
    """
    B0: Always select the arm with the lowest mean training cost.

    Purpose: cost lower bound / trivial static baseline.
    Requires the raw cost matrix at fit() time to identify the cheapest arm.
    """

    def __init__(self, n_arms: int, arm_names: Optional[list[str]] = None):
        super().__init__(n_arms, arm_names)
        self._cheapest_arm: int = 0

    def fit(
        self,
        X: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        lam: Optional[float] = None,
    ) -> "CheapestRouter":
        """
        cost_matrix : (n_train, n_arms) array of raw training costs.
                      Required — raises if absent.
        """
        if cost_matrix is None:
            raise ValueError(
                "CheapestRouter.fit() requires cost_matrix=(n_train, n_arms). "
                "Pass the raw (not normalised) training cost matrix."
            )
        mean_costs = np.nanmean(cost_matrix, axis=0)
        self._cheapest_arm = int(np.argmin(mean_costs))
        return self

    def select(self, context: np.ndarray) -> int:
        return self._cheapest_arm

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass

    def reset(self) -> None:
        super().reset()
