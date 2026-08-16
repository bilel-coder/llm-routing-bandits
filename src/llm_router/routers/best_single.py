"""B1 — Best-single-model router. Selects the arm with highest mean training utility."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Router


class BestSingleRouter(Router):
    """
    B1: Identify the single LLM with the highest mean utility on TRAINING data
    and always select it, regardless of context.

    Purpose: strong static baseline.
    The selected arm must be determined from training data only.
    """

    def __init__(self, n_arms: int, arm_names: Optional[list[str]] = None):
        super().__init__(n_arms, arm_names)
        self._best_arm: int = 0
        self._fitted_lam: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        lam: Optional[float] = None,
    ) -> "BestSingleRouter":
        """
        rewards : (n_train_queries, n_arms) full reward matrix on training data.
        lam     : lambda used to compute rewards (stored for mismatch detection).
        """
        if rewards is None:
            raise ValueError("BestSingleRouter.fit() requires the full training reward matrix.")
        mean_rewards = np.nanmean(rewards, axis=0)
        self._best_arm = int(np.argmax(mean_rewards))
        self._fitted_lam = lam
        return self

    def select(self, context: np.ndarray) -> int:
        return self._best_arm

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        # Best arm is fixed at fit time; preserved across reset.
