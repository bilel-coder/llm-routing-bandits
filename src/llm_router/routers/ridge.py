"""B3 — Per-arm Ridge regression router."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from .base import Router


class RidgeRouter(Router):
    """
    B3: For each arm a, train a Ridge regressor:

        r_hat_a(x) = theta_a^T x

    using FULL-INFORMATION training rewards (all arms observed for each query).
    At inference, select argmax_a r_hat_a(x_t).

    Purpose: matched linear offline baseline. Uses the same linear reward-model
    family as LinUCB but receives stronger full-information offline supervision.
    """

    def __init__(
        self,
        n_arms: int,
        arm_names: Optional[list[str]] = None,
        alpha: float = 1.0,
    ):
        super().__init__(n_arms, arm_names)
        self.alpha = alpha
        self._models: list[Optional[Ridge]] = [None] * n_arms
        self._fitted_lam: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        lam: Optional[float] = None,
    ) -> "RidgeRouter":
        """
        X       : (n_train, context_dim)
        rewards : (n_train, n_arms) full reward matrix on training data
        lam     : lambda used to compute rewards (stored for mismatch detection).
        """
        if rewards is None:
            raise ValueError("RidgeRouter.fit() requires the full training reward matrix.")

        for a in range(self.n_arms):
            y = rewards[:, a]
            mask = ~np.isnan(y)
            if mask.sum() == 0:
                continue
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X[mask], y[mask])
            self._models[a] = model
        self._fitted_lam = lam
        return self

    def select(self, context: np.ndarray) -> int:
        x = context.reshape(1, -1)
        scores = np.full(self.n_arms, -np.inf)
        for a, model in enumerate(self._models):
            if model is not None:
                scores[a] = model.predict(x)[0]
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        # Models fixed at fit time; preserved across reset.
