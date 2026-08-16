"""B2 — kNN router. Selects the arm preferred by the k nearest training neighbours."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import Router


class KNNRouter(Router):
    """
    B2: For each test context, find k nearest training contexts and select
    the arm with the highest mean reward among those neighbours.

    Purpose: strong simple non-parametric offline router.
    Uses full-information training rewards to identify the locally best arm.
    """

    def __init__(
        self,
        n_arms: int,
        arm_names: Optional[list[str]] = None,
        k: int = 10,
        metric: str = "cosine",
    ):
        super().__init__(n_arms, arm_names)
        self.k = k
        self.metric = metric
        self._nn: Optional[NearestNeighbors] = None
        self._X_train: Optional[np.ndarray] = None
        self._R_train: Optional[np.ndarray] = None
        self._fitted_lam: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        lam: Optional[float] = None,
    ) -> "KNNRouter":
        """
        X       : (n_train, context_dim)
        rewards : (n_train, n_arms) full reward matrix on training data
        lam     : lambda used to compute rewards (stored for mismatch detection).
        """
        if rewards is None:
            raise ValueError("KNNRouter.fit() requires the full training reward matrix.")
        self._X_train = X
        self._R_train = rewards
        self._fitted_lam = lam
        self._nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric, algorithm="auto")
        self._nn.fit(X)
        return self

    def select(self, context: np.ndarray) -> int:
        if self._nn is None:
            raise RuntimeError("KNNRouter must be fitted before calling select().")
        context_2d = context.reshape(1, -1)
        _, indices = self._nn.kneighbors(context_2d)
        neighbour_rewards = self._R_train[indices[0]]  # (k, n_arms)
        mean_arm_rewards = np.nanmean(neighbour_rewards, axis=0)
        return int(np.argmax(mean_arm_rewards))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        # kNN index is fixed at fit time; preserved across reset.
