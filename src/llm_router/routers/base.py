"""
Abstract base class for all routing policies.

Scientific constraint: update() must NEVER receive outcomes for arms other
than the one that was actually selected. Violating this gives an offline
router access to bandit-inaccessible information and invalidates the comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Router(ABC):
    """Common interface for all routing policies (B0–B6)."""

    # Subclasses set this to True if they may access full outcome matrix.
    # Only Oracle should set this to True. Used in leakage checks.
    REQUIRES_FULL_INFORMATION: bool = False

    def __init__(self, n_arms: int, arm_names: Optional[list[str]] = None):
        self.n_arms = n_arms
        self.arm_names = arm_names or [str(i) for i in range(n_arms)]
        self._t = 0  # internal time step counter

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        lam: Optional[float] = None,
    ) -> "Router":
        """
        Offline pre-training (if applicable).

        Parameters
        ----------
        X           : (n_queries, context_dim) context matrix from TRAINING data only
        rewards     : (n_queries, n_arms) full reward matrix — ONLY for offline
                      policies (B1, B2, B3). Online policies (B4, B5) must ignore
                      this or raise if accidentally passed training rewards they
                      should not use.
        cost_matrix : (n_queries, n_arms) raw cost matrix — required by B0.
        lam         : lambda used to compute rewards = quality - lam*cost_norm.
                      Offline policies (B1, B2, B3) must store this so the
                      evaluator can detect lambda mismatches at run time.
        """

    @abstractmethod
    def select(self, context: np.ndarray) -> int:
        """
        Select an arm given a context vector.

        Parameters
        ----------
        context : (context_dim,) feature vector for the current query

        Returns
        -------
        arm index in [0, n_arms)
        """

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Update internal state from ONE observed (context, action, reward) triple.

        CRITICAL: the reward is ONLY for the selected action.
        Reward for other arms must NOT be passed or accessed here.
        Offline/static policies implement this as a no-op.
        """

    def reset(self) -> None:
        """Reset internal state to allow fresh re-runs with new seeds."""
        self._t = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_arms={self.n_arms})"
