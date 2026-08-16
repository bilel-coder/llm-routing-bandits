"""
B4 — LinUCB (stationary contextual bandit).

Reference: Li et al. (2010) "A Contextual-Bandit Approach to Personalized
News Article Recommendation", WWW 2010.
https://arxiv.org/abs/1003.0146

Per-arm disjoint formulation. Each arm maintains:
  A_a  : (d x d) regularised gram matrix, initialised to I_d
  b_a  : (d,) reward-context accumulator, initialised to 0

At each step:
  theta_hat_a = A_a^{-1} b_a
  score_a = x^T theta_hat_a + alpha * sqrt(x^T A_a^{-1} x)
  a_t = argmax_a score_a

Update (only for selected arm):
  A_{a_t} += x_t x_t^T
  b_{a_t} += r_t x_t

Online policies NEVER receive rewards for unselected arms.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

from .base import Router


class LinUCBRouter(Router):
    """B4: Disjoint LinUCB for stationary contextual bandits."""

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        arm_names: Optional[list[str]] = None,
        alpha: float = 1.0,
    ):
        super().__init__(n_arms, arm_names)
        self.context_dim = context_dim
        self.alpha = alpha
        self._init_matrices()

    def _init_matrices(self) -> None:
        d = self.context_dim
        self._A: list[np.ndarray] = [np.eye(d) for _ in range(self.n_arms)]
        self._b: list[np.ndarray] = [np.zeros(d) for _ in range(self.n_arms)]
        # Cache inverses; recomputed lazily when A changes
        self._A_inv: list[Optional[np.ndarray]] = [np.eye(d)] * self.n_arms
        self._dirty: list[bool] = [False] * self.n_arms

    def _get_A_inv(self, a: int) -> np.ndarray:
        if self._dirty[a]:
            self._A_inv[a] = np.linalg.inv(self._A[a])
            self._dirty[a] = False
        return self._A_inv[a]

    def fit(
        self,
        X: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        lam: Optional[float] = None,
    ) -> "LinUCBRouter":
        if rewards is not None:
            raise ValueError(
                "LinUCBRouter is an online policy. "
                "Do not pass training rewards to fit() — "
                "this would constitute full-information leakage."
            )
        return self

    def select(self, context: np.ndarray) -> int:
        x = context.reshape(-1)
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = self._get_A_inv(a)
            theta = A_inv @ self._b[a]
            ucb = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[a] = x @ theta + ucb
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Update ONLY the selected arm with the observed reward."""
        x = context.reshape(-1)
        self._A[action] += np.outer(x, x)
        self._b[action] += reward * x
        self._dirty[action] = True
        self._t += 1

    def reset(self) -> None:
        super().reset()
        self._init_matrices()
