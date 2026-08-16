"""
B5 — Discounted LinUCB (D-LinUCB) for non-stationary contextual bandits.

Faithful implementation of:
  Russac, Y., Vernade, C., & Cappé, O. (2019).
  "Weighted Linear Bandits for Non-Stationary Environments."
  NeurIPS 2019. https://arxiv.org/abs/1909.09595

Algorithm (per arm, disjoint formulation):
  Maintain per arm:
    V   : discounted gram matrix,        V_0 = reg_lambda * I
    V~  : doubly-discounted gram matrix, V~_0 = reg_lambda * I
    b   : discounted reward accumulator, b_0 = 0

  When arm a is selected at step t with context x_t, reward r_t:
    V_a   ← γ  · V_a   + x_t x_t^T
    V~_a  ← γ² · V~_a  + x_t x_t^T
    b_a   ← γ  · b_a   + r_t · x_t

  Parameter estimate:  θ̂_a = V_a^{-1} b_a
  Uncertainty (Russac eq. 4):  σ_a(x) = sqrt( x^T V_a^{-1} V~_a V_a^{-1} x )
  Score:  x^T θ̂_a + α · σ_a(x)

Reduction property: when γ = 1.0, V~ = V at all times (identical updates),
so σ_a(x) = sqrt( x^T V_a^{-1} x ), which recovers standard LinUCB (B4) exactly.

Only the SELECTED arm's matrices are updated per step; unselected arms retain
their matrices unchanged (equivalently, their forgetting is implicit: old
observations carry a smaller weight relative to new ones accumulating in V).

The discount factor γ MUST be tuned on validation streams only.
Do NOT select γ by evaluating on test data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Router


class DiscountedLinUCBRouter(Router):
    """B5: Disjoint Discounted LinUCB (Russac et al. 2019)."""

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        arm_names: Optional[list[str]] = None,
        alpha: float = 1.0,
        gamma: float = 0.99,
        reg_lambda: float = 1.0,
    ):
        super().__init__(n_arms, arm_names)
        self.context_dim = context_dim
        self.alpha = alpha
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self._init_matrices()

    def _init_matrices(self) -> None:
        d = self.context_dim
        lam = self.reg_lambda
        self._V:      list[np.ndarray] = [lam * np.eye(d) for _ in range(self.n_arms)]
        self._V_tilde: list[np.ndarray] = [lam * np.eye(d) for _ in range(self.n_arms)]
        self._b:      list[np.ndarray] = [np.zeros(d)    for _ in range(self.n_arms)]
        self._V_inv:  list[Optional[np.ndarray]] = [np.eye(d) / lam] * self.n_arms
        self._dirty:  list[bool] = [False] * self.n_arms

    def _get_V_inv(self, a: int) -> np.ndarray:
        if self._dirty[a]:
            self._V_inv[a] = np.linalg.inv(self._V[a])
            self._dirty[a] = False
        return self._V_inv[a]

    def fit(
        self,
        X: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        lam: Optional[float] = None,
    ) -> "DiscountedLinUCBRouter":
        if rewards is not None:
            raise ValueError(
                "DiscountedLinUCBRouter is an online policy. "
                "Do not pass training rewards to fit() — full-information leakage."
            )
        return self

    def select(self, context: np.ndarray) -> int:
        x = context.reshape(-1)
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            V_inv = self._get_V_inv(a)
            theta = V_inv @ self._b[a]
            # Russac et al. eq. 4: σ = sqrt( x^T V^{-1} V~ V^{-1} x )
            sigma = np.sqrt(max(0.0, x @ V_inv @ self._V_tilde[a] @ V_inv @ x))
            scores[a] = x @ theta + self.alpha * sigma
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """
        Update only the selected arm.

        V_a   ← γ  · V_a   + x x^T
        V~_a  ← γ² · V~_a  + x x^T
        b_a   ← γ  · b_a   + r · x
        """
        x = context.reshape(-1)
        g  = self.gamma
        g2 = g * g
        self._V[action]       = g  * self._V[action]       + np.outer(x, x)
        self._V_tilde[action] = g2 * self._V_tilde[action] + np.outer(x, x)
        self._b[action]       = g  * self._b[action]       + reward * x
        self._dirty[action]   = True
        self._t += 1

    def reset(self) -> None:
        super().reset()
        self._init_matrices()
