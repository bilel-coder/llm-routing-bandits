"""
B6 — Oracle router (theoretical upper bound).

IMPORTANT: the Oracle is NOT a deployable router.
It accesses the full hidden outcome matrix at every step and selects
the arm with the highest true reward.

It exists ONLY to compute the theoretical regret ceiling. All regret
calculations compare other policies against Oracle performance.

This class is intentionally segregated and must never be used as a
router in production-evaluation code paths. The evaluator enforces
this via Router.REQUIRES_FULL_INFORMATION.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Router


class OracleRouter(Router):
    """
    B6: Omniscient oracle that always picks the best arm.
    Requires full reward matrix at each step — forbidden for deployable policies.
    """

    REQUIRES_FULL_INFORMATION: bool = True

    def __init__(self, n_arms: int, arm_names: Optional[list[str]] = None):
        super().__init__(n_arms, arm_names)
        self._current_rewards: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        cost_matrix: Optional[np.ndarray] = None,
        lam: Optional[float] = None,
    ) -> "OracleRouter":
        return self

    def set_step_rewards(self, rewards_all_arms: np.ndarray) -> None:
        """
        Provide the full reward vector for the CURRENT step.
        Called by the evaluator before select(); not accessible to other policies.
        """
        self._current_rewards = rewards_all_arms

    def select(self, context: np.ndarray) -> int:
        if self._current_rewards is None:
            raise RuntimeError("OracleRouter: call set_step_rewards() before select().")
        return int(np.argmax(self._current_rewards))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        # Oracle doesn't learn — it already knows everything.
        self._current_rewards = None  # Clear for safety

    def reset(self) -> None:
        super().reset()
        self._current_rewards = None
