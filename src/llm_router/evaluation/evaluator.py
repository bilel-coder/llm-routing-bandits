"""
Simulation evaluator: runs a routing policy over a stream of (context, reward) pairs.

Scientific constraints enforced here:
1. Online policies only receive reward for their selected arm.
2. Oracle receives full reward vector via set_step_rewards() — this call path
   is blocked for all non-Oracle policies.
3. The hidden reward matrix R[t, a] is NEVER passed to policy.update().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from llm_router.routers.base import Router
from llm_router.routers.oracle import OracleRouter
from llm_router.evaluation.metrics import instant_regret, cumulative_regret

logger = logging.getLogger(__name__)


@dataclass
class StreamResult:
    """Tidy per-step results for one (router, seed, lambda) run."""
    experiment: str = ""
    scenario: str = ""
    seed: int = 0
    lam: float = 0.0
    router: str = ""
    records: list[dict] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def run_stream(
    router: Router,
    X: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    C_norm: np.ndarray,
    oracle: OracleRouter,
    lam: float,
    seed: int,
    scenario: str = "S0",
    experiment: str = "",
    shift_schedule: Optional[dict] = None,
) -> StreamResult:
    """
    Run a single routing policy over a query stream.

    Parameters
    ----------
    router    : policy to evaluate (must implement Router interface)
    X         : (T, d) context matrix — all time steps
    R         : (T, K) reward matrix R[t,a] = quality[t,a] - lam*cost_norm[t,a]
    Q         : (T, K) quality matrix
    C_norm    : (T, K) normalised cost matrix
    oracle    : OracleRouter instance (for regret computation)
    lam       : cost-quality trade-off parameter
    seed      : random seed used to build the stream (stored in output)
    scenario  : scenario identifier (S0/S1/S2/S3)
    experiment: experiment label
    shift_schedule : optional dict mapping time step → modification callable
                     (used by environment modules to inject shifts)
    """
    if router.REQUIRES_FULL_INFORMATION and not isinstance(router, OracleRouter):
        raise ValueError(
            f"{router} sets REQUIRES_FULL_INFORMATION=True but is not an OracleRouter. "
            "This would constitute illegal full-information access."
        )

    # Lambda lifecycle check: offline routers store the lambda used during fit().
    # If a router is reused with a different lambda the reward signal is mismatched.
    fitted_lam = getattr(router, "_fitted_lam", None)
    if fitted_lam is not None and abs(fitted_lam - lam) > 1e-9:
        raise ValueError(
            f"{router.__class__.__name__} was fitted with lam={fitted_lam} but "
            f"run_stream() was called with lam={lam}. "
            "Re-fit the router with the correct lambda before evaluation."
        )

    T, K = R.shape
    result = StreamResult(
        experiment=experiment,
        scenario=scenario,
        seed=seed,
        lam=lam,
        router=router.__class__.__name__,
    )

    cumreg = 0.0

    for t in range(T):
        x_t = X[t]

        # Apply any scheduled environment modifications (shifts)
        if shift_schedule and t in shift_schedule:
            R, Q, C_norm = shift_schedule[t](t, R, Q, C_norm)

        # Oracle selection (uses full reward vector — evaluator only)
        oracle.set_step_rewards(R[t])
        oracle_action = oracle.select(x_t)
        oracle_reward = float(R[t, oracle_action])

        # Policy selection (context only — no reward access)
        action = router.select(x_t)

        # Observe reward for selected arm only
        reward = float(R[t, action])
        quality = float(Q[t, action])
        cost_n = float(C_norm[t, action])

        # Regret
        inst_reg = float(max(0.0, oracle_reward - reward))
        cumreg += inst_reg

        result.records.append({
            "t": t,
            "action": action,
            "oracle_action": oracle_action,
            "quality": quality,
            "cost_norm": cost_n,
            "reward": reward,
            "oracle_reward": oracle_reward,
            "instant_regret": inst_reg,
            "cumulative_regret": cumreg,
        })

        # Online update — ONLY selected arm reward is passed
        router.update(x_t, action, reward)

    return result
