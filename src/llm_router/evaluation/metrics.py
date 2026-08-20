"""
Evaluation metrics for the dissertation.

All metrics operate on tidy result DataFrames with columns:
  t, router, seed, lam, action, quality, cost_norm, reward,
  oracle_action, oracle_reward, instant_regret, cumulative_regret
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-step metrics (applied at result construction time in evaluator.py)
# ---------------------------------------------------------------------------

def instant_regret(oracle_reward: np.ndarray, policy_reward: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, oracle_reward - policy_reward)


def cumulative_regret(inst_regret: np.ndarray) -> np.ndarray:
    return np.cumsum(inst_regret)


# ---------------------------------------------------------------------------
# Aggregate metrics from result DataFrame
# ---------------------------------------------------------------------------

def mean_quality(df: pd.DataFrame) -> float:
    return float(df["quality"].mean())


def mean_cost_norm(df: pd.DataFrame) -> float:
    return float(df["cost_norm"].mean())


def mean_utility(df: pd.DataFrame) -> float:
    return float(df["reward"].mean())


def total_regret(df: pd.DataFrame) -> float:
    return float(df["instant_regret"].sum())


def recovery_time(
    df: pd.DataFrame,
    shift_t: int,
    pre_shift_utility: float,
    threshold_fraction: float = 0.95,
) -> int:
    """
    Number of steps after shift_t until mean rolling utility (window=50)
    recovers to threshold_fraction * pre_shift_utility.

    Returns -1 if recovery is never achieved in the remaining stream.
    """
    post = df[df["t"] >= shift_t].copy()
    if post.empty:
        return -1
    target = threshold_fraction * pre_shift_utility
    rolling = post["reward"].rolling(window=50, min_periods=1).mean().values
    hits = np.where(rolling >= target)[0]
    if len(hits) == 0:
        return -1
    return int(hits[0])


def post_shift_regret(df: pd.DataFrame, shift_t: int, window: int = 200) -> float:
    """Mean regret in the `window` steps immediately following the shift point."""
    post = df[(df["t"] >= shift_t) & (df["t"] < shift_t + window)]
    if post.empty:
        return float("nan")
    return float(post["instant_regret"].mean())


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_resamples: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Returns (point_estimate, lower_ci, upper_ci) via percentile bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values)
    point = float(stat_fn(values))
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Oracle tie-aware routing shares
# ---------------------------------------------------------------------------

def oracle_tie_stats(
    R: np.ndarray,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute tie-aware Oracle routing shares.

    Under the naive `argmax(R, axis=1)` recipe, ties in R (very common when
    Q is binary and λ is small) are broken by numpy's stable first-index
    rule — so shares depend on the column ORDER of the model pool, not on
    any semantically meaningful preference. This distorts share reporting
    even though Oracle utility is unaffected (max is well-defined).

    This function instead assigns FRACTIONAL credit: for each row where k
    arms tie at the max reward, each tied arm receives 1/k of a vote.

    Parameters
    ----------
    R    : (n, K) reward matrix (typically Q - λ·C_norm).
    tol  : two arms are considered tied if |R_a - R_max| ≤ tol.

    Returns
    -------
    shares         : (K,) fractional Oracle routing share per arm — sums to 1.
    tie_rate       : fraction of rows with ≥ 2 tied arms at the max.
    ties_per_row   : (n,) integer count of tied arms per row (1 = no tie).
    """
    max_r = R.max(axis=1, keepdims=True)
    is_max = np.abs(R - max_r) <= tol
    ties_per_row = is_max.sum(axis=1)
    fractional = is_max / ties_per_row[:, None]  # (n, K), rows sum to 1
    shares = fractional.sum(axis=0) / R.shape[0]
    tie_rate = float((ties_per_row > 1).mean())
    return shares, tie_rate, ties_per_row


# ---------------------------------------------------------------------------
# Macro / micro averaging
# ---------------------------------------------------------------------------

def macro_mean(df: pd.DataFrame, metric: str, dataset_col: str = "dataset") -> float:
    """
    Mean of per-dataset means — equal weight per dataset regardless of size.

    This is the PRIMARY aggregation for dissertation results. It prevents
    large datasets (e.g. simpleqa with 4310 queries) from dominating the
    overall metric over small ones (e.g. aime with 56 queries).

    Requires df to contain a dataset column (join to outcomes if absent).
    """
    if dataset_col not in df.columns:
        raise ValueError(
            f"Column '{dataset_col}' not found. "
            "Join the result DataFrame with dataset labels from outcomes.parquet."
        )
    return float(df.groupby(dataset_col)[metric].mean().mean())


def micro_mean(df: pd.DataFrame, metric: str) -> float:
    """
    Mean over all rows — equal weight per query step.

    Secondary aggregation. Dominated by large datasets.
    """
    return float(df[metric].mean())


def macro_summary(
    df: pd.DataFrame,
    metrics: list[str],
    dataset_col: str = "dataset",
) -> dict[str, float]:
    """Return macro mean for each metric."""
    return {m: macro_mean(df, m, dataset_col) for m in metrics if m in df.columns}


def micro_summary(
    df: pd.DataFrame,
    metrics: list[str],
) -> dict[str, float]:
    """Return micro mean for each metric."""
    return {m: micro_mean(df, m) for m in metrics if m in df.columns}


# ---------------------------------------------------------------------------
# Summary table across seeds
# ---------------------------------------------------------------------------

def summarise_across_seeds(
    results: list[pd.DataFrame],
    metrics: list[str] | None = None,
    averaging: str = "micro",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    """
    Given a list of per-seed result DataFrames, compute mean ± std for each metric.

    Parameters
    ----------
    averaging : 'micro' (default) or 'macro'.
                Macro averaging requires a dataset column in each DataFrame.
    """
    if metrics is None:
        metrics = ["quality", "cost_norm", "reward", "instant_regret"]

    rows = []
    for df in results:
        if averaging == "macro" and dataset_col in df.columns:
            row = macro_summary(df, metrics, dataset_col)
        else:
            row = {m: df[m].mean() for m in metrics if m in df.columns}
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.agg(["mean", "std"])
