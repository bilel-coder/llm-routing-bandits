"""
Stream generators for the four experimental scenarios.

Interface: each generator produces a `Stream` — a deterministic ordering of
query indices plus (optionally) a shift schedule that modifies the Q/C/C_norm
matrices at specified time steps.

Determinism guarantee: `make_stream(seed=s)` returns the same permutation
and same shift_schedule every time it is called with the same seed and same
underlying matrices. This ensures all routers under one seed see the same
stream, and reruns are reproducible.

Scientific note (S3 cost drift): the frozen CostNormaliser scalar is NEVER
refit during a drift. Instead, C is multiplied by per-arm drift factors and
C_norm is recomputed with the same frozen scalar. Post-drift C_norm can
therefore exceed 1.0 by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Type alias for shift callables passed to the evaluator's shift_schedule.
# Signature: (t, R, Q, C_norm) -> (R', Q', C_norm')
ShiftFn = Callable[[int, np.ndarray, np.ndarray, np.ndarray],
                   tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class Stream:
    """
    A deterministic query stream + optional matrix modification schedule.

    Attributes
    ----------
    order          : (T,) permutation of row indices into the source matrices.
                     The stream visits row `order[t]` at step t.
    shift_schedule : optional dict[t → ShiftFn]. Evaluator applies fn(t, R,Q,C_n)
                     BEFORE selecting the arm at step t.
    scenario       : label ("S0", "S1", "S2", "S3").
    seed           : the seed used to build this stream (for provenance).
    metadata       : any extra info (change points, mixtures, drift factors).
    """
    order: np.ndarray
    shift_schedule: dict[int, ShiftFn] = field(default_factory=dict)
    scenario: str = "S0"
    seed: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# S0 — Stationary
# ---------------------------------------------------------------------------

def make_S0_stream(n: int, seed: int) -> Stream:
    """
    Stationary reference: deterministic random permutation of the n queries.
    No shift schedule.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return Stream(order=order, shift_schedule={}, scenario="S0", seed=seed)


# ---------------------------------------------------------------------------
# S1 — Abrupt covariate shift (dataset mixture change)
# ---------------------------------------------------------------------------

def make_S1_stream(
    datasets: np.ndarray,
    seed: int,
    pre_shift_datasets: list[str],
    post_shift_datasets: list[str],
    shift_frac: float = 0.5,
    length: Optional[int] = None,
) -> Stream:
    """
    Abrupt covariate shift: at t = int(length * shift_frac) the dataset
    mixture switches from `pre_shift_datasets` to `post_shift_datasets`.

    Only P(x) changes — the outcome mapping (Q, C) is untouched.

    Parameters
    ----------
    datasets            : (n,) dataset label per row of the source matrices.
    pre_shift_datasets  : datasets sampled uniformly for t < shift point.
    post_shift_datasets : datasets sampled uniformly for t >= shift point.
    shift_frac          : where the shift occurs, as a fraction of length.
    length              : stream length (default = len(datasets)).
    """
    n = len(datasets)
    T = length if length is not None else n
    change_point = int(round(T * shift_frac))
    rng = np.random.default_rng(seed)

    idx_by_ds = {ds: np.where(datasets == ds)[0] for ds in np.unique(datasets)}
    for ds in pre_shift_datasets + post_shift_datasets:
        if ds not in idx_by_ds or len(idx_by_ds[ds]) == 0:
            raise ValueError(f"S1: dataset '{ds}' not present in source matrix")

    def sample_from(dslist: list[str], k: int) -> np.ndarray:
        # Uniform mixture over the listed datasets, sampling WITH replacement
        # (streams are conceptually infinite; sampling with replacement lets
        # the stream length exceed any single dataset's size).
        pool = np.concatenate([idx_by_ds[d] for d in dslist])
        return rng.choice(pool, size=k, replace=True)

    pre  = sample_from(pre_shift_datasets,  change_point)
    post = sample_from(post_shift_datasets, T - change_point)
    order = np.concatenate([pre, post])

    return Stream(
        order=order,
        shift_schedule={},   # no matrix modification; only P(x) shifts
        scenario="S1",
        seed=seed,
        metadata={
            "change_point": change_point,
            "pre_shift_datasets": pre_shift_datasets,
            "post_shift_datasets": post_shift_datasets,
            "shift_frac": shift_frac,
            "length": T,
        },
    )


# ---------------------------------------------------------------------------
# S2 — Gradual covariate shift
# ---------------------------------------------------------------------------

def make_S2_stream(
    datasets: np.ndarray,
    seed: int,
    pre_shift_datasets: list[str],
    post_shift_datasets: list[str],
    shift_start_frac: float = 0.3,
    shift_end_frac: float = 0.7,
    length: Optional[int] = None,
) -> Stream:
    """
    Gradual covariate shift: the mixture weight on post_shift_datasets
    interpolates linearly from 0 at t=shift_start to 1 at t=shift_end.

    Only P(x) changes — Q and C are untouched.
    """
    n = len(datasets)
    T = length if length is not None else n
    t_start = int(round(T * shift_start_frac))
    t_end   = int(round(T * shift_end_frac))
    rng = np.random.default_rng(seed)

    idx_by_ds = {ds: np.where(datasets == ds)[0] for ds in np.unique(datasets)}
    for ds in pre_shift_datasets + post_shift_datasets:
        if ds not in idx_by_ds or len(idx_by_ds[ds]) == 0:
            raise ValueError(f"S2: dataset '{ds}' not present in source matrix")

    pre_pool  = np.concatenate([idx_by_ds[d] for d in pre_shift_datasets])
    post_pool = np.concatenate([idx_by_ds[d] for d in post_shift_datasets])

    order = np.empty(T, dtype=np.int64)
    for t in range(T):
        if t < t_start:
            w_post = 0.0
        elif t >= t_end:
            w_post = 1.0
        else:
            w_post = (t - t_start) / max(1, (t_end - t_start))
        pool = post_pool if rng.random() < w_post else pre_pool
        order[t] = rng.choice(pool)

    return Stream(
        order=order,
        shift_schedule={},
        scenario="S2",
        seed=seed,
        metadata={
            "shift_start": t_start,
            "shift_end": t_end,
            "pre_shift_datasets": pre_shift_datasets,
            "post_shift_datasets": post_shift_datasets,
            "length": T,
        },
    )


# ---------------------------------------------------------------------------
# S3 — Cost/reward drift (cost multipliers per arm, Q unchanged)
# ---------------------------------------------------------------------------

def _make_cost_shift_fn(
    drift_factors: np.ndarray,
    lam: float,
) -> ShiftFn:
    """
    Build a ShiftFn that permanently multiplies C_norm by per-arm drift factors
    at the time step it is applied. Q is untouched. R is recomputed from the
    modified C_norm using the SAME lambda (no renormalisation of costs).

    CALLER CONTRACT — read before calling outside the evaluator
    -----------------------------------------------------------
    The returned function IGNORES its `t` argument: it scales EVERY row of the
    array it is handed. It does not itself split pre- from post-change-point
    rows. Two consequences:

    * Inside `run_stream()` this is correct. The loop consumes R[t] one step at
      a time and invokes the schedule at t == change_point, so rows < cp have
      already been read at their pre-drift values by the time the matrix is
      mutated. Effective semantics: pre-drift before cp, post-drift after.

    * Any OTHER caller that processes the whole array at once (e.g. a vectorised
      oracle computing `R.argmax(axis=1)` over the full stream) MUST slice
      explicitly, or every row silently gets post-drift costs:

          R_post, _, Cn_post = fn(cp, R[cp:], Q[cp:], Cn[cp:])
          R  = np.concatenate([R[:cp],  R_post])
          Cn = np.concatenate([Cn[:cp], Cn_post])

    This exact bug corrupted the Oracle's pre-shift S3 rows in the first P5 run
    (see scripts/12b_fix_oracle_s3.py). Guarded by
    tests/test_m2_pipeline.py::test_cost_shift_fn_scales_every_row_it_receives.
    """
    def shift(t: int, R: np.ndarray, Q: np.ndarray,
              C_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        C_norm_new = C_norm * drift_factors  # broadcasts (T,K) x (K,)
        R_new      = Q - lam * C_norm_new
        return R_new, Q, C_norm_new
    return shift


def make_S3_stream(
    n: int,
    seed: int,
    lam: float,
    drift_factors: np.ndarray,
    change_point_frac: float = 0.5,
    length: Optional[int] = None,
) -> Stream:
    """
    Cost/reward drift: at t = int(length * change_point_frac), multiply the
    normalised cost of each arm by `drift_factors[a]`. Q remains unchanged.

    The frozen CostNormaliser is NOT refit — the scalar `s_train` stays fixed
    and post-drift C_norm values above 1.0 are the intended signal for
    cost-aware routers to detect the change.

    Parameters
    ----------
    n              : number of source rows (used only for order permutation).
    seed           : RNG seed for deterministic stream order.
    lam            : lambda used to compute R = Q - lam * C_norm (needed so
                     the shift function can recompute R with the same lambda).
    drift_factors  : (K,) multiplicative factor per arm. e.g. [1,1,1,1,1,2,1,1]
                     doubles the normalised cost of arm 5 from the change
                     point onward.
    change_point_frac : where the drift kicks in, as fraction of length.
    length         : stream length (default = n).
    """
    T = length if length is not None else n
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    if T != n:
        # Extend or truncate with replacement sampling from source rows
        if T > n:
            extra = rng.choice(n, size=T - n, replace=True)
            order = np.concatenate([order, extra])
        else:
            order = order[:T]

    change_point = int(round(T * change_point_frac))
    drift_factors = np.asarray(drift_factors, dtype=np.float32)

    schedule = {change_point: _make_cost_shift_fn(drift_factors, lam)}

    return Stream(
        order=order,
        shift_schedule=schedule,
        scenario="S3",
        seed=seed,
        metadata={
            "change_point": change_point,
            "drift_factors": drift_factors.tolist(),
            "lam": lam,
            "length": T,
        },
    )
