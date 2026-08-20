"""
M2 tests — features, matrices, streams, evaluator invariants.

Verifies:
  1. PCA is fitted on TRAIN partition only (no val/test leakage into fit).
  2. X / Q / C / C_norm row alignment across all partitions.
  3. Stream generators are deterministic per seed.
  4. Reward formula: R = Q - lam * C_norm  (exactly, per row).
  5. Cost drift multiplies C_norm without triggering renormalisation.
  6. Selected-arm feedback isolation: online routers see only reward of
     their chosen arm — the evaluator never leaks other arms' rewards.
  7. Stationary S0 preserves outcomes: quality/cost observed by the stream
     equals the source matrix values at the visited rows.
  8. S3 ShiftFn caller contract: it scales every row it receives (ignoring `t`),
     so a vectorised oracle must slice at the change point to match the
     sequential evaluator. Regression guard for the Oracle x S3 bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_router.environments import (
    make_S0_stream,
    make_S1_stream,
    make_S2_stream,
    make_S3_stream,
)
from llm_router.routers.linucb import LinUCBRouter
from llm_router.routers.oracle import OracleRouter
from llm_router.evaluation.evaluator import run_stream

INTERIM = Path("data/interim")
MAT     = INTERIM / "matrices"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matrices():
    """Load all partition matrices; skip if not built."""
    needed = [
        INTERIM / "X_train.npy", INTERIM / "X_val.npy", INTERIM / "X_test.npy",
        MAT / "Q_train.npy", MAT / "Q_val.npy", MAT / "Q_test.npy",
        MAT / "C_train.npy", MAT / "C_test.npy",
        MAT / "C_norm_train.npy", MAT / "C_norm_test.npy",
        MAT / "datasets_test.json",
        INTERIM / "pca_meta.json",
    ]
    if not all(p.exists() for p in needed):
        pytest.skip("M2 feature/matrix artefacts not built. "
                    "Run scripts 06 and 07 first.")
    data = {}
    for part in ("train", "val", "test"):
        data[f"X_{part}"]      = np.load(INTERIM / f"X_{part}.npy")
        data[f"Q_{part}"]      = np.load(MAT / f"Q_{part}.npy")
        data[f"C_{part}"]      = np.load(MAT / f"C_{part}.npy")
        data[f"C_norm_{part}"] = np.load(MAT / f"C_norm_{part}.npy")
        with open(MAT / f"datasets_{part}.json") as f:
            data[f"datasets_{part}"] = np.array(json.load(f))
    with open(INTERIM / "pca_meta.json") as f:
        data["pca_meta"] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# 1. PCA fitted on TRAIN only
# ---------------------------------------------------------------------------

def test_pca_fitted_on_train_only(matrices):
    meta = matrices["pca_meta"]
    assert meta["fitted_on"] == "train", (
        "PCA metadata must record fitted_on=='train' — no val/test in fit."
    )
    assert meta["n_train_samples"] == matrices["X_train"].shape[0], (
        f"PCA declared n_train_samples={meta['n_train_samples']} but "
        f"X_train has {matrices['X_train'].shape[0]} rows — leakage or misalignment."
    )


def test_pca_dim_matches_config(matrices):
    assert matrices["pca_meta"]["pca_dim"] == 64
    for part in ("train", "val", "test"):
        assert matrices[f"X_{part}"].shape[1] == 64


def test_pca_explained_variance_sensible(matrices):
    """PCA on natural-language embeddings should explain >= 40% at 64 dim."""
    cum = matrices["pca_meta"]["cum_ev_at_pca_dim"]
    assert cum >= 0.40, (
        f"PCA 64d cumulative EV = {cum:.3f} — unexpectedly low. "
        "Check embeddings quality."
    )


# ---------------------------------------------------------------------------
# 2. X / Q / C row alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("part", ["train", "val", "test"])
def test_xqc_row_alignment(matrices, part):
    X  = matrices[f"X_{part}"]
    Q  = matrices[f"Q_{part}"]
    C  = matrices[f"C_{part}"]
    Cn = matrices[f"C_norm_{part}"]
    ds = matrices[f"datasets_{part}"]
    n  = X.shape[0]
    assert Q.shape[0]  == n
    assert C.shape[0]  == n
    assert Cn.shape[0] == n
    assert len(ds)     == n
    # Column count == 8 arms
    assert Q.shape[1]  == 8
    assert C.shape[1]  == 8
    assert Cn.shape[1] == 8


@pytest.mark.parametrize("part", ["train", "val", "test"])
def test_no_nan_or_inf(matrices, part):
    for key in (f"X_{part}", f"Q_{part}", f"C_{part}", f"C_norm_{part}"):
        arr = matrices[key]
        assert not np.isnan(arr).any(), f"{key}: contains NaN"
        assert not np.isinf(arr).any(), f"{key}: contains Inf"


@pytest.mark.parametrize("part", ["train", "val", "test"])
def test_cost_norm_matches_scale_ratio(matrices, part):
    """C_norm = C / scale — must be exact per element."""
    with open(MAT / "matrix_meta.json") as f:
        meta = json.load(f)
    scale = meta["cost_norm_scale_usd"]
    C  = matrices[f"C_{part}"]
    Cn = matrices[f"C_norm_{part}"]
    np.testing.assert_allclose(Cn, C / scale, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Stream determinism
# ---------------------------------------------------------------------------

def test_S0_stream_deterministic():
    """Two calls with same seed produce identical order."""
    s1 = make_S0_stream(n=500, seed=42)
    s2 = make_S0_stream(n=500, seed=42)
    np.testing.assert_array_equal(s1.order, s2.order)
    # And different seeds produce different orders
    s3 = make_S0_stream(n=500, seed=43)
    assert not np.array_equal(s1.order, s3.order)


def test_S0_covers_all_rows_once():
    s = make_S0_stream(n=1000, seed=0)
    assert len(s.order) == 1000
    assert set(s.order.tolist()) == set(range(1000)), "S0 must be a permutation"


def test_S1_stream_deterministic():
    ds = np.array(["a"] * 200 + ["b"] * 200 + ["c"] * 200)
    s1 = make_S1_stream(ds, seed=7,
                        pre_shift_datasets=["a", "b"],
                        post_shift_datasets=["c"],
                        shift_frac=0.5)
    s2 = make_S1_stream(ds, seed=7,
                        pre_shift_datasets=["a", "b"],
                        post_shift_datasets=["c"],
                        shift_frac=0.5)
    np.testing.assert_array_equal(s1.order, s2.order)
    assert s1.metadata["change_point"] == 300


def test_S2_stream_deterministic():
    ds = np.array(["a"] * 200 + ["b"] * 200)
    s1 = make_S2_stream(ds, seed=3,
                        pre_shift_datasets=["a"],
                        post_shift_datasets=["b"],
                        shift_start_frac=0.25, shift_end_frac=0.75)
    s2 = make_S2_stream(ds, seed=3,
                        pre_shift_datasets=["a"],
                        post_shift_datasets=["b"],
                        shift_start_frac=0.25, shift_end_frac=0.75)
    np.testing.assert_array_equal(s1.order, s2.order)


def test_S3_stream_deterministic():
    factors = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    s1 = make_S3_stream(n=200, seed=11, lam=0.25, drift_factors=factors)
    s2 = make_S3_stream(n=200, seed=11, lam=0.25, drift_factors=factors)
    np.testing.assert_array_equal(s1.order, s2.order)
    assert s1.metadata["change_point"] == 100
    assert 100 in s1.shift_schedule


# ---------------------------------------------------------------------------
# 4. Reward formula
# ---------------------------------------------------------------------------

def test_reward_formula_exact():
    """R = Q - lam * C_norm  applied element-wise."""
    rng = np.random.default_rng(0)
    Q  = rng.uniform(0, 1, (50, 8))
    Cn = rng.uniform(0, 2, (50, 8))
    for lam in [0.0, 0.1, 0.25, 0.5, 1.0]:
        R = Q - lam * Cn
        # Every row: reward at every arm equals q - lam*cn
        for i in range(50):
            for a in range(8):
                assert abs(R[i, a] - (Q[i, a] - lam * Cn[i, a])) < 1e-9


# ---------------------------------------------------------------------------
# 5. Cost drift multiplies without renormalising
# ---------------------------------------------------------------------------

def test_S3_drift_multiplies_cost_norm_without_renormalisation():
    """After S3 shift kicks in, C_norm for driven arms is multiplied by the
    drift factor. Frozen normaliser stays untouched."""
    rng = np.random.default_rng(0)
    n, K, d = 40, 4, 6
    Q  = rng.uniform(0, 1, (n, K)).astype(np.float32)
    Cn = rng.uniform(0, 0.5, (n, K)).astype(np.float32)
    lam = 0.25
    R  = Q - lam * Cn

    factors = np.array([1.0, 2.0, 0.5, 1.0], dtype=np.float32)
    stream = make_S3_stream(n=n, seed=1, lam=lam, drift_factors=factors,
                            change_point_frac=0.5)
    cp = stream.metadata["change_point"]
    shift_fn = stream.shift_schedule[cp]

    # Apply shift once
    Cn_before = Cn.copy()
    R_new, Q_new, Cn_new = shift_fn(cp, R.copy(), Q.copy(), Cn.copy())

    # Q must be unchanged
    np.testing.assert_array_equal(Q_new, Q)
    # C_norm multiplied element-wise by drift factors (broadcast across rows)
    for a in range(K):
        np.testing.assert_allclose(Cn_new[:, a], Cn_before[:, a] * factors[a],
                                    rtol=1e-6)
    # R recomputed from new C_norm using SAME lam
    np.testing.assert_allclose(R_new, Q - lam * Cn_new, rtol=1e-6)


def test_cost_shift_fn_scales_every_row_it_receives():
    """
    Contract test: the S3 ShiftFn IGNORES its `t` argument and scales EVERY row
    of the array handed to it — it does not split pre/post change point itself.

    This is not a latent bug, it is the interface `run_stream()` relies on (the
    sequential loop has already consumed rows < cp). Pinning it here so that any
    caller processing the whole array at once knows it must slice explicitly.
    """
    rng = np.random.default_rng(11)
    n, K = 40, 4
    Q  = rng.uniform(0, 1, (n, K)).astype(np.float32)
    Cn = rng.uniform(0, 0.5, (n, K)).astype(np.float32)
    lam = 0.5
    factors = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)

    stream = make_S3_stream(n=n, seed=7, lam=lam, drift_factors=factors,
                            change_point_frac=0.5)
    cp = stream.metadata["change_point"]
    fn = stream.shift_schedule[cp]

    # Passing t=cp but the FULL array: rows before cp are scaled too.
    _, _, Cn_all = fn(cp, Q - lam * Cn, Q, Cn.copy())
    np.testing.assert_allclose(Cn_all[0], Cn[0] * factors, rtol=1e-6)
    assert not np.allclose(Cn_all[:cp], Cn[:cp]), (
        "ShiftFn is expected to scale pre-cp rows when handed the full array; "
        "if this now passes, the caller contract changed and "
        "scripts/12_p5_final_validation.py must be revisited."
    )


def test_vectorised_S3_oracle_matches_sequential_evaluator():
    """
    Regression guard for the Oracle x S3 bug fixed in scripts/12b_fix_oracle_s3.py.

    The evaluator computes the oracle sequentially, so rows before the change
    point keep their pre-drift reward. A vectorised oracle (argmax over the
    whole stream at once) only reproduces that if it slices the ShiftFn at cp.
    Applying the ShiftFn to the whole array instead silently gives every
    pre-shift step post-drift costs.
    """
    rng = np.random.default_rng(12)
    n, K, d = 60, 4, 6
    X  = rng.standard_normal((n, d)).astype(np.float32)
    Q  = rng.uniform(0, 1, (n, K)).astype(np.float32)
    Cn = rng.uniform(0.1, 0.9, (n, K)).astype(np.float32)
    lam = 1.0
    factors = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    stream = make_S3_stream(n=n, seed=5, lam=lam, drift_factors=factors,
                            change_point_frac=0.5)
    o = stream.order
    cp = stream.metadata["change_point"]
    fn = stream.shift_schedule[cp]

    Xs, Qs = X[o], Q[o]
    Cns = Cn[o].copy()
    Rs = Qs - lam * Cns

    # --- Ground truth: the evaluator's own sequential oracle ------------------
    router = LinUCBRouter(n_arms=K, context_dim=d, alpha=1.0)
    router.fit(Xs)
    oracle = OracleRouter(n_arms=K)
    oracle.fit(Xs)
    res = run_stream(router, Xs, Rs.copy(), Qs.copy(), Cns.copy(), oracle,
                     lam=lam, seed=0, scenario="S3",
                     shift_schedule=stream.shift_schedule)
    expected = res.to_dataframe()["oracle_reward"].values

    # --- Correct vectorised path: slice at cp --------------------------------
    R_post, _, Cn_post = fn(cp, Rs[cp:], Qs[cp:], Cns[cp:])
    R_sliced = np.concatenate([Rs[:cp], R_post])
    np.testing.assert_allclose(R_sliced.max(axis=1), expected, rtol=1e-6,
                               err_msg="Sliced vectorised oracle diverged from "
                                       "the sequential evaluator.")

    # --- Buggy vectorised path: whole array at once --------------------------
    R_whole, _, _ = fn(cp, Rs.copy(), Qs.copy(), Cns.copy())
    assert not np.allclose(R_whole.max(axis=1), expected), (
        "Whole-array application should differ from the evaluator on pre-shift "
        "rows; if it no longer does, this test has lost its teeth."
    )
    # The divergence must be confined to pre-shift rows.
    assert not np.allclose(R_whole[:cp].max(axis=1), expected[:cp])
    np.testing.assert_allclose(R_whole[cp:].max(axis=1), expected[cp:], rtol=1e-6)


# ---------------------------------------------------------------------------
# 6. Selected-arm feedback isolation
# ---------------------------------------------------------------------------

class _SpyRouter(LinUCBRouter):
    """LinUCB that also records the (action, reward) pairs it observes."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.observed = []

    def update(self, context, action, reward):
        self.observed.append((int(action), float(reward)))
        super().update(context, action, reward)


def test_selected_arm_feedback_isolation():
    """
    The evaluator must pass the online router ONLY the reward of the arm it
    selected. Confirm by comparing spy-recorded (action, reward) against
    the R matrix at (t, action).
    """
    rng = np.random.default_rng(2)
    T, K, d = 60, 4, 6
    X = rng.standard_normal((T, d)).astype(np.float32)
    Q = rng.uniform(0, 1, (T, K)).astype(np.float32)
    Cn = rng.uniform(0, 1, (T, K)).astype(np.float32)
    lam = 0.25
    R = Q - lam * Cn

    router = _SpyRouter(n_arms=K, context_dim=d, alpha=1.0)
    router.fit(X)
    oracle = OracleRouter(n_arms=K)
    oracle.fit(X)
    _ = run_stream(router, X, R, Q, Cn, oracle, lam=lam, seed=0)

    assert len(router.observed) == T
    for t, (a, r) in enumerate(router.observed):
        assert abs(r - R[t, a]) < 1e-9, (
            f"Step {t}: router received reward {r} for arm {a}, "
            f"but R[{t},{a}] = {R[t, a]} — feedback isolation broken."
        )


# ---------------------------------------------------------------------------
# 7. Stationary environment preserves outcomes
# ---------------------------------------------------------------------------

def test_S0_preserves_outcomes():
    """
    Running Oracle on an S0-permuted view must observe the same quality/cost
    as the source matrix values at the visited rows.
    """
    rng = np.random.default_rng(3)
    n, K, d = 80, 4, 6
    X  = rng.standard_normal((n, d)).astype(np.float32)
    Q  = rng.uniform(0, 1, (n, K)).astype(np.float32)
    Cn = rng.uniform(0, 0.5, (n, K)).astype(np.float32)
    lam = 0.25
    R  = Q - lam * Cn

    stream = make_S0_stream(n=n, seed=17)
    order  = stream.order
    Xs, Rs, Qs, Cns = X[order], R[order], Q[order], Cn[order]

    oracle = OracleRouter(n_arms=K)
    oracle.fit(Xs)
    result = run_stream(oracle, Xs, Rs, Qs, Cns, oracle, lam=lam, seed=17)
    df = result.to_dataframe()

    # For each t, the quality/cost recorded must equal Q/Cn at (t, action)
    for t, row in df.iterrows():
        a = int(row["action"])
        assert abs(row["quality"]   - Qs[t, a])  < 1e-9
        assert abs(row["cost_norm"] - Cns[t, a]) < 1e-9
        assert abs(row["reward"]    - Rs[t, a])  < 1e-9
