"""
M2.5 tests — protocol audit.

Verifies:
  1. Partition roles: TRAIN → 'train', DEV → 'test', FINAL_TEST → 'val';
     FINAL_TEST is guarded behind an explicit flag.
  2. Oracle tie-aware routing shares (fractional credit + tie-rate).
  3. Warm-up carries online-router state (LinUCB after warm-up is not
     identical to fresh LinUCB).
  4. Pilot script does not reference FINAL_TEST (grep-level).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from llm_router.data.roles import partition_for, load_role_map, VALID_ROLES
from llm_router.evaluation.metrics import oracle_tie_stats


# ---------------------------------------------------------------------------
# 1. Partition roles
# ---------------------------------------------------------------------------

def test_role_map_expected_assignment():
    m = load_role_map()
    assert m == {"TRAIN": "train", "DEV": "test", "FINAL_TEST": "val"}, (
        f"Unexpected role map: {m}"
    )


def test_partition_for_train_and_dev_no_guard():
    assert partition_for("TRAIN") == "train"
    assert partition_for("DEV")   == "test"


def test_partition_for_final_test_requires_explicit_flag():
    with pytest.raises(RuntimeError, match="allow_final_test"):
        partition_for("FINAL_TEST")
    # Explicit override must succeed
    assert partition_for("FINAL_TEST", allow_final_test=True) == "val"


def test_partition_for_rejects_unknown_role():
    with pytest.raises(ValueError):
        partition_for("holdout")


def test_pilot_script_does_not_reference_final_test():
    """
    Any physical reference to the 'val' partition inside the pilot would
    load FINAL_TEST. Guard against accidental exposure at the source level.
    """
    src = Path("scripts/09_pilot_S0.py").read_text(encoding="utf-8")
    # Allow the docstring line documenting the rule, but no code path
    # should hit 'val' as a partition literal (X_val, Q_val, etc.).
    forbidden = re.findall(r"['\"]val['\"]", src) + re.findall(r"_val\b", src)
    assert not forbidden, (
        f"Pilot script references FINAL_TEST partition literally: {forbidden}"
    )
    assert "FINAL_TEST" not in src or "not load" in src.lower(), (
        "Pilot should not attempt to load FINAL_TEST."
    )


# ---------------------------------------------------------------------------
# 2. Oracle tie-aware routing shares
# ---------------------------------------------------------------------------

def test_oracle_tie_stats_no_ties():
    # Distinct max per row
    R = np.array([[0.1, 0.9, 0.2, 0.4],
                  [0.7, 0.2, 0.3, 0.5],
                  [0.1, 0.1, 0.9, 0.1]])
    shares, tie_rate, tpr = oracle_tie_stats(R)
    assert tie_rate == 0.0
    np.testing.assert_array_equal(tpr, [1, 1, 1])
    # Row 0: arm 1; row 1: arm 0; row 2: arm 2  → shares (1/3, 1/3, 1/3, 0)
    np.testing.assert_allclose(shares, [1/3, 1/3, 1/3, 0], atol=1e-9)


def test_oracle_tie_stats_all_ties():
    # Every row: all 4 arms equal → each gets 1/4
    R = np.ones((10, 4))
    shares, tie_rate, tpr = oracle_tie_stats(R)
    assert tie_rate == 1.0
    np.testing.assert_array_equal(tpr, [4] * 10)
    np.testing.assert_allclose(shares, [0.25, 0.25, 0.25, 0.25], atol=1e-9)


def test_oracle_tie_stats_shares_sum_to_one():
    rng = np.random.default_rng(0)
    R = rng.uniform(0, 1, (200, 6))
    # Introduce some ties by rounding
    R = np.round(R, 1)
    shares, tie_rate, tpr = oracle_tie_stats(R)
    assert abs(shares.sum() - 1.0) < 1e-9
    assert 0.0 <= tie_rate <= 1.0
    assert (tpr >= 1).all() and (tpr <= 6).all()


def test_oracle_tie_stats_fractional_vs_argmax_on_binary_quality():
    """Reproduce the reported bias: with binary Q and λ=0, argmax picks the
    first column; fractional credit spreads equally over tied arms."""
    Q = np.array([[1, 1, 0, 0],  # arms 0 and 1 tie
                  [1, 1, 1, 1],  # all tie
                  [0, 0, 1, 0]]) # only arm 2 wins
    shares_frac, tie_rate, tpr = oracle_tie_stats(Q.astype(float))
    # Row0: 0.5, 0.5, 0, 0 → col-sums / 3
    # Row1: 0.25 each → col-sums / 3
    # Row2: 0, 0, 1, 0 → col-sums / 3
    expected = (np.array([0.5, 0.5, 0, 0])
                + np.array([0.25, 0.25, 0.25, 0.25])
                + np.array([0.0, 0.0, 1.0, 0.0])) / 3
    np.testing.assert_allclose(shares_frac, expected, atol=1e-9)
    # Naive argmax would give (0, 0, 1, 0) each row → shares (2/3, 0, 1/3, 0)
    # which is clearly biased by column order.
    naive = np.bincount(Q.argmax(axis=1), minlength=4) / len(Q)
    assert not np.allclose(shares_frac, naive), (
        "Fractional shares must differ from naive argmax when ties exist."
    )


# ---------------------------------------------------------------------------
# 3. Warm-up carries state
# ---------------------------------------------------------------------------

def test_warmup_carries_state_for_online_router():
    """
    A LinUCB fed a warm-up stream must NOT be equivalent to a fresh LinUCB
    at select() time — its V matrices must differ.
    """
    from llm_router.routers.linucb import LinUCBRouter
    from llm_router.routers.oracle import OracleRouter
    from llm_router.evaluation.evaluator import run_stream
    from llm_router.environments import make_S0_stream

    rng = np.random.default_rng(0)
    T, K, d = 100, 4, 6
    X = rng.standard_normal((T, d)).astype(np.float32)
    Q = rng.uniform(0, 1, (T, K)).astype(np.float32)
    Cn = rng.uniform(0, 0.5, (T, K)).astype(np.float32)
    lam = 0.25
    R = Q - lam * Cn

    warm = LinUCBRouter(n_arms=K, context_dim=d, alpha=1.0)
    fresh = LinUCBRouter(n_arms=K, context_dim=d, alpha=1.0)
    warm.fit(X); fresh.fit(X)
    stream = make_S0_stream(n=T, seed=42)
    oracle = OracleRouter(n_arms=K); oracle.fit(X)
    run_stream(warm, X[stream.order], R[stream.order], Q[stream.order],
                Cn[stream.order], oracle, lam=lam, seed=42, scenario="warmup")
    # Fresh has NOT been warmed
    same = all(np.allclose(warm._A[a], fresh._A[a]) for a in range(K))
    assert not same, "Warm-up did not modify router state — carry is broken."


def test_warmup_deterministic_for_same_seed():
    """Two warm-ups with identical seed produce identical final router state."""
    from llm_router.routers.linucb import LinUCBRouter
    from llm_router.routers.oracle import OracleRouter
    from llm_router.evaluation.evaluator import run_stream
    from llm_router.environments import make_S0_stream

    rng = np.random.default_rng(0)
    T, K, d = 80, 4, 6
    X = rng.standard_normal((T, d)).astype(np.float32)
    Q = rng.uniform(0, 1, (T, K)).astype(np.float32)
    Cn = rng.uniform(0, 0.5, (T, K)).astype(np.float32)
    lam = 0.25
    R = Q - lam * Cn

    def warm_one():
        r = LinUCBRouter(n_arms=K, context_dim=d, alpha=1.0); r.fit(X)
        stream = make_S0_stream(n=T, seed=7)
        oracle = OracleRouter(n_arms=K); oracle.fit(X)
        run_stream(r, X[stream.order], R[stream.order], Q[stream.order],
                    Cn[stream.order], oracle, lam=lam, seed=7, scenario="warmup")
        return r

    r1 = warm_one(); r2 = warm_one()
    for a in range(K):
        np.testing.assert_allclose(r1._A[a], r2._A[a], atol=1e-12)
        np.testing.assert_allclose(r1._b[a], r2._b[a], atol=1e-12)
