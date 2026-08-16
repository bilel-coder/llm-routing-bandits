"""
Unit tests for all routing policy implementations (B0–B6).

Tests use synthetic data — no dataset required.
Focus: interface compliance, update isolation, leakage prevention.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_router.routers.base import Router
from llm_router.routers.cheapest import CheapestRouter
from llm_router.routers.best_single import BestSingleRouter
from llm_router.routers.knn import KNNRouter
from llm_router.routers.ridge import RidgeRouter
from llm_router.routers.linucb import LinUCBRouter
from llm_router.routers.discounted_linucb import DiscountedLinUCBRouter
from llm_router.routers.oracle import OracleRouter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ARMS = 4
DIM = 8
N_TRAIN = 50
N_TEST = 20

rng = np.random.default_rng(0)
X_train = rng.standard_normal((N_TRAIN, DIM))
R_train = rng.uniform(0, 1, (N_TRAIN, N_ARMS))
C_train = rng.uniform(0.01, 0.5, (N_TRAIN, N_ARMS))  # cost matrix
X_test  = rng.standard_normal((N_TEST, DIM))
R_test  = rng.uniform(0, 1, (N_TEST, N_ARMS))
MEAN_COSTS = rng.uniform(0.01, 0.5, N_ARMS)


# ---------------------------------------------------------------------------
# Test: all routers implement the Router interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("RouterClass,kwargs", [
    (CheapestRouter, {"n_arms": N_ARMS}),
    (BestSingleRouter, {"n_arms": N_ARMS}),
    (KNNRouter, {"n_arms": N_ARMS, "k": 3}),
    (RidgeRouter, {"n_arms": N_ARMS, "alpha": 1.0}),
    (LinUCBRouter, {"n_arms": N_ARMS, "context_dim": DIM}),
    (DiscountedLinUCBRouter, {"n_arms": N_ARMS, "context_dim": DIM, "gamma": 0.99}),
])
def test_router_interface(RouterClass, kwargs):
    router = RouterClass(**kwargs)
    assert isinstance(router, Router)
    assert hasattr(router, "fit")
    assert hasattr(router, "select")
    assert hasattr(router, "update")
    assert hasattr(router, "reset")


# ---------------------------------------------------------------------------
# Test: select returns valid arm index
# ---------------------------------------------------------------------------

def test_cheapest_select():
    """B0: fit() requires cost_matrix; selects arm with lowest mean cost."""
    router = CheapestRouter(N_ARMS)
    router.fit(X_train, cost_matrix=C_train)
    expected = int(np.argmin(np.mean(C_train, axis=0)))
    for x in X_test:
        assert router.select(x) == expected


def test_cheapest_raises_without_cost_matrix():
    """B0: fit() without cost_matrix must raise ValueError."""
    router = CheapestRouter(N_ARMS)
    with pytest.raises(ValueError, match="cost_matrix"):
        router.fit(X_train)


def test_best_single_select():
    router = BestSingleRouter(N_ARMS)
    router.fit(X_train, R_train)
    expected = int(np.argmax(np.nanmean(R_train, axis=0)))
    for x in X_test:
        assert router.select(x) == expected


def test_knn_select_valid_arm():
    router = KNNRouter(N_ARMS, k=3)
    router.fit(X_train, R_train)
    for x in X_test:
        a = router.select(x)
        assert 0 <= a < N_ARMS


def test_ridge_select_valid_arm():
    router = RidgeRouter(N_ARMS, alpha=1.0)
    router.fit(X_train, R_train)
    for x in X_test:
        a = router.select(x)
        assert 0 <= a < N_ARMS


def test_linucb_select_valid_arm():
    router = LinUCBRouter(N_ARMS, DIM, alpha=1.0)
    router.fit(X_train)
    for x in X_test:
        a = router.select(x)
        assert 0 <= a < N_ARMS


def test_discounted_linucb_select_valid_arm():
    router = DiscountedLinUCBRouter(N_ARMS, DIM, gamma=0.95)
    router.fit(X_train)
    for x in X_test:
        a = router.select(x)
        assert 0 <= a < N_ARMS


# ---------------------------------------------------------------------------
# Test: online policies raise if rewards passed to fit()
# ---------------------------------------------------------------------------

def test_linucb_rejects_training_rewards():
    router = LinUCBRouter(N_ARMS, DIM)
    with pytest.raises(ValueError, match="online policy"):
        router.fit(X_train, rewards=R_train)


def test_discounted_linucb_rejects_training_rewards():
    router = DiscountedLinUCBRouter(N_ARMS, DIM)
    with pytest.raises(ValueError, match="online policy"):
        router.fit(X_train, rewards=R_train)


# ---------------------------------------------------------------------------
# Test: offline policies ignore update() calls (no-op)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("RouterClass,fit_kwargs,fit_rewards", [
    (CheapestRouter, {"n_arms": N_ARMS}, False),
    (BestSingleRouter, {"n_arms": N_ARMS}, True),
    (RidgeRouter, {"n_arms": N_ARMS}, True),
])
def test_offline_update_noop(RouterClass, fit_kwargs, fit_rewards):
    router = RouterClass(**fit_kwargs)
    if fit_rewards:
        router.fit(X_train, R_train)
    else:
        router.fit(X_train, cost_matrix=C_train)

    x = X_test[0]
    a_before = router.select(x)
    router.update(x, a_before, 1.0)
    a_after = router.select(x)
    assert a_before == a_after, "Offline router changed selection after update()"


# ---------------------------------------------------------------------------
# Test: LinUCB updates only selected arm
# ---------------------------------------------------------------------------

def test_linucb_updates_only_selected_arm():
    router = LinUCBRouter(N_ARMS, DIM, alpha=1.0)
    router.fit(X_train)

    A_before = [m.copy() for m in router._A]

    x = X_test[0]
    a = router.select(x)
    router.update(x, a, 1.0)

    for arm in range(N_ARMS):
        if arm == a:
            assert not np.allclose(router._A[arm], A_before[arm]), \
                f"Arm {arm} (selected) should have updated A"
        else:
            assert np.allclose(router._A[arm], A_before[arm]), \
                f"Arm {arm} (NOT selected) should NOT have updated A"


# ---------------------------------------------------------------------------
# Test: D-LinUCB updates only selected arm (Russac: per-observation, not per-step)
# ---------------------------------------------------------------------------

def test_discounted_linucb_updates_only_selected_arm():
    router = DiscountedLinUCBRouter(N_ARMS, DIM, alpha=1.0, gamma=0.9)
    router.fit(X_train)

    V_before = [m.copy() for m in router._V]

    x = X_test[0]
    a = router.select(x)
    router.update(x, a, 1.0)

    for arm in range(N_ARMS):
        if arm == a:
            assert not np.allclose(router._V[arm], V_before[arm]), \
                f"Arm {arm} (selected) V should have changed"
        else:
            assert np.allclose(router._V[arm], V_before[arm]), \
                f"Arm {arm} (NOT selected) V should be unchanged"


def test_discounted_linucb_v_tilde_updates_with_gamma_squared():
    """V~ is discounted by γ², V by γ — check the ratio after one update."""
    g = 0.9
    reg = 1.0
    router = DiscountedLinUCBRouter(N_ARMS, DIM, alpha=1.0, gamma=g, reg_lambda=reg)
    router.fit(X_train)

    x = X_test[0]
    a = router.select(x)
    router.update(x, a, 1.0)

    # V_a  = γ  * reg*I + x x^T
    # V~_a = γ² * reg*I + x x^T
    # Difference = (γ - γ²) * reg * I = γ(1-γ) * reg * I
    diff = router._V[a] - router._V_tilde[a]
    expected_diff = (g - g * g) * reg * np.eye(DIM)
    assert np.allclose(diff, expected_diff, atol=1e-10), \
        "V - V~ should equal γ(1-γ)·reg·I after first update"


# ---------------------------------------------------------------------------
# Test: D-LinUCB with gamma=1.0 behaves like LinUCB (reduction property)
# ---------------------------------------------------------------------------

def test_discounted_linucb_gamma1_equals_linucb():
    """When gamma=1.0, V~=V so D-LinUCB UCB reduces to standard LinUCB UCB."""
    lin  = LinUCBRouter(N_ARMS, DIM, alpha=1.0)
    dlin = DiscountedLinUCBRouter(N_ARMS, DIM, alpha=1.0, gamma=1.0)
    lin.fit(X_train)
    dlin.fit(X_train)

    for t in range(15):
        x = X_test[t % N_TEST]
        a_lin  = lin.select(x)
        a_dlin = dlin.select(x)
        assert a_lin == a_dlin, (
            f"Step {t}: LinUCB chose arm {a_lin} but D-LinUCB(gamma=1) chose {a_dlin}"
        )
        r = float(R_test[t % N_TEST, a_lin])
        lin.update(x, a_lin, r)
        dlin.update(x, a_dlin, r)


# ---------------------------------------------------------------------------
# Test: Oracle selects true best arm
# ---------------------------------------------------------------------------

def test_oracle_selects_best_arm():
    oracle = OracleRouter(N_ARMS)
    oracle.fit(X_test)
    for t, x in enumerate(X_test):
        rewards = R_test[t]
        oracle.set_step_rewards(rewards)
        a = oracle.select(x)
        assert a == int(np.argmax(rewards))


def test_oracle_requires_full_information_flag():
    oracle = OracleRouter(N_ARMS)
    assert oracle.REQUIRES_FULL_INFORMATION is True


def test_oracle_raises_without_set_step_rewards():
    oracle = OracleRouter(N_ARMS)
    oracle.fit(X_test)
    with pytest.raises(RuntimeError):
        oracle.select(X_test[0])


# ---------------------------------------------------------------------------
# Test: reset() clears online state
# ---------------------------------------------------------------------------

def test_linucb_reset_clears_state():
    router = LinUCBRouter(N_ARMS, DIM)
    router.fit(X_train)

    for t in range(10):
        x = X_test[t]
        a = router.select(x)
        router.update(x, a, 0.5)

    router.reset()
    for arm in range(N_ARMS):
        assert np.allclose(router._A[arm], np.eye(DIM)), \
            f"Arm {arm} A matrix not reset to identity"


def test_discounted_linucb_reset_clears_state():
    router = DiscountedLinUCBRouter(N_ARMS, DIM, gamma=0.9, reg_lambda=1.0)
    router.fit(X_train)

    for t in range(10):
        x = X_test[t]
        a = router.select(x)
        router.update(x, a, 0.5)

    router.reset()
    for arm in range(N_ARMS):
        assert np.allclose(router._V[arm], np.eye(DIM)), \
            f"Arm {arm} V matrix not reset to reg_lambda*I"
        assert np.allclose(router._V_tilde[arm], np.eye(DIM)), \
            f"Arm {arm} V~ matrix not reset to reg_lambda*I"


# ---------------------------------------------------------------------------
# Test: lambda lifecycle — mismatch raises in evaluator
# ---------------------------------------------------------------------------

def test_lambda_mismatch_raises():
    """Offline routers store _fitted_lam; mismatch must raise in run_stream()."""
    import pandas as pd
    from llm_router.evaluation.evaluator import run_stream

    router = BestSingleRouter(N_ARMS)
    router.fit(X_train, rewards=R_train, lam=0.1)
    oracle = OracleRouter(N_ARMS)
    oracle.fit(X_train)

    Q = rng.uniform(0, 1, (N_TEST, N_ARMS))
    C = rng.uniform(0.01, 0.5, (N_TEST, N_ARMS))
    lam_eval = 0.5   # different from fitted lam=0.1
    R = Q - lam_eval * C

    with pytest.raises(ValueError, match="lam="):
        run_stream(router, X_test, R, Q, C, oracle, lam=lam_eval, seed=0)


def test_lambda_match_does_not_raise():
    """Same lambda at fit and eval time must not raise."""
    from llm_router.evaluation.evaluator import run_stream

    router = BestSingleRouter(N_ARMS)
    lam = 0.25
    router.fit(X_train, rewards=R_train, lam=lam)
    oracle = OracleRouter(N_ARMS)
    oracle.fit(X_train)

    Q = rng.uniform(0, 1, (N_TEST, N_ARMS))
    C = rng.uniform(0.01, 0.5, (N_TEST, N_ARMS))
    R = Q - lam * C

    result = run_stream(router, X_test, R, Q, C, oracle, lam=lam, seed=0)
    assert len(result.records) == N_TEST


# ---------------------------------------------------------------------------
# Test: offline routers with lam=None do NOT trigger mismatch check
# ---------------------------------------------------------------------------

def test_lambda_none_skips_check():
    """lam=None at fit time means no enforcement (backwards-compatible)."""
    from llm_router.evaluation.evaluator import run_stream

    router = BestSingleRouter(N_ARMS)
    router.fit(X_train, rewards=R_train)  # no lam passed
    oracle = OracleRouter(N_ARMS)
    oracle.fit(X_train)

    Q = rng.uniform(0, 1, (N_TEST, N_ARMS))
    C = rng.uniform(0.01, 0.5, (N_TEST, N_ARMS))
    R = Q - 0.5 * C

    result = run_stream(router, X_test, R, Q, C, oracle, lam=0.5, seed=0)
    assert len(result.records) == N_TEST
