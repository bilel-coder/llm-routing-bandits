"""
P5a benchmark — measure per-router runtime for one (config, seed, lambda, scenario)
run so the full sweep budget can be planned. Uses only TRAIN + DEV.
"""

from __future__ import annotations

import io
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from llm_router.data.roles import partition_for
from llm_router.routers.knn import KNNRouter
from llm_router.routers.ridge import RidgeRouter
from llm_router.routers.linucb import LinUCBRouter
from llm_router.routers.discounted_linucb import DiscountedLinUCBRouter
from llm_router.routers.oracle import OracleRouter
from llm_router.evaluation.evaluator import run_stream
from llm_router.environments import make_S0_stream

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INTERIM = Path("data/interim")
MAT = INTERIM / "matrices"


def load(role):
    p = partition_for(role)
    return (
        np.load(INTERIM / f"X_{p}.npy"),
        np.load(MAT / f"Q_{p}.npy"),
        np.load(MAT / f"C_norm_{p}.npy"),
    )


def bench_online(RouterCls, kwargs, X_tr, Q_tr, Cn_tr, X_dv, Q_dv, Cn_dv,
                 lam=0.25, seed=42, n_scenarios=3):
    K = Q_tr.shape[1]
    d = X_tr.shape[1]
    router = RouterCls(n_arms=K, context_dim=d, **kwargs)
    R_tr = Q_tr - lam * Cn_tr
    R_dv = Q_dv - lam * Cn_dv
    orc = OracleRouter(n_arms=K)
    orc.fit(X_tr)

    t0 = time.perf_counter()
    router.fit(X_tr)
    # warmup on train
    stream_wu = make_S0_stream(n=X_tr.shape[0], seed=seed)
    o = stream_wu.order
    run_stream(router, X_tr[o], R_tr[o], Q_tr[o], Cn_tr[o],
                orc, lam=lam, seed=seed, scenario="warmup")
    t_wu = time.perf_counter() - t0

    # deepcopy warmed state (to reuse across scenarios)
    t1 = time.perf_counter()
    warmed_state = deepcopy(router)
    t_dc = time.perf_counter() - t1

    # simulate n_scenarios evals from the warmed state
    orc2 = OracleRouter(n_arms=K)
    orc2.fit(X_dv)
    stream_dv = make_S0_stream(n=X_dv.shape[0], seed=seed)
    o2 = stream_dv.order
    t_eval_total = 0.0
    for _ in range(n_scenarios):
        r = deepcopy(warmed_state)
        t2 = time.perf_counter()
        run_stream(r, X_dv[o2], R_dv[o2], Q_dv[o2], Cn_dv[o2],
                    orc2, lam=lam, seed=seed, scenario="S0")
        t_eval_total += time.perf_counter() - t2

    return t_wu, t_dc, t_eval_total


def bench_offline(RouterCls, kwargs, X_tr, Q_tr, Cn_tr, X_dv, Q_dv, Cn_dv,
                   lam=0.25, seed=42, n_scenarios=3):
    K = Q_tr.shape[1]
    router = RouterCls(n_arms=K, **kwargs)
    R_tr = Q_tr - lam * Cn_tr
    R_dv = Q_dv - lam * Cn_dv
    orc = OracleRouter(n_arms=K)
    orc.fit(X_dv)
    stream_dv = make_S0_stream(n=X_dv.shape[0], seed=seed)
    o = stream_dv.order

    t0 = time.perf_counter()
    router.fit(X_tr, rewards=R_tr, lam=lam)
    t_fit = time.perf_counter() - t0

    t_eval_total = 0.0
    for _ in range(n_scenarios):
        t2 = time.perf_counter()
        run_stream(router, X_dv[o], R_dv[o], Q_dv[o], Cn_dv[o],
                    orc, lam=lam, seed=seed, scenario="S0")
        t_eval_total += time.perf_counter() - t2

    return t_fit, t_eval_total


def main():
    X_tr, Q_tr, Cn_tr = load("TRAIN")
    X_dv, Q_dv, Cn_dv = load("DEV")
    print(f"TRAIN: {X_tr.shape}   DEV: {X_dv.shape}")

    # Warm the imports and numpy caches with a tiny call
    _ = np.linalg.inv(np.eye(64))

    # kNN
    t_fit, t_eval = bench_offline(KNNRouter, {"k": 10}, X_tr, Q_tr, Cn_tr,
                                    X_dv, Q_dv, Cn_dv)
    print(f"kNN(k=10):       fit={t_fit:.2f}s   eval(x3 scenarios)={t_eval:.2f}s")

    # Ridge
    t_fit, t_eval = bench_offline(RidgeRouter, {"alpha": 1.0}, X_tr, Q_tr, Cn_tr,
                                    X_dv, Q_dv, Cn_dv)
    print(f"Ridge(alpha=1):  fit={t_fit:.2f}s   eval(x3 scenarios)={t_eval:.2f}s")

    # LinUCB
    t_wu, t_dc, t_eval = bench_online(LinUCBRouter, {"alpha": 1.0},
                                        X_tr, Q_tr, Cn_tr, X_dv, Q_dv, Cn_dv)
    print(f"LinUCB(a=1):     warmup={t_wu:.2f}s   deepcopy={t_dc:.3f}s   "
          f"eval(x3 scenarios)={t_eval:.2f}s")

    # D-LinUCB
    t_wu, t_dc, t_eval = bench_online(DiscountedLinUCBRouter,
                                        {"alpha": 1.0, "gamma": 0.99},
                                        X_tr, Q_tr, Cn_tr, X_dv, Q_dv, Cn_dv)
    print(f"D-LinUCB(a=1,g=.99): warmup={t_wu:.2f}s   deepcopy={t_dc:.3f}s   "
          f"eval(x3 scenarios)={t_eval:.2f}s")


if __name__ == "__main__":
    main()
