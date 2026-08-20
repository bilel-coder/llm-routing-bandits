"""
P5 — Final comparative validation on DEV.

Runs all 7 routers (Cheapest, BestSingle, kNN, Ridge, LinUCB, D-LinUCB, Oracle)
with frozen hyperparameters across:
  * 10 matched seeds: 42..51
  * 5 lambdas: {0.0, 0.1, 0.25, 0.5, 1.0}
  * 4 scenarios: S0 (stationary), S1 (abrupt covariate), S2 (gradual covariate),
                 S3 (cost drift)

FINAL_TEST is NOT accessed. All hyperparameters loaded from
configs/algorithms/*.yaml (frozen in P5a).

Protocol:
  * Offline routers (kNN, Ridge, BestSingle) fit on TRAIN with full-info rewards.
  * Cheapest fits on TRAIN cost matrix.
  * Online routers (LinUCB, D-LinUCB) warm-up on TRAIN (deterministic ordering
    per seed), then evaluate. Warmed state deep-copied per scenario so each
    scenario starts from the same warm state.
  * Oracle utility per step is R.max(axis=1); regret is 0 by definition.

Per-run outputs: macro/micro utility, quality, cost_norm, cum_regret,
total_regret, post_shift_regret_200 (S1/S2/S3 only), recovery_time
(S1/S2/S3 only), per-arm routing share.

Audit provisos carried forward:
  * recovery_time on S3 λ=0 is a metric artefact (see A1) — retained here for
    completeness but must be filtered in the analysis report.
"""

from __future__ import annotations

import io
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from llm_router.data.roles import partition_for
from llm_router.routers.cheapest import CheapestRouter
from llm_router.routers.best_single import BestSingleRouter
from llm_router.routers.knn import KNNRouter
from llm_router.routers.ridge import RidgeRouter
from llm_router.routers.linucb import LinUCBRouter
from llm_router.routers.discounted_linucb import DiscountedLinUCBRouter
from llm_router.routers.oracle import OracleRouter
from llm_router.evaluation.evaluator import run_stream
from llm_router.environments import (
    make_S0_stream,
    make_S1_stream,
    make_S2_stream,
    make_S3_stream,
)

INTERIM = Path("data/interim")
MAT     = INTERIM / "matrices"
CFG_DIR = Path("configs/algorithms")

SEEDS     = list(range(42, 52))                   # 10 matched seeds
LAMBDAS   = [0.0, 0.1, 0.25, 0.5, 1.0]
SCENARIOS = ["S0", "S1", "S2", "S3"]
ROUTERS   = ["Cheapest", "BestSingle", "kNN", "Ridge", "LinUCB", "D-LinUCB", "Oracle"]
ONLINE    = {"LinUCB", "D-LinUCB"}

# Covariate-shift dataset partition (same for S1 and S2)
COV_PRE  = ["arenahard", "hle", "mmlupro", "simpleqa"]
COV_POST = ["aime", "arc-agi", "gpqa", "livecodebench",
            "livemathbench", "swe-bench", "tau2"]

# S3 cost drift: double kimi-k2 (idx 5) and deepseek-v3-0324 (idx 7) at t=T/2
# Pool order:
#   [claude-sonnet-4, gemini-2.5-flash, gemini-2.5-pro, gpt-5,
#    deepseek-r1-0528, kimi-k2-0905, glm-4.6, deepseek-v3-0324]
S3_DRIFT = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)


def load_matrices(role):
    p = partition_for(role)
    X = np.load(INTERIM / f"X_{p}.npy")
    Q = np.load(MAT / f"Q_{p}.npy")
    C = np.load(MAT / f"C_{p}.npy")
    Cn = np.load(MAT / f"C_norm_{p}.npy")
    with open(MAT / f"datasets_{p}.json") as f:
        ds = json.load(f)
    return X, Q, C, Cn, np.array(ds)


def load_frozen_hps():
    hps = {}
    for name in ("knn", "ridge", "linucb", "discounted_linucb"):
        with open(CFG_DIR / f"{name}.yaml") as f:
            cfg = yaml.safe_load(f)
        hps[cfg["router"]] = cfg["hyperparameters"]
    return hps


def build_router(name, hps, K, dim):
    if name == "Cheapest":   return CheapestRouter(n_arms=K)
    if name == "BestSingle": return BestSingleRouter(n_arms=K)
    if name == "kNN":        return KNNRouter(n_arms=K, **hps["kNN"])
    if name == "Ridge":      return RidgeRouter(n_arms=K, **hps["Ridge"])
    if name == "LinUCB":     return LinUCBRouter(n_arms=K, context_dim=dim, **hps["LinUCB"])
    if name == "D-LinUCB":   return DiscountedLinUCBRouter(n_arms=K, context_dim=dim,
                                                            **hps["D-LinUCB"])
    if name == "Oracle":     return OracleRouter(n_arms=K)
    raise ValueError(name)


def fit_or_warmup(name, router, X_tr, Q_tr, Cn_tr, C_tr, R_tr, seed, lam):
    if name == "Cheapest":
        router.fit(X_tr, cost_matrix=C_tr)
    elif name in ("BestSingle", "kNN", "Ridge"):
        router.fit(X_tr, rewards=R_tr, lam=lam)
    elif name == "Oracle":
        router.fit(X_tr)
    elif name in ONLINE:
        router.fit(X_tr)
        stream = make_S0_stream(n=X_tr.shape[0], seed=seed)
        o = stream.order
        orc = OracleRouter(n_arms=router.n_arms); orc.fit(X_tr)
        run_stream(router, X_tr[o], R_tr[o], Q_tr[o], Cn_tr[o], orc,
                    lam=lam, seed=seed, scenario="warmup")


def build_stream(scenario, X_dv, ds_dv, seed, lam):
    n = X_dv.shape[0]
    if scenario == "S0":
        return make_S0_stream(n=n, seed=seed)
    if scenario == "S1":
        return make_S1_stream(datasets=ds_dv, seed=seed,
                              pre_shift_datasets=COV_PRE,
                              post_shift_datasets=COV_POST,
                              shift_frac=0.5, length=n)
    if scenario == "S2":
        return make_S2_stream(datasets=ds_dv, seed=seed,
                              pre_shift_datasets=COV_PRE,
                              post_shift_datasets=COV_POST,
                              shift_start_frac=0.3, shift_end_frac=0.7,
                              length=n)
    if scenario == "S3":
        return make_S3_stream(n=n, seed=seed, lam=lam,
                              drift_factors=S3_DRIFT,
                              change_point_frac=0.5, length=n)
    raise ValueError(scenario)


def summarise_run(df, stream_meta, scenario, K, ds_labels):
    macro_util = float(df.groupby("dataset")["reward"].mean().mean())
    macro_qual = float(df.groupby("dataset")["quality"].mean().mean())
    row = {
        "macro_utility":   macro_util,
        "macro_quality":   macro_qual,
        "micro_utility":   float(df["reward"].mean()),
        "micro_quality":   float(df["quality"].mean()),
        "micro_cost_norm": float(df["cost_norm"].mean()),
        "cum_regret":      float(df["cumulative_regret"].iloc[-1]),
        "total_regret":    float(df["instant_regret"].sum()),
    }
    for a in range(K):
        row[f"share_arm_{a}"] = float((df["action"] == a).mean())

    # Shift-related metrics
    if "change_point" in stream_meta:
        cp = int(stream_meta["change_point"])
        post = df[(df["t"] >= cp) & (df["t"] < cp + 200)]
        row["post_shift_regret_200"] = float(post["instant_regret"].mean()) \
                                         if len(post) else np.nan
        pre_len = min(cp, 500)
        if pre_len >= 50 and cp + 50 < len(df):
            pre_u = df.loc[cp - pre_len:cp, "reward"].mean()
            target = 0.95 * pre_u
            post_series = df[df["t"] >= cp]["reward"].rolling(50, min_periods=1).mean().values
            hits = np.where(post_series >= target)[0]
            row["recovery_time"] = int(hits[0]) if len(hits) else -1
        else:
            row["recovery_time"] = -1
    elif scenario == "S2":
        # Gradual shift — we use shift_start as the reference point for PSR
        cp = int(stream_meta.get("shift_start", -1))
        if cp > 0:
            post = df[(df["t"] >= cp) & (df["t"] < cp + 200)]
            row["post_shift_regret_200"] = float(post["instant_regret"].mean()) \
                                             if len(post) else np.nan
            pre_len = min(cp, 500)
            if pre_len >= 50 and cp + 50 < len(df):
                pre_u = df.loc[cp - pre_len:cp, "reward"].mean()
                target = 0.95 * pre_u
                post_series = df[df["t"] >= cp]["reward"].rolling(50, min_periods=1).mean().values
                hits = np.where(post_series >= target)[0]
                row["recovery_time"] = int(hits[0]) if len(hits) else -1
            else:
                row["recovery_time"] = -1
        else:
            row["post_shift_regret_200"] = np.nan
            row["recovery_time"] = -1
    else:
        row["post_shift_regret_200"] = np.nan
        row["recovery_time"] = -1

    return row


def oracle_summary_row(Rs, Qs, Cns, ds_s, K, scenario, stream_meta):
    """Oracle is the per-step argmax of R. Compute metrics directly (no evaluator)."""
    n = len(Rs)
    action = Rs.argmax(axis=1)
    rows = np.arange(n)
    quality   = Qs[rows, action]
    cost_norm = Cns[rows, action]
    reward    = Rs[rows, action]
    df = pd.DataFrame({
        "t": rows, "action": action, "quality": quality,
        "cost_norm": cost_norm, "reward": reward,
        "instant_regret": np.zeros(n),
        "cumulative_regret": np.zeros(n),
        "dataset": ds_s,
    })
    return summarise_run(df, stream_meta, scenario, K, ds_s)


def main():
    hps = load_frozen_hps()
    print("Frozen HPs:")
    for r, cfg in hps.items():
        print(f"  {r}: {cfg}")

    X_tr, Q_tr, C_tr, Cn_tr, _ = load_matrices("TRAIN")
    X_dv, Q_dv, C_dv, Cn_dv, ds_dv = load_matrices("DEV")
    print(f"\nTRAIN {X_tr.shape}   DEV {X_dv.shape}")
    print(f"Routers ({len(ROUTERS)}): {ROUTERS}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Lambdas: {LAMBDAS}   Scenarios: {SCENARIOS}")

    total = len(ROUTERS) * len(SEEDS) * len(LAMBDAS) * len(SCENARIOS)
    print(f"Total runs: {total}\n", flush=True)

    K, dim = Q_tr.shape[1], X_tr.shape[1]
    out_path = Path("results/tables/p5_final_dev.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    t_global = time.perf_counter()
    for router_name in ROUTERS:
        t_router = time.perf_counter()
        for seed in SEEDS:
            for lam in LAMBDAS:
                R_tr = Q_tr - lam * Cn_tr
                R_dv = Q_dv - lam * Cn_dv

                if router_name == "Oracle":
                    # Oracle uses no fit/warm-up. Per-scenario, compute per-step
                    # argmax on the shifted reward stream directly.
                    for scen in SCENARIOS:
                        stream = build_stream(scen, X_dv, ds_dv, seed, lam)
                        o = stream.order
                        # Apply shift schedule to (Q, Cn) once before argmax
                        Q_stream = Q_dv[o]
                        Cn_stream = Cn_dv[o].copy()
                        R_stream = Q_stream - lam * Cn_stream
                        # For S3 the shift multiplies Cn permanently from cp onward.
                        # _make_cost_shift_fn ignores its `t` argument and scales
                        # every row it receives — correct inside run_stream (rows
                        # < cp are already consumed by the sequential loop), but
                        # wrong here where the whole array is processed at once.
                        # Slice explicitly so rows < cp keep their pre-drift cost.
                        if scen == "S3" and stream.shift_schedule:
                            cp = list(stream.shift_schedule.keys())[0]
                            fn = stream.shift_schedule[cp]
                            R_post, _, Cn_post = fn(cp, R_stream[cp:],
                                                     Q_stream[cp:], Cn_stream[cp:])
                            R_stream  = np.concatenate([R_stream[:cp], R_post])
                            Cn_stream = np.concatenate([Cn_stream[:cp], Cn_post])
                        row = oracle_summary_row(R_stream, Q_stream, Cn_stream,
                                                  ds_dv[o], K, scen, stream.metadata)
                        row.update({"router": router_name, "seed": seed,
                                     "lambda": lam, "scenario": scen})
                        all_rows.append(row)
                    continue

                router = build_router(router_name, hps, K, dim)
                fit_or_warmup(router_name, router, X_tr, Q_tr, Cn_tr, C_tr, R_tr,
                              seed, lam)
                snap = deepcopy(router) if router_name in ONLINE else None

                for scen in SCENARIOS:
                    r = deepcopy(snap) if snap is not None else router
                    stream = build_stream(scen, X_dv, ds_dv, seed, lam)
                    o = stream.order
                    Xs = X_dv[o]; Qs = Q_dv[o]; Cns = Cn_dv[o]; Rs = R_dv[o]
                    ds_s = ds_dv[o]
                    orc = OracleRouter(n_arms=K); orc.fit(Xs)
                    res = run_stream(r, Xs, Rs, Qs, Cns, orc,
                                      lam=lam, seed=seed, scenario=scen,
                                      shift_schedule=stream.shift_schedule)
                    df = res.to_dataframe()
                    df["dataset"] = ds_s
                    row = summarise_run(df, stream.metadata, scen, K, ds_s)
                    row.update({"router": router_name, "seed": seed,
                                 "lambda": lam, "scenario": scen,
                                 **{f"hp_{k}": v for k, v in
                                    (hps.get(router_name, {}) or {}).items()}})
                    all_rows.append(row)

        # per-router checkpoint
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        dt = time.perf_counter() - t_router
        elapsed = (time.perf_counter() - t_global) / 60
        print(f"=== {router_name:11s} done  ({dt:.1f}s)   "
              f"elapsed={elapsed:.1f}min   rows={len(all_rows)}", flush=True)

    total_min = (time.perf_counter() - t_global) / 60
    print(f"\nP5 complete in {total_min:.1f} min. Rows: {len(all_rows)}. "
          f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
