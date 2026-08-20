"""
P5a — hyperparameter sweep on DEV only (TRAIN for fitting/warmup).

Protocol:
  * Configs per router:
      kNN    : k     ∈ {5, 10, 20, 50}
      Ridge  : alpha ∈ {0.01, 0.1, 1.0, 10.0}
      LinUCB : alpha ∈ {0.1, 0.25, 0.5, 1.0, 2.0}
      D-LinUCB : alpha × gamma  (5 × 5 = 25 configs)
  * Scenarios: S0 (stationary), S1 (abrupt covariate shift),
               S3 (cost drift — kimi & deepseek-v3 costs doubled at t=T/2)
  * Lambdas: {0.0, 0.1, 0.25, 0.5, 1.0}  (frozen grid)
  * Seeds:  configurable via --seeds N  (default 5)

Selection principle: ONE robust config per router, chosen by macro mean
utility across all (scenario, lambda, seed) DEV conditions.

FINAL_TEST is not accessed.
"""

from __future__ import annotations

import argparse
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
from llm_router.routers.knn import KNNRouter
from llm_router.routers.ridge import RidgeRouter
from llm_router.routers.linucb import LinUCBRouter
from llm_router.routers.discounted_linucb import DiscountedLinUCBRouter
from llm_router.routers.oracle import OracleRouter
from llm_router.evaluation.evaluator import run_stream
from llm_router.environments import (
    make_S0_stream,
    make_S1_stream,
    make_S3_stream,
)

INTERIM = Path("data/interim")
MAT     = INTERIM / "matrices"
CONFIG  = Path("configs/model_pool.yaml")

LAMBDAS   = [0.0, 0.1, 0.25, 0.5, 1.0]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# ---- Scenarios --------------------------------------------------------------
# S1 covariate-shift dataset split (both halves are non-empty on DEV)
S1_PRE  = ["arenahard", "hle", "mmlupro", "simpleqa"]
S1_POST = ["aime", "arc-agi", "gpqa", "livecodebench",
           "livemathbench", "swe-bench", "tau2"]

# S3 cost drift: double the cost of the two cheapest arms at t=T/2
# Pool order: [claude-s4, gem-flash, gem-pro, gpt-5, dsk-r1, kimi, glm-4.6, dsk-v3]
S3_DRIFT_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)

# ---- Hyperparameter grids ---------------------------------------------------
KNN_CONFIGS      = [{"k": k}     for k in [5, 10, 20, 50]]
RIDGE_CONFIGS    = [{"alpha": a} for a in [0.01, 0.1, 1.0, 10.0]]
LINUCB_CONFIGS   = [{"alpha": a} for a in [0.1, 0.25, 0.5, 1.0, 2.0]]
DLINUCB_CONFIGS  = [{"alpha": a, "gamma": g}
                    for a in [0.1, 0.25, 0.5, 1.0, 2.0]
                    for g in [0.95, 0.98, 0.99, 0.995, 0.999]]

ONLINE_NAMES = {"LinUCB", "D-LinUCB"}


def load_matrices(role: str):
    p = partition_for(role)
    X = np.load(INTERIM / f"X_{p}.npy")
    Q = np.load(MAT / f"Q_{p}.npy")
    C = np.load(MAT / f"C_{p}.npy")
    Cn = np.load(MAT / f"C_norm_{p}.npy")
    with open(MAT / f"datasets_{p}.json") as f:
        ds = json.load(f)
    return X, Q, C, Cn, np.array(ds)


def build_router(router_name: str, cfg: dict, K: int, dim: int):
    if router_name == "kNN":
        return KNNRouter(n_arms=K, k=cfg["k"])
    if router_name == "Ridge":
        return RidgeRouter(n_arms=K, alpha=cfg["alpha"])
    if router_name == "LinUCB":
        return LinUCBRouter(n_arms=K, context_dim=dim, alpha=cfg["alpha"])
    if router_name == "D-LinUCB":
        return DiscountedLinUCBRouter(
            n_arms=K, context_dim=dim,
            alpha=cfg["alpha"], gamma=cfg["gamma"],
        )
    raise ValueError(router_name)


def fit_or_warmup(router_name, router, X_tr, Q_tr, Cn_tr, R_tr, seed, lam):
    """Offline: full-info fit. Online: warmup on TRAIN with selected-arm feedback."""
    if router_name in ONLINE_NAMES:
        router.fit(X_tr)
        stream = make_S0_stream(n=X_tr.shape[0], seed=seed)
        o = stream.order
        orc = OracleRouter(n_arms=router.n_arms); orc.fit(X_tr)
        run_stream(router, X_tr[o], R_tr[o], Q_tr[o], Cn_tr[o],
                    orc, lam=lam, seed=seed, scenario="warmup")
    else:
        router.fit(X_tr, rewards=R_tr, lam=lam)


def build_stream(scenario, X_dv, ds_dv, seed, lam):
    n = X_dv.shape[0]
    if scenario == "S0":
        return make_S0_stream(n=n, seed=seed)
    if scenario == "S1":
        return make_S1_stream(datasets=ds_dv, seed=seed,
                              pre_shift_datasets=S1_PRE,
                              post_shift_datasets=S1_POST,
                              shift_frac=0.5, length=n)
    if scenario == "S3":
        return make_S3_stream(n=n, seed=seed, lam=lam,
                              drift_factors=S3_DRIFT_FACTORS,
                              change_point_frac=0.5, length=n)
    raise ValueError(scenario)


def eval_one(router, stream, X_dv, Q_dv, Cn_dv, ds_dv, lam, seed, scenario):
    """Run one deterministic evaluation. Returns tidy per-step DataFrame."""
    K = Q_dv.shape[1]
    R_dv = Q_dv - lam * Cn_dv
    o = stream.order
    Xs, Qs, Cns, Rs, ds_s = X_dv[o], Q_dv[o], Cn_dv[o], R_dv[o], ds_dv[o]
    orc = OracleRouter(n_arms=K); orc.fit(Xs)
    res = run_stream(router, Xs, Rs, Qs, Cns, orc,
                     lam=lam, seed=seed, scenario=scenario,
                     shift_schedule=stream.shift_schedule)
    df = res.to_dataframe()
    df["dataset"] = ds_s
    return df, stream.metadata


def summarise_run(df: pd.DataFrame, stream_meta: dict, scenario: str,
                  ds_labels: np.ndarray) -> dict:
    """Extract scalar metrics from a single-run tidy DataFrame."""
    macro_util = float(df.groupby("dataset")["reward"].mean().mean())
    macro_qual = float(df.groupby("dataset")["quality"].mean().mean())
    micro_util = float(df["reward"].mean())
    micro_qual = float(df["quality"].mean())
    micro_cn   = float(df["cost_norm"].mean())
    cum_regret = float(df["cumulative_regret"].iloc[-1])
    total_regret = float(df["instant_regret"].sum())

    out = {
        "macro_utility": macro_util,
        "macro_quality": macro_qual,
        "micro_utility": micro_util,
        "micro_quality": micro_qual,
        "micro_cost_norm": micro_cn,
        "cum_regret": cum_regret,
        "total_regret": total_regret,
    }

    # Post-shift regret + recovery time when a change point exists
    if scenario in ("S1", "S3") and "change_point" in stream_meta:
        cp = int(stream_meta["change_point"])
        # Mean instant regret in the 200-step window after the shift
        post = df[(df["t"] >= cp) & (df["t"] < cp + 200)]
        out["post_shift_regret_200"] = float(post["instant_regret"].mean()) \
                                        if len(post) else float("nan")

        # Recovery time: steps after cp until rolling-50 utility reaches
        # 95% of pre-shift utility (window pre_len = min(cp, 500))
        pre_len = min(cp, 500)
        if pre_len >= 50 and cp + 50 < len(df):
            pre_u = df.loc[cp - pre_len:cp, "reward"].mean()
            target = 0.95 * pre_u
            post_series = df[df["t"] >= cp]["reward"].rolling(50, min_periods=1).mean().values
            hits = np.where(post_series >= target)[0]
            out["recovery_time"] = int(hits[0]) if len(hits) else -1
        else:
            out["recovery_time"] = -1
    else:
        out["post_shift_regret_200"] = float("nan")
        out["recovery_time"] = -1

    return out


def run_config(router_name, cfg, seeds, lambdas, scenarios,
               X_tr, Q_tr, Cn_tr, X_dv, Q_dv, C_dv, Cn_dv, ds_dv):
    """Run all (seed × lambda × scenario) evaluations for one config."""
    K = Q_tr.shape[1]
    d = X_tr.shape[1]
    rows = []
    for seed in seeds:
        for lam in lambdas:
            R_tr = Q_tr - lam * Cn_tr
            router = build_router(router_name, cfg, K, d)
            # Fit (offline) or warm-up (online) — depends on lambda
            fit_or_warmup(router_name, router, X_tr, Q_tr, Cn_tr, R_tr, seed, lam)
            # Snapshot warmed state so each scenario starts from the same point
            snap = deepcopy(router) if router_name in ONLINE_NAMES else None
            for scen in scenarios:
                r = deepcopy(snap) if snap is not None else router
                stream = build_stream(scen, X_dv, ds_dv, seed, lam)
                df, meta = eval_one(r, stream, X_dv, Q_dv, Cn_dv, ds_dv,
                                     lam, seed, scen)
                m = summarise_run(df, meta, scen, ds_dv)
                m.update({
                    "router": router_name,
                    "seed": seed, "lambda": lam, "scenario": scen,
                    **{f"hp_{k}": v for k, v in cfg.items()},
                    "config_id": json.dumps(cfg, sort_keys=True),
                })
                rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of seeds (starts at 42, sequential).")
    ap.add_argument("--scenarios", nargs="+", default=["S0", "S1", "S3"])
    ap.add_argument("--only", nargs="+", default=None,
                    help="Restrict to router names (e.g. --only kNN Ridge).")
    ap.add_argument("--out", default="results/tables/hp_sweep_dev.csv")
    args = ap.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    scenarios = args.scenarios

    X_tr, Q_tr, _, Cn_tr, _ = load_matrices("TRAIN")
    X_dv, Q_dv, C_dv, Cn_dv, ds_dv = load_matrices("DEV")
    print(f"TRAIN: {X_tr.shape}   DEV: {X_dv.shape}")
    print(f"Seeds: {seeds}   Lambdas: {LAMBDAS}   Scenarios: {scenarios}")

    router_grids = {
        "kNN":       KNN_CONFIGS,
        "Ridge":     RIDGE_CONFIGS,
        "LinUCB":    LINUCB_CONFIGS,
        "D-LinUCB":  DLINUCB_CONFIGS,
    }
    if args.only:
        router_grids = {k: v for k, v in router_grids.items() if k in args.only}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    total_configs = sum(len(v) for v in router_grids.values())
    total_runs = total_configs * len(seeds) * len(LAMBDAS) * len(scenarios)
    print(f"Total: {total_configs} configs × {len(seeds)} seeds × "
          f"{len(LAMBDAS)} lambdas × {len(scenarios)} scenarios = {total_runs} runs")

    t_global = time.perf_counter()
    completed_configs = 0
    for router_name, configs in router_grids.items():
        for i, cfg in enumerate(configs, 1):
            t0 = time.perf_counter()
            rows = run_config(router_name, cfg, seeds, LAMBDAS, scenarios,
                              X_tr, Q_tr, Cn_tr, X_dv, Q_dv, C_dv, Cn_dv, ds_dv)
            all_rows.extend(rows)
            completed_configs += 1
            dt = time.perf_counter() - t0
            elapsed = time.perf_counter() - t_global
            remaining = total_configs - completed_configs
            eta_min = (elapsed / completed_configs) * remaining / 60.0
            print(f"[{completed_configs:>2}/{total_configs}] "
                  f"{router_name:8s} {i}/{len(configs)} {cfg}  "
                  f"({dt:5.1f}s)  "
                  f"macro_util={np.mean([r['macro_utility'] for r in rows]):.4f}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta_min:.1f}min",
                  flush=True)

            # Incremental save so partial results are always on disk
            pd.DataFrame(all_rows).to_csv(out_path, index=False)

    total = time.perf_counter() - t_global
    print(f"\nSweep complete in {total/60:.1f} min. Rows: {len(all_rows)}. "
          f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
