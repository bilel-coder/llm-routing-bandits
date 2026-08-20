"""
M2.5 — Stationary pilot on DEV (scenario=S0, seed=42, lambda=0.25).

Protocol:
  * Offline routers (Cheapest, BestSingle, kNN, Ridge) fit on TRAIN with
    full-information rewards.
  * Online routers (LinUCB, D-LinUCB) additionally run a WARM-UP pass over
    TRAIN with selected-arm feedback only (deterministic ordering per seed);
    their learned state is carried into the DEV evaluation stream.
  * All routers share ONE deterministic S0 stream over DEV for one seed.
  * Oracle routing shares are reported with fractional-tie credit; naive
    argmax shares are also shown for contrast.
  * FINAL_TEST is not loaded.

Does NOT run the full seeds × lambdas × scenarios grid — this is a single
end-to-end verification that (a) all routers run, (b) numbers pass a smell
test, (c) evaluator + stream + warm-up integration is correct.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_router.data.roles import partition_for
from llm_router.routers.cheapest import CheapestRouter
from llm_router.routers.best_single import BestSingleRouter
from llm_router.routers.knn import KNNRouter
from llm_router.routers.ridge import RidgeRouter
from llm_router.routers.linucb import LinUCBRouter
from llm_router.routers.discounted_linucb import DiscountedLinUCBRouter
from llm_router.routers.oracle import OracleRouter
from llm_router.evaluation.evaluator import run_stream
from llm_router.evaluation.metrics import oracle_tie_stats
from llm_router.environments import make_S0_stream

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INTERIM = Path("data/interim")
MAT     = INTERIM / "matrices"
CONFIG  = Path("configs/model_pool.yaml")

SEED   = 42
LAM    = 0.25
LINUCB_ALPHA  = 1.0
DLINUCB_ALPHA = 1.0
DLINUCB_GAMMA = 0.99
KNN_K = 10

ONLINE_ROUTERS = {"LinUCBRouter", "DiscountedLinUCBRouter"}


def load_matrices(role: str):
    part = partition_for(role)  # role → physical partition
    X = np.load(INTERIM / f"X_{part}.npy")
    Q = np.load(MAT / f"Q_{part}.npy")
    C = np.load(MAT / f"C_{part}.npy")
    Cn = np.load(MAT / f"C_norm_{part}.npy")
    with open(MAT / f"datasets_{part}.json") as f:
        ds = json.load(f)
    return X, Q, C, Cn, np.array(ds)


def build_routers(K: int, dim: int) -> dict[str, object]:
    return {
        "Cheapest":     CheapestRouter(n_arms=K),
        "BestSingle":   BestSingleRouter(n_arms=K),
        "kNN":          KNNRouter(n_arms=K, k=KNN_K),
        "Ridge":        RidgeRouter(n_arms=K, alpha=1.0),
        "LinUCB":       LinUCBRouter(n_arms=K, context_dim=dim, alpha=LINUCB_ALPHA),
        "D-LinUCB":     DiscountedLinUCBRouter(n_arms=K, context_dim=dim,
                                               alpha=DLINUCB_ALPHA, gamma=DLINUCB_GAMMA),
    }


def fit_offline(router, X_train, R_train, C_train, lam):
    """Full-information offline fit (BestSingle, kNN, Ridge, Cheapest)."""
    name = router.__class__.__name__
    if name == "CheapestRouter":
        router.fit(X_train, cost_matrix=C_train)
    elif name in ("BestSingleRouter", "KNNRouter", "RidgeRouter"):
        router.fit(X_train, rewards=R_train, lam=lam)


def warmup_online(router, X_train, Q_train, Cn_train, R_train, seed, lam):
    """
    Warm-up: run online router through TRAIN sequentially with selected-arm
    feedback only. State is carried into subsequent evaluation calls.

    The warm-up stream is a deterministic permutation of TRAIN under the
    given seed, so matched algorithms/seeds see identical orderings.
    A throwaway Oracle is used to satisfy the evaluator signature (its
    step-rewards are discarded — Oracle is not the router being warmed).
    """
    name = router.__class__.__name__
    if name not in ONLINE_ROUTERS:
        return
    router.fit(X_train)  # no-op for online, but keeps interface uniform
    n_train = X_train.shape[0]
    stream = make_S0_stream(n=n_train, seed=seed)
    order = stream.order
    dummy_oracle = OracleRouter(n_arms=router.n_arms)
    dummy_oracle.fit(X_train)
    run_stream(
        router,
        X_train[order],
        R_train[order],
        Q_train[order],
        Cn_train[order],
        dummy_oracle,
        lam=lam, seed=seed, scenario="warmup", experiment="pilot_warmup",
    )


def main() -> None:
    with open(CONFIG) as f:
        pool = [m["name"] for m in yaml.safe_load(f)["pool"]]
    K = len(pool)

    # ---- TRAIN (for fitting and warm-up) ----------------------------------
    X_tr, Q_tr, C_tr, Cn_tr, _ = load_matrices("TRAIN")
    R_tr = Q_tr - LAM * Cn_tr
    print(f"TRAIN: X={X_tr.shape}  Q={Q_tr.shape}  C={C_tr.shape}  Cn={Cn_tr.shape}")

    # ---- DEV (the pilot evaluation partition — NEVER FINAL_TEST) ----------
    X_dv, Q_dv, C_dv, Cn_dv, ds_dv = load_matrices("DEV")
    R_dv = Q_dv - LAM * Cn_dv
    print(f"DEV:   X={X_dv.shape}  Q={Q_dv.shape}  R={R_dv.shape}")

    # ---- Deterministic S0 stream over DEV (shared by all routers) ---------
    stream = make_S0_stream(n=X_dv.shape[0], seed=SEED)
    order = stream.order
    Xs   = X_dv[order]
    Qs   = Q_dv[order]
    Cs   = C_dv[order]
    Cns  = Cn_dv[order]
    Rs   = R_dv[order]
    ds_s = ds_dv[order]
    T = len(order)
    print(f"S0 stream on DEV: T={T}  (seed={SEED})")

    # Oracle segregated code path
    oracle = OracleRouter(n_arms=K)
    oracle.fit(Xs)

    routers = build_routers(K=K, dim=X_tr.shape[1])
    results: dict[str, pd.DataFrame] = {}

    for name, router in routers.items():
        fit_offline(router, X_tr, R_tr, C_tr, LAM)
        warmup_online(router, X_tr, Q_tr, Cn_tr, R_tr, seed=SEED, lam=LAM)
        res = run_stream(router, Xs, Rs, Qs, Cns, oracle, lam=LAM, seed=SEED,
                         scenario="S0", experiment="pilot")
        df = res.to_dataframe()
        df["router"] = name
        df["dataset"] = ds_s
        results[name] = df
        print(f"  [{name:12s}]  mean_util={df['reward'].mean():.4f}  "
              f"mean_quality={df['quality'].mean():.4f}  "
              f"mean_cost_norm={df['cost_norm'].mean():.4f}  "
              f"cum_regret={df['cumulative_regret'].iloc[-1]:.2f}")

    # ---- Oracle utility row (regret zero by construction) -----------------
    oracle_best_arm = Rs.argmax(axis=1)
    rows = np.arange(T)
    oracle_result_df = pd.DataFrame({
        "t":                 np.arange(T),
        "action":            oracle_best_arm,
        "quality":           Qs[rows, oracle_best_arm],
        "cost_norm":         Cns[rows, oracle_best_arm],
        "reward":            Rs.max(axis=1),
        "instant_regret":    np.zeros(T),
        "cumulative_regret": np.zeros(T),
        "router":            "Oracle",
        "dataset":           ds_s,
    })
    results["Oracle"] = oracle_result_df

    # ---- Oracle tie-aware shares ------------------------------------------
    frac_shares, tie_rate, ties_per_row = oracle_tie_stats(Rs)
    print("\nOracle tie analysis (on DEV stream, λ={:.2f}):".format(LAM))
    print(f"  tie_rate: {tie_rate*100:.2f}% of rows have ≥2 tied arms at max R")
    print(f"  max ties on a single row: {int(ties_per_row.max())}")
    print(f"  mean ties per row: {ties_per_row.mean():.3f}")

    # ---- Summary table ----------------------------------------------------
    rows = []
    for name, df in results.items():
        rows.append({
            "router":         name,
            "mean_quality":   round(df["quality"].mean(), 4),
            "mean_cost_norm": round(df["cost_norm"].mean(), 4),
            "mean_utility":   round(df["reward"].mean(), 4),
            "cum_regret":     round(df["cumulative_regret"].iloc[-1], 2),
        })
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(f"PILOT — S0 stationary, DEV, seed={SEED}, lambda={LAM}")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nRouting shares per router (naive argmax; column order dependent):")
    for name, df in results.items():
        shares = np.array([(df["action"] == a).mean() for a in range(K)])
        share_str = ", ".join(f"{pool[a]}:{shares[a]*100:.1f}%" for a in range(K))
        print(f"  [{name:12s}]  {share_str}")

    print("\nOracle FRACTIONAL routing shares (ties split equally):")
    frac_share_str = ", ".join(f"{pool[a]}:{frac_shares[a]*100:.1f}%" for a in range(K))
    print(f"  [Oracle-frac] {frac_share_str}")

    # ---- Figures ----------------------------------------------------------
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for name, df in results.items():
        if name == "Oracle":
            continue
        plt.plot(df["t"], df["cumulative_regret"], label=name, linewidth=1.4)
    plt.xlabel("Stream step t")
    plt.ylabel("Cumulative regret")
    plt.title(f"S0 stationary (DEV) — cumulative regret (seed={SEED}, λ={LAM})")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "pilot_S0_cumulative_regret.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 5))
    router_names = list(results.keys())
    x = np.arange(len(pool))
    width = 0.11
    for i, name in enumerate(router_names):
        df = results[name]
        shares = np.array([(df["action"] == a).mean() for a in range(K)])
        ax.bar(x + i * width, shares * 100, width, label=name)
    ax.set_xticks(x + (len(router_names) - 1) * width / 2)
    ax.set_xticklabels(pool, rotation=25, ha="right")
    ax.set_ylabel("Routing share (%)")
    ax.set_title(f"S0 stationary (DEV) — naive routing shares (seed={SEED}, λ={LAM})")
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()
    plt.savefig(fig_dir / "pilot_S0_routing_shares.png", dpi=120)
    plt.close()

    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "pilot_S0_summary.csv", index=False)

    # Save tie analysis
    tie_report = {
        "seed": SEED, "lambda": LAM, "partition": "DEV",
        "n_rows": int(T), "tie_rate": float(tie_rate),
        "max_ties_on_row": int(ties_per_row.max()),
        "mean_ties_per_row": float(ties_per_row.mean()),
        "fractional_oracle_shares": {pool[a]: float(frac_shares[a])
                                       for a in range(K)},
    }
    with open(out / "pilot_S0_oracle_tie_report.json", "w") as f:
        json.dump(tie_report, f, indent=2)

    print(f"\nFigures: {fig_dir}/pilot_S0_cumulative_regret.png")
    print(f"         {fig_dir}/pilot_S0_routing_shares.png")
    print(f"Table:   {out}/pilot_S0_summary.csv")
    print(f"Ties:    {out}/pilot_S0_oracle_tie_report.json")


if __name__ == "__main__":
    main()
