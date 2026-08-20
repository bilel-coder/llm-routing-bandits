"""
P5a analysis — aggregate sweep results, select best HP per router, freeze.

Reads:   results/tables/hp_sweep_dev.csv
Writes:  results/tables/hp_selection_summary.csv
         results/tables/hp_sensitivity_*.csv
         configs/algorithms/{knn,ridge,linucb,discounted_linucb}.yaml  (frozen)
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CSV = Path("results/tables/hp_sweep_dev.csv")
OUT_TABLES = Path("results/tables")
OUT_CONFIGS = Path("configs/algorithms")
OUT_CONFIGS.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(x: np.ndarray, n=2000, ci=0.95, rng=None) -> tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x)
    boot = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(x.mean()), float(lo), float(hi)


def aggregate_configs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per (router, config_id) across all (scenario, lambda, seed).
    Primary metric: macro_utility. Report mean + 95% CI + stability.
    """
    rows = []
    for (router, cid), g in df.groupby(["router", "config_id"]):
        cfg = json.loads(cid)
        # 1) Per-seed, average macro_utility over (scenario × lambda)
        per_seed = g.groupby("seed")["macro_utility"].mean().values
        mean, lo, hi = bootstrap_ci(per_seed)
        # 2) Post-shift regret aggregated on drift scenarios
        ps = g[g["scenario"].isin(["S1", "S3"])]["post_shift_regret_200"].dropna().values
        # 3) Stationary macro
        stat = g[g["scenario"] == "S0"]["macro_utility"].mean()
        # 4) Drift macro
        drift = g[g["scenario"].isin(["S1", "S3"])]["macro_utility"].mean()
        # 5) Std across seeds (stability)
        std_seed = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0
        rows.append({
            "router": router,
            "config_id": cid,
            **cfg,
            "macro_util_mean":  round(mean, 5),
            "macro_util_ci_lo": round(lo, 5),
            "macro_util_ci_hi": round(hi, 5),
            "macro_util_std_seed": round(std_seed, 5),
            "macro_util_S0":    round(float(stat), 5),
            "macro_util_drift": round(float(drift), 5),
            "post_shift_regret_mean": round(float(ps.mean()), 5) if len(ps) else float("nan"),
            "n_seeds": int(len(per_seed)),
        })
    return pd.DataFrame(rows)


def rank_and_print(agg: pd.DataFrame) -> pd.DataFrame:
    winners = []
    for router, sub in agg.groupby("router"):
        sub = sub.sort_values("macro_util_mean", ascending=False)
        print(f"\n=== {router} ranked by macro_util_mean ===")
        cols = ["config_id", "macro_util_mean", "macro_util_ci_lo",
                "macro_util_ci_hi", "macro_util_std_seed", "macro_util_S0",
                "macro_util_drift", "post_shift_regret_mean"]
        print(sub[cols].to_string(index=False))
        winners.append(sub.iloc[0])
    return pd.DataFrame(winners)


def sensitivity_tables(agg: pd.DataFrame):
    """Marginal effect of each hyperparameter, averaging over the others."""
    tables = {}
    for router, sub in agg.groupby("router"):
        if router == "kNN":
            piv = sub.pivot_table(index="k", values="macro_util_mean")
        elif router in ("Ridge", "LinUCB"):
            piv = sub.pivot_table(index="alpha", values="macro_util_mean")
        elif router == "D-LinUCB":
            piv = sub.pivot_table(index="alpha", columns="gamma",
                                   values="macro_util_mean")
        else:
            continue
        tables[router] = piv
        print(f"\nSensitivity — {router}\n{piv.round(4)}")
    return tables


def stationary_vs_drift(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for router, sub in agg.groupby("router"):
        best = sub.sort_values("macro_util_mean", ascending=False).iloc[0]
        rows.append({
            "router": router,
            "best_cfg": best["config_id"],
            "S0_macro_util":    best["macro_util_S0"],
            "drift_macro_util": best["macro_util_drift"],
            "delta_S0_vs_drift": round(best["macro_util_S0"] - best["macro_util_drift"], 4),
        })
    df = pd.DataFrame(rows)
    print("\n=== Stationary vs Drift for BEST config per router ===")
    print(df.to_string(index=False))
    return df


def freeze_configs(winners: pd.DataFrame) -> None:
    """Write yaml configs for the frozen HP choice per router."""
    for _, r in winners.iterrows():
        cfg = json.loads(r["config_id"])
        yaml_path = {
            "kNN":      OUT_CONFIGS / "knn.yaml",
            "Ridge":    OUT_CONFIGS / "ridge.yaml",
            "LinUCB":   OUT_CONFIGS / "linucb.yaml",
            "D-LinUCB": OUT_CONFIGS / "discounted_linucb.yaml",
        }[r["router"]]
        payload = {
            "router": r["router"],
            "hyperparameters": cfg,
            "selection": {
                "criterion": "macro_utility_mean_across_DEV_conditions",
                "macro_util_mean": float(r["macro_util_mean"]),
                "macro_util_ci_lo": float(r["macro_util_ci_lo"]),
                "macro_util_ci_hi": float(r["macro_util_ci_hi"]),
                "macro_util_std_seed": float(r["macro_util_std_seed"]),
                "n_seeds": int(r["n_seeds"]),
                "frozen_partition": "DEV",
                "final_test_touched": False,
            },
        }
        with open(yaml_path, "w") as f:
            yaml.dump(payload, f, sort_keys=False)
        print(f"  Frozen: {yaml_path}")


def main():
    if not CSV.exists():
        print(f"ERROR: {CSV} not found. Run scripts/10_hp_sweep.py first.")
        return
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows.")
    print(f"Router counts:\n{df['router'].value_counts()}")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")
    print(f"Lambdas:   {sorted(df['lambda'].unique())}")
    print(f"Seeds:     {sorted(df['seed'].unique())}")

    agg = aggregate_configs(df)
    agg.to_csv(OUT_TABLES / "hp_selection_summary.csv", index=False)

    winners = rank_and_print(agg)
    sens = sensitivity_tables(agg)
    for r, tbl in sens.items():
        tbl.to_csv(OUT_TABLES / f"hp_sensitivity_{r.lower().replace('-','')}.csv")

    stat_drift = stationary_vs_drift(agg)
    stat_drift.to_csv(OUT_TABLES / "hp_stationary_vs_drift.csv", index=False)

    print("\n" + "=" * 70)
    print("FREEZING BEST HP PER ROUTER")
    print("=" * 70)
    freeze_configs(winners)


if __name__ == "__main__":
    main()
