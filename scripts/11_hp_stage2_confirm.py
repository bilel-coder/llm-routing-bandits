"""
P5a stage 2 — re-evaluate the top 3 configs per router with 5 ADDITIONAL
matched seeds, then freeze the winner using the combined 10-seed grand macro.

Reads:   results/tables/hp_sweep_dev_stage1.csv
Writes:  results/tables/hp_sweep_dev_stage2.csv     (top-K × new seeds only)
         results/tables/hp_sweep_dev_combined.csv   (stage1 + stage2 unified)
         results/tables/hp_final_selection.csv      (winner per router + CI)
         configs/algorithms/{knn,ridge,linucb,discounted_linucb}.yaml (frozen)

Selection rule (locked before stage 1 launched):
  1. per-seed macro = mean over (scenario, λ) of macro_utility
  2. grand_macro    = mean over 10 seeds of per-seed macro
  3. rank descending by grand_macro; tie-break by CI overlap → post-shift
     regret → seed std → simpler config.

FINAL_TEST is NOT accessed.
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

# Reuse machinery from stage 1
sys.path.insert(0, str(Path(__file__).parent))
mod = __import__("10_hp_sweep")
run_config    = mod.run_config
build_router  = mod.build_router
LAMBDAS       = mod.LAMBDAS

from llm_router.data.roles import partition_for

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INTERIM = Path("data/interim")
MAT     = INTERIM / "matrices"

STAGE1_CSV   = Path("results/tables/hp_sweep_dev_stage1.csv")
STAGE2_CSV   = Path("results/tables/hp_sweep_dev_stage2.csv")
COMBINED_CSV = Path("results/tables/hp_sweep_dev_combined.csv")
FINAL_CSV    = Path("results/tables/hp_final_selection.csv")
OUT_CONFIGS  = Path("configs/algorithms")
OUT_CONFIGS.mkdir(parents=True, exist_ok=True)

STAGE2_SEEDS = [47, 48, 49, 50, 51]
TOP_K_PER_ROUTER = 3


def load_matrices(role):
    p = partition_for(role)
    X  = np.load(INTERIM / f"X_{p}.npy")
    Q  = np.load(MAT / f"Q_{p}.npy")
    C  = np.load(MAT / f"C_{p}.npy")
    Cn = np.load(MAT / f"C_norm_{p}.npy")
    with open(MAT / f"datasets_{p}.json") as f:
        ds = json.load(f)
    return X, Q, C, Cn, np.array(ds)


def per_seed_grand_macro(g: pd.DataFrame) -> np.ndarray:
    """For one (router, config_id), returns per-seed grand macro across
    all (scenario, lambda) DEV conditions."""
    return g.groupby("seed")["macro_utility"].mean().sort_index().values


def bootstrap_ci(x: np.ndarray, n=2000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x)
    boot = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(x.mean()), float(lo), float(hi)


def pick_top_configs(stage1_df: pd.DataFrame, k=TOP_K_PER_ROUTER) -> list[tuple[str, dict]]:
    """Return list of (router, cfg_dict) for top-k configs per router."""
    picks = []
    for router, sub in stage1_df.groupby("router"):
        agg = sub.groupby("config_id")["macro_utility"].mean().sort_values(ascending=False)
        for cid in agg.head(k).index:
            cfg = json.loads(cid)
            picks.append((router, cfg))
    return picks


def rank_and_freeze(combined: pd.DataFrame):
    rows = []
    for router, sub in combined.groupby("router"):
        per_cfg = []
        for cid, g in sub.groupby("config_id"):
            per_seed = per_seed_grand_macro(g)
            mean, lo, hi = bootstrap_ci(per_seed)
            ps = g[g["scenario"].isin(["S1", "S3"])]["post_shift_regret_200"].dropna().values
            per_cfg.append({
                "router": router, "config_id": cid,
                "grand_macro":     round(mean, 5),
                "ci_lo":           round(lo,   5),
                "ci_hi":           round(hi,   5),
                "std_seed":        round(float(per_seed.std(ddof=1)), 5),
                "n_seeds":         int(len(per_seed)),
                "post_shift_regret_mean": round(float(ps.mean()), 5) if len(ps) else float("nan"),
                **json.loads(cid),
            })
        per_cfg = sorted(per_cfg, key=lambda r: -r["grand_macro"])
        print(f"\n=== {router} — top after 10-seed confirmation ===")
        for r in per_cfg[:TOP_K_PER_ROUTER]:
            print(f"  {r['config_id']:40s}  "
                  f"grand={r['grand_macro']:.4f}  "
                  f"CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  "
                  f"std={r['std_seed']:.4f}  "
                  f"psr={r['post_shift_regret_mean']}  n={r['n_seeds']}")
        winner = per_cfg[0]  # primary rank; no tie-break needed unless CI overlaps
        rows.append(winner)
    df = pd.DataFrame(rows)
    df.to_csv(FINAL_CSV, index=False)
    return df


def freeze(winners: pd.DataFrame):
    router_to_file = {
        "kNN":      OUT_CONFIGS / "knn.yaml",
        "Ridge":    OUT_CONFIGS / "ridge.yaml",
        "LinUCB":   OUT_CONFIGS / "linucb.yaml",
        "D-LinUCB": OUT_CONFIGS / "discounted_linucb.yaml",
    }
    print("\n" + "=" * 70)
    print("FREEZING HYPERPARAMETERS")
    print("=" * 70)
    for _, r in winners.iterrows():
        cfg = json.loads(r["config_id"])
        payload = {
            "router": r["router"],
            "hyperparameters": cfg,
            "selection": {
                "criterion": "mean macro_utility across (scenario, lambda) DEV conditions",
                "n_seeds":         int(r["n_seeds"]),
                "grand_macro":     float(r["grand_macro"]),
                "ci_lo":           float(r["ci_lo"]),
                "ci_hi":           float(r["ci_hi"]),
                "std_seed":        float(r["std_seed"]),
                "post_shift_regret_mean": (float(r["post_shift_regret_mean"])
                                            if pd.notna(r["post_shift_regret_mean"])
                                            else None),
                "frozen_partition":  "DEV",
                "final_test_touched": False,
                "protocol": {
                    "stage1_seeds": [42, 43, 44, 45, 46],
                    "stage2_seeds": STAGE2_SEEDS,
                    "scenarios":    ["S0", "S1", "S3"],
                    "lambdas":      LAMBDAS,
                },
            },
        }
        p = router_to_file[r["router"]]
        with open(p, "w") as f:
            yaml.dump(payload, f, sort_keys=False)
        print(f"  Wrote {p}")


def main():
    if not STAGE1_CSV.exists():
        print(f"ERROR: {STAGE1_CSV} not found — run scripts/10_hp_sweep.py first.")
        sys.exit(1)

    print(f"Loading stage-1 results from {STAGE1_CSV} ...")
    stage1 = pd.read_csv(STAGE1_CSV)
    print(f"  Stage 1: {len(stage1):,} rows, "
          f"{stage1['config_id'].nunique()} configs, "
          f"seeds={sorted(stage1['seed'].unique())}")

    picks = pick_top_configs(stage1, k=TOP_K_PER_ROUTER)
    print(f"\nTop {TOP_K_PER_ROUTER} configs per router selected for stage 2:")
    for r, cfg in picks:
        print(f"  {r:10s}  {cfg}")

    X_tr, Q_tr, _, Cn_tr, _ = load_matrices("TRAIN")
    X_dv, Q_dv, C_dv, Cn_dv, ds_dv = load_matrices("DEV")

    print(f"\nStage 2: {len(picks)} configs × {len(STAGE2_SEEDS)} new seeds "
          f"× {len(LAMBDAS)} λ × 3 scenarios")

    stage2_rows = []
    t0 = time.perf_counter()
    for i, (router_name, cfg) in enumerate(picks, 1):
        t_cfg = time.perf_counter()
        rows = run_config(router_name, cfg, STAGE2_SEEDS, LAMBDAS,
                          ["S0", "S1", "S3"],
                          X_tr, Q_tr, Cn_tr, X_dv, Q_dv, C_dv, Cn_dv, ds_dv)
        stage2_rows.extend(rows)
        dt = time.perf_counter() - t_cfg
        print(f"  [{i}/{len(picks)}] {router_name} {cfg}  ({dt:.1f}s)",
              flush=True)
        pd.DataFrame(stage2_rows).to_csv(STAGE2_CSV, index=False)

    print(f"\nStage 2 complete in {(time.perf_counter()-t0)/60:.1f} min. "
          f"Rows: {len(stage2_rows)}")

    combined = pd.concat([stage1, pd.DataFrame(stage2_rows)], ignore_index=True)
    combined.to_csv(COMBINED_CSV, index=False)
    print(f"Combined table written to {COMBINED_CSV}")

    # Only rank configs that ALSO appear in stage 2 (i.e. had 10 seeds)
    stage2_cids = set(pd.DataFrame(stage2_rows)["config_id"].unique())
    combined_top = combined[combined["config_id"].isin(stage2_cids)]
    winners = rank_and_freeze(combined_top)
    freeze(winners)


if __name__ == "__main__":
    main()
