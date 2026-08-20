"""
M2 — Pre-experiment sanity checks.

For each lambda in {0.0, 0.1, 0.25, 0.5, 1.0}, evaluated on the TRAIN partition
(so we do not touch validation/test), report:
  - Best Single arm (argmax mean utility on train)
  - Oracle routing share by model (per-query argmax over R)
  - Best Single utility, quality, cost, cost_norm
  - Oracle    utility, quality, cost, cost_norm
  - Utility gap (Oracle - BestSingle)
  - Macro and micro averages (macro = mean of per-dataset means)

Sanity narrative:
  - Larger lambda increases cost sensitivity: with r = q - lam*c_norm,
    higher lambda penalises cost more heavily, shifting Best Single and
    Oracle preferences toward cheaper arms.
  - The Oracle-vs-BestSingle utility gap is the ceiling on what any
    contextual router can gain over the strongest static baseline.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from llm_router.evaluation.metrics import oracle_tie_stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MAT_DIR = Path("data/interim/matrices")
CONFIG  = Path("configs/model_pool.yaml")
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0]


def load_partition(partition: str):
    Q = np.load(MAT_DIR / f"Q_{partition}.npy")
    C = np.load(MAT_DIR / f"C_{partition}.npy")
    C_norm = np.load(MAT_DIR / f"C_norm_{partition}.npy")
    with open(MAT_DIR / f"datasets_{partition}.json") as f:
        datasets = json.load(f)
    return Q, C, C_norm, np.array(datasets)


def macro_by_dataset(values: np.ndarray, datasets: np.ndarray) -> float:
    """Mean-of-per-dataset-means."""
    unique = np.unique(datasets)
    return float(np.mean([values[datasets == d].mean() for d in unique]))


def evaluate_lambda(Q, C, C_norm, datasets, pool, lam: float) -> dict:
    R = Q - lam * C_norm

    # Best Single (offline, chosen on train utility)
    per_arm_mean_util = R.mean(axis=0)
    best_arm = int(np.argmax(per_arm_mean_util))

    # Oracle (per-query argmax)
    oracle_arm = R.argmax(axis=1)

    def stats(arm_selection: np.ndarray) -> dict:
        # arm_selection is either shape () (single arm) or (n,) (per-row)
        if arm_selection.ndim == 0:
            a = int(arm_selection)
            q_sel   = Q[:, a]
            c_sel   = C[:, a]
            cn_sel  = C_norm[:, a]
            u_sel   = q_sel - lam * cn_sel
        else:
            rows = np.arange(len(arm_selection))
            q_sel  = Q[rows, arm_selection]
            c_sel  = C[rows, arm_selection]
            cn_sel = C_norm[rows, arm_selection]
            u_sel  = q_sel - lam * cn_sel
        return {
            "quality_micro":   float(q_sel.mean()),
            "cost_usd_micro":  float(c_sel.mean()),
            "cost_norm_micro": float(cn_sel.mean()),
            "utility_micro":   float(u_sel.mean()),
            "quality_macro":   macro_by_dataset(q_sel,   datasets),
            "cost_usd_macro":  macro_by_dataset(c_sel,   datasets),
            "cost_norm_macro": macro_by_dataset(cn_sel,  datasets),
            "utility_macro":   macro_by_dataset(u_sel,   datasets),
        }

    bs_stats = stats(np.array(best_arm))
    oracle_stats = stats(oracle_arm)

    # Oracle routing shares — BOTH naive argmax and tie-aware fractional
    n = len(oracle_arm)
    naive_share = {pool[a]: float((oracle_arm == a).sum() / n) for a in range(len(pool))}
    frac_shares, tie_rate, ties_per_row = oracle_tie_stats(R)
    share = {pool[a]: float(frac_shares[a]) for a in range(len(pool))}

    return {
        "lambda": lam,
        "best_single_arm":  pool[best_arm],
        "best_single_idx":  best_arm,
        "oracle_share":            share,             # fractional (tie-aware)
        "oracle_share_naive":      naive_share,       # argmax (column-order-biased at ties)
        "oracle_tie_rate":         float(tie_rate),
        "oracle_mean_ties_per_row": float(ties_per_row.mean()),
        "best_single":      bs_stats,
        "oracle":           oracle_stats,
        "utility_gap_micro": oracle_stats["utility_micro"] - bs_stats["utility_micro"],
        "utility_gap_macro": oracle_stats["utility_macro"] - bs_stats["utility_macro"],
    }


def main() -> None:
    with open(CONFIG) as f:
        pool = [m["name"] for m in yaml.safe_load(f)["pool"]]

    Q, C, C_norm, datasets = load_partition("train")
    print(f"Sanity checks on TRAIN partition: n={Q.shape[0]}, K={Q.shape[1]}")
    print(f"Pool order: {pool}\n")

    results = []
    for lam in LAMBDAS:
        r = evaluate_lambda(Q, C, C_norm, datasets, pool, lam)
        results.append(r)

    # Print summary table
    print("=" * 110)
    print("BEST SINGLE vs ORACLE — utility gap and routing share by lambda")
    print("=" * 110)
    header = f"{'lambda':>7} {'best_arm':<20} {'BS util':>9} {'Oracle util':>12} " \
             f"{'gap':>9} {'BS qual':>9} {'BS c$':>10} {'BS c_norm':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['lambda']:>7.2f} {r['best_single_arm']:<20} "
              f"{r['best_single']['utility_micro']:>9.4f} "
              f"{r['oracle']['utility_micro']:>12.4f} "
              f"{r['utility_gap_micro']:>+9.4f} "
              f"{r['best_single']['quality_micro']:>9.4f} "
              f"${r['best_single']['cost_usd_micro']:>9.6f} "
              f"{r['best_single']['cost_norm_micro']:>10.4f}")

    # Macro table
    print()
    print("=" * 110)
    print("MACRO (per-dataset-mean-of-means) summary")
    print("=" * 110)
    print(f"{'lambda':>7} {'BS util macro':>15} {'Oracle util macro':>19} "
          f"{'gap macro':>11} {'BS qual macro':>15} {'BS c_norm macro':>17}")
    for r in results:
        print(f"{r['lambda']:>7.2f} "
              f"{r['best_single']['utility_macro']:>15.4f} "
              f"{r['oracle']['utility_macro']:>19.4f} "
              f"{r['utility_gap_macro']:>+11.4f} "
              f"{r['best_single']['quality_macro']:>15.4f} "
              f"{r['best_single']['cost_norm_macro']:>17.4f}")

    # Oracle routing share table (tie-aware fractional)
    print()
    print("=" * 110)
    print("ORACLE FRACTIONAL ROUTING SHARE (%) BY LAMBDA  (tie-aware — ties split equally)")
    print("=" * 110)
    header = f"{'lambda':>7}" + "".join(f" {m:>18}" for m in pool) + f"  {'tie_rate':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['lambda']:>7.2f}"
        for m in pool:
            row += f" {r['oracle_share'][m]*100:>17.2f}%"
        row += f"  {r['oracle_tie_rate']*100:>9.2f}%"
        print(row)

    # Cost sensitivity sanity check
    print()
    print("=" * 110)
    print("COST-SENSITIVITY CHECK  (r = q - lam*c_norm → higher lambda = MORE cost sensitive)")
    print("=" * 110)
    print("Oracle mean cost_norm should DECREASE monotonically as lambda increases:")
    for r in results:
        print(f"  lam={r['lambda']:.2f}  oracle_cost_norm_micro={r['oracle']['cost_norm_micro']:.4f}  "
              f"oracle_quality_micro={r['oracle']['quality_micro']:.4f}")

    # Persist
    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sanity_by_lambda.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}/sanity_by_lambda.json")


if __name__ == "__main__":
    main()
