"""
P5 analysis — ranked routers with paired CIs, effect sizes, Pareto plots, A* candidates.

Reads:  results/tables/p5_final_dev.csv
Writes:
  results/tables/p5_router_ranking.csv
  results/tables/p5_pairwise_effect_sizes.csv
  results/tables/p5_per_scenario_lambda.csv
  results/tables/p5_drift_adaptation.csv        (post-shift regret + recovery)
  results/figures/p5_pareto_by_lambda.png
  results/figures/p5_router_utility_by_scenario.png
  results/figures/p5_cumulative_regret_S3.png   (illustrative — one seed / lambda)

Provisos carried from the P5a audit:
  * recovery_time excluded for S3 λ=0 (metric artefact under no drift).
  * HP selection ambiguity flagged for LinUCB / D-LinUCB.
  * kNN k=50 and D-LinUCB γ=0.999 at grid edge — flagged as limitation.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CSV = Path("results/tables/p5_final_dev.csv")
OUT_TABLES = Path("results/tables")
OUT_FIGURES = Path("results/figures")
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# Routers reported as reference bounds but ineligible as A* candidates.
# Oracle only: it needs hidden full-information outcomes (non-deployable).
# See a_star_shortlist() for why Cheapest is deliberately NOT in this set.
REFERENCE_ONLY = {"Oracle"}


def per_seed_grand(g: pd.DataFrame, col: str = "macro_utility") -> np.ndarray:
    """Per-seed mean of `col` across (scenario, lambda)."""
    return g.groupby("seed")[col].mean().sort_index().values


def bootstrap_ci(x: np.ndarray, n_boot=2000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boot = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(x.mean()), float(lo), float(hi)


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def rank_routers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for router, g in df.groupby("router"):
        per = per_seed_grand(g)
        mean, lo, hi = bootstrap_ci(per)
        rows.append({
            "router": router,
            "grand_macro":  round(mean, 4),
            "ci_lo":        round(lo, 4),
            "ci_hi":        round(hi, 4),
            "std_seed":     round(float(per.std(ddof=1)), 4),
            "n_seeds":      int(len(per)),
            # sub-metrics
            "macro_quality":   round(float(g.groupby("seed")["macro_quality"].mean().mean()), 4),
            "micro_cost_norm": round(float(g.groupby("seed")["micro_cost_norm"].mean().mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("grand_macro", ascending=False)


def pairwise_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """All pairs (router_a vs router_b) grand_macro comparison with paired
    CI on the difference and Cohen's d."""
    routers = sorted(df["router"].unique())
    rows = []
    for i, a in enumerate(routers):
        for b in routers:
            if a == b: continue
            xa = per_seed_grand(df[df["router"] == a])
            xb = per_seed_grand(df[df["router"] == b])
            if len(xa) != len(xb): continue
            diff = xa - xb
            mean_diff, lo, hi = bootstrap_ci(diff)
            try:
                w, p = stats.wilcoxon(xa, xb)
            except ValueError:
                w, p = np.nan, np.nan
            rows.append({
                "A": a, "B": b,
                "mean(A-B)":  round(mean_diff, 4),
                "diff_ci_lo": round(lo, 4),
                "diff_ci_hi": round(hi, 4),
                "wilcoxon_p": round(p, 4) if not np.isnan(p) else np.nan,
                "cohens_d":   round(cohens_d_paired(xa, xb), 3),
                "n_seeds":    len(xa),
            })
    return pd.DataFrame(rows)


def per_scenario_lambda(df: pd.DataFrame) -> pd.DataFrame:
    """Mean macro_utility per (router, scenario, lambda), 95% CI over seeds."""
    rows = []
    for (router, scen, lam), g in df.groupby(["router", "scenario", "lambda"]):
        per = g.groupby("seed")["macro_utility"].mean().values
        m, lo, hi = bootstrap_ci(per)
        rows.append({
            "router": router, "scenario": scen, "lambda": lam,
            "macro_util":  round(m, 4),
            "ci_lo":       round(lo, 4),
            "ci_hi":       round(hi, 4),
            "cost_norm":   round(float(g["micro_cost_norm"].mean()), 4),
            "quality":     round(float(g["micro_quality"].mean()), 4),
        })
    return pd.DataFrame(rows)


def drift_adaptation(df: pd.DataFrame) -> pd.DataFrame:
    """Post-shift regret and recovery time for S1, S2, S3 (excluding S3 λ=0)."""
    keep = df[df["scenario"].isin(["S1", "S2", "S3"])].copy()
    # Proviso A1: exclude S3 λ=0 recovery/PSR — metric artefact under no drift.
    keep = keep[~((keep["scenario"] == "S3") & (keep["lambda"] == 0.0))]

    rows = []
    for (router, scen, lam), g in keep.groupby(["router", "scenario", "lambda"]):
        psr = g["post_shift_regret_200"].dropna().values
        rt  = g["recovery_time"].values
        rt_valid = rt[rt >= 0]
        rows.append({
            "router": router, "scenario": scen, "lambda": lam,
            "psr_mean": round(float(psr.mean()), 4) if len(psr) else np.nan,
            "psr_std":  round(float(psr.std(ddof=1)), 4) if len(psr) > 1 else np.nan,
            "rt_mean":  round(float(rt_valid.mean()), 1) if len(rt_valid) else np.nan,
            "rt_n_neg": int((rt == -1).sum()),
            "rt_max":   int(rt_valid.max()) if len(rt_valid) else np.nan,
        })
    return pd.DataFrame(rows)


def pareto_by_lambda(df: pd.DataFrame):
    """Cost vs quality per router × lambda, one point per (router, lambda) = mean over seeds/scenarios."""
    fig, axes = plt.subplots(1, len(sorted(df["lambda"].unique())),
                              figsize=(4 * len(df["lambda"].unique()), 4),
                              sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    lambdas = sorted(df["lambda"].unique())
    routers = df["router"].unique().tolist()
    colours = plt.cm.tab10(np.linspace(0, 1, len(routers)))
    for ax, lam in zip(axes, lambdas):
        g = df[df["lambda"] == lam]
        for r, col in zip(routers, colours):
            gr = g[g["router"] == r]
            x = gr["micro_cost_norm"].mean()
            y = gr["micro_quality"].mean()
            ax.scatter(x, y, s=90, color=col, label=r)
            ax.annotate(r, (x, y), xytext=(5, 3), textcoords="offset points",
                         fontsize=7)
        ax.set_xlabel("mean cost_norm")
        ax.set_title(f"λ={lam}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean quality")
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "p5_pareto_by_lambda.png", dpi=120)
    plt.close()


def utility_by_scenario_bar(df: pd.DataFrame):
    """Grouped bar: mean macro_utility per (router, scenario), averaged over seeds×lambdas."""
    means = df.groupby(["router", "scenario"])["macro_utility"].mean().unstack("scenario")
    means = means.reindex(index=["Cheapest", "BestSingle", "kNN", "Ridge",
                                    "LinUCB", "D-LinUCB", "Oracle"])
    ax = means.plot(kind="bar", figsize=(11, 4.5), width=0.85,
                     colormap="tab10")
    ax.set_ylabel("mean macro_utility  (avg over 10 seeds × 5 λ)")
    ax.set_title("Router utility by scenario  (DEV, frozen HPs)")
    ax.legend(loc="lower right", title="scenario", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "p5_router_utility_by_scenario.png", dpi=120)
    plt.close()


def a_star_shortlist(ranked: pd.DataFrame, pairwise: pd.DataFrame,
                      drift: pd.DataFrame) -> list[str]:
    """
    Pick A* candidates: routers whose grand_macro CI overlaps the best
    DEPLOYABLE router; final choice needs richer criteria.

    Eligibility
    -----------
    Oracle (B6) is excluded: it consumes hidden full-information outcomes and
    is not deployable, so it is a reference upper bound only. Left in, it always
    ranks first (grand_macro 0.78 vs 0.58 for the best real router) and the
    CI-overlap filter collapses the shortlist to ["Oracle"].

    Cheapest (B0) is NOT excluded. It is the cost lower bound, but unlike Oracle
    it is a genuinely deployable policy — under a sufficiently cost-averse
    lambda it could legitimately win, and excluding it a priori would hide that
    result. On current DEV evidence it never reaches the shortlist on its own
    (ci_hi 0.41 vs top ci_lo 0.58).
    """
    eligible = ranked[~ranked["router"].isin(REFERENCE_ONLY)]
    if eligible.empty:
        return []
    top_lo = eligible.iloc[0]["ci_lo"]
    return eligible[eligible["ci_hi"] >= top_lo]["router"].tolist()


def main():
    if not CSV.exists():
        print(f"ERROR: {CSV} not found. Run scripts/12_p5_final_validation.py first.")
        sys.exit(1)
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df):,} rows.  "
          f"routers={sorted(df['router'].unique())}  "
          f"seeds={sorted(df['seed'].unique())}  "
          f"lambdas={sorted(df['lambda'].unique())}  "
          f"scenarios={sorted(df['scenario'].unique())}")

    # ---- Ranking --------------------------------------------------------------
    print("\n=== 1. Router ranking (grand_macro across DEV conditions) ===")
    ranked = rank_routers(df)
    print(ranked.to_string(index=False))
    ranked.to_csv(OUT_TABLES / "p5_router_ranking.csv", index=False)

    # ---- Pairwise -------------------------------------------------------------
    print("\n=== 2. Pairwise A vs B (paired 10-seed) ===")
    pairwise = pairwise_comparison(df)
    print(pairwise.to_string(index=False))
    pairwise.to_csv(OUT_TABLES / "p5_pairwise_effect_sizes.csv", index=False)

    # ---- Per-scenario, per-lambda --------------------------------------------
    print("\n=== 3. Per (router, scenario, lambda) — head ===")
    per_sl = per_scenario_lambda(df)
    print(per_sl.head(20).to_string(index=False))
    per_sl.to_csv(OUT_TABLES / "p5_per_scenario_lambda.csv", index=False)

    # ---- Drift adaptation ----------------------------------------------------
    print("\n=== 4. Drift adaptation (S1/S2/S3; S3 λ=0 excluded per audit A1) ===")
    drift = drift_adaptation(df)
    piv = drift.pivot_table(index=["scenario", "lambda"], columns="router",
                              values="psr_mean")
    print("Post-shift regret (200-step window):")
    print(piv.round(4).to_string())
    drift.to_csv(OUT_TABLES / "p5_drift_adaptation.csv", index=False)

    # ---- Figures --------------------------------------------------------------
    print("\n=== 5. Figures ===")
    pareto_by_lambda(df)
    utility_by_scenario_bar(df)
    print(f"  Saved: {OUT_FIGURES}/p5_pareto_by_lambda.png")
    print(f"         {OUT_FIGURES}/p5_router_utility_by_scenario.png")

    # ---- A* shortlist --------------------------------------------------------
    print("\n=== 6. A* shortlist (CI-overlapping with the top deployable router) ===")
    shortlist = a_star_shortlist(ranked, pairwise, drift)
    print(f"  excluded as reference-only: {sorted(REFERENCE_ONLY)}")
    print(f"  {shortlist}")
    print("  NOTE: A* selection must weigh multi-criteria evidence:")
    print("        quality, cost, utility, cum_regret, post_shift_regret,")
    print("        recovery_time, Pareto efficiency — jointly.")


if __name__ == "__main__":
    main()
