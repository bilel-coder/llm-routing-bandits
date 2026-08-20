"""
M1.5 → M2 gate: candidate pool sizing analysis.

Compares 6-, 8-, 10-model pools on the 11 retained datasets using the same
filter pipeline as scripts/02_build_dataset.py. Reports coverage, cost
reliability, and diversity to inform the pool-size decision.

Does NOT modify any frozen data (splits stay untouched).
"""

from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW_DIR = Path("data/raw/bench-release")

RETAINED_DATASETS = [
    "aime", "arc-agi", "arenahard", "gpqa", "hle", "livecodebench",
    "livemathbench", "mmlupro", "simpleqa", "swe-bench", "tau2",
]
CANONICAL_SPLIT = {"simpleqa": "test", "mmlupro": "test_1000", "hle": "test"}


# ---------------------------------------------------------------------------
# Ingest (same filter chain as 02_build_dataset.py)
# ---------------------------------------------------------------------------

def ingest() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW_DIR.rglob("*.json")):
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                d = json.load(fh)
        except Exception:
            continue
        m, ds, sp = d.get("model_name", ""), d.get("dataset_name", ""), d.get("split", "")
        recs = d.get("records", [])
        if not m or ds not in RETAINED_DATASETS or not recs:
            continue
        # Canonical split filter
        can = CANONICAL_SPLIT.get(ds)
        if can and sp != can:
            continue
        for r in recs:
            rows.append({
                "query_id":          f"{ds}__{sp}__{r.get('index')}",
                "dataset":           ds,
                "model":             m,
                "quality":           float(r["score"]) if r.get("score") is not None else np.nan,
                "cost":              float(r.get("cost", 0.0) or 0.0),
                "completion_tokens": r.get("completion_tokens"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-model statistics
# ---------------------------------------------------------------------------

def provider_of(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):     return "Anthropic"
    if m.startswith("gpt"):        return "OpenAI"
    if m.startswith("gemini"):     return "Google"
    if m.startswith("deepseek"):   return "DeepSeek"
    if m.startswith("kimi"):       return "Moonshot"
    if m.startswith("qwen"):       return "Alibaba/Qwen"
    if m.startswith("glm"):        return "Zhipu/GLM"
    if m.startswith("intern"):     return "InternLM/Shanghai AI Lab"
    return "Other"


def per_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("model"):
        n_records = len(g)
        n_datasets = g["dataset"].nunique()
        # Case A: gen failure (cost=0 & tokens<=0)
        tok = g["completion_tokens"].fillna(0)
        case_a = ((g["cost"] == 0.0) & (tok <= 0)).sum()
        # Case B: missing cost (cost=0 & tokens>0)
        case_b = ((g["cost"] == 0.0) & (tok > 0)).sum()
        pos = g.loc[g["cost"] > 0, "cost"]
        mean_c = float(pos.mean()) if len(pos) else np.nan
        median_c = float(pos.median()) if len(pos) else np.nan
        max_c = float(pos.max()) if len(pos) else np.nan
        pct_zero = 100.0 * (g["cost"] == 0.0).mean()
        rows.append({
            "model": m,
            "provider": provider_of(m),
            "n_records": n_records,
            "n_datasets_covered": n_datasets,
            "case_A_gen_fail": int(case_a),
            "case_B_missing_cost": int(case_b),
            "pct_zero_cost": round(pct_zero, 3),
            "mean_cost_usd": round(mean_c, 6) if not np.isnan(mean_c) else None,
            "median_cost_usd": round(median_c, 6) if not np.isnan(median_c) else None,
            "max_cost_usd": round(max_c, 4) if not np.isnan(max_c) else None,
        })
    return pd.DataFrame(rows).sort_values("mean_cost_usd", ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# Complete-coverage query count for an arbitrary pool
# ---------------------------------------------------------------------------

def complete_coverage(df: pd.DataFrame, pool: list[str]) -> tuple[int, dict[str, int]]:
    """Return (n_complete_queries, per_dataset_count) after removing failures."""
    sub = df[df["model"].isin(pool)].copy()
    # Remove Case A (gen failure) and Case B (missing cost) rows
    tok = sub["completion_tokens"].fillna(0)
    bad = (sub["cost"] == 0.0) & ((tok <= 0) | (tok > 0))
    sub = sub[~bad]
    # Require all pool models present with valid quality
    valid = sub[sub["quality"].notna()]
    cov = valid.groupby("query_id")["model"].nunique()
    complete = set(cov[cov == len(pool)].index)
    per_ds = (
        sub[sub["query_id"].isin(complete)]
        .drop_duplicates(["query_id"])
        .groupby("dataset")
        .size()
        .to_dict()
    )
    return len(complete), per_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Ingesting raw benchmark (11 retained datasets, canonical splits)...")
    df = ingest()
    print(f"  Total rows: {len(df):,}  |  models: {df['model'].nunique()}  |  "
          f"datasets: {df['dataset'].nunique()}  |  unique queries: {df['query_id'].nunique():,}")

    # ---- Per-model stats ---------------------------------------------------
    stats = per_model_stats(df)
    print("\n" + "=" * 100)
    print("PER-MODEL STATISTICS (11 retained datasets, canonical splits)")
    print("=" * 100)
    print(stats.to_string(index=False))
    stats.to_csv("results/tables/pool_candidate_stats.csv", index=False)

    # Filter to models with reasonable coverage (≥ 8 of 11 datasets)
    plausible = stats[stats["n_datasets_covered"] >= 8]["model"].tolist()
    print(f"\nModels covering ≥8/11 datasets ({len(plausible)}): {plausible}")

    # ---- Candidate pools ---------------------------------------------------
    pool6 = [
        "claude-sonnet-4", "gemini-2.5-flash", "gemini-2.5-pro",
        "gpt-5", "deepseek-r1-0528", "kimi-k2-0905",
    ]
    # All candidates for expansion have 11/11 dataset coverage AND non-catastrophic
    # failure rates. Excluded from consideration:
    #   - gpt-4.1 (covers only 5/11 datasets)
    #   - claude-opus-4.1 (covers only 1/11: swe-bench)
    #   - gpt-5-chat, deepseek-v3.1-terminus, openrouter (10/11 coverage — would
    #     force a dataset drop)
    #   - qwen3-235b-a22b-thinking-2507 (12.26% failure rate — data-quality risk)

    candidate_pools = {
        "pool_8A_glm+intern": pool6 + ["glm-4.6", "intern-s1"],
        "pool_8D_glm+dsv3":   pool6 + ["glm-4.6", "deepseek-v3-0324"],
        "pool_8E_intern+dsv3": pool6 + ["intern-s1", "deepseek-v3-0324"],
    }
    pool10 = pool6 + ["glm-4.6", "intern-s1", "qwen3-235b-a22b-2507", "deepseek-v3-0324"]

    print("\n" + "=" * 100)
    print("COMPLETE-COVERAGE ANALYSIS PER CANDIDATE POOL")
    print("=" * 100)
    results = []
    for name, pool in candidate_pools.items():
        # Verify all members exist in raw data
        missing = [m for m in pool if m not in stats["model"].values]
        if missing:
            print(f"\n[{name}] MISSING FROM RAW DATA: {missing}  — SKIPPING")
            continue

        n_complete, per_ds = complete_coverage(df, pool)
        # Cost stats within this pool
        pool_stats = stats[stats["model"].isin(pool)]
        min_mean = pool_stats["mean_cost_usd"].min()
        max_mean = pool_stats["mean_cost_usd"].max()
        cost_ratio = max_mean / min_mean if min_mean else float("inf")
        n_providers = pool_stats["provider"].nunique()
        avg_pct_zero = pool_stats["pct_zero_cost"].mean()
        max_pct_zero = pool_stats["pct_zero_cost"].max()

        results.append({
            "pool": name,
            "n_arms": len(pool),
            "complete_queries": n_complete,
            "n_datasets_retained": len(per_ds),
            "min_datasets_kept": min(per_ds.values()) if per_ds else 0,
            "cost_min_usd": round(min_mean, 5),
            "cost_max_usd": round(max_mean, 5),
            "cost_max_min_ratio": round(cost_ratio, 1),
            "n_providers": n_providers,
            "avg_pct_zero": round(avg_pct_zero, 3),
            "max_pct_zero": round(max_pct_zero, 3),
        })
        print(f"\n[{name}] members={pool}")
        print(f"  complete queries: {n_complete:,}   datasets covered: {len(per_ds)}/11")
        print(f"  cost range: ${min_mean:.5f} .. ${max_mean:.5f}  (ratio {cost_ratio:.1f}×)")
        print(f"  providers: {n_providers}   avg zero-cost %: {avg_pct_zero:.2f}   "
              f"max zero-cost %: {max_pct_zero:.2f}")
        print(f"  per-dataset kept: {per_ds}")

    summary = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    summary.to_csv("results/tables/pool_comparison.csv", index=False)

    # ---- Specific model investigation --------------------------------------
    print("\n" + "=" * 100)
    print("SPECIFIC MODEL INVESTIGATION")
    print("=" * 100)
    for m in ["qwen3-235b-a22b-2507", "deepseek-v3-0324", "intern-s1"]:
        row = stats[stats["model"] == m]
        if row.empty:
            print(f"\n[{m}] NOT PRESENT in raw data on retained datasets.")
            continue
        r = row.iloc[0]
        print(f"\n[{m}]  provider={r['provider']}")
        print(f"  n_records={r['n_records']:,}  datasets_covered={r['n_datasets_covered']}/11")
        print(f"  case_A_gen_fail={r['case_A_gen_fail']}  case_B_missing_cost={r['case_B_missing_cost']}")
        print(f"  pct_zero_cost={r['pct_zero_cost']}%")
        print(f"  mean_cost=${r['mean_cost_usd']}  median=${r['median_cost_usd']}  max=${r['max_cost_usd']}")

    # ---- Zero-cost pattern per model ---------------------------------------
    print("\n" + "=" * 100)
    print("ZERO-COST BREAKDOWN (candidate pool members + investigated)")
    print("=" * 100)
    for m in sorted(set(pool10 + ["intern-s1"])):
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        tok = sub["completion_tokens"].fillna(0)
        n = len(sub)
        n_ca = int(((sub["cost"] == 0.0) & (tok <= 0)).sum())
        n_cb = int(((sub["cost"] == 0.0) & (tok > 0)).sum())
        n_pos = int((sub["cost"] > 0).sum())
        print(f"  {m:32s}  n={n:5d}  case_A={n_ca:4d}  case_B={n_cb:4d}  "
              f"positive_cost={n_pos:5d}")


if __name__ == "__main__":
    main()
