"""
M1.5 — Build Canonical Outcome Matrix.

Applies the corrected pipeline:
  1. Load raw data
  2. Filter to pool models and retained datasets
  3. Exclude generation failures (cost=0 AND completion_tokens=0)
  4. Resolve duplicate/leakage issues (arenahard subsets, mmlupro splits, tau2 grouping)
  5. Build complete Q[i,a] and C[i,a] matrices (no imputation)
  6. Save outcomes.parquet and outcome_matrix metadata

Usage:
    python scripts/02_build_dataset.py

Outputs:
    data/processed/outcomes.parquet          -- flat (query_id, model) rows
    data/processed/outcome_matrix_meta.json  -- Q/C matrix dimensions and index
    results/tables/model_summary.csv
    results/tables/dataset_summary.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw/bench-release")
PROCESSED_DIR = Path("data/processed")
INTERIM_DIR   = Path("data/interim")
TABLES_DIR    = Path("results/tables")
for d in [PROCESSED_DIR, INTERIM_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path("configs/model_pool.yaml")

# Canonical split per dataset (for datasets with multiple sub-splits)
CANONICAL_SPLIT: dict[str, str] = {
    "simpleqa": "test",
    "mmlupro":  "test_1000",
    "hle":      "test",
}


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
def load_pool_config() -> tuple[list[str], list[str]]:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    models   = [m["name"] for m in cfg["pool"]]
    datasets = cfg["datasets"]
    return models, datasets


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def ingest_all() -> pd.DataFrame:
    all_files = sorted(RAW_DIR.rglob("*.json"))
    log.info("Processing %d raw JSON files...", len(all_files))
    rows: list[dict] = []
    for fpath in all_files:
        try:
            import json as _json
            with open(fpath, encoding="utf-8", errors="replace") as f:
                data = _json.load(f)
        except Exception as e:
            log.warning("Skipping %s: %s", fpath, e)
            continue
        model   = data.get("model_name", "")
        dataset = data.get("dataset_name", "")
        split   = data.get("split", "")
        records = data.get("records", [])
        if not model or not dataset or not records:
            continue
        for r in records:
            rows.append({
                "query_id":          f"{dataset}__{split}__{r.get('index')}",
                "dataset":           dataset,
                "split":             split,
                "index_in_dataset":  r.get("index"),
                "prompt":            r.get("origin_query") or r.get("prompt", ""),
                "model":             model,
                "quality":           float(r["score"]) if r.get("score") is not None else np.nan,
                "cost":              float(r.get("cost", 0.0) or 0.0),
                "prompt_tokens":     r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
            })
    df = pd.DataFrame(rows)
    log.info("Ingested %d total rows before filtering", len(df))
    return df


# ---------------------------------------------------------------------------
# Filtering steps
# ---------------------------------------------------------------------------
def filter_pool_and_datasets(df: pd.DataFrame,
                              pool: list[str],
                              datasets: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df[df["model"].isin(pool) & df["dataset"].isin(datasets)]
    log.info("After pool+dataset filter: %d rows (dropped %d)", len(df), before - len(df))
    return df.reset_index(drop=True)


def filter_canonical_splits(df: pd.DataFrame) -> pd.DataFrame:
    """For datasets with multiple sub-splits, keep only the canonical one."""
    parts: list[pd.DataFrame] = []
    for dataset, grp in df.groupby("dataset"):
        canonical = CANONICAL_SPLIT.get(dataset)
        if canonical:
            available = grp["split"].unique()
            if canonical not in available:
                log.warning("Dataset %s: canonical split '%s' not found (available: %s)",
                            dataset, canonical, available)
                parts.append(grp)
            else:
                parts.append(grp[grp["split"] == canonical])
        else:
            parts.append(grp)
    result = pd.concat(parts, ignore_index=True)
    log.info("After canonical-split filter: %d rows", len(result))
    return result


def filter_generation_failures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records with unreliable cost data. Two cases:

    Case A — Generation failure: cost=0 AND completion_tokens=0
      Model produced no output. Quality=0 is vacuous; cost=0 is correct but
      the query is not useful for routing learning.

    Case B — Missing cost data: cost=0 AND completion_tokens>0
      Model produced real output but cost was not recorded in the benchmark.
      Treating these as free-to-run would corrupt the cost-aware routing signal.
      Identified occurrence: gpt-5 on HLE (499 records).

    Both cases are excluded. Affected queries are dropped from the complete
    coverage requirement in the next step.
    """
    # completion_tokens <= 0 covers both zero and negative values
    # (negative values are benchmark integer-overflow artefacts in gpt-5 HLE/simpleqa)
    failure_mask      = (df["cost"] == 0.0) & (df["completion_tokens"].fillna(0) <= 0)
    missing_cost_mask = (df["cost"] == 0.0) & (df["completion_tokens"].fillna(0) > 0)

    n_failures = failure_mask.sum()
    n_missing  = missing_cost_mask.sum()

    if n_failures > 0:
        by_model = df[failure_mask].groupby("model").size()
        log.info("Case A (generation failures, cost=0 & tokens=0): %d\n%s",
                 n_failures, by_model.to_string())
    if n_missing > 0:
        by_model = df[missing_cost_mask].groupby(["model","dataset"]).size()
        log.info("Case B (missing cost data, cost=0 & tokens>0): %d\n%s",
                 n_missing, by_model.to_string())

    drop_mask = failure_mask | missing_cost_mask
    df = df[~drop_mask].reset_index(drop=True)
    log.info("After failure+missing-cost filter: %d rows (removed %d)",
             len(df), drop_mask.sum())
    return df


# ---------------------------------------------------------------------------
# Duplicate / leakage resolution
# ---------------------------------------------------------------------------
def build_prompt_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a group_id to each unique prompt text (after Unicode normalization).
    Queries sharing a prompt get the same group_id.
    This enables group-aware splitting: the same prompt cannot appear in both
    train and test under different query_ids.

    tau2 has 278 queries but only 167 unique prompts (repeated scenario contexts).
    mmlupro test_1000 already selected — no cross-split overlap.
    arenahard sub-datasets already excluded.
    """
    import unicodedata

    def normalize(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.replace(" ", "\n").replace(" ", "\n")
        return unicodedata.normalize("NFKC", s).strip()

    df = df.copy()
    df["prompt_norm"] = df["prompt"].apply(normalize)

    # Build prompt → group_id mapping from unique queries
    unique_queries = df.drop_duplicates("query_id")[["query_id", "prompt_norm"]].copy()
    unique_prompts = unique_queries["prompt_norm"].unique()
    prompt_to_group = {p: i for i, p in enumerate(sorted(unique_prompts))}
    query_to_group  = {row.query_id: prompt_to_group[row.prompt_norm]
                       for row in unique_queries.itertuples()}

    df["prompt_group_id"] = df["query_id"].map(query_to_group)
    n_groups = df["prompt_group_id"].nunique()
    n_queries = df["query_id"].nunique()
    log.info("Prompt grouping: %d unique queries → %d unique prompt groups (%d collapsed)",
             n_queries, n_groups, n_queries - n_groups)
    return df


# ---------------------------------------------------------------------------
# Complete outcome matrix
# ---------------------------------------------------------------------------
def require_complete_coverage(df: pd.DataFrame, pool: list[str]) -> pd.DataFrame:
    """
    Keep only queries where ALL pool models have a valid (non-NaN) quality AND
    a non-failure cost entry. No imputation.
    """
    n_required = len(pool)
    # Count how many pool models have valid quality per query
    valid = df[df["quality"].notna()]
    coverage = valid.groupby("query_id")["model"].nunique()
    complete_qids = coverage[coverage == n_required].index
    before = df["query_id"].nunique()
    df = df[df["query_id"].isin(complete_qids)].reset_index(drop=True)
    after = df["query_id"].nunique()
    log.info("Complete coverage filter: %d → %d unique queries (dropped %d)",
             before, after, before - after)
    return df


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, grp in df.groupby("model"):
        rows.append({
            "model":                    model,
            "number_of_queries":        grp["query_id"].nunique(),
            "number_of_datasets":       grp["dataset"].nunique(),
            "mean_quality":             round(grp["quality"].mean(), 4),
            "median_quality":           round(grp["quality"].median(), 4),
            "mean_cost_usd":            round(grp["cost"].mean(), 8),
            "median_cost_usd":          round(grp["cost"].median(), 8),
            "max_cost_usd":             round(grp["cost"].max(), 6),
            "pct_zero_cost":            round((grp["cost"] == 0.0).mean() * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("mean_cost_usd", ascending=False)


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, grp in df.groupby("dataset"):
        q = grp["quality"].dropna()
        rows.append({
            "dataset":      dataset,
            "n_queries":    grp["query_id"].nunique(),
            "n_models":     grp["model"].nunique(),
            "score_type":   "ternary" if set(q.unique()) == {0.0, 0.5, 1.0}
                            else ("binary" if q.isin([0.0, 1.0]).all() else "continuous"),
            "mean_quality": round(q.mean(), 4),
        })
    return pd.DataFrame(rows).sort_values("n_queries", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found. Run download step first.")
        sys.exit(1)

    pool, datasets = load_pool_config()
    log.info("Pool (%d models): %s", len(pool), pool)
    log.info("Datasets (%d): %s", len(datasets), datasets)

    # 1. Ingest
    df = ingest_all()

    # 2. Filter to pool and datasets
    df = filter_pool_and_datasets(df, pool, datasets)

    # 3. Canonical split selection
    df = filter_canonical_splits(df)

    # 4. Remove generation failures
    df = filter_generation_failures(df)

    # 5. Require complete model coverage per query
    df = require_complete_coverage(df, pool)

    # 6. Assign prompt groups for leakage-safe splitting
    df = build_prompt_groups(df)

    # 7. Validate
    dup = df.duplicated(subset=["query_id", "model"]).sum()
    assert dup == 0, f"Duplicate (query_id, model) pairs: {dup}"
    missing_quality = df["quality"].isna().sum()
    assert missing_quality == 0, f"Missing quality after complete-coverage filter: {missing_quality}"
    neg_cost = (df["cost"] < 0).sum()
    assert neg_cost == 0, f"Negative costs: {neg_cost}"
    log.info("All integrity checks passed.")

    # 8. Save
    out_path = PROCESSED_DIR / "outcomes.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(df), out_path)

    # 9. Save outcome matrix metadata
    query_ids = sorted(df["query_id"].unique().tolist())
    models    = pool  # fixed order from config
    meta = {
        "n_queries":  len(query_ids),
        "n_models":   len(models),
        "pool":       models,
        "datasets":   datasets,
        "shape_Q":    [len(query_ids), len(models)],
        "shape_C":    [len(query_ids), len(models)],
        "n_prompt_groups": int(df["prompt_group_id"].nunique()),
        "note": "Q[i,a] = quality, C[i,a] = cost_usd. No missing values.",
    }
    with open(PROCESSED_DIR / "outcome_matrix_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 10. Summary tables
    build_model_summary(df).to_csv(TABLES_DIR / "model_summary.csv", index=False)
    build_dataset_summary(df).to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

    # 11. Print report
    print("\n" + "=" * 70)
    print("OUTCOME MATRIX BUILD COMPLETE")
    print("=" * 70)
    print(f"  Pool models:    {len(models)}")
    print(f"  Datasets:       {len(datasets)}")
    print(f"  Complete rows:  {len(df):,}")
    print(f"  Q shape:        {meta['shape_Q']}  (queries × models)")
    print(f"  C shape:        {meta['shape_C']}  (queries × models)")
    print(f"  Prompt groups:  {meta['n_prompt_groups']:,}  (unique prompt texts)")
    print(f"  Collapsed:      {len(query_ids) - meta['n_prompt_groups']} queries share a prompt with another query")
    print()
    print(build_model_summary(df)[["model","number_of_queries","mean_cost_usd","pct_zero_cost"]].to_string(index=False))
    print()
    print(build_dataset_summary(df)[["dataset","n_queries","score_type","mean_quality"]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
