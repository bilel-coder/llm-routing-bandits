"""
M1 — Audit Script: inspect raw LLMRouterBench data and report schema.

Usage:
    python scripts/01_audit_data.py

Outputs:
    data/interim/audit_summary.json
    data/interim/audit_summary.csv (per-model statistics)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/bench-release")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Failed to load %s: %s", path, e)
        return None


def audit() -> dict:
    """
    Walk bench-release/, load every JSON, report schema and coverage statistics.
    """
    all_files = sorted(RAW_DIR.rglob("*.json"))
    log.info("Found %d JSON files under %s", len(all_files), RAW_DIR)

    # Accumulators
    datasets: set[str] = set()
    models: set[str] = set()
    malformed: list[str] = []

    # Per (dataset, model): list of record-level info
    # coverage_matrix[dataset][model] = list of (index, score, cost, pt, ct)
    coverage: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    # Unique query keys
    query_keys: dict[str, str] = {}  # key -> prompt (to detect duplicates)

    total_records = 0
    missing_cost_files = 0
    schema_fields_seen: dict[str, int] = defaultdict(int)

    for fpath in all_files:
        data = load_json(fpath)
        if data is None:
            malformed.append(str(fpath))
            continue

        model_name = data.get("model_name", "")
        dataset_name = data.get("dataset_name", "")
        split = data.get("split", "")
        file_cost = data.get("cost", None)
        records = data.get("records", [])

        if not records:
            log.debug("No records in %s", fpath)
            continue

        if file_cost is None or file_cost == 0.0:
            missing_cost_files += 1

        datasets.add(dataset_name)
        models.add(model_name)

        for r in records:
            for k in r.keys():
                schema_fields_seen[k] += 1

            idx = r.get("index", None)
            score = r.get("score", None)
            cost = r.get("cost", 0.0)
            pt = r.get("prompt_tokens", None)
            ct = r.get("completion_tokens", None)
            origin = r.get("origin_query", r.get("prompt", ""))

            # Build globally unique query key
            qkey = f"{dataset_name}__{split}__{idx}"
            if qkey in query_keys and query_keys[qkey] != origin[:200]:
                log.warning("Duplicate key %s maps to different prompts!", qkey)
            else:
                query_keys[qkey] = origin[:200]

            coverage[dataset_name][model_name].append({
                "index": idx,
                "score": score,
                "cost": cost,
                "prompt_tokens": pt,
                "completion_tokens": ct,
            })
            total_records += 1

    # -----------------------------------------------------------------------
    # Build per-model statistics table
    model_stats: list[dict] = []
    for dataset, model_data in sorted(coverage.items()):
        for model, recs in sorted(model_data.items()):
            scores = [r["score"] for r in recs if r["score"] is not None]
            costs  = [r["cost"]  for r in recs if r["cost"]  is not None]
            pts    = [r["prompt_tokens"] for r in recs if r["prompt_tokens"] is not None]
            cts    = [r["completion_tokens"] for r in recs if r["completion_tokens"] is not None]
            model_stats.append({
                "dataset":              dataset,
                "model":                model,
                "n_records":            len(recs),
                "mean_quality":         round(np.mean(scores), 4) if scores else None,
                "median_quality":       round(np.median(scores), 4) if scores else None,
                "min_quality":          round(np.min(scores), 4) if scores else None,
                "max_quality":          round(np.max(scores), 4) if scores else None,
                "mean_cost":            round(np.mean(costs), 6) if costs else None,
                "median_cost":          round(np.median(costs), 6) if costs else None,
                "max_cost":             round(np.max(costs), 6) if costs else None,
                "n_zero_cost":          sum(1 for c in costs if c == 0.0),
                "mean_prompt_tokens":   round(np.mean(pts), 1) if pts else None,
                "mean_completion_tokens": round(np.mean(cts), 1) if cts else None,
            })

    stats_df = pd.DataFrame(model_stats)

    # -----------------------------------------------------------------------
    # Coverage matrix: for each dataset, which models cover which queries?
    coverage_report: dict[str, dict] = {}
    for dataset, model_data in sorted(coverage.items()):
        model_query_sets = {m: set(r["index"] for r in recs)
                            for m, recs in model_data.items()}
        all_queries = set().union(*model_query_sets.values()) if model_query_sets else set()
        # Intersection: queries covered by ALL models
        if model_query_sets:
            common_queries = set.intersection(*model_query_sets.values())
        else:
            common_queries = set()

        coverage_report[dataset] = {
            "n_models":        len(model_data),
            "n_queries_union": len(all_queries),
            "n_queries_intersection": len(common_queries),
            "models":          sorted(model_data.keys()),
            "coverage_fraction": {
                m: round(len(qs) / len(all_queries), 4) if all_queries else 0.0
                for m, qs in model_query_sets.items()
            },
        }

    # -----------------------------------------------------------------------
    # Global unique queries
    total_unique_queries = len(query_keys)

    # Score distribution summary
    all_scores = [r["score"]
                  for ds in coverage.values()
                  for m_recs in ds.values()
                  for r in m_recs
                  if r["score"] is not None]
    all_costs = [r["cost"]
                 for ds in coverage.values()
                 for m_recs in ds.values()
                 for r in m_recs
                 if r["cost"] is not None]

    # -----------------------------------------------------------------------
    # Methodological notes
    zero_cost_count = sum(1 for c in all_costs if c == 0.0)
    zero_cost_pct   = zero_cost_count / len(all_costs) * 100 if all_costs else 0.0

    summary = {
        "raw_dir":           str(RAW_DIR),
        "total_json_files":  len(all_files),
        "malformed_files":   malformed,
        "total_records":     total_records,
        "total_unique_queries": total_unique_queries,
        "n_datasets":        len(datasets),
        "datasets":          sorted(datasets),
        "n_models":          len(models),
        "models":            sorted(models),
        "record_schema_fields": sorted(schema_fields_seen.keys()),
        "score_distribution": {
            "min":    round(np.min(all_scores), 4)  if all_scores else None,
            "max":    round(np.max(all_scores), 4)  if all_scores else None,
            "mean":   round(np.mean(all_scores), 4) if all_scores else None,
            "median": round(np.median(all_scores), 4) if all_scores else None,
            "pct_correct": round(np.mean([s == 1.0 for s in all_scores]) * 100, 2) if all_scores else None,
        },
        "cost_distribution": {
            "min":       round(np.min(all_costs), 8)  if all_costs else None,
            "max":       round(np.max(all_costs), 6)  if all_costs else None,
            "mean":      round(np.mean(all_costs), 8) if all_costs else None,
            "median":    round(np.median(all_costs), 8) if all_costs else None,
            "n_zero":    zero_cost_count,
            "pct_zero":  round(zero_cost_pct, 2),
        },
        "methodological_notes": [
            "query_id = '{dataset_name}__{split}__{index}' (constructed key)",
            "quality = record['score'] (0.0 or 1.0 binary in most datasets; may be float for open-ended)",
            "cost = record['cost'] in USD per query (0.0 for open-source/local models)",
            f"{zero_cost_pct:.1f}% of records have cost=0.0 — open-source models running locally",
            "origin_query used as canonical prompt (original un-templated query)",
            "Incomplete coverage: not all models evaluated on all datasets — check coverage_report",
        ],
        "coverage_report": coverage_report,
    }

    # -----------------------------------------------------------------------
    # Save outputs
    out_json = INTERIM_DIR / "audit_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Audit summary saved to %s", out_json)

    out_csv = INTERIM_DIR / "audit_per_model.csv"
    stats_df.to_csv(out_csv, index=False)
    log.info("Per-model stats saved to %s", out_csv)

    # -----------------------------------------------------------------------
    # Print human-readable summary
    print("\n" + "=" * 70)
    print("LLMRouterBench AUDIT SUMMARY")
    print("=" * 70)
    print(f"  JSON files found:        {len(all_files)}")
    print(f"  Malformed files:         {len(malformed)}")
    print(f"  Total records:           {total_records:,}")
    print(f"  Unique query keys:       {total_unique_queries:,}")
    print(f"  Datasets:                {len(datasets)}")
    print(f"  Models:                  {len(models)}")
    print(f"\nRecord schema fields: {sorted(schema_fields_seen.keys())}")
    print(f"\nScore distribution:  min={summary['score_distribution']['min']}"
          f"  mean={summary['score_distribution']['mean']}"
          f"  max={summary['score_distribution']['max']}")
    print(f"Cost distribution:   min={summary['cost_distribution']['min']}"
          f"  mean={summary['cost_distribution']['mean']}"
          f"  max={summary['cost_distribution']['max']}"
          f"  ({zero_cost_pct:.1f}% zero)")
    print(f"\nDatasets ({len(datasets)}):")
    for ds in sorted(datasets):
        cr = coverage_report.get(ds, {})
        print(f"  {ds:<30} models={cr.get('n_models',0):>3}  "
              f"queries_union={cr.get('n_queries_union',0):>6}  "
              f"queries_intersection={cr.get('n_queries_intersection',0):>6}")
    print(f"\nModels ({len(models)}):")
    for m in sorted(models):
        print(f"  {m}")
    print("\n" + "=" * 70)
    print(f"Methodological notes:")
    for n in summary["methodological_notes"]:
        print(f"  - {n}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DIR}")
        print("Run the download step first (see README).")
        sys.exit(1)
    audit()
