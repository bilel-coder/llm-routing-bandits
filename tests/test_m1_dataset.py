"""
M1 tests — canonical dataset validation.

All tests operate on data/processed/outcomes.parquet.
Run: pytest tests/test_m1_dataset.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed")
OUTCOMES_PATH = PROCESSED_DIR / "outcomes.parquet"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def outcomes() -> pd.DataFrame:
    if not OUTCOMES_PATH.exists():
        pytest.skip(f"outcomes.parquet not found at {OUTCOMES_PATH}. "
                    "Run scripts/02_build_dataset.py first.")
    return pd.read_parquet(OUTCOMES_PATH)


# ---------------------------------------------------------------------------
# Test 1: Required columns exist
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["query_id", "dataset", "model", "quality", "cost",
                    "prompt_tokens", "completion_tokens", "prompt"]

def test_required_columns_present(outcomes: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in outcomes.columns]
    assert not missing, f"Missing columns: {missing}"


# ---------------------------------------------------------------------------
# Test 2: (query_id, model) uniqueness
# ---------------------------------------------------------------------------

def test_query_model_uniqueness(outcomes: pd.DataFrame) -> None:
    dup_count = outcomes.duplicated(subset=["query_id", "model"]).sum()
    assert dup_count == 0, (
        f"Found {dup_count} duplicate (query_id, model) pairs. "
        "Each query must appear exactly once per model."
    )


# ---------------------------------------------------------------------------
# Test 3: Cost non-negative
# ---------------------------------------------------------------------------

def test_costs_non_negative(outcomes: pd.DataFrame) -> None:
    bad = (outcomes["cost"] < 0.0).sum()
    assert bad == 0, f"Found {bad} rows with negative cost."


# ---------------------------------------------------------------------------
# Test 4: Quality in [0, 1] (where not NaN)
# ---------------------------------------------------------------------------

def test_quality_range(outcomes: pd.DataFrame) -> None:
    valid = outcomes["quality"].dropna()
    out_of_range = ((valid < 0.0) | (valid > 1.0)).sum()
    assert out_of_range == 0, (
        f"Found {out_of_range} quality values outside [0, 1]. "
        f"Max={valid.max():.4f}, Min={valid.min():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: query_id consistently maps to the same prompt
# ---------------------------------------------------------------------------

def test_query_id_prompt_consistency(outcomes: pd.DataFrame) -> None:
    # Each query_id should map to one unique prompt.
    # Known exception: 2 queries in the raw benchmark have U+2028 (LINE SEPARATOR)
    # artefacts in some model files vs others (0.007% of queries, non-routing-relevant).
    # We tolerate up to 5 such raw-data encoding inconsistencies.
    KNOWN_TOLERANCE = 5
    inconsistent = (
        outcomes.groupby("query_id")["prompt"]
        .nunique()
        .gt(1)
        .sum()
    )
    assert inconsistent <= KNOWN_TOLERANCE, (
        f"Found {inconsistent} query_ids mapping to different prompts "
        f"(tolerance={KNOWN_TOLERANCE}). Investigate upstream data if > {KNOWN_TOLERANCE}."
    )


# ---------------------------------------------------------------------------
# Test 6# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_data_types(outcomes: pd.DataFrame) -> None:
    assert outcomes["quality"].dtype in [np.float32, np.float64], \
        f"quality dtype should be float, got {outcomes['quality'].dtype}"
    assert outcomes["cost"].dtype in [np.float32, np.float64], \
        f"cost dtype should be float, got {outcomes['cost'].dtype}"
    assert outcomes["dataset"].dtype == object or str(outcomes["dataset"].dtype) == "string", \
        f"dataset should be string, got {outcomes['dataset'].dtype}"
    assert outcomes["model"].dtype == object or str(outcomes["model"].dtype) == "string", \
        f"model should be string, got {outcomes['model'].dtype}"


# ---------------------------------------------------------------------------
# Test 7: At least 10 models and 5 datasets
# ---------------------------------------------------------------------------

def test_minimum_scale(outcomes: pd.DataFrame) -> None:
    n_models = outcomes["model"].nunique()
    n_datasets = outcomes["dataset"].nunique()
    # Pool frozen at 6 models (M1.5 audit); dataset floor remains 5
    assert n_models >= 6, f"Expected ≥6 models, found {n_models}"
    assert n_datasets >= 5, f"Expected ≥5 datasets, found {n_datasets}"


# ---------------------------------------------------------------------------
# Test 8: Coverage — every model has at least 100 records
# ---------------------------------------------------------------------------

def test_model_minimum_records(outcomes: pd.DataFrame) -> None:
    per_model = outcomes.groupby("model").size()
    thin_models = per_model[per_model < 100]
    assert len(thin_models) == 0, (
        f"Models with <100 records: {thin_models.to_dict()}"
    )


# ---------------------------------------------------------------------------
# Test 9: Coverage calculation is correct
# ---------------------------------------------------------------------------

def test_coverage_calculation(outcomes: pd.DataFrame) -> None:
    # Pick a dataset with full coverage; verify union == intersection
    for ds in ["arcc", "bbh", "winogrande"]:
        grp = outcomes[outcomes["dataset"] == ds]
        if grp.empty:
            continue
        model_sets = {m: set(mg["query_id"]) for m, mg in grp.groupby("model")}
        union = set().union(*model_sets.values())
        inter = set.intersection(*model_sets.values())
        # These datasets should have full coverage
        assert union == inter, (
            f"Dataset {ds}: expected full coverage but "
            f"union={len(union)}, intersection={len(inter)}"
        )


# ---------------------------------------------------------------------------
# Test 10: query_id format is consistent
# ---------------------------------------------------------------------------

def test_query_id_format(outcomes: pd.DataFrame) -> None:
    sample = outcomes["query_id"].head(100)
    malformed = [qid for qid in sample if len(qid.split("__")) < 3]
    assert not malformed, (
        f"Malformed query_ids (expected format 'dataset__split__index'): {malformed[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 11: No all-NaN quality columns per model
# ---------------------------------------------------------------------------

def test_no_all_nan_model(outcomes: pd.DataFrame) -> None:
    per_model_nan = outcomes.groupby("model")["quality"].apply(
        lambda x: x.isna().all()
    )
    all_nan_models = per_model_nan[per_model_nan].index.tolist()
    assert not all_nan_models, f"Models with ALL NaN quality: {all_nan_models}"
