"""
M1.5 tests — cost normalisation and macro/micro metrics.

Run: pytest tests/test_m15_preprocessing.py -v
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_router.data.preprocessing import CostNormaliser, compute_reward
from llm_router.evaluation.metrics import (
    macro_mean,
    micro_mean,
    macro_summary,
    micro_summary,
    summarise_across_seeds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cost_df(costs: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for model, cs in costs.items():
        for c in cs:
            rows.append({"model": model, "cost": c})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CostNormaliser: basic correctness
# ---------------------------------------------------------------------------

def test_global_p95_scalar():
    """Scale should be the 95th percentile of all positive training costs."""
    costs = list(range(1, 101))  # 1..100 — p95 = 95.05
    df = pd.DataFrame({"cost": costs, "model": "m"})
    cn = CostNormaliser()
    cn.fit(df)
    expected_p95 = float(np.percentile(costs, 95))
    assert abs(cn.scale_ - expected_p95) < 1e-10


def test_inter_model_ratios_preserved():
    """
    Global scalar must preserve the cost ratio between models.
    If model A costs 2× model B, c_norm_A / c_norm_B == 2 for all queries.
    """
    costs_A = [0.010, 0.012, 0.011]
    costs_B = [0.005, 0.006, 0.0055]
    df_train = make_cost_df({"A": costs_A, "B": costs_B})
    cn = CostNormaliser().fit(df_train)

    df_eval = pd.DataFrame({
        "model": ["A", "B"],
        "cost":  [0.010, 0.005],
    })
    norms = cn.transform(df_eval).values
    ratio = norms[0] / norms[1]
    assert abs(ratio - 2.0) < 1e-10, f"Expected ratio=2.0, got {ratio}"


def test_no_clipping_above_p95():
    """Costs above the p95 training value produce c_norm > 1.0 (no clipping)."""
    df_train = pd.DataFrame({"cost": list(range(1, 101)), "model": "m"})
    cn = CostNormaliser().fit(df_train)

    df_high = pd.DataFrame({"cost": [200.0], "model": "m"})
    c_norm = cn.transform(df_high).values[0]
    assert c_norm > 1.0, "Cost above p95 should produce c_norm > 1.0"


def test_drift_2x_cost_gives_2x_norm():
    """
    Post-drift cost doubling should produce exactly 2× normalized values.
    c_norm_drift / c_norm_original = 2.0 for the same query.
    """
    df_train = pd.DataFrame({"cost": np.linspace(0.001, 0.1, 100), "model": "m"})
    cn = CostNormaliser().fit(df_train)

    original_cost = 0.05
    drift_cost = 0.10
    df_orig  = pd.DataFrame({"cost": [original_cost], "model": "m"})
    df_drift = pd.DataFrame({"cost": [drift_cost],    "model": "m"})
    norm_orig  = cn.transform(df_orig).values[0]
    norm_drift = cn.transform(df_drift).values[0]
    assert abs(norm_drift / norm_orig - 2.0) < 1e-10


def test_fit_requires_positive_costs():
    """fit() must raise if no positive costs present (all zero = unfiltered failures)."""
    df = pd.DataFrame({"cost": [0.0, 0.0, 0.0], "model": "m"})
    with pytest.raises(ValueError, match="No positive costs"):
        CostNormaliser().fit(df)


def test_transform_raises_if_not_fitted():
    cn = CostNormaliser()
    df = pd.DataFrame({"cost": [0.01], "model": "m"})
    with pytest.raises(RuntimeError, match="fitted"):
        cn.transform(df)


def test_save_load_roundtrip():
    """Saved and reloaded normaliser produces identical transforms."""
    df_train = pd.DataFrame({"cost": np.linspace(0.001, 0.1, 50), "model": "m"})
    cn = CostNormaliser().fit(df_train)

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        cn.save(path)
        cn2 = CostNormaliser.load(path)
        assert abs(cn.scale_ - cn2.scale_) < 1e-12
        assert cn2.fitted_ is True

        df_eval = pd.DataFrame({"cost": [0.05, 0.12], "model": "m"})
        np.testing.assert_array_almost_equal(
            cn.transform(df_eval).values,
            cn2.transform(df_eval).values,
        )
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_rejects_wrong_strategy():
    """Loading a YAML with a different strategy must raise."""
    import yaml
    bad_data = {"strategy": "per_model_minmax", "scale": 1.0, "fitted": True}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(bad_data, f)
        path = f.name
    try:
        with pytest.raises(ValueError, match="strategy"):
            CostNormaliser.load(path)
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CostNormaliser: only positive costs contribute to p95
# ---------------------------------------------------------------------------

def test_zero_costs_excluded_from_p95():
    """
    Zero costs must be excluded from the p95 calculation.
    Only positive costs define the scale.
    """
    positive = list(range(1, 101))
    df = pd.DataFrame({"cost": [0.0] * 50 + positive, "model": "m"})
    cn_with_zeros = CostNormaliser().fit(df)

    df_pos_only = pd.DataFrame({"cost": positive, "model": "m"})
    cn_pos_only = CostNormaliser().fit(df_pos_only)

    assert abs(cn_with_zeros.scale_ - cn_pos_only.scale_) < 1e-10


# ---------------------------------------------------------------------------
# Macro vs micro averaging
# ---------------------------------------------------------------------------

def make_result_df() -> pd.DataFrame:
    """3 datasets of unequal size; metric values differ by dataset."""
    rows = []
    # ds_A: 100 rows, metric=0.8
    for _ in range(100):
        rows.append({"dataset": "ds_A", "quality": 0.8})
    # ds_B: 10 rows, metric=0.2
    for _ in range(10):
        rows.append({"dataset": "ds_B", "quality": 0.2})
    # ds_C: 1 row, metric=0.5
    rows.append({"dataset": "ds_C", "quality": 0.5})
    return pd.DataFrame(rows)


def test_macro_mean_equal_dataset_weight():
    df = make_result_df()
    # Macro: (0.8 + 0.2 + 0.5) / 3 = 0.5
    assert abs(macro_mean(df, "quality") - 0.5) < 1e-10


def test_micro_mean_dominated_by_large_dataset():
    df = make_result_df()
    # Micro: (100*0.8 + 10*0.2 + 1*0.5) / 111 ≈ 0.7432...
    expected = (100 * 0.8 + 10 * 0.2 + 1 * 0.5) / 111
    assert abs(micro_mean(df, "quality") - expected) < 1e-10


def test_macro_and_micro_differ_for_unequal_datasets():
    df = make_result_df()
    assert abs(macro_mean(df, "quality") - micro_mean(df, "quality")) > 0.1


def test_macro_mean_raises_without_dataset_column():
    df = pd.DataFrame({"quality": [0.5, 0.6]})
    with pytest.raises(ValueError, match="dataset"):
        macro_mean(df, "quality")


def test_macro_summary_returns_dict():
    df = make_result_df()
    result = macro_summary(df, ["quality"])
    assert "quality" in result
    assert abs(result["quality"] - 0.5) < 1e-10


def test_summarise_across_seeds_macro():
    """summarise_across_seeds with averaging='macro' uses per-dataset weighting."""
    df = make_result_df()
    result = summarise_across_seeds([df, df], metrics=["quality"],
                                    averaging="macro")
    assert "quality" in result.columns
    # Mean across 2 identical seeds = macro_mean = 0.5
    assert abs(result.loc["mean", "quality"] - 0.5) < 1e-10


def test_summarise_across_seeds_micro():
    df = make_result_df()
    result = summarise_across_seeds([df, df], metrics=["quality"],
                                    averaging="micro")
    expected_micro = (100 * 0.8 + 10 * 0.2 + 1 * 0.5) / 111
    assert abs(result.loc["mean", "quality"] - expected_micro) < 1e-10
