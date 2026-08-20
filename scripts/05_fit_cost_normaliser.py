"""
M1.5 — Fit and freeze the global p95 CostNormaliser on the TRAIN partition only.

Loads:  data/processed/outcomes.parquet
        data/interim/splits.json
Writes: data/interim/cost_normaliser.yaml

Scientific constraints:
  - Only training-partition costs are used to fit the scalar.
  - Scalar = 95th percentile of all positive costs across all 8 pool models.
  - Same scalar applied to val/test at transform time — never refitted.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

from llm_router.data.preprocessing import CostNormaliser

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROCESSED = Path("data/processed/outcomes.parquet")
SPLITS    = Path("data/interim/splits.json")
OUT       = Path("data/interim/cost_normaliser.yaml")


def main() -> None:
    df = pd.read_parquet(PROCESSED)
    with open(SPLITS) as f:
        splits = json.load(f)
    train_ids = set(splits["train"])
    train_df = df[df["query_id"].isin(train_ids)].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows, {df['model'].nunique()} models, "
          f"{df['query_id'].nunique():,} queries.")
    print(f"Train partition: {len(train_df):,} rows, "
          f"{train_df['query_id'].nunique():,} queries.")

    cn = CostNormaliser().fit(train_df)
    cn.save(OUT)

    print()
    print("=" * 60)
    print("COST NORMALISER FROZEN")
    print("=" * 60)
    print(f"  Strategy:        global_p95")
    print(f"  Scale (p95):     ${cn.scale_:.6f} USD")
    print(f"  Train rows:      {cn.n_train_rows_:,}")
    print(f"  Positive-cost:   {cn.n_positive_costs_:,}")
    print(f"  Pool size:       {len(cn.pool_)} models")
    print(f"  Saved to:        {OUT}")
    print()
    print("  Sanity check — c_norm at each cost decile of train:")
    positive = train_df.loc[train_df["cost"] > 0, "cost"].sort_values()
    for pct in [1, 25, 50, 75, 95, 99, 100]:
        q = positive.quantile(pct / 100)
        print(f"    p{pct:>3d}: cost=${q:.6f}  →  c_norm={q/cn.scale_:.4f}")


if __name__ == "__main__":
    main()
