"""
M2 — Build aligned Q[i,a] and C[i,a] outcome matrices per partition.

Uses:
  - Frozen 8-model canonical order from configs/model_pool.yaml
  - Feature-partition query_id order from data/interim/feature_query_ids.json
    (guarantees row alignment with X_{train,val,test}.npy)
  - Frozen global-p95 CostNormaliser from data/interim/cost_normaliser.yaml
    (LOADED, not refitted)

Writes per partition to data/interim/matrices/:
  Q_{train,val,test}.npy       — (n, 8) quality
  C_{train,val,test}.npy       — (n, 8) raw cost (USD)
  C_norm_{train,val,test}.npy  — (n, 8) = C / scale
  datasets_{train,val,test}.json — (n,) list of dataset labels per row
  matrix_meta.json — pool order, scale, shapes, checksum
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from llm_router.data.preprocessing import CostNormaliser

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUTCOMES  = Path("data/processed/outcomes.parquet")
CONFIG    = Path("configs/model_pool.yaml")
FEAT_QIDS = Path("data/interim/feature_query_ids.json")
NORMALIZER = Path("data/interim/cost_normaliser.yaml")
OUT_DIR   = Path("data/interim/matrices")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pool_order() -> list[str]:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    return [m["name"] for m in cfg["pool"]]


def build_partition(
    df: pd.DataFrame,
    qids: list[str],
    pool: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (Q, C, datasets) with rows aligned to qids order, cols to pool order."""
    n_q, n_a = len(qids), len(pool)
    qid_to_row = {q: i for i, q in enumerate(qids)}
    arm_to_col = {m: j for j, m in enumerate(pool)}

    Q = np.full((n_q, n_a), np.nan, dtype=np.float32)
    C = np.full((n_q, n_a), np.nan, dtype=np.float32)

    sub = df[df["query_id"].isin(qid_to_row) & df["model"].isin(arm_to_col)]
    for row in sub.itertuples():
        i = qid_to_row[row.query_id]
        j = arm_to_col[row.model]
        Q[i, j] = row.quality
        C[i, j] = row.cost

    # Dataset labels per row
    qid_to_ds = dict(zip(sub.drop_duplicates("query_id")["query_id"],
                          sub.drop_duplicates("query_id")["dataset"]))
    datasets = [qid_to_ds[q] for q in qids]
    return Q, C, datasets


def main() -> None:
    df = pd.read_parquet(OUTCOMES)
    with open(FEAT_QIDS) as f:
        feat_qids = json.load(f)
    pool = load_pool_order()

    print(f"Pool ({len(pool)}): {pool}")

    # Load frozen normaliser
    cn = CostNormaliser.load(NORMALIZER)
    if cn.pool_ and sorted(cn.pool_) != sorted(pool):
        raise ValueError(
            f"Pool mismatch: normaliser fitted on {cn.pool_} but current pool is {pool}"
        )
    print(f"CostNormaliser scale (p95) = ${cn.scale_:.6f} USD  (frozen, not refitted)")

    meta = {
        "pool_order": pool,
        "cost_norm_scale_usd": cn.scale_,
        "cost_norm_strategy": "global_p95",
        "partitions": {},
    }

    for partition in ("train", "val", "test"):
        qids = feat_qids[partition]
        Q, C, datasets = build_partition(df, qids, pool)

        # Alignment & sanity checks
        assert Q.shape == C.shape == (len(qids), len(pool))
        n_nan_Q = int(np.isnan(Q).sum())
        n_nan_C = int(np.isnan(C).sum())
        n_inf_Q = int(np.isinf(Q).sum())
        n_inf_C = int(np.isinf(C).sum())
        n_neg_C = int((C < 0).sum())
        assert n_nan_Q == 0 and n_nan_C == 0, (
            f"{partition}: NaN found — Q={n_nan_Q}, C={n_nan_C}. "
            "Complete-coverage filter was supposed to remove these."
        )
        assert n_inf_Q == 0 and n_inf_C == 0
        assert n_neg_C == 0

        C_norm = (C / cn.scale_).astype(np.float32)

        np.save(OUT_DIR / f"Q_{partition}.npy",      Q)
        np.save(OUT_DIR / f"C_{partition}.npy",      C)
        np.save(OUT_DIR / f"C_norm_{partition}.npy", C_norm)
        with open(OUT_DIR / f"datasets_{partition}.json", "w") as f:
            json.dump(datasets, f)

        # Consistency check: X row count matches Q row count
        X = np.load(Path("data/interim") / f"X_{partition}.npy")
        assert X.shape[0] == Q.shape[0], (
            f"{partition}: X has {X.shape[0]} rows, Q has {Q.shape[0]}. Alignment broken."
        )

        meta["partitions"][partition] = {
            "n": int(Q.shape[0]),
            "n_arms": int(Q.shape[1]),
            "X_shape": list(X.shape),
            "Q_shape": list(Q.shape),
            "C_shape": list(C.shape),
            "C_norm_shape": list(C_norm.shape),
            "n_datasets": int(len(set(datasets))),
            "Q_min": float(Q.min()),  "Q_max": float(Q.max()),  "Q_mean": float(Q.mean()),
            "C_min_usd": float(C.min()), "C_max_usd": float(C.max()),
            "C_mean_usd": float(C.mean()),
            "C_norm_min": float(C_norm.min()), "C_norm_max": float(C_norm.max()),
            "C_norm_mean": float(C_norm.mean()),
        }

        print(f"\n[{partition}]  X={X.shape}  Q={Q.shape}  C={C.shape}  C_norm={C_norm.shape}")
        print(f"  Q: min={Q.min():.3f}  mean={Q.mean():.3f}  max={Q.max():.3f}")
        print(f"  C (USD):    min=${C.min():.6f}  mean=${C.mean():.6f}  max=${C.max():.4f}")
        print(f"  C_norm:     min={C_norm.min():.4f}  mean={C_norm.mean():.4f}  "
              f"max={C_norm.max():.4f}")
        print(f"  Datasets in partition: {sorted(set(datasets))}")

    with open(OUT_DIR / "matrix_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll matrices saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
