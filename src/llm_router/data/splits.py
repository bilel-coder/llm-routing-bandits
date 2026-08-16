"""
Fixed train / validation / test splits for LLMRouterBench.

Splits are built from query_ids (not rows), so every model's outcomes
for a given query land in the same partition — no leakage at the query level.

Splits are frozen to disk as data/interim/splits.json before any algorithm
is trained or tuned. The same file is loaded for all downstream experiments.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_PATH = Path("data/interim/splits.json")


def make_splits(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
    stratify_by: Optional[str] = "dataset",
) -> dict[str, list]:
    """
    Assign each unique query_id to train, validation, or test.

    Parameters
    ----------
    df : canonical outcomes DataFrame (must contain 'query_id' and optionally 'dataset')
    train_frac, val_frac : fractions; test_frac = 1 - train_frac - val_frac
    seed : for reproducibility
    stratify_by : column to stratify on (preserves dataset distribution across splits)

    Returns
    -------
    dict with keys 'train', 'val', 'test' each mapping to a list of query_ids
    """
    if "query_id" not in df.columns:
        raise ValueError("df must contain 'query_id'")

    test_frac = 1.0 - train_frac - val_frac
    if test_frac < 0:
        raise ValueError("train_frac + val_frac must be <= 1.0")

    rng = np.random.default_rng(seed)

    query_meta = df.drop_duplicates("query_id")[["query_id"]].copy()
    if stratify_by and stratify_by in df.columns:
        query_meta = (
            df.drop_duplicates("query_id")[["query_id", stratify_by]]
            .reset_index(drop=True)
        )

    train_ids, val_ids, test_ids = [], [], []

    if stratify_by and stratify_by in query_meta.columns:
        for _, grp in query_meta.groupby(stratify_by):
            qids = grp["query_id"].tolist()
            rng.shuffle(qids)
            n = len(qids)
            n_train = int(round(n * train_frac))
            n_val   = int(round(n * val_frac))
            train_ids.extend(qids[:n_train])
            val_ids.extend(qids[n_train:n_train + n_val])
            test_ids.extend(qids[n_train + n_val:])
    else:
        qids = query_meta["query_id"].tolist()
        rng.shuffle(qids)
        n = len(qids)
        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))
        train_ids = qids[:n_train]
        val_ids   = qids[n_train:n_train + n_val]
        test_ids  = qids[n_train + n_val:]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    logger.info(
        "Splits: train=%d, val=%d, test=%d (total queries=%d)",
        len(train_ids), len(val_ids), len(test_ids), len(query_meta),
    )
    return splits


def save_splits(splits: dict, path: str | Path = DEFAULT_SPLIT_PATH) -> None:
    """Persist splits to disk so they are frozen for all experiments."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info("Splits frozen to %s", path)


def load_splits(path: str | Path = DEFAULT_SPLIT_PATH) -> dict[str, list]:
    """Load previously frozen splits."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Splits file not found at {path}. "
            "Run scripts/03_create_splits.py first."
        )
    with open(path) as f:
        splits = json.load(f)
    logger.info(
        "Loaded splits: train=%d, val=%d, test=%d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


def apply_split(df: pd.DataFrame, splits: dict, partition: str) -> pd.DataFrame:
    """Filter df to rows whose query_id is in the given partition."""
    if partition not in splits:
        raise ValueError(f"Unknown partition '{partition}'. Choose from {list(splits)}")
    ids = set(splits[partition])
    return df[df["query_id"].isin(ids)].reset_index(drop=True)
