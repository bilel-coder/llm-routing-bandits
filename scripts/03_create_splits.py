"""
M1.5 — Create fixed group-aware train/validation/test splits.

Split unit: prompt_group_id (not raw query_id).
This guarantees the same prompt text cannot appear in both train and test
under different query_ids (the arenahard/tau2/mmlupro leakage pattern).

Procedure:
  1. Collect unique (prompt_group_id, dataset) pairs.
  2. Stratify by dataset: each dataset's groups are split 60/20/20.
  3. Map group assignments back to query_ids.
  4. Freeze to data/interim/splits.json.
  5. Generate leakage report.

Usage:
    python scripts/03_create_splits.py

Outputs:
    data/interim/splits.json
    data/interim/split_stats.json
    data/interim/leakage_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
INTERIM_DIR   = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
SEED       = 42


def make_group_aware_splits(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Split by prompt_group_id stratified by dataset.
    Returns dict: partition name → list of query_ids.
    """
    rng = np.random.default_rng(SEED)

    # One row per (prompt_group_id, dataset) — groups may span one dataset only
    group_meta = (
        df.drop_duplicates("query_id")[["query_id", "prompt_group_id", "dataset"]]
        .groupby("prompt_group_id")
        .agg(
            dataset=("dataset", "first"),    # all queries in same group share dataset
            query_ids=("query_id", list),
        )
        .reset_index()
    )

    train_qids, val_qids, test_qids = [], [], []

    for dataset, grp in group_meta.groupby("dataset"):
        group_ids = grp["prompt_group_id"].tolist()
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_train = int(round(n * TRAIN_FRAC))
        n_val   = int(round(n * VAL_FRAC))

        train_groups = set(group_ids[:n_train])
        val_groups   = set(group_ids[n_train:n_train + n_val])
        test_groups  = set(group_ids[n_train + n_val:])

        for _, row in grp.iterrows():
            gid = row["prompt_group_id"]
            qids = row["query_ids"]
            if gid in train_groups:
                train_qids.extend(qids)
            elif gid in val_groups:
                val_qids.extend(qids)
            else:
                test_qids.extend(qids)

    log.info("Splits (by prompt_group → query_ids): train=%d, val=%d, test=%d",
             len(train_qids), len(val_qids), len(test_qids))
    return {"train": train_qids, "val": val_qids, "test": test_qids}


def generate_leakage_report(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
) -> dict:
    """
    Verify no prompt_group_id and no normalized prompt text appears in
    more than one partition.
    """
    qid_to_group  = df.drop_duplicates("query_id").set_index("query_id")["prompt_group_id"]
    qid_to_prompt = df.drop_duplicates("query_id").set_index("query_id")["prompt_norm"]

    partition_groups: dict[str, set] = {}
    partition_prompts: dict[str, set] = {}
    for part, qids in splits.items():
        partition_groups[part]  = set(qid_to_group.reindex(qids).dropna())
        partition_prompts[part] = set(qid_to_prompt.reindex(qids).dropna())

    parts = list(splits.keys())
    group_leaks, prompt_leaks = [], []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            pa, pb = parts[i], parts[j]
            g_overlap = partition_groups[pa] & partition_groups[pb]
            p_overlap = partition_prompts[pa] & partition_prompts[pb]
            if g_overlap:
                group_leaks.append({"partitions": [pa, pb], "n_shared_groups": len(g_overlap)})
            if p_overlap:
                prompt_leaks.append({"partitions": [pa, pb], "n_shared_prompts": len(p_overlap)})

    report = {
        "group_leaks":  group_leaks,
        "prompt_leaks": prompt_leaks,
        "status": "PASS" if not group_leaks and not prompt_leaks else "FAIL",
    }
    if report["status"] == "PASS":
        log.info("Leakage report: PASS — no shared groups or prompts across partitions.")
    else:
        log.error("Leakage report: FAIL — %s", report)
    return report


def main() -> None:
    outcomes_path = PROCESSED_DIR / "outcomes.parquet"
    if not outcomes_path.exists():
        print(f"ERROR: {outcomes_path} not found. Run 02_build_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(outcomes_path)
    log.info("Loaded %d rows, %d queries, %d models",
             len(df), df["query_id"].nunique(), df["model"].nunique())

    # Require prompt_group_id
    if "prompt_group_id" not in df.columns:
        print("ERROR: prompt_group_id missing. Re-run 02_build_dataset.py.")
        sys.exit(1)

    # Also need prompt_norm for leakage report — rebuild if absent
    if "prompt_norm" not in df.columns:
        import unicodedata
        def normalize(s):
            if not isinstance(s, str): return ""
            s = s.replace(" ", "\n").replace(" ", "\n")
            return unicodedata.normalize("NFKC", s).strip()
        df["prompt_norm"] = df["prompt"].apply(normalize)

    # Create splits
    splits = make_group_aware_splits(df)

    # Leakage report
    leakage = generate_leakage_report(df, splits)
    assert leakage["status"] == "PASS", f"Leakage detected: {leakage}"

    # Save splits
    splits_path = INTERIM_DIR / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    log.info("Splits frozen to %s", splits_path)

    # Per-dataset split stats
    qid_to_ds = df.drop_duplicates("query_id").set_index("query_id")["dataset"]
    stats_per_ds: dict[str, dict] = {}
    for part, qids in splits.items():
        ds_counts = qid_to_ds.reindex(qids).value_counts().to_dict()
        for ds, n in ds_counts.items():
            if ds not in stats_per_ds:
                stats_per_ds[ds] = {}
            stats_per_ds[ds][part] = n

    stats = {
        "seed": SEED,
        "total_queries": df["query_id"].nunique(),
        "total_prompt_groups": int(df["prompt_group_id"].nunique()),
        "n_train": len(splits["train"]),
        "n_val":   len(splits["val"]),
        "n_test":  len(splits["test"]),
        "n_models": df["model"].nunique(),
        "n_datasets": df["dataset"].nunique(),
        "per_dataset": stats_per_ds,
        "leakage_status": leakage["status"],
    }
    with open(INTERIM_DIR / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(INTERIM_DIR / "leakage_report.json", "w") as f:
        json.dump(leakage, f, indent=2)

    print("\n" + "=" * 65)
    print("SPLITS FROZEN (group-aware, dataset-stratified)")
    print("=" * 65)
    print(f"  Seed:           {SEED}")
    print(f"  Total queries:  {stats['total_queries']:,}")
    print(f"  Prompt groups:  {stats['total_prompt_groups']:,}")
    print(f"  Train:          {stats['n_train']:,}  ({stats['n_train']/stats['total_queries']*100:.1f}%)")
    print(f"  Validation:     {stats['n_val']:,}  ({stats['n_val']/stats['total_queries']*100:.1f}%)")
    print(f"  Test:           {stats['n_test']:,}  ({stats['n_test']/stats['total_queries']*100:.1f}%)")
    print(f"  Leakage check:  {leakage['status']}")
    print()
    print(f"  {'Dataset':<25} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*48}")
    for ds, counts in sorted(stats_per_ds.items()):
        tr = counts.get("train", 0)
        va = counts.get("val", 0)
        te = counts.get("test", 0)
        print(f"  {ds:<25} {tr:>7} {va:>7} {te:>7}")
    print("=" * 65)


if __name__ == "__main__":
    main()
