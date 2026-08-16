"""
Load raw LLMRouterBench files from data/raw/bench/ into a unified DataFrame.

Schema is inferred from the actual files during audit (scripts/01_audit_data.py).
This module must NOT be finalised until the audit confirms the raw schema.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw-file traversal
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".parquet"}


def iter_raw_files(raw_dir: str | Path) -> Iterator[Path]:
    """Yield all data files under raw_dir recursively."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_file(path: Path) -> pd.DataFrame:
    """Load a single raw benchmark file into a DataFrame."""
    ext = path.suffix.lower()
    try:
        if ext == ".parquet":
            return pd.read_parquet(path)
        elif ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".jsonl":
            return pd.read_json(path, lines=True)
        elif ext == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Some benchmarks store as {key: [records]}
                for v in data.values():
                    if isinstance(v, list):
                        return pd.DataFrame(v)
                return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported extension: {ext}")
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return pd.DataFrame()


def load_all_raw(raw_dir: str | Path, max_files: int | None = None) -> pd.DataFrame:
    """
    Load all raw benchmark files and concatenate into one DataFrame.

    Adds a `_source_file` column tracking which file each row came from.
    Does NOT apply canonical mapping — call preprocessing.canonicalise() for that.
    """
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    n = 0
    for path in iter_raw_files(raw_dir):
        if max_files is not None and n >= max_files:
            break
        df = load_file(path)
        if df.empty:
            continue
        df["_source_file"] = str(path.relative_to(raw_dir))
        # Infer dataset name from directory structure: bench/<dataset>/...
        parts = path.relative_to(raw_dir).parts
        df["_inferred_dataset"] = parts[0] if len(parts) > 1 else "unknown"
        frames.append(df)
        n += 1
    if not frames:
        raise RuntimeError(f"No data files found under {raw_dir}")
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d files → %d rows from %s", n, len(combined), raw_dir)
    return combined


# ---------------------------------------------------------------------------
# Processed outcomes loader
# ---------------------------------------------------------------------------

def load_outcomes(processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Load the canonical outcomes.parquet built by scripts/02_build_dataset.py."""
    path = Path(processed_dir) / "outcomes.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical outcomes not found at {path}. "
            "Run scripts/02_build_dataset.py first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded outcomes: %d rows, %d models, %d queries",
                len(df),
                df["model"].nunique(),
                df["query_id"].nunique())
    return df
