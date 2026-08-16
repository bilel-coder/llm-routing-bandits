"""
Canonical schema mapping and preprocessing for LLMRouterBench.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "query_id",
    "dataset",
    "prompt",
    "model",
    "quality",
    "cost",
    "prompt_tokens",
    "completion_tokens",
]

RAW_FIELD_CANDIDATES: dict[str, list[str]] = {
    "query_id":          ["query_id", "id", "idx", "question_id", "index"],
    "dataset":           ["dataset", "task", "benchmark", "source", "_inferred_dataset"],
    "prompt":            ["prompt", "question", "query", "input", "instruction"],
    "model":             ["model", "model_name", "llm", "provider"],
    "quality":           ["score", "quality", "accuracy", "correct", "pass", "result"],
    "cost":              ["cost", "price", "total_cost", "inference_cost"],
    "prompt_tokens":     ["prompt_tokens", "input_tokens", "n_prompt_tokens"],
    "completion_tokens": ["completion_tokens", "output_tokens", "n_completion_tokens",
                          "generated_tokens"],
}


def _resolve_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    for candidate in RAW_FIELD_CANDIDATES[canonical]:
        if candidate in df.columns:
            return candidate
    return None


def canonicalise(df: pd.DataFrame, source_hint: str = "") -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    unresolved: list[str] = []

    for canon in CANONICAL_COLUMNS:
        raw_col = _resolve_column(df, canon)
        if raw_col is not None:
            out[canon] = df[raw_col].values
        else:
            unresolved.append(canon)
            out[canon] = np.nan

    if unresolved:
        logger.warning("[%s] Could not resolve canonical columns: %s",
                       source_hint, unresolved)

    result = pd.DataFrame(out)

    for col in ("_source_file", "_inferred_dataset"):
        if col in df.columns:
            result[col] = df[col].values

    return result


# ---------------------------------------------------------------------------
# Cost normalisation
# ---------------------------------------------------------------------------

@dataclass
class CostNormaliser:
    """
    Global 95th-percentile cost normalisation.

    One scalar s = p95 of all positive training costs across all pool models.
    c_norm[i,a] = c[i,a] / s

    Properties:
    - Preserves inter-model cost ratios exactly.
    - s is fitted on training data only; same scalar applied to val and test.
    - c_norm > 1 is valid and expected for costs above the training 95th percentile
      (e.g. post-drift price increases). Do NOT clip.
    - Using a global (not per-model) scalar is intentional: per-model scaling
      would destroy the cost differences between models that the router must
      learn to exploit.
    """

    scale_: float = 1.0
    fitted_: bool = False
    n_train_rows_: int = 0
    n_positive_costs_: int = 0
    pool_: list = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "CostNormaliser":
        """
        Fit from training rows. df must contain 'cost' column.
        Only positive costs are used to compute the percentile (zero costs
        indicate generation failures and must have been removed upstream).
        """
        if "cost" not in df.columns:
            raise ValueError("df must have 'cost' column")

        positive = df.loc[df["cost"] > 0, "cost"]
        if len(positive) == 0:
            raise ValueError(
                "No positive costs found in training data. "
                "Ensure generation failures are filtered before normalisation."
            )

        self.scale_ = float(np.percentile(positive, 95))
        self.n_train_rows_ = len(df)
        self.n_positive_costs_ = len(positive)
        if "model" in df.columns:
            self.pool_ = sorted(df["model"].unique().tolist())
        self.fitted_ = True
        logger.info(
            "CostNormaliser fitted: p95=%.6f USD from %d positive-cost rows, %d models",
            self.scale_, self.n_positive_costs_, len(self.pool_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Return c / s for each row. No clipping — costs above p95 produce c_norm > 1.
        """
        if not self.fitted_:
            raise RuntimeError("CostNormaliser must be fitted before transform()")
        return pd.Series(
            df["cost"].values / self.scale_,
            index=df.index,
            name="cost_norm",
        )

    def save(self, path: str | Path) -> None:
        """Persist fitted parameters to YAML for reproducibility audit."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                {
                    "strategy":          "global_p95",
                    "scale":             self.scale_,
                    "fitted":            self.fitted_,
                    "n_train_rows":      self.n_train_rows_,
                    "n_positive_costs":  self.n_positive_costs_,
                    "pool":              self.pool_,
                },
                f,
            )
        logger.info("CostNormaliser saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CostNormaliser":
        """Load a previously fitted CostNormaliser from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        if data.get("strategy") != "global_p95":
            raise ValueError(
                f"Unexpected normalisation strategy '{data.get('strategy')}'. "
                "Expected 'global_p95'. Re-run the normaliser fitting step."
            )
        obj = cls()
        obj.scale_ = float(data["scale"])
        obj.fitted_ = bool(data["fitted"])
        obj.n_train_rows_ = int(data.get("n_train_rows", 0))
        obj.n_positive_costs_ = int(data.get("n_positive_costs", 0))
        obj.pool_ = data.get("pool", [])
        return obj


# ---------------------------------------------------------------------------
# Reward construction
# ---------------------------------------------------------------------------

def compute_reward(
    quality: pd.Series,
    cost_norm: pd.Series,
    lam: float,
) -> pd.Series:
    """r = quality - lambda * cost_norm"""
    return quality - lam * cost_norm
