"""
M2 — Prompt feature construction.

Pipeline:
  1. Load outcomes.parquet + splits.json.
  2. Extract unique (query_id, prompt_norm) pairs.
  3. Encode with sentence-transformers/all-MiniLM-L6-v2  → 384-dim.
  4. Cache embeddings to data/interim/embeddings_384.npz (query_id, matrix).
  5. Fit PCA(64) on TRAIN embeddings ONLY.
  6. Transform train / val / test with the frozen PCA.
  7. Save X_train.npy, X_val.npy, X_test.npy, query_id order files, PCA meta.

Scientific constraints:
  - PCA is fitted on train rows only.
  - Val/test are transformed with the frozen PCA — never re-fitted.
  - Row order in X_{train,val,test} matches sorted splits["<partition>"]
    to guarantee alignment with Q/C matrices built downstream.
"""

from __future__ import annotations

# IMPORTANT: sentence_transformers (→ torch) must be imported BEFORE sklearn
# on Windows to avoid a libomp/MKL DLL conflict that breaks torch's c10.dll.
from sentence_transformers import SentenceTransformer  # noqa: E402

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUTCOMES = Path("data/processed/outcomes.parquet")
SPLITS   = Path("data/interim/splits.json")
INTERIM  = Path("data/interim")

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM     = 384
PCA_DIM       = 64
EMBED_CACHE   = INTERIM / "embeddings_384.npz"
X_TRAIN_PATH  = INTERIM / "X_train.npy"
X_VAL_PATH    = INTERIM / "X_val.npy"
X_TEST_PATH   = INTERIM / "X_test.npy"
QIDS_PATH     = INTERIM / "feature_query_ids.json"
PCA_META_PATH = INTERIM / "pca_meta.json"


def encode_prompts(qid_to_prompt: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """Return (ordered_query_ids, embeddings [n, 384])."""
    if EMBED_CACHE.exists():
        cached = np.load(EMBED_CACHE, allow_pickle=True)
        qids_cached = list(cached["query_ids"])
        emb_cached = cached["embeddings"]
        cache_map = {q: emb_cached[i] for i, q in enumerate(qids_cached)}
        ordered = sorted(qid_to_prompt.keys())
        missing = [q for q in ordered if q not in cache_map]
        if not missing:
            print(f"  Cache hit: {len(ordered):,} embeddings loaded from {EMBED_CACHE}")
            emb = np.stack([cache_map[q] for q in ordered])
            return ordered, emb
        print(f"  Cache incomplete: {len(missing):,} missing embeddings — re-encoding all.")

    print(f"  Loading {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    ordered = sorted(qid_to_prompt.keys())
    prompts = [qid_to_prompt[q] for q in ordered]
    print(f"  Encoding {len(prompts):,} prompts (batch=64) ...")
    emb = model.encode(
        prompts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    assert emb.shape == (len(ordered), EMBED_DIM), f"Unexpected shape {emb.shape}"
    print(f"  Caching embeddings to {EMBED_CACHE} ...")
    np.savez_compressed(EMBED_CACHE, query_ids=np.array(ordered), embeddings=emb)
    return ordered, emb


def main() -> None:
    df = pd.read_parquet(OUTCOMES)
    with open(SPLITS) as f:
        splits = json.load(f)

    # Unique prompts by query_id. Use prompt_norm if available (canonical text).
    prompt_col = "prompt_norm" if "prompt_norm" in df.columns else "prompt"
    unique = df.drop_duplicates("query_id")[["query_id", prompt_col]]
    qid_to_prompt = dict(zip(unique["query_id"], unique[prompt_col]))
    print(f"Unique queries: {len(qid_to_prompt):,}")

    # Encode
    ordered_qids, embeddings = encode_prompts(qid_to_prompt)
    qid_to_row = {q: i for i, q in enumerate(ordered_qids)}
    print(f"Embeddings: shape={embeddings.shape}  dtype={embeddings.dtype}")

    # Partition-aligned matrices (sorted query_id order per partition)
    train_qids = sorted(splits["train"])
    val_qids   = sorted(splits["val"])
    test_qids  = sorted(splits["test"])

    def slice_emb(qids):
        idx = np.array([qid_to_row[q] for q in qids])
        return embeddings[idx]

    E_train = slice_emb(train_qids)
    E_val   = slice_emb(val_qids)
    E_test  = slice_emb(test_qids)

    print(f"Pre-PCA shapes: train={E_train.shape}  val={E_val.shape}  test={E_test.shape}")

    # Fit PCA on TRAIN only
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(E_train)
    ev = pca.explained_variance_ratio_
    cum_ev = np.cumsum(ev)
    print(f"\nPCA (fitted on TRAIN, {PCA_DIM} components):")
    print(f"  First 5 components explained variance: {ev[:5].round(4)}")
    print(f"  Cumulative EV at 10 comps: {cum_ev[9]:.4f}")
    print(f"  Cumulative EV at 32 comps: {cum_ev[31]:.4f}")
    print(f"  Cumulative EV at 64 comps: {cum_ev[63]:.4f}")

    X_train = pca.transform(E_train).astype(np.float32)
    X_val   = pca.transform(E_val).astype(np.float32)
    X_test  = pca.transform(E_test).astype(np.float32)

    print(f"Post-PCA shapes: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    # Persist
    np.save(X_TRAIN_PATH, X_train)
    np.save(X_VAL_PATH,   X_val)
    np.save(X_TEST_PATH,  X_test)
    with open(QIDS_PATH, "w") as f:
        json.dump({"train": train_qids, "val": val_qids, "test": test_qids}, f)
    with open(PCA_META_PATH, "w") as f:
        json.dump({
            "embed_model": EMBED_MODEL,
            "embed_dim": EMBED_DIM,
            "pca_dim": PCA_DIM,
            "fitted_on": "train",
            "seed": 42,
            "cumulative_ev": cum_ev.tolist(),
            "explained_variance_ratio": ev.tolist(),
            "cum_ev_at_pca_dim": float(cum_ev[-1]),
            "n_train_samples": int(X_train.shape[0]),
        }, f, indent=2)

    print(f"\nSaved: {X_TRAIN_PATH.name}, {X_VAL_PATH.name}, {X_TEST_PATH.name}, "
          f"{QIDS_PATH.name}, {PCA_META_PATH.name}")


if __name__ == "__main__":
    main()
