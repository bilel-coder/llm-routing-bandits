"""
Targeted rerun: Oracle x S3 only (10 seeds x 5 lambdas = 50 rows).

Why this exists
---------------
The original P5 run applied the S3 cost-drift ShiftFn to the ENTIRE Oracle
reward stream in one call. `_make_cost_shift_fn` ignores its `t` argument and
scales every row it is handed — correct inside `run_stream()` (the sequential
loop has already consumed rows < cp before the mutation), but wrong in the
Oracle's vectorised path, where the whole array is processed at once. Rows
before the change point therefore saw post-drift costs, corrupting the Oracle's
pre-shift actions, costs and rewards on S3.

Fixed in scripts/12_p5_final_validation.py by slicing at cp. This script
recomputes ONLY the affected rows and splices them back into
results/tables/p5_final_dev.csv. All other routers are untouched: their regret
against the Oracle is computed inside run_stream() on a separate, already
correct code path.

Run scripts/13_p5_analysis.py afterwards to refresh the derived tables/figures.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CSV = Path("results/tables/p5_final_dev.csv")
P5 = Path("scripts/12_p5_final_validation.py")


def load_p5_module():
    """Import scripts/12_p5_final_validation.py (leading digit blocks a normal import)."""
    spec = importlib.util.spec_from_file_location("p5_final_validation", P5)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if not CSV.exists():
        print(f"ERROR: {CSV} not found. Run scripts/12_p5_final_validation.py first.")
        sys.exit(1)

    p5 = load_p5_module()
    df = pd.read_csv(CSV)
    columns = list(df.columns)

    mask = (df["router"] == "Oracle") & (df["scenario"] == "S3")
    n_old = int(mask.sum())
    expected = len(p5.SEEDS) * len(p5.LAMBDAS)
    print(f"Loaded {len(df):,} rows; Oracle x S3 rows to replace: {n_old} "
          f"(expected {expected})")
    if n_old != expected:
        print(f"ERROR: expected {expected} Oracle x S3 rows, found {n_old}. Aborting.")
        sys.exit(1)

    old = df[mask].set_index(["seed", "lambda"]).sort_index()

    X_dv, Q_dv, _, Cn_dv, ds_dv = p5.load_matrices("DEV")
    K = Q_dv.shape[1]

    new_rows = []
    for seed in p5.SEEDS:
        for lam in p5.LAMBDAS:
            stream = p5.build_stream("S3", X_dv, ds_dv, seed, lam)
            o = stream.order
            Q_stream = Q_dv[o]
            Cn_stream = Cn_dv[o].copy()
            R_stream = Q_stream - lam * Cn_stream

            cp = list(stream.shift_schedule.keys())[0]
            fn = stream.shift_schedule[cp]
            R_post, _, Cn_post = fn(cp, R_stream[cp:], Q_stream[cp:], Cn_stream[cp:])
            R_stream = np.concatenate([R_stream[:cp], R_post])
            Cn_stream = np.concatenate([Cn_stream[:cp], Cn_post])

            row = p5.oracle_summary_row(R_stream, Q_stream, Cn_stream,
                                        ds_dv[o], K, "S3", stream.metadata)
            row.update({"router": "Oracle", "seed": seed,
                        "lambda": lam, "scenario": "S3"})
            new_rows.append(row)

    new = pd.DataFrame(new_rows)

    # Report the correction before writing anything.
    cmp = new.set_index(["seed", "lambda"]).sort_index()
    print("\nEffect of the fix (mean over 50 rows):")
    for col in ("macro_utility", "micro_utility", "micro_cost_norm",
                "macro_quality", "recovery_time"):
        if col in old.columns and col in cmp.columns:
            o_m, n_m = old[col].mean(), cmp[col].mean()
            print(f"  {col:18s} {o_m:9.4f} -> {n_m:9.4f}   (delta {n_m - o_m:+.4f})")
    print(f"\n  recovery_time before: {sorted(old['recovery_time'].unique())}")
    print(f"  recovery_time after : {sorted(cmp['recovery_time'].unique())}")

    out = pd.concat([df[~mask], new], ignore_index=True)
    missing = set(columns) - set(out.columns)
    extra = set(out.columns) - set(columns)
    if missing or extra:
        print(f"ERROR: column mismatch. missing={missing} extra={extra}")
        sys.exit(1)
    out = out[columns]

    # Restore the original row ordering (router, seed, lambda, scenario).
    order_key = {r: i for i, r in enumerate(p5.ROUTERS)}
    scen_key = {s: i for i, s in enumerate(p5.SCENARIOS)}
    out = (out.assign(_r=out["router"].map(order_key),
                      _s=out["scenario"].map(scen_key))
              .sort_values(["_r", "seed", "lambda", "_s"])
              .drop(columns=["_r", "_s"])
              .reset_index(drop=True))

    assert len(out) == len(df), f"row count changed: {len(df)} -> {len(out)}"

    backup = CSV.with_suffix(".csv.bak_oracle_s3")
    if not backup.exists():
        backup.write_bytes(CSV.read_bytes())
        print(f"\nBacked up original to {backup}")

    out.to_csv(CSV, index=False)
    print(f"\nWrote {len(out):,} rows to {CSV} ({n_old} Oracle x S3 rows replaced).")
    print("Next: python scripts/13_p5_analysis.py")


if __name__ == "__main__":
    main()
