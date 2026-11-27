#!/usr/bin/env python
"""Robustness battery (E6): missing observations, temporal density, fills, GSD.

Drives the registered extraction arms on a fixed seeded SUBSET of fields
(robustness comparisons do not need the full archive; the subset and seed are
recorded), then reports:
- embedding drift: mean cosine(full-arm embedding, arm embedding) per obs,
  plus the mean abs difference;
- stage-probe macro-F1 delta vs the default arm on the same subset.

Arms: fill05, b11dup, up144 (registered sensitivity arms) and drop1/drop2/
drop4/drop6, rebin30, rebin60 (robustness modifiers). Each extraction is
invoked through extract_embeddings.py so the artifacts stay reusable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oe_eval_common as common  # noqa: E402

ARMS = [
    # (suffix, extract args)
    ("fill05", ["--arm", "fill05"]),
    ("b11dup", ["--arm", "b11dup"]),
    ("up144", ["--arm", "up144"]),
    ("drop1", ["--drop-bins", "1"]),
    ("drop2", ["--drop-bins", "2"]),
    ("drop4", ["--drop-bins", "4"]),
    ("drop6", ["--drop-bins", "6"]),
    ("rebin30", ["--rebin-days", "30"]),
    ("rebin60", ["--rebin-days", "60"]),
]
ROBUSTNESS_FIELDS_N = 120
ROBUSTNESS_SEED = 2026


def pick_fields() -> list[int]:
    import zarr

    store = zarr.open_group(str(common.PATCH_STORE), mode="r")
    ids = sorted(int(f) for f in store["field_ids"][:])
    rng = np.random.default_rng(ROBUSTNESS_SEED)
    return sorted(rng.choice(ids, size=ROBUSTNESS_FIELDS_N, replace=False)
                  .tolist())


def run_arm(suffix: str, args: list[str], fields: list[int]) -> None:
    cmd = [sys.executable, "-X", "utf8",
           str(HERE / "extract_embeddings.py"),
           "--fields", *[str(f) for f in fields],
           "--suffix", suffix, *args]
    print("RUN:", " ".join(cmd[:8]), f"... ({len(fields)} fields)")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"arm {suffix} failed with exit {proc.returncode}")
    for line in proc.stdout.splitlines():
        if line.startswith("SUMMARY "):
            s = json.loads(line[8:])
            print(f"  arm {suffix}: {s['n_obs_rows']} obs, "
                  f"{s['wall_time_s']} s")


def cosine_drift(base: np.ndarray, alt: np.ndarray) -> float:
    num = (base * alt).sum(1)
    den = np.linalg.norm(base, axis=1) * np.linalg.norm(alt, axis=1) + 1e-12
    return float((num / den).mean())


def probe_macro_f1(df: pd.DataFrame, train_fields: set[int],
                   dev_fields: set[int], seed: int = 42) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = common.add_derived_columns(df)
    d = d[d["field_id"].isin(train_fields | dev_fields)]
    y = d["phase"].astype("category").cat.codes.values
    tr = d["field_id"].isin(train_fields).values
    te = d["field_id"].isin(dev_fields).values
    X = d[common.EMBED_COLS].values
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    clf.fit(X[tr], y[tr])
    return float(f1_score(y[te], clf.predict(X[te]), average="macro"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[a for a, _ in ARMS])
    ap.add_argument("--skip-extraction", action="store_true",
                    help="only analyze already-extracted arms")
    ap.add_argument("--seed", type=int, default=ROBUSTNESS_SEED)
    args = ap.parse_args(argv)

    fields = pick_fields()
    if not args.skip_extraction:
        for suffix, eargs in ARMS:
            if suffix in args.arms:
                run_arm(suffix, eargs, fields)

    train_fields, dev_fields = common.split_field_sets()
    base = common.load_obs_embeddings("")
    base_sub = base[base["field_id"].isin(fields)].set_index(
        ["field_id", "cycle_index", "bin_index"])
    base_probe = probe_macro_f1(base[base["field_id"].isin(fields)],
                                train_fields, dev_fields)

    rows = []
    for suffix, _ in ARMS:
        if suffix not in args.arms:
            continue
        path = common.OUT_DIR / f"embeddings_v1_2_nano{suffix}.parquet"
        if not path.exists():
            rows.append(dict(arm=suffix, status="absent"))
            continue
        alt = pd.read_parquet(path).set_index(
            ["field_id", "cycle_index", "bin_index"])
        joined = base_sub.join(alt[common.EMBED_COLS], how="inner",
                               rsuffix="_alt")
        if len(joined) == 0:
            rows.append(dict(arm=suffix, status="no-aligned-obs"))
            continue
        drift = cosine_drift(joined[common.EMBED_COLS].values,
                             joined[[c + "_alt" for c in common.EMBED_COLS]].values)
        f1 = probe_macro_f1(pd.read_parquet(path), train_fields, dev_fields)
        rows.append(dict(
            arm=suffix, status="ok",
            n_aligned_obs=int(len(joined)),
            mean_cosine_to_full=round(drift, 4),
            probe_macro_f1=round(f1, 4),
            probe_f1_delta_vs_full=round(f1 - base_probe, 4)))

    out = dict(
        subset_n_fields=ROBUSTNESS_FIELDS_N, subset_seed=ROBUSTNESS_SEED,
        baseline_probe_macro_f1=round(base_probe, 4),
        arms=rows,
        note="drop-k arms re-extract with k fewer valid bins per cycle; "
             "rebin arms aggregate to coarser medians; drift is mean cosine "
             "to the default-arm embedding on aligned bins",
    )
    path = common.write_table("robustness", out)
    print(f"wrote {path}")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
