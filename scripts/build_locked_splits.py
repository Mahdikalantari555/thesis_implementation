#!/usr/bin/env python3
"""Build data/processed/splits_locked.json once (design D2, task 1.1/1.2).

Run exactly once before any training; the manifest is never regenerated inside
the change. Logs split sizes and the cycles dropped by the temporal cutoff
(task 1.4 checkpoint: >15 % of uncensored cycles dropped must be raised).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ssl_anomaly.monitor.splits import (  # noqa: E402
    SPLITS_LOCKED_PATH,
    build_splits_locked,
)


def main() -> int:
    if SPLITS_LOCKED_PATH.exists():
        print(f"REFUSING: {SPLITS_LOCKED_PATH} already exists (built once, never regenerated).")
        return 1

    manifest = build_splits_locked()
    n_uncensored = sum(
        manifest["counts"][s] for s in ("train", "dev", "test")
    )
    dropped = manifest["n_dropped_train_cutoff"] + manifest["n_dropped_dev_cutoff"]
    drop_frac = dropped / max(1, n_uncensored + dropped)

    SPLITS_LOCKED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_LOCKED_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {SPLITS_LOCKED_PATH}")
    print(f"manifest_sha256: {manifest['manifest_sha256'][:16]}…")
    print(f"git_sha: {manifest['git_sha']}")
    print(f"kmeans: seed={manifest['kmeans']['seed']}, k={manifest['kmeans']['n_blocks']}, "
          f"test_block={manifest['kmeans']['test_block']} "
          f"({manifest['kmeans']['block_sizes'][str(manifest['kmeans']['test_block'])]} fields), "
          f"dev_block={manifest['kmeans']['dev_block']} "
          f"({manifest['kmeans']['block_sizes'][str(manifest['kmeans']['dev_block'])]} fields)")
    for name in ("train", "dev", "test", "monitoring_only"):
        n_fields = len(manifest[f"{name}_fields"])
        print(f"  {name:16s}: {manifest['counts'][name]:5d} cycles / {n_fields} fields")
    print(f"Temporal-cutoff drops: train {manifest['n_dropped_train_cutoff']}, "
          f"dev {manifest['n_dropped_dev_cutoff']} "
          f"({100 * drop_frac:.1f}% of uncensored non-block cycles)")

    if drop_frac > 0.15:
        print(
            "WARNING (task 1.4 checkpoint): the temporal cutoff drops more than 15 % of "
            "uncensored non-block cycles — this must be raised with the user."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
