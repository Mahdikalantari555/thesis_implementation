#!/usr/bin/env python3
"""Record the pre-registration freeze and the v2 artifact bindings (tasks 0.2-0.4).

Change: regional-ssl-representation-evaluation. Writes
``reports/regional_representation/preregistration_freeze.json`` — the committed
evidence that every capstone run binds against the SAME frozen chain:

- **0.2** — every pre-registered artifact is hashed as it sits on disk and
  compared to its frozen sha256 prefix. Any mismatch exits non-zero. A missing
  artifact is recorded ``absent`` (never substituted); the artifacts the
  field-mean path REQUIRES also fail the run when absent.
- **0.3** — the v2 normalization sidecar is verified twice: its byte hash
  against the pre-registration prefix AND its internal bindings
  (``splits_sha256``/``holdout_sha256`` prefixes, ``n_train_cycles`` = the
  locked train count — the train-split-only guarantee).
- **0.4** — the pre-registration document's own sha256 and the last commit that
  touched it are recorded, so any later deviation is detectable mechanically.

Usage
-----
::

    /home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
        scripts/evaluation/record_preregistration_freeze.py

Expected wall time: ~10-40 s (hashes the 320 MB patch store and the cycle
parquet once). Exit codes: 0 = all bindings hold; 1 = mismatch or a required
artifact is absent; the JSON is still written so the failure is inspectable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ssl_anomaly.evaluation.binding import (  # noqa: E402
    bind_artifacts,
    load_preregistration,
)
from ssl_anomaly.evaluation.runner import environment_block  # noqa: E402

#: Artifacts without which the field-mean capstone path cannot run at all.
REQUIRED_FOR_FIELD_MEAN = ("cycles_490_v2", "splits_locked_v2",
                           "holdout_manifest_v2", "norm_stats_full_v2")

#: Locked train-cycle count recorded by the v2 split manifest (finding 25 set).
EXPECTED_TRAIN_CYCLES = 2105

OUT_PATH = REPO_ROOT / "reports/regional_representation/preregistration_freeze.json"


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--base-dir", type=Path, default=None,
                   help="artifact root (default: repository root)")
    args = p.parse_args(argv)
    started = time.time()

    prereg = load_preregistration()
    binding = bind_artifacts(prereg=prereg, base_dir=args.base_dir,
                             strict=False, compute_hashes=True,
                             require=REQUIRED_FOR_FIELD_MEAN)
    blob = binding.to_dict()

    # ---- 0.3: the norm sidecar's internal bindings ------------------------
    norm_checks: dict = {"status": "absent", "detail": "artifact not bound"}
    norm_status = next((a for a in binding.artifacts if a.name == "norm_stats_full_v2"), None)
    if norm_status is not None and norm_status.ok and norm_status.actual_sha256:
        sidecar_path = REPO_ROOT / prereg.artifacts["norm_stats_full_v2"]["path"]
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_self = str(payload.get("sha256", "")).lower()
        splits_field = str(payload.get("splits_sha256", "")).lower()
        holdout_field = str(payload.get("holdout_sha256", "")).lower()
        n_train = payload.get("n_train_cycles")
        checks = {
            "self_sha256_matches_bytes": sidecar_self == norm_status.actual_sha256,
            "binds_splits_locked_v2": splits_field.startswith("30ecbe10"),
            "binds_holdout_manifest_v2": holdout_field.startswith("d501bf81"),
            "train_split_only": n_train == EXPECTED_TRAIN_CYCLES,
            "computed_from": payload.get("computed_from"),
        }
        norm_checks = {
            "status": "verified" if all(v is True for k, v in checks.items()
                                        if isinstance(v, bool)) else "failed",
            "n_train_cycles": n_train,
            "n_train_fields": payload.get("n_train_fields"),
            "checks": checks,
        }

    # ---- 0.4: freeze identity --------------------------------------------
    prereg_commit = _git("log", "-1", "--format=%H", "--",
                         "configs/evaluation/pre_registration.yaml")
    freeze = {
        "change": prereg.payload.get("change"),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistration": {
            "path": "configs/evaluation/pre_registration.yaml",
            "document_sha256": prereg.document_sha256,
            "last_commit_touching_file": prereg_commit or "absent (not a git checkout)",
            "version": prereg.payload.get("version"),
            "created": prereg.payload.get("created"),
            "amendments": prereg.payload.get("amendments", []),
        },
        "artifact_binding": blob,
        "norm_sidecar_v2": norm_checks,
        "required_for_field_mean": list(REQUIRED_FOR_FIELD_MEAN),
        "verdict": ("bound" if (binding.ok and norm_checks["status"] == "verified")
                    else "failed"),
        "environment": environment_block(),
        "wall_time_s": round(time.time() - started, 2),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")

    for a in binding.artifacts:
        line = f"  {a.name:24s} {a.status:8s}"
        if a.actual_sha256:
            line += f" {a.actual_sha256[:12]}"
        if a.detail:
            line += f" — {a.detail}"
        print(line)
    print(f"norm sidecar v2: {norm_checks['status']}")
    print(f"prereg document sha256: {prereg.document_sha256}")
    print(f"verdict: {freeze['verdict']}")

    ok = binding.ok and norm_checks["status"] == "verified"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
