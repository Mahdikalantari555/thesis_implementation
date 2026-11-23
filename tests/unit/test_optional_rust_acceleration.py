"""Smoke checks for the opt-in Rust acceleration contract.

These tests verify the *contract*, not a scientific result: Python stays in the
repository and is reachable; a clean/opted-out run uses it; an enabled run
produces identical values. All checks are L0 (infrastructure).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

SWAPS = [
    (ROOT / "src/ssl_anomaly/baseline/score.py", "ewma_detector"),
    (ROOT / "src/ssl_anomaly/baseline/score.py", "cusum_detector"),
    (ROOT / "src/ssl_anomaly/io/patches.py", "aggregate_bin"),
]

# One deterministic payload across all three functions, printed as a digest by
# a fresh interpreter so import-time env flags take effect.
_PROBE = r"""
import hashlib, sys
import numpy as np
import ssl_anomaly.baseline.score as s
import ssl_anomaly.io.patches as p
mods = [s.ewma_detector.__module__, s.cusum_detector.__module__, p.aggregate_bin.__module__]
r = np.random.default_rng(0)
x = r.normal(0, 1, 36); x[5] = np.nan
f_ewma, _ = s.ewma_detector(x, alpha=0.35)
f_cusum, _ = s.cusum_detector(x, drift_k=1.0)
scenes = [r.uniform(0.1, 0.5, (8, 8, 8)).astype(np.float32) for _ in range(2)]
masks = [r.random((8, 8)) > 0.2 for _ in range(2)]
agg = p.aggregate_bin(scenes, masks, in_field=np.ones((8, 8), bool), dtype="float16")
h = hashlib.sha256()
for a in (f_ewma, f_cusum, agg.values.astype(np.float64), agg.pixel_valid):
    h.update(np.ascontiguousarray(a).tobytes())
print("|".join(mods) + "|" + h.hexdigest())
"""


def _run(*, disabled: bool) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env.pop("FAST_ANYTHING_DISABLE", None)
    if disabled:
        env["FAST_ANYTHING_DISABLE"] = "1"
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_original_python_bodies_are_retained_and_blocks_are_reversible():
    """No Python function is deleted or rewritten; the swap is an append."""
    for path, func in SWAPS:
        text = path.read_text(encoding="utf-8")
        assert f"def {func}(" in text, f"{func} definition missing from {path}"
        assert "FAST_ANYTHING_DISABLE" in text
        assert "def _fast_anything_load():" in text
        assert f"{func} = _fast_anything_load() or {func}" in text
        assert f"del _fast_anything_load" in text
        # the swap block is the LAST thing in the file: removing it restores
        # the original module exactly (no edits above it). The tool closes each
        # block with a `# <<< fast-anything: <id> <<<` marker.
        lines = text.rstrip().splitlines()
        assert lines[-1].startswith("# <<< fast-anything:")
        assert lines[-2] == "del _fast_anything_load"


def test_opt_out_runs_the_canonical_python_functions():
    """FAST_ANYTHING_DISABLE=1 binds the original Python defs, in any checkout."""
    digest = _run(disabled=True)
    modules = digest.split("|")[:3]
    assert modules == [
        "ssl_anomaly.baseline.score", "ssl_anomaly.baseline.score",
        "ssl_anomaly.io.patches",
    ], modules


def test_enabled_and_python_paths_produce_identical_values():
    """The Rust path (when present) must be numerically identical to Python.

    On a clean clone without binaries the default run is also Python; the
    digest equality still holds, so this check needs no skip marker.
    """
    disabled = _run(disabled=True).split("|")[-1]
    enabled = _run(disabled=False).split("|")[-1]
    assert disabled == enabled


def test_default_path_is_python_when_no_binary_is_available(tmp_path, monkeypatch):
    """Simulate a clean clone: hide the compiled libs and confirm fail-open.

    ``sitecustomize`` and the swap block only import from ``src/.fast/lib``, so
    temporarily pointing that import at an empty directory exercises exactly
    the clean-clone path without deleting or moving any file.
    """
    code = r"""
import ssl_anomaly.baseline.score as s, ssl_anomaly.io.patches as p
print("|".join([s.ewma_detector.__module__, s.cusum_detector.__module__, p.aggregate_bin.__module__]))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(SRC)
    env.pop("FAST_ANYTHING_DISABLE", None)
    # Put an empty shadow of the extension module names ahead of .fast/lib so
    # `from fa_... import ...` raises ImportError -> Python fallback.
    shadow = tmp_path / "fa_ssl_anomaly_baseline_score_ewma_detector.py"
    shadow.write_text("raise ImportError('simulated clean clone')\n")
    (tmp_path / "fa_ssl_anomaly_baseline_score_cusum_detector.py").write_text(
        "raise ImportError('simulated clean clone')\n")
    (tmp_path / "fa_ssl_anomaly_io_patches_aggregate_bin.py").write_text(
        "raise ImportError('simulated clean clone')\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().split("|") == [
        "ssl_anomaly.baseline.score", "ssl_anomaly.baseline.score",
        "ssl_anomaly.io.patches",
    ], out.stdout.strip()
