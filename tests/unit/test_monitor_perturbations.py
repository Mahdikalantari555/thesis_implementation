"""Tests for MAD-scaled perturbations (task 3.1) and the dev-metric pieces
(detection delay, FAR calibration, forbidden targets — task 3.3)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ssl_anomaly.monitor import perturbations as mpert  # noqa: E402
from ssl_anomaly.monitor import scoring as mscoring  # noqa: E402


T, C = 36, 8
MAD_SCALE = np.full((T, C), 1.4826, dtype=np.float64)  # scale → MAD = 1.0 everywhere


def make_cycle(seed=0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.5, 0.05, size=(T, C)).astype(np.float32)
    x[:, 7] = rng.normal(28, 2, size=T)
    m = np.ones((T, C), dtype=bool)
    m[33:, :] = False  # tail unobserved
    m[10, 7] = False  # isolated unobserved LST bin
    x[~m] = np.nan  # values are NaN exactly where unobserved
    return x, m


# ---------------------------------------------------------------------------
# Injection correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ptype", mpert.INTERIM_TYPES)
def test_injection_changes_intended_channels_only(ptype: str) -> None:
    x, m = make_cycle()
    res = mpert.inject_perturbation(
        x, m, pert_type=ptype, onset_bin=12, duration=3, severity=1.0, mad=MAD_SCALE,
    )
    channels = set(mpert.TYPE_CHANNELS[ptype])
    changed = {c for c in range(C) if not np.allclose(res.x[:, c][m[:, c]], x[:, c][m[:, c]])}
    assert changed <= channels, f"{ptype} modified foreign channels {changed - channels}"
    assert changed, f"{ptype} changed nothing"
    assert res.onset_bin == 12


def test_injection_respects_observation_mask() -> None:
    """Unobserved entries must never be created or destroyed."""
    x, m = make_cycle()
    for ptype in mpert.INTERIM_TYPES:
        res = mpert.inject_perturbation(
            x, m, pert_type=ptype, onset_bin=8, duration=4, severity=2.0, mad=MAD_SCALE,
        )
        assert np.array_equal(res.m, m)
        assert np.all(np.isnan(res.x[~m]))


def test_severity_scales_monotonically() -> None:
    """A 2×MAD perturbation moves values at least as far as 1×MAD."""
    x, m = make_cycle()
    r1 = mpert.inject_perturbation(x, m, pert_type="ndvi_drop", onset_bin=10, duration=3, severity=1.0, mad=MAD_SCALE)
    r2 = mpert.inject_perturbation(x, m, pert_type="ndvi_drop", onset_bin=10, duration=3, severity=2.0, mad=MAD_SCALE)
    d1 = np.abs(r1.x[:, 0][m[:, 0]] - x[:, 0][m[:, 0]]).sum()
    d2 = np.abs(r2.x[:, 0][m[:, 0]] - x[:, 0][m[:, 0]]).sum()
    assert d2 > d1 > 0


def test_phenology_delay_never_erases_observations() -> None:
    x, m = make_cycle()
    res = mpert.inject_perturbation(
        x, m, pert_type="phenology_delay", onset_bin=6, duration=5, severity=1.0, mad=MAD_SCALE,
    )
    was_obs = m[6:, :]
    still_finite = np.isfinite(res.x[6:, :]) | ~was_obs
    assert np.all(still_finite)


def test_mad_sizing_uses_train_stats() -> None:
    """severity × MAD: with scale=1.4826 (MAD=1), a 1-bin lst_rise adds 1.0 K."""
    x, m = make_cycle()
    before = float(x[12, 7])
    res = mpert.inject_perturbation(x, m, pert_type="lst_rise", onset_bin=12, duration=1, severity=1.0, mad=MAD_SCALE)
    assert res.x[12, 7] == pytest.approx(before + 1.0)


def test_inject_dataset_onset_and_phases() -> None:
    rng = np.random.default_rng(0)
    n = 20
    X = np.stack([make_cycle(seed=i)[0] for i in range(n)])
    M = np.stack([make_cycle(seed=i)[1] for i in range(n)])
    cfg = mpert.PerturbationConfig(n_per_stratum=n)
    results = mpert.inject_dataset(X, M, MAD_SCALE, cfg, rng)
    assert len(results) == n
    for r in results:
        assert r.pert_type in mpert.INTERIM_TYPES
        assert r.onset_bin in (6, 18, 30)  # early / mid / late


# ---------------------------------------------------------------------------
# Alarm rule and detection delay (hand-made cycles)
# ---------------------------------------------------------------------------
def test_first_alarm_bin_delay_handmade() -> None:
    """A hand-made cycle with an injected spike: delay measured from onset."""
    s = np.full(36, 0.5)
    s[20] = 5.0  # onset bin 20 carries the anomaly
    assert mscoring.first_alarm_bin(s, tau=2.0, k=1, onset_bin=20) == 20
    assert mscoring.first_alarm_bin(s, tau=2.0, k=1, onset_bin=0) == 20
    # k=2 needs two consecutive bins above τ.
    s2 = np.full(36, 0.5)
    s2[20] = 5.0
    assert mscoring.first_alarm_bin(s2, tau=2.0, k=2, onset_bin=0) == -1
    s2[21] = 5.0
    assert mscoring.first_alarm_bin(s2, tau=2.0, k=2, onset_bin=0) == 20
    # Unobserved bins break runs (NaN): the only above-τ pair straddles a NaN.
    s3 = np.full(36, 0.5)
    s3[14] = 5.0
    s3[15] = np.nan
    s3[16] = 5.0
    assert mscoring.first_alarm_bin(s3, tau=2.0, k=2, onset_bin=0) == -1
    s3[17] = 5.0
    assert mscoring.first_alarm_bin(s3, tau=2.0, k=2, onset_bin=0) == 16


def test_alarm_per_cycle_runs() -> None:
    s = np.array([np.nan, 1, 1, np.nan, 1, 1, 1])
    assert mscoring.alarm_per_cycle(s, tau=0.5, k=3)
    assert not mscoring.alarm_per_cycle(s, tau=0.5, k=4)
    s2 = np.array([1, np.nan, 1])  # NaN breaks the only candidate pair
    assert not mscoring.alarm_per_cycle(s2, tau=0.5, k=2)
    assert mscoring.alarm_per_cycle(np.array([1, 1]), tau=0.5, k=2)


# ---------------------------------------------------------------------------
# FAR calibration hits 5 % on synthetic normal scores
# ---------------------------------------------------------------------------
def test_far_calibration_hits_target() -> None:
    rng = np.random.default_rng(42)
    clean = [rng.normal(0.0, 1.0, size=36) for _ in range(400)]
    tau, achieved = mscoring.calibrate_threshold(clean, k=1, far_target=0.05)
    assert achieved <= 0.05
    # With n=400 the empirical FAR should be close to the target.
    assert achieved >= 0.02, f"tau too conservative: FAR {achieved}"


def test_far_calibration_k2() -> None:
    rng = np.random.default_rng(1)
    clean = [rng.normal(0.0, 1.0, size=36) for _ in range(400)]
    tau, achieved = mscoring.calibrate_threshold(clean, k=2, far_target=0.05)
    assert achieved <= 0.05


def test_choose_threshold_prefers_higher_detection() -> None:
    """k chosen by detection rate; tie → smaller k."""
    rng = np.random.default_rng(3)
    clean = [rng.normal(0.0, 1.0, size=36) for _ in range(200)]
    injected = []
    onsets = []
    for _ in range(60):
        s = rng.normal(0.0, 1.0, size=36)
        onset = 10
        s[onset:onset + 4] += 6.0  # easily detected at any k
        injected.append(s)
        onsets.append(onset)
    thr, per_k = mscoring.choose_threshold(clean, injected, onsets, far_target=0.05)
    assert thr.far_achieved <= 0.05
    assert thr.k in (1, 2)
    assert per_k["k1"]["detection_rate"] >= 0.9


# ---------------------------------------------------------------------------
# Partial AUC-PR
# ---------------------------------------------------------------------------
def test_aucpr_perfect_and_chance() -> None:
    clean = np.array([0.1, 0.2, 0.3, 0.4])
    inj = np.array([0.9, 1.0])
    assert mscoring.partial_auc_pr(clean, inj) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    c = rng.normal(0, 1, 500)
    i = rng.normal(0, 1, 50)
    ap = mscoring.partial_auc_pr(c, i)
    assert 0.0 <= ap <= 1.0
    # Chance level ≈ prevalence (0.5 / 5.5 ≈ 0.09), within noise.
    assert ap < 0.3


# ---------------------------------------------------------------------------
# Forbidden targets in development code (task 3.3)
# ---------------------------------------------------------------------------
FORBIDDEN_NAMES = {"cwsi", "eta_deficit", "y_stress", "eta_daily"}
ALLOWED_PATHS = [
    REPO_ROOT / "src/ssl_anomaly/monitor",
    REPO_ROOT / "scripts/train_main.py",
    REPO_ROOT / "scripts/eval_dev.py",
    REPO_ROOT / "scripts/run_dev_matrix.py",
    REPO_ROOT / "scripts/check_dev_env.py",
    REPO_ROOT / "scripts/freeze_main_model.py",
]


def test_no_forbidden_targets_in_dev_code() -> None:
    """AST scan: dev code must not reference CWSI/ETa/y_stress anywhere
    (names, attributes, string keys). Docstrings are excluded — they document
    the prohibition; any *use* (name, attribute, subscript key) fails."""
    offenders: list[str] = []
    for path in ALLOWED_PATHS:
        files = sorted(path.glob("*.py")) if path.is_dir() else [path]
        for f in files:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            docstring_nodes = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node, clean=False)
                    if doc is not None:
                        for stmt in node.body:
                            if (
                                isinstance(stmt, ast.Expr)
                                and isinstance(stmt.value, ast.Constant)
                                and isinstance(stmt.value.value, str)
                            ):
                                docstring_nodes.add(id(stmt.value))
            for node in ast.walk(tree):
                bad = None
                if isinstance(node, ast.Name) and node.id.lower() in FORBIDDEN_NAMES:
                    bad = node.id
                elif isinstance(node, ast.Attribute) and node.attr.lower() in FORBIDDEN_NAMES:
                    bad = node.attr
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and node.value.lower() in FORBIDDEN_NAMES and id(node) not in docstring_nodes:
                    bad = node.value
                if bad:
                    offenders.append(f"{f.name}:{node.lineno}: {bad}")
    assert not offenders, f"Forbidden targets referenced in dev code: {offenders}"
