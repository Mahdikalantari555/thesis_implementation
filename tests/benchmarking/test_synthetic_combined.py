"""Task 4.9 — the designed combined scenarios.

The contract pre-registers two multi-family scenarios; until 4.9 they existed only as a
note. These tests pin what a combined cell must now be: one event carrying every
constituent family at once, per-family ground truth recorded, labels = the union of the
component windows, and the headline macro still over the pre-registered 5 × 3 grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.benchmarking.synthetic import (
    FAMILY_CHANNELS,
    FAMILY_MECHANISMS,
    apply_panel,
)


class _Clean:
    """Minimal clean tensor set (mask-aware) standing in for the v2 archive."""

    def __init__(self, n=80, t=36, seed=5):
        rng = np.random.default_rng(seed)
        self.channel_names = ("ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst")
        self.m = rng.random((n, t, len(self.channel_names))) > 0.15
        self.x = (rng.normal(size=(n, t, len(self.channel_names))) * 0.1 * self.m).astype(np.float32)


@pytest.fixture(scope="module")
def panel(cfg):
    clean = _Clean()
    return apply_panel(clean, cfg, seed=12345, pool="dev_cycles_only", role="evaluation"), clean


def _combined_cells(panel):
    return [c for c in panel.cells if c.is_combined]


def test_every_declared_scenario_becomes_one_cell_per_severity(panel):
    p, _ = panel
    combo = _combined_cells(p)
    assert combo, "no combined-scenario cell was generated; task 4.9 is not wired"
    names = {c.cell for c in combo}
    assert {"combined:persistent_spectral_multivariate__weak",
            "combined:persistent_spectral_multivariate__moderate",
            "combined:persistent_spectral_multivariate__strong",
            "combined:temporal_phenological__weak",
            "combined:temporal_phenological__moderate",
            "combined:temporal_phenological__strong"} <= names
    for c in combo:
        assert c.scenario in ("persistent_spectral_multivariate", "temporal_phenological")
        assert set(c.scenario_families) <= set(FAMILY_MECHANISMS)
        assert len(c.components) == len(c.scenario_families) * c.n_events, (
            "every event must carry one component per constituent family")


def test_component_metadata_covers_every_family_and_window(panel):
    p, _ = panel
    for c in _combined_cells(p):
        fams = {comp["family"] for comp in c.components}
        assert fams == set(c.scenario_families)
        for comp in c.components:
            assert 0 <= comp["onset_bin"] < 36 and 1 <= comp["length_bins"] <= 36
            assert set(comp["channels"]) == set(FAMILY_CHANNELS[comp["family"]])
        # the union of the component windows IS the event's label set. `cycle_position`
        # follows the cell's own cycle order (`c.cycle_index[i]`), which is NOT sorted.
        rng_pos = {}
        for comp in c.components:
            k = comp["cycle_position"]
            rng_pos.setdefault(k, np.zeros(36, dtype=bool))
            rng_pos[k][comp["onset_bin"]: comp["onset_bin"] + comp["length_bins"]] = True
        assert len(c.cycle_index) == c.n_events
        for i, gi in enumerate(c.cycle_index):
            assert np.array_equal(c.labels[gi], rng_pos[i]), (
                f"{c.cell}: labels are not the union of the recorded component windows")


def test_combined_injection_lands_inside_the_recorded_ground_truth(panel):
    p, clean = panel
    moved = (np.abs(p.x - np.asarray(clean.x, dtype=np.float32)).sum(axis=-1) > 0)
    for c in _combined_cells(p):
        # Cycles carry events from several cells at once, so the correct attribution
        # question is panel-level: no injection may land outside the recorded ground
        # truth of the panel (`labels_bin` is the union over every cell, combined included).
        stray = moved[c.cycle_index] & ~p.labels_bin[c.cycle_index]
        assert not stray.any(), (
            f"{c.cell}: {int(stray.sum())} bins changed outside the panel's recorded labels")


def test_single_family_grid_is_unchanged_and_macro_stays_5x3(panel):
    p, _ = panel
    single = [c for c in p.cells if not c.is_combined]
    assert len(single) == 15, "the pre-registered headline grid is 5 families × 3 severities"
    assert {c.family for c in single} == set(FAMILY_MECHANISMS)
    assert all(c.scenario is None and not c.components for c in single)


def test_combined_cells_pass_the_same_trivial_separability_audit(panel):
    p, _ = panel
    for c in _combined_cells(p):
        assert c.audit, f"{c.cell}: combined cells must carry the realism audit"
        assert "frac_delta_above_threshold" in c.audit and "trivial_rule_auprc" in c.audit
    assert all(c.cell in p.usable_cells for c in _combined_cells(p)), (
        "a combined cell excluded by the audit must appear in excluded_cells, not vanish")


def test_panel_dict_records_scenario_provenance(panel):
    p, _ = panel
    d = p.to_dict()
    combo_rows = [row for row in d["cells"] if row["cell"].startswith("combined:")]
    assert combo_rows and all("scenario" in row and "scenario_families" in row for row in combo_rows)
    assert d["audit_summary"]["combined_scenarios"]["temporal_phenological"]["families"] == \
        ["temporal", "phenological"]


def test_combined_generation_is_deterministic(cfg):
    a = apply_panel(_Clean(), cfg, seed=777, pool="dev_cycles_only")
    b = apply_panel(_Clean(), cfg, seed=777, pool="dev_cycles_only")
    ca = [(c.cell, c.cycle_index.tolist(), [cp["onset_bin"] for cp in c.components])
          for c in a.cells if c.is_combined]
    cb = [(c.cell, c.cycle_index.tolist(), [cp["onset_bin"] for cp in c.components])
          for c in b.cells if c.is_combined]
    assert ca == cb, "the same seed must redraw the same combined events"
    assert np.array_equal(a.x, b.x) and np.array_equal(a.labels_bin, b.labels_bin)


def test_unknown_combined_family_is_an_error_not_a_drop(cfg):
    import copy

    raw = copy.deepcopy(cfg.raw)
    raw["synthetic"]["combined_scenarios"] = [{"name": "bad", "families": ["not_a_family"]}]
    from ssl_anomaly.benchmarking.config import BenchmarkConfig

    cfg2 = object.__new__(BenchmarkConfig)
    cfg2.raw = raw
    with pytest.raises(KeyError, match="not_a_family"):
        apply_panel(_Clean(), cfg2, seed=1, pool="dev_cycles_only")
