"""Task 1.1b — representation monitors and frozen-encoder probes."""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.train.monitoring import (
    dims_above_floor,
    effective_rank,
    gap_reconstruction_probe,
    linear_cka,
    next_ndvi_probe,
    offdiag_cov_energy,
    phase_linear_probe,
    rep_health,
    same_field_same_phase_knn,
    topk_rank_stability,
)


def _rich_embedding(n=200, d=16, seed=0):
    """Embedding with 4 informative dims, 12 noise dims — healthy, not collapsed."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 4))
    w = rng.normal(size=(4, d)) * 0.5
    z = latent @ w + rng.normal(size=(n, d)) * 0.01
    return z


class TestCollapseSignals:
    def test_effective_rank_of_full_rank_embedding(self):
        z = _rich_embedding()
        er = effective_rank(z)
        assert 3.0 < er <= 16.0  # 4 latent dims, uneven weights  # informative latent spans ~4 dims of variance

    def test_effective_rank_of_collapsed_embedding_is_low(self):
        # constant embedding → rank 1 (finding-28 near-collapse analogue)
        z = np.tile(np.arange(8.0), (64, 1))
        assert effective_rank(z) == pytest.approx(1.0)

    def test_effective_rank_handles_few_rows_and_nan(self):
        assert np.isnan(effective_rank(np.ones((1, 4))))
        z = _rich_embedding(50, 8)
        z[3] = np.nan
        assert np.isfinite(effective_rank(z))

    def test_dims_above_floor(self):
        healthy = dims_above_floor(_rich_embedding())
        assert healthy["dims_above_floor_frac"] == pytest.approx(1.0)
        dead = np.hstack([_rich_embedding(64, 8), np.zeros((64, 4))])
        rec = dims_above_floor(dead)
        assert rec["dims_above_floor_frac"] == pytest.approx(8 / 12)
        assert rec["per_dim_std_min"] == pytest.approx(0.0)

    def test_offdiag_cov_energy_bounds(self):
        # independent dims → near-zero off-diagonal energy
        rng = np.random.default_rng(1)
        indep = rng.normal(size=(500, 6))
        assert offdiag_cov_energy(indep) < 0.25  # finite-sample floor for d=6, n=500
        # all rows identical projection → dominated by one direction, off-diag high
        u = rng.normal(size=(500, 1))
        corr = np.hstack([u, u, u]) * rng.normal()
        assert offdiag_cov_energy(corr) > 0.5

    def test_rep_health_combines_signals(self):
        rec = rep_health(_rich_embedding(), std_floor=1e-3)
        assert {"effective_rank", "dims_above_floor_frac", "per_dim_std_min",
                "offdiag_cov_energy"} <= set(rec)


class TestProbes:
    def test_phase_probe_recovers_linear_phase(self):
        # season phase is probed through its circular (sin/cos) representation,
        # which IS linear in the features — scalar DOY is not (documented)
        rng = np.random.default_rng(2)
        n, d = 200, 8
        phase = rng.uniform(0, 1, size=n)
        y = np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], axis=1)
        z = np.tile(y, 4)  # informative dims repeat the circular phase
        z += rng.normal(size=z.shape) * 0.01
        rec = phase_linear_probe(z, y)
        assert rec["probe_phase_r2"] > 0.99

    def test_phase_probe_near_zero_on_noise(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(200, 8))
        phase = rng.uniform(size=200)
        # R² on noise targets is ~0 (not negative-noise artefact)
        assert -0.1 < phase_linear_probe(z, phase)["probe_phase_r2"] < 0.15

    def test_next_ndvi_probe(self):
        rng = np.random.default_rng(4)
        n, d = 200, 6
        base = rng.normal(size=(n, d))
        next_ndvi = base[:, 0] * 0.9 + rng.normal(size=n) * 0.05
        rec = next_ndvi_probe(base, next_ndvi)
        assert rec["probe_next_ndvi_r2"] > 0.9

    def test_knn_retrieval_beats_chance_on_informative_embedding(self):
        rng = np.random.default_rng(5)
        n = 240
        field = rng.integers(0, 3, size=n)              # 3 fields
        phase = rng.integers(0, 4, size=n)              # 4 phase bins
        z = np.zeros((n, 6))
        z[np.arange(n), field % 6] += 3.0               # field encoded strongly
        z[np.arange(n), (phase + 3) % 6] += 2.0         # phase encoded
        rec = same_field_same_phase_knn(z, field, phase)
        assert rec["probe_knn_r_at_1"] > 0.8
        assert rec["probe_knn_chance"] == pytest.approx(1 / 12, abs=0.05)  # max-group sampling noise
        assert rec["probe_knn_r_at_1"] > rec["probe_knn_chance"]

    def test_knn_on_noise_sits_near_chance(self):
        rng = np.random.default_rng(6)
        n = 120
        field = rng.integers(0, 3, size=n)
        phase = rng.integers(0, 4, size=n)
        rec = same_field_same_phase_knn(rng.normal(size=(n, 6)), field, phase)
        assert rec["probe_knn_r_at_1"] < rec["probe_knn_chance"] + 0.15

    def test_gap_reconstruction_beats_mean_predictor(self):
        rng = np.random.default_rng(7)
        n, d = 200, 8
        z = rng.normal(size=(n, d))
        gap = z[:, :3] @ rng.normal(size=(3, 4))  # linearly decodable
        rec = gap_reconstruction_probe(z, gap)
        assert rec["probe_gap_mse_ratio"] < 0.5
        assert rec["probe_gap_r2"] > 0.5

    def test_probe_budget_caps_samples(self):
        z = np.random.default_rng(8).normal(size=(5000, 4))
        phase = np.random.default_rng(9).uniform(size=5000)
        rec = phase_linear_probe(z, phase)
        assert np.isfinite(rec["probe_phase_r2"])  # capped deterministically, no blow-up


class TestStability:
    def test_cka_identical_is_one(self):
        z = _rich_embedding(100, 8)
        assert linear_cka(z, z) == pytest.approx(1.0, abs=1e-6)

    def test_cka_rotation_invariant(self):
        rng = np.random.default_rng(10)
        z = _rich_embedding(100, 8)
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        assert linear_cka(z, z @ q) == pytest.approx(1.0, abs=1e-6)

    def test_cka_unrelated_is_low(self):
        a = _rich_embedding(100, 8, seed=1)
        b = _rich_embedding(100, 8, seed=2)
        assert linear_cka(a, b) < 0.6

    def test_topk_stability_identical(self):
        s = np.linspace(0, 1, 50)
        rec = topk_rank_stability(s, s, k=10)
        assert rec["topk_spearman"] == pytest.approx(1.0)
        assert rec["topk_jaccard"] == 1.0

    def test_topk_stability_shuffled(self):
        rng = np.random.default_rng(11)
        s = rng.normal(size=200)
        rec = topk_rank_stability(s, rng.permutation(s), k=10)
        assert rec["topk_spearman"] < 0.5
        assert rec["topk_jaccard"] < 0.5
