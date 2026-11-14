# Tasks — reconstruction-invariance-diagnostic

## 1. Diagnostics module (TDD-first)

- [ ] 1.1 `src/ssl_anomaly/monitor/diagnostics.py`: signal-path decomposition
      — per-injection stage quantities (onset residual |r| in σ units, offset
      subtraction, normalized contribution vs cell q95, trailing path vs τ,
      SNR₁/SNR₂), absorption ratio A at post-onset bins against the
      counterfactual natural tensor, and stage-wise AUC-PR (raw onset
      residual vs final path). Unit tests with synthetic residual paths:
      onset context clean ⇒ subtraction ≈ 0; injected post-onset context ⇒
      absorption measurable; waterfall ordering preserved.
- [ ] 1.2 `scripts/diagnose_reconstruction.py`: runs the decomposition for
      `gru_ae_main` + `b1_phasez_causal` (masked_transformer opt-in) on the
      dev evaluation subset with the EXISTING severity-table injections
      (deterministic, same subset hash), writes
      `reports/dev/<run_id>/diagnostic_<model>.json` (medians + IQR per cell,
      decision-matrix attribution, provenance: seeds, git sha, splits hash,
      subset hash, severity table hash, wall time) + progress.jsonl. Smoke
      mode. No threshold/model/platform writes.

## 2. Run + analysis

- [ ] 2.1 Run the diagnostic on dev (agent-run; read-only eval, ~minutes):
      gru_ae_main + b1; verify stage cross-checks (stage-4 detection equals
      the sweep's detection at the same cells).
- [ ] 2.2 Decision matrix: attribute the dominant loss per (family, index)
      for each arm; quantify absorption on the persistent family.

## 3. Finding + gated follow-ups

- [ ] 3.1 Finding 20 (`docs/findings/20_reconstruction_invariance.md`,
      Persian, persian-writing skill): hypothesis verdict with measured
      evidence, waterfall table, decision matrix, explicit scope limit
      (single-index drop family only); + 00_index row.
- [ ] 3.2 `todo.md`: record gated follow-ups as open questions — path B
      (multivariate/relational + phenology-shape injection families; design
      informed by this diagnostic) and path C (forecasting/predictive-coding
      objective arm) — no implementation in this change.

## 4. Gates

- [ ] 4.1 Unit tests green; existing severity/bank tests untouched and green.
- [ ] 4.2 `openspec validate reconstruction-invariance-diagnostic`; no
      platform impact (diagnostic writes only `reports/`).
