# Tasks — patch-contamination-absorption

## 1. Machinery (TDD-first)

- [ ] 1.1 Unit tests: patch-tensor contamination injection (exact fraction,
      seed-fixed cycle selection, S4 persistent/ndvi profile); A_patch
      computation from two checkpoints (clean vs contaminated) with
      bootstrap CI over injected cycles.
- [ ] 1.2 `scripts/train_patch.py`: `--contaminate-fraction`,
      `--trimmed-loss {on,off}`, `--curation {on,off}` (curation = two-phase
      run: score → remove top-2% → retrain, removed ids in manifest);
      config-hash covers the contamination configuration (resume safety).
- [ ] 1.3 Part-C patch adapter: three-checkpoint decomposition on patch
      scorer outputs (`diagnostics.decompose_injection` reuse).

## 2. Runs (user-executed, exact commands + estimates handed over)

- [ ] 2.1 Fraction axis first: {0, 5, 10}% × trimmed=on × curation=off,
      seeds {13, 42, 2026} on the PoC subset; agent verifies smoke +
      reviews produced reports before handover.
- [ ] 2.2 Winning fraction: remaining trimmed/curation cells.
- [ ] 2.3 Report per cell: A_patch ± CI, three-checkpoint profiles, clean
      MDM80, vs gru_ae_main α ≈ 0.45 / gru_pred A; dev_log rows complete.

## 3. Decision

- [ ] 3.1 Pre-registered decision rule applied verbatim; winning
      configuration recorded as the full-run default.
- [ ] 3.2 Persian finding (persian-writing skill): absorption at patch
      level vs field level (the aerial-MIM "extended anomalies get
      reconstructed" prediction tested), honest negatives explicit, L2
      rung labelling.
