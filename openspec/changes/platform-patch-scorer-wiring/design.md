# Design: platform-patch-scorer-wiring

## Context

Finding 23 froze the trial result (MDM80 = S1, both patch arms, eval hash
6186edf8); the G4 localization probe and the full-archive severity run
decide which arm (if any) gets wired. This change is written before those
verdicts, so every arm-specific decision is a rule, not a constant.

## Goals / Non-Goals

**Goals**
- One new platform score slot backed by the frozen patch checkpoint, with
  honest trial-stage provenance end to end.
- Reuse the EXACT trial scoring path (`PatchDeepScorer.predict_pixels` →
  top-20% in-field aggregation → empirical-p vs calibration paths) — the
  platform shows the same statistic the gates measured, not a re-derivation.
- Keep every existing payload field byte-identical.

**Non-Goals**
- Replacing `gru_ae_main` as the stress-flag source (owner-gated task).
- Patch XAI, ensembles, or any `patch-*` openspec change's pending work.
- Pixel-level UI (maps of per-pixel z²) — the platform stays field/obs.

## Decisions

### D1 — Cycle-bin → observation-date attribution

The patch statistic exists per (field, cycle, bin). The platform axis is
every acquisition date (contract v4/v5). Rule: an observation date d of
field f is attributed to the bin b such that `cycle_start(f) + b·15 ≤ d <
cycle_start(f) + (b+1)·15` of the admissible cycle c containing d (the same
half-open grid `extract_patches.py`/`bin_of_day` uses, same epoch
`splits_locked.base_date`); the slot value at d is the score of (f, c, b).
Dates outside every admissible cycle → `null`/`absent`. Multi-acquisition
dates inside one bin share the bin's score (documented; the alternative —
per-date sub-binning — would create a statistic the gates never measured).

### D2 — Which checkpoint the slot backs

`patch_<winner>_main`, produced by the existing freeze protocol extended to
patch checkpoints (frozen seed, sidecar with τ/k/σ/norm/store hashes). The
freeze step runs AFTER the full-archive severity evaluation and the G4
verdict; the slot name is fixed there. If both arms remain tied after G4,
the simplicity rule (the established tie-break) selects `patch_pixelwise`
and records the tie.

### D3 — Contract version

Additive slot + additive per-observation arrays + one new per-score `stage`
marker. Existing consumers tolerate an extra MODELS entry (registry-driven);
old payload FILES (v5) load with the slot absent (loader treats a missing
array as all-`absent` — the pre-existing degradation path). Therefore
`CONTRACT_VERSION` stays 5 unless apply-time work uncovers a breaking
semantic (then 6 with a migration note). The 51-test contract+bilingual
gate is updated for the new slot and remains the gate.

### D4 — Compute and degradation

Exporter cost ≈ +20–30 min (799 cycles × causal LOO on 48×48×8 tensors on
CPU, checkpoint inference only — no training). If `patches.zarr`, the
frozen checkpoint, norm or σ artifact is missing at export time, the slot
exports all-`absent` and the export SUCCEEDS with a logged warning line
(the platform degradation contract: absence is a first-class state); the
artifact-mismatch case (hash binding fails) is the exception and ABORTS.

### D5 — Provenance surface

Per observation: score + provenance RLE entry (`model`) + the existing
`split` column. Per field: nothing new (ladderRung already exists). Per
payload: the score registry meta gains `stage: "trial"` + the provenance
block gains the patch artifact hashes. UI: badge on the slot (EN/FA).

## Risks / Trade-offs

- **Payload size**: one more fp array over 263,620 obs — bounded (~1 MB
  after RLE + fp16-ish encoding); acceptable.
- **Double-statistic confusion**: two "anomaly" scores (gru_ae + patch) on
  screen could read as competing truths — mitigated by the trial badge and
  the score_methods doc.
- **CPU export time doubles** — accepted; the exporter is a batch job.
- **Owner-gated flag switch** recorded as an explicit unticked task, so the
  change cannot silently change what "suspected_stress" means on screen.

## Migration Plan

1. Freeze winner → 2. apply tasks (TDD order) → 3. full export + gates →
4. owner reviews the platform → 5. owner-gated flag-source task. Rollback:
revert the registry slot; the payload without the slot is the pre-change
state (additive).

## Open Questions

- OQ1 (owner): after the platform is up — switch `stressFlagModel` to the
  patch checkpoint, or keep gru_ae flags and show the patch score beside?
  Default until answered: keep gru_ae flags.
