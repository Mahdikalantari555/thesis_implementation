# Design — consistent-spectral-injection

## Context

External audit (2026-11-20, verified against the local tree the same day)
identified that both severity-sweep injection arms produce inputs that are
internally inconsistent between the stored NDVI channel and the stored raw
bands:

- ndvi arm: channel 0 shifts, Red/NIR untouched → `ndvi_from(red, nir)` no
  longer equals the stored NDVI.
- savi arm: Red/NIR solved, NDVI channel stale → same violation in the other
  direction.

The model consumes both planes, so gate success could partly reflect
inconsistency detection. **Counter-evidence recorded** (audit hypothesis is
plausible but unproven): the field-level arms received identical inconsistent
injections under findings 18/21/22 and still censored >S6, so inconsistency
alone does not produce S1 detection — architecture matters too. The fix is
still required either way: it makes the protocol's semantics explicit and
separates the two explanations empirically (basis becomes a reported axis).

## Decisions

### D1 — One band-level solve serves both arms

- `index="ndvi"` under `bands_consistent`: new sum-preserving NDVI solve on
  (red, nir) — keep `nir + red` constant, move mass from nir to red until the
  realized NDVI drop equals the requested δ (clipped at physical bounds;
  floored values counted). Because the sum is preserved, SAVI shifts as a
  deterministic side effect; its realized drop is recorded.
- `index="savi"` under `bands_consistent`: the existing `_solve_savi_drop`
  solver (unchanged); the stored NDVI channel is then recomputed from the new
  bands (its realized drop recorded).
- Both arms therefore satisfy ONE invariant: **stored NDVI ≡
  `ndvi_from(red, nir)` after injection** — the only derived stored channel in
  the 8-channel layout (`CH_NDVI=0`, raw Blue/Green/Red/NIR/SWIR1/SWIR2/LST).

### D2 — Gate semantics unchanged on the requested index

Severity δ keeps its meaning ("requested drop of the named index at the bin"),
with realized per-bin deltas already recorded today. The side-effect drop of
the other index is metadata, never a gate input. This keeps S1–S6 comparable
across bases while making the physical interpretation of each row explicit.

### D3 — Field-level applier intentionally NOT changed here

`monitor/perturbations.py` (field-level `inject_index_drop`) keeps its current
behavior. Findings 18/22 are historical field-level records; the patch protocol
is the thesis primary. The cross-arm comparability bridge after this change is
the `index_only` sensitivity row (patch) vs the historical field-level rows.
Changing the field-level applier would silently invalidate those historical
comparisons for zero thesis benefit — recorded as out of scope.

### D4 — Eval-only; no retraining triggered

Finding-24 run manifests show `contaminate_fraction = 0.0` and
`trim_after_epoch = 10` with `epochs = 10` (trimming never activated): the
existing checkpoints never saw injections during training. σ tables are fitted
on clean dev residuals and never see injections
(`test_patch_scores.py` pins this). So this change is eval-side only; the
retrained v3 checkpoints (score-semantics-realignment, on the v2 store) are the
first to be evaluated primarily under `bands_consistent`.

### D5 — Protocol identity

`severity_sweep_patch_v2` is emitted when basis = `bands_consistent`; v1 id is
kept verbatim for `index_only`. The eval-subset hash machinery is unchanged
(injection-set identity already hashes shape/footprint/levels; the basis joins
the hash payload via `level_id`/meta so two bases cannot share an identity).

## Risks / trade-offs

- **Risk:** consistent injections may raise MDM80 (the S1-at-1.000 result could
  weaken). *Accepted by design* — that is the honest number; if MDM80 moves,
  the delta vs `index_only` rows becomes a reported finding, not a defect.
- **Risk:** NDVI solve at physical bounds (very dark red/nir pixels) can floor;
  floored bins under-realize δ. *Mitigation:* floored counts already recorded;
  the realized-Δ series is reported per cell as today.
- **Note:** the localization probe (footprint 0.3) and the white-null
  diagnostic inherit the basis unchanged; no separate handling.

## Migration

No data migration. Old reports keep v1 ids; new primary runs emit v2. The
sensitivity-row command is documented in the finding template so the v3 finding
carries both bases side by side.
