# Change: field-id-provenance-fix

**Change ID:** `field-id-provenance-fix` · **Created:** 2026-11-25 · **Status:** In progress
**Scope:** mini dataset builders only; the frozen 490-field archive is untouched.

## Why
The 2026-11-25 rebuild recorded `archive_alignment` IoU for the three farms of
`3farms-dev.geojson` and found, for all three, that pixels stored under archive
field id *k* belong to the polygon of reference id *k−1* (IoU 1.0 vs 0.0 own).
This is an **ID off-by-one**, not a georeferencing shift. Any farm name, variety,
age or management attribute attached to those pixels is wrong, and any L4
linkage to company records is invalid. The frozen archive and its hash-bound
splits/findings cannot be re-indexed inside this change.

## What Changes
- Resolve each selection-layer farm to **two** ids: `reference_fid` (polygon and
  farm identity from `Farms.geojson`) and `archive_fid` (pixel field id). The
  measured, local rule is `archive_fid = reference_fid + 1`, recorded with the
  IoU evidence, never assumed silently.
- `archive_alignment` compares archive pixels against the **reference** polygon
  and its ±1 neighbours; `best_is_own_fid` must be true after the fix.
- Store attrs + manifest carry both ids, the alias rule, the IoU evidence and
  the caveat that the canonical 490-field mapping stays shifted until a
  separate owner-gated provenance change.
- Downstream mini builders (weather/phenology/validation/weak labels) and the
  probe keep using farm **names**, so they need no id logic change.
- A sample verifier checks the alias on N reference farms before applying it.

## Capabilities
- New: `field-id-provenance` (alias resolution + IoU evidence + fail-closed
  verification for the mini pipeline).
- Modified: `mini-research-dataset` (farm resolution and attrs only).

## Impact
- **Affected:** new mini store under `Data/Processed/mini_research_dataset/`,
  `scripts/build_mini_dataset.py`, `tests/unit/test_mini_farm_source.py`.
- **Frozen / untouched:** canonical `pixels/`, `zonal_stats/`, `cycles_490_v2`,
  `splits_locked_v2.json`, holdout manifest, findings 00–25, checkpoints, platform.
- **Risk:** if the +1 rule is not uniform beyond the 3 sampled farms, the fix is
  scoped to sampled farms only and the run fails closed rather than guessing.
