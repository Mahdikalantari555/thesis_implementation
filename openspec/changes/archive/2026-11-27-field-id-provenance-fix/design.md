# Design — field-id-provenance-fix

## Decision
Use an **additive alias** in the mini builder, not a re-index of the canonical
archive. Reason: findings 18–25 and the v2 splits/holdout are hash-bound to the
canonical ids; rewriting them would silently invalidate prior results (prime
directive: honest provenance, scope discipline).

## Mapping evidence (2026-11-25, 3 farms)
| farm (reference fid) | archive pixels best-match fid | IoU best / own |
|---|---|---|
| L02-02 (300) | 299 (R10-31) | 1.0000 / 0.0000 |
| R12-08 (245) | 244 (R12-06) | 1.0000 / 0.0000 |
| R15-18 (17)  | 16 (R15-20)  | 1.0000 / 0.0000 |

Therefore `archive_fid = reference_fid + 1` for these farms. The rule is stored
as evidence, not inferred at run time; a farm with no evidence fails closed.

## Implementation
1. `resolve_farms` returns `reference_fid` (reference `OGC_FID`) and
   `archive_fid = reference_fid + 1` (from an evidence map keyed by farm name).
2. `build_farm` selects pixels with `archive_fid` and passes `reference_fid` to
   `archive_alignment`, which now reports `best_reference_fid == reference_fid`.
3. Store attrs/manifest record both ids, `id_alias_rule`, `alias_evidence`
   (IoU pairs) and the canonical-shift caveat.
4. Unit tests: resolution returns both ids; alignment passes for alias-applied
   mask; missing evidence raises; canonical files untouched.

## Non-goals
No change to canonical pixels/zonal stats, the 490-field splits, the platform,
or the pre-registered gates. No claim above mini-screening (L1/L2 at most).
