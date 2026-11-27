# field-id-provenance Specification

The mini dataset SHALL distinguish a reference polygon id from the archive pixel
field id, and SHALL NOT present a shifted mapping as verified.

## Requirements

### R1 — Two ids per farm
When a farm is resolved from the selection layer, the resolver SHALL return both
`reference_fid` (from `Farms.geojson`, used for the polygon and farm identity)
and `archive_fid` (used to select pixels). The alias rule and its evidence SHALL
be recorded in the store attrs and manifest.

### R2 — Fail-closed alias
The builder SHALL NOT apply an alias rule unless it has IoU evidence for each
farm in the selection layer. If no evidence rule applies, it SHALL fail with a
message naming the farm and the observed IoUs, never fall back to the
unshifted id.

### R3 — Alignment diagnostic
`archive_alignment` SHALL compare archive pixel membership to the **reference**
polygon of `reference_fid` and its ±1 neighbours, and report `best_is_own_fid`
relative to that reference id. A best match at a neighbour SHALL be treated as
a failed verification, not a warning that is ignored.

### R4 — No canonical mutation
This capability SHALL NOT modify canonical `pixels/`, `zonal_stats/`, frozen
splits, holdout manifests or prior findings. All effects are confined to the
mini store and its reports.

### R5 — Honest provenance
Every mini report derived from the store SHALL state that the canonical
490-field archive still carries the off-by-one id, and that only the mini store
has the alias applied. No field attribute SHALL be described as L4-validated.
