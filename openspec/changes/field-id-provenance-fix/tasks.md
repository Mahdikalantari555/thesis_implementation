# Tasks — field-id-provenance-fix

- [x] 1. Add evidence-backed alias map (`reference_fid -> archive_fid`) for the
      three `3farms-dev` farms; fail closed for unmapped farms.
- [x] 2. `resolve_farms` returns both ids; update store attrs + manifest.
- [x] 3. `archive_alignment` uses `reference_fid`; verified IoU 1.0 own / 0.0
      neighbours after applying `archive_fid = reference_fid + 1`.
- [x] 4. Unit tests pass (18 mini tests green, including fail-closed and alias
      verification).
- [x] 5. Rebuilt mini store and downstream artifacts; parity now passes
      11,242/11,242 (the prior 2 "QA mismatch" failures were also the ID shift).
- [x] 6. Update `docs/audits/mini_rebuild_3farms_probe_2026-11-25.md` with a
      supersession note; record in `todo.md` and finding 28.
