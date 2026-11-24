# Inspector full-operation roadmap

Status: **2026-11-24** · Scope: `tools/inspector/` + artifacts it reads.
This is a remediation handover, not a claim that the Inspector is finished.
It records what was fixed, what still blocks a clean run, and the exact
commands used to verify each stage.

## 1. What "fully well" means

The Inspector is operating at full capacity when all of the following hold:

1. `python tools/inspector/app.py` (and `--report-only`) completes the startup
   scan without an unhandled exception.
2. `integration.status` is `PASS`, or `WARNING` **only** for findings that have
   been explicitly reviewed and accepted in writing; `critical_failures` is
   always empty.
3. Every JSON artifact under `reports/`, `checkpoints/`, and
   `platform/src/data/` parses as **strict** JSON (no bare `NaN`, `Infinity`,
   or `-Infinity`), because readers such as `ijson` and `JSON.parse` reject
   them even though Python's `json.loads` accepts them.
4. `pytest tests/inspector` and the two platform contract test modules are green.
5. The runtime UI gate (tile server + screenshot capture) reports zero console
   errors and zero page errors.

## 2. Baseline after the 2026-11-24 fix

The crash in the original report was two defects, not one:

- **Producer:** `scripts/freeze_main_model.py` wrote sidecars with plain
  `json.dumps`, emitting bare `NaN` for "no median delay" / censored cells.
- **Consumer:** `ijson.common.JSONError` derives from `Exception`, not
  `ValueError`. The `except (OSError, ValueError, …)` clauses in
  `tools/inspector/core/integration.py` did not catch it, so one malformed
  sidecar aborted the whole scan.

Applied fixes:

| File | Change |
|---|---|
| `tools/inspector/core/integration.py` | Added `STREAM_ERRORS` including `ijson.common.JSONError`, used at all six JSON/stream read sites (`_read_farms`, `_axis_and_version`, `_checkpoint_alignment`, `_read_test_masks`, `_validate_series_values`). Malformed input now yields a finding, never a scan abort. |
| `scripts/freeze_main_model.py` | Added `_jsonable()` (non-finite → `null`) and `allow_nan=False` on every sidecar/report dump and summary print. This applies the project's already-adopted audit-14 policy (`docs/audits/code_audit_2026-11-21.md` item 14) to the freeze path. |
| `checkpoints/gru_ae_main.json`, `checkpoints/gru_pred_main.json` | Sanitized 6 and 4 bare `NaN` values to `null` using a JSON-aware round-trip (string snapshots preserved). Both now pass strict parsing and `ijson`. **These files are gitignored** (`.gitignore:33`) — see item R5. |
| `tests/inspector/test_integration.py`, `tests/unit/test_freeze_json_policy.py` | Regression tests: `NaN` in a sidecar or series is reported, not fatal; `_jsonable` maps non-finite floats at every depth. |

Observed scan after the fix (`--report-only`):

- `integration.status = WARNING`, `critical_failures = []`.
- Remaining warnings: `farm_crs`, and `checkpoint_sidecars` for 4 registered
  checkpoints with no sidecar file.
- All contract and series checks PASS: 490 farm IDs, 538 ISO dates, split RLE,
  channel ranges, provenance, ladder rung.

Test baseline after the fix:

- `tests/unit/test_platform_data_contract.py tests/unit/test_platform_bilingual.py` → **51 passed**.
- `tests/inspector/test_integration.py tests/unit/test_freeze_json_policy.py` → **10 passed**.
- `pytest tests/inspector` → `3 failed, 47 passed, 1 skipped` (see R3).

## 3. Remaining work, prioritized

### R0 — Decide whether the remaining warnings are acceptable (owner decision)

A `WARNING` is not a failure, but the Inspector cannot silently promote it to
`PASS`. Either fix the underlying artifact or record an explicit acceptance
with a rationale tied to the validation ladder. Do not delete a check just to
turn the status green.

### R1 — Sanitize the remaining non-strict JSON reports (P0)

**Finding:** 44 JSON files under `reports/` still contain bare `NaN`
(e.g. `reports/data_scarcity.json`, `reports/rq2_pretexts.json`,
`reports/rq5_early_detection.json`, `reports/rq6_xai.json`,
`reports/rq7_rq8.json`, `reports/dev/trial_gru_pred_20261114/metrics_gru_pred_sev_recon_only.json`).
The Inspector scans them as artifacts but does not stream-parse them, so they
do not crash startup; they are still invalid strict JSON and violate AGENTS.md
§6 ("never emit placeholder/NaN-filled numbers") and §7.

**What to do:**
1. Enumerate: `D:\Dev\Miniforge3\envs\geo\python.exe -X utf8 -c "import pathlib; print('\n'.join(str(p) for p in pathlib.Path('reports').rglob('*.json') if 'NaN' in p.read_text(encoding='utf-8', errors='ignore')))"`.
2. Apply the same JSON-aware round-trip used for the sidecars: load with
   `parse_constant=lambda s: None`, dump with `indent=2, allow_nan=False`.
   This changes only numeric `NaN` → `null`; strings and measured values stay
   byte-identical.
3. Audit every remaining writer of report JSON and switch it to
   `allow_nan=False` plus a `_jsonable` sanitizer (the pattern in
   `scripts/eval_dev.py` and `scripts/freeze_main_model.py`).
4. Add a CI/verification step that rejects any committed JSON containing a
   bare `NaN`.

**Caveat:** this rewrites committed scientific artifacts. Coordinate with any
hash-pinned report records (`data_source`, n, date range, git SHA, split
hashes) before re-exporting; the numeric evidence is unchanged, but the file
bytes and therefore any whole-file digest will change.

### R2 — Resolve the 4 registered-but-absent model checkpoints (P0)

**Finding:** `scripts/platform_channels.py:105-114` registers
`masked_transformer_main`, `sts_net_490`, `tiny_cnn_mamba_490`, and
`time2agri_490` as model-backed scores, but `checkpoints/` has no sidecar for
them. `integration.py` therefore reports "Sidecar metadata missing/unavailable
for 4 registered checkpoint(s); weights were not loaded".

**What to do:** pick one, explicitly:
- **Produce** real frozen checkpoints + sidecars for those four (training is a
  Colab job per AGENTS.md §6 — validate locally first, hand over the exact
  notebook cell, and return a complete Drive bundle per §6.2); or
- **Demote** the registry slots to `checkpoint=None` until the artifacts exist.

**Do not** fabricate sidecars or present a proxy formula as checkpoint
inference. Prime directive §2/§3: a registry slot without an artifact must be
labelled `absent`, not `model`.

### R3 — Fix the 3 pre-existing inspector test failures (P0, test-only)

`pytest tests/inspector` currently fails in three fixtures that embed Windows
temporary paths with raw backslashes into TOML inline tables. `tomllib` reads
`\U`, `\A`, … as invalid escape sequences:

| Test | Location | Error |
|---|---|---|
| `test_cli.py::test_refresh_works_from_other_cwd_and_writes_inspector_output_only` | `tests/inspector/test_cli.py:24-25` | `configuration error: Invalid hex value (at line 4, column 44)` |
| `test_cli.py::test_report_only_writes_report_and_does_not_start_http_server` | `tests/inspector/test_cli.py:43` | `configuration error: Invalid hex value (at line 2, column 44)` |
| `test_core.py::ConfigManifestTests::test_external_roots_are_supported` | `tests/inspector/test_core.py:35` | `tomllib.TOMLDecodeError: Invalid hex value (at line 3, column 44)` |

**What to do:** write those paths with a TOML single-quoted literal (no escape
processing) or use `str(source).replace('\\', '/')` in the fixture. These are
test-fixture bugs, not Inspector runtime bugs; they were present before the
2026-11-24 fix and are unrelated to it. Fix them so the whole inspector suite
is green and the regression tests cannot be drowned out by noise.

### R4 — Resolve or formally accept the `farm_crs` warning (P1)

`farms_khuzestan.json` is written as a bare array by
`scripts/export_platform_data.py:819-820`; it carries no CRS field, so
`integration.py` can only state that coordinates are interpreted as WGS84
latitude/longitude. Either add explicit CRS metadata to the export (requires
coordinated platform loader changes — `platform/src/data/data.ts` imports the
array directly) or record a signed acceptance of the WGS84 assumption and
keep the warning as informational.

### R5 — Account for gitignored checkpoint sidecars (P1)

`.gitignore:33` excludes `checkpoints/`, so the local `NaN → null` sanitization
of the two sidecars is **not** in git and will not appear on another machine or
after a clean clone. Anyone reproducing the Inspector must regenerate the
sidecars with the fixed `freeze_main_model.py`, or sanitize the copies that
arrive in Drive bundles (AGENTS.md §6.2). Add this to the bundle checklist:
a sidecar containing bare `NaN` is an incomplete bundle.

### R6 — Confirm optional readers and the UI runtime gate (P1)

- `ijson` is required for the series/contract checks; without it the Inspector
  reports `NOT_EVALUATED` rather than `PASS`. Confirm it is installed in the
  `geo` env on every machine that runs the Inspector.
- `numpy`, `pyarrow`, `zarr`, `rasterio` control per-format coverage; missing
  packages produce `UNSUPPORTED` states, not failures, but reduce coverage.
- After artifact changes, run the runtime gate: start
  `scripts/serve_tiles.py --port 8765`, then
  `node scripts/capture_platform_screenshots.mjs --out reports/screenshots/audit --locale en`
  (+ `--no-tile-server` for degradation paths). Require zero console/page
  errors.

## 4. Verification recipe

```bash
# from thesis_implementation, geo env
D:\Dev\Miniforge3\envs\geo\python.exe -X utf8 -m pytest ^
  tests/unit/test_platform_data_contract.py tests/unit/test_platform_bilingual.py -q --no-cov
D:\Dev\Miniforge3\envs\geo\python.exe -X utf8 -m pytest ^
  tests/inspector/test_integration.py tests/unit/test_freeze_json_policy.py -q --no-cov
D:\Dev\Miniforge3\envs\geo\python.exe -X utf8 -m pytest tests/inspector -q --no-cov
D:\Dev\Miniforge3\envs\geo\python.exe -X utf8 tools/inspector/app.py --report-only
```

Accept the scan only if `critical_failures` is empty and every listed warning
is either fixed or explicitly accepted. The full report is written under
`reports/inspector/` (gitignored, safe to delete to reset the cache).

## 5. Definition of done

- [ ] No bare `NaN` in any JSON under `reports/`, `checkpoints/`, or `platform/src/data/`.
- [ ] `pytest tests/inspector` is fully green (R3).
- [ ] Every registered model score either has a real sidecar or is registered as `absent` (R2).
- [ ] `farm_crs` is fixed or formally accepted in writing (R4).
- [ ] Sidecar regeneration / bundle sanitization is documented for other machines (R5).
- [ ] Optional readers and the UI runtime gate verified (R6).
- [ ] `integration.status` is `PASS`, or `WARNING` with a written acceptance for each remaining item.
