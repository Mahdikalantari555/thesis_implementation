# Design — data-fix-and-colab-prep

## D1: Canonical-first placement with mirror, not move

v2 artifacts currently live under `thesis_implementation/data/` and inside the
`google_drive` bundle, while canonical `Data/Processed/` does not hold them.
Rather than moving (which would break every script path and the bundle), this
change **mirrors** them into `Data/Processed/` at the same relative paths and
records both locations in the bundle manifest. The originals stay authoritative
for existing scripts; the mirror makes the canonical data root self-contained for
reviewers and for the Colab notebook. No in-place migration anywhere (preserves
the v2 preservation rule).

## D2: Deferred, never synthesized

Any artifact that cannot be produced locally or that depends on an owner upload
is recorded as `Deferred` with a reason and owner action in `tasks.md`. The
benchmark matrix already labels `hierarchical_patch` as `BLOCKED`; this change
keeps that label and adds the reason string. No proxy formula, placeholder, or
`absent → zeros` value is ever written to fill a gap (prime directives 2 & 3).

## D3: Amendments discipline for config changes

`benchmark_config.yaml` is the committed contract and stores
`benchmark_config_sha256` in every run manifest. Any binding change to artifact
paths or new bindings (e.g. once `weather_daily.parquet` arrives) is appended to
the `amendments:` list with date + target + reason + approved_by. Silent edits
are forbidden. Optional artifacts (`stl_features`) keep `required_for: optional`
and are reported as `absent` when missing.

## D4: Colab bundle validation, not rebuild

The `google_drive` bundles (`thesis_data_full_v2.zip` 293 MB, eval bundles 4.4 MB)
are treated as immutable inputs. This change adds a validation step (list
contents + sha256 of the zip and of the extracted key files) and documents the
extraction order for the Colab notebook. Rebuilding or re-uploading bundles is
out of scope and is left to the owner if a new artifact is added.

## D5: WSL as the Rust build/parity host

Windows has no `cargo`/`rustc` and `ssl_anomaly_rs` is not importable there.
WSL (geospatial conda env) is the build and parity-verification host. Before
any WSL work the clone must be fast-forwarded to Windows HEAD (`349e5ba`,
currently 15 commits ahead; merge-base `becb6f7b`, no unique WSL commits). Build
command: `PYO3_PYTHON=/home/.../geospatial/bin/python cargo build` (~30 s),
then `maturin develop`. Parity gate: `tests/parity/test_backends_parity.py`
(2 passed) + `tests/unit/test_backends.py` (4 passed).

## D6: Tolerance alignment decision

The spec states `1e-9` relative tolerance; the tests assert `rtol=1e-6`. Rather
than silently tightening either side, this change records the decision to align
the **spec text to `1e-6`** (matching the enforced test), because the ULP/exactness
traps noted in the PERF audit make a `1e-9` claim unsupportable for the
vectorized hot paths. The test file is left unchanged. If a future PR tightens
the implementation's numerical exactness, the spec can be lowered in a separate
amendment with evidence.

## D7: Inspector surface (recorded, not expanded)

The local thesis research inspector (`tools/inspector/*`, FastAPI + UI) exposes
local data. This change records the data-exposure surface as a risk and defers
any hardening to a separate change; no inspector code is modified here. The
inspector is not part of the Colab path.

## D8: Perf audit follow-ups are out of scope

The PERF 1–10 status and exactness traps recorded in `docs(audit)` are not
reopened by this change. Only the documentation of the WSL build and the
tolerance mismatch is in scope; numerical fixes remain tracked by their own
changes.