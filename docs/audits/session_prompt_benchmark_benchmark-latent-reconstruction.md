# Session prompt — apply `benchmark-latent-reconstruction-prediction-architectures`

Paste everything below the line into a **new** Arena session on this repository.

---

You are continuing work on `Mahdikalantari555/thesis_implementation`. This session's job is to **apply** the OpenSpec change `benchmark-latent-reconstruction-prediction-architectures` — currently `PROPOSED, 0/37` — to the standard this repository already holds for the companion change.

## 1. Read these first, in this order

Do not write code before you have read all of them. They are short and each one
constrains something below.

1. `AGENTS.md` — binding. Particularly §1 prime directives, §6/§6.1 (the local
   gate; the agent never launches a long job), §7 (the ML review checklist).
2. `openspec/changes/benchmark-latent-reconstruction-prediction-architectures/`
   — `proposal.md`, `design.md`, `tasks.md`, and the four `specs/*/spec.md`.
   `design.md` is declared the authority where the owner's draft conflicts with
   the repo. `tasks.md` has 37 checkboxes across 16 phases; Phase 0 → 15.
3. `openspec/changes/INDEX.md` and `todo.md` — where this change sits relative
   to everything else.
4. `docs/findings/00_index.md`, then findings `18`, `20`, `21`, `23`, `24`, `25`
   — the negative results this benchmark exists to answer. Read the *numbers*,
   not just the summaries: `MDM80` censored `>S6` for every SSL arm and both
   patch arms, best detection anywhere `0.456`, `B1 ≥ SSL` everywhere.

## 2. What was already built — reuse it, do not duplicate it

The companion change (`regional-ssl-representation-evaluation`) is **PR #17,
which is still OPEN — not merged into `main`.** Its evaluation package exists
only on the branch `arena/539c13a7-thesis-implementation`. Resolve this before
you start:

- If PR #17 has merged by the time you read this, the package is on `main` and
  you simply have it.
- If it is still open, **branch your work from
  `arena/539c13a7-thesis-implementation`, not from `main`** — otherwise none of
  the infrastructure in this section exists in your tree and you will
  re-implement it (or, worse, import from a path that isn't there).

Check with `gh pr view 17 --json state`. Do not assume.

That infrastructure is **not** for this change to re-implement:

- `src/ssl_anomaly/evaluation/` — `binding.py` (artifact hashing,
  `bind_artifacts`, `load_preregistration`), `data.py` (`CycleData`,
  `load_cycle_table`, `synthetic_evaluation_split`, `TestSplitAccessError`),
  `metrics.py` (clustered bootstrap, Holm correction), `candidates.py`
  (`CycleEncoder`, `MaskedReconHead`, `IncumbentAE`, `BarlowTwinsModel`,
  `TrajectoryPredictor`, `HybridModel`, `CapacityReport`, `TrainConfig`,
  `train_candidate`), `gates.py`, `runner.py`
  (`RunPaths`/`RunManifest`/`build_manifest`/`write_run_outputs`,
  `make_smoke_fixture`, `SmokeExpectation`, `environment_block`), `provenance.py`.
- `scripts/evaluation/_common.py` — `add_common_args`, `RunContext`,
  `resolve_run`, `require_real_data`, `finish_smoke`.
- Test conventions in `tests/evaluation/` — read `test_candidates.py` and
  `test_runner.py` before writing new ones.

Two things follow from this. First, `A1 GRU-AE` in this benchmark is the same
*pattern* as `IncumbentAE` in `candidates.py` (a GRU encoder with a causal
per-bin masked-reconstruction head). It is not literally the frozen
`gru_ae_main` — that lives in `ssl_anomaly.monitor` and is the physics-aware
incumbent, which this change reuses as a **baseline row**, never redefines
(`design.md` §Context is explicit on this). So: adapt `IncumbentAE` behind the
new common interface for A1, and wire the frozen `gru_ae_main` in as the
incumbent baseline. Two different jobs, two different mechanisms. Second, the
smoke-output and smoke-expectation machinery already exists — a new script that
invents its own is a defect.

Also note `configs/benchmarks/matrix.yaml` does **not** exist yet — creating it
is task 0.4, the first thing that must land.

**Do not modify** `src/ssl_anomaly/evaluation/gates.py`,
`configs/evaluation/pre_registration.yaml`, or any frozen finding. If this
benchmark needs a gate or a threshold, it gets its own, in its own
pre-registration, and says so.

## 3. Environment

- Tests must run as:
  ```
  PYTHONPATH=src:tests python -m pytest <path> -q -o addopts="" -p no:cacheprovider
  ```
  The repo's `addopts` interfere with a plain invocation.
- `torch` is CPU-only here. `xarray`, `zarr<3` and `torch` may need installing
  into a fresh venv; a venv outside the repo is not persisted between sessions,
  so expect to rebuild it.
- Mamba selective-scan kernels need CUDA. On this machine they are **CPU
  fallback, unit tests only** — which is exactly what `design.md` already says
  (§Backbone notes). Do not let Mamba correctness rest on a CPU approximation
  you have not bounded.
- `hydra-core`/`omegaconf` are already dependencies. Use them for the config
  schema rather than inventing one; the design says "matching repo conventions".

## 4. Non-negotiables — these are the whole point of the change

`design.md` §Research integrity lists twelve binding constraints. Four of them
will decide whether this work is quotable at all, and each has a specific way it
gets violated by accident:

1. **The locked test split is never read.** `splits_locked_v2.json` sha
   `30ecbe10…`; test = 773 cycles. Every protocol here is V1/V2/V3 *within
   train+dev*. `holdout_manifest_v2.json`'s 37 excluded fields stay excluded
   everywhere. Write a test that asserts the loader refuses test, and run it.
2. **The committed matrix is the contract.** `configs/benchmarks/matrix.yaml`
   is generated, committed and hashed *before* any execution (task 0.4). Any
   change afterwards is an amendment with a recorded reason — not an edit.
3. **`Not Applicable` ≠ `Deferred` ≠ `Failed`.** Three distinct registry
   states, each needing a machine-checkable reason. "Not applicable" requires a
   mathematical reason (an invalid Ax×Sy pair, or a Mamba feasibility-gate
   trigger with recorded evidence). Never substitute, shrink or re-scope a
   non-applicable cell — task 3.5 and the Mamba gate both say so explicitly.
4. **Synthetic ≠ stress; proxy ≠ ground truth.** Every proxy table carries its
   circularity caveat, and CWSI_LST/TVDI/CWSI_ET/ETa are *shared-input*
   consistency checks, never confirmation. The report's rung ceiling is L3.

## 5. Working standard

Follow the standard the capstone work set, because it is what caught the real
bugs:

- **TDD-first.** Every implementation task ships with tests. Nothing reaches a
  training run before the local gate.
- **Never emit a fabricated value.** An absent input is `absent`/`Deferred` with
  a reason — never `0.0`, never `NaN`, never a silent default. A gate or cell
  with no usable input reports *undetermined*, not *failed*; reporting failure
  claims a measured loss that never happened.
- **Never commit a red suite.** If you break something, fix it in the same
  commit, or revert.
- **Producers and consumers share one schema.** The capstone's most expensive
  bug was a producer emitting `farm_separability_top1` while the consumer read
  `farm_separability_ssl`/`_rand`: the gate matched nothing and reported
  "missing evidence" forever, however much was measured. When you add a
  producer/consumer pair — metric → table, manifest → report, registry →
  summary — write the test that drives one into the other.
- **Smoke output is not evidence.** It goes under `reports/_smoke/` and is
  gitignored. Check `.gitignore` covers wherever your scripts actually write;
  a synthetic artifact slipped into a commit once because the pattern pointed
  at the wrong path.
- **Ticked means done.** A `tasks.md` checkbox is ticked against code that
  exists and runs. If a box asks for an evaluation and only the plumbing
  exists, leave it unchecked and say why in the commit message. The capstone
  left `5.1` unchecked for exactly this reason.
- **Report honestly.** If a result is negative, that is the result. Findings
  18–25 are all negative and they were published as such with equal prominence.

## 6. Order of work, and where to stop before the owner

Work Phase 0 → 1 → 2 → 3 → 4 first. That is the core the rest depends on and
it is all CPU-testable.

- **Phase 0 (0.1–0.4)** — inventory, regression gate, freeze references, and
  the committed `matrix.yaml` + hash. Nothing executes before 0.4 is done.
- **Phase 1** — `CommonArchitecture` interface (train/infer/score/checkpoint),
  config + manifest schema, seed/split/checkpoint/`progress.jsonl`/`--resume`,
  metric interfaces, and the experiment registry. The registry is where
  `Pending/Running/Completed/Failed/Interrupted/Deferred/Not Applicable` live;
  build state transitions resume-safe, because the whole budget story depends
  on it.
- **Phase 2 (A1–A10)** — A1 first, reusing `IncumbentAE`. Then A3/A4 latent
  reconstruction, which force a decision you must document: *which*
  collapse-prevention mechanism (stop-gradient or EMA target) and why. A7/A8
  JEPA need a content-preservation probe or the arm is `invalid`. A2/A6/A8/A10
  sit behind the Mamba feasibility gate.
- **Phase 3 (S0–S3)** — S1's bounded weights must be **non-reinforcing**: the
  design forbids weights derived only from the current residual. S3's train
  generator must be independent of the eval generator (separate seeds,
  separate samplers). Task 3.5's applicability matrix is binding and committed
  before execution.
- **Phase 4** — the five families, three severities scaled to historical
  variability, exact ground-truth metadata, and the train/test stream isolation
  check.

Then **stop and report** before Phase 5 onwards. Those phases depend on owner
actions that do not exist yet:

| blocker | what it gates |
|---|---|
| v2 patch store + norm v2 sidecar (owner action in `todo.md`) | all of Phase 5 — patch-level cells are `Deferred` until bound; field-mean path unaffected |
| T4 hour budget for the first cycle | Phase 12 budget path; screening breadth (1 vs 2 seeds) |
| ERA5 daily features | Phase 6 |
| decomposition features | Phase 6 |
| owner decision on `ndvi_lst_rule` as an extra baseline row | Phase 8 |

Build against them with the `Deferred` path tested — task 12.7 asks for exactly
that. Do not invent stand-ins.

## 7. Two questions to resolve early, with the owner if needed

`design.md` §Open questions for the owner lists five. Two will block you inside
the first three phases:

1. **Mamba feasibility thresholds.** The design says "numerical stability or 2×
   the T4 wall-time budget of its GRU twin" — but the wall-time half cannot be
   measured locally without CUDA. Decide and record what the *local* half of
   that gate is (a stability check that is meaningfully stricter than "it ran"),
   so A2/A6/A8/A10 are not admitted by default merely because CUDA was absent.
2. **Screening breadth.** The design mentions "1 seed" in the scheduling order
   and the task list leaves it open. Pick one, record it in the matrix, and do
   not let it drift per architecture — unequal screening is unequal budget.

## 8. Handover rules

Every phase ends with, in this order:

1. `pytest` green — state the count.
2. `--smoke` end-to-end on the local machine, **output read**, not just exit 0.
3. The smoke output inspected: metrics present, no `NaN`/`Inf`, shapes and
   counts as expected, manifest complete.
4. Exact commands/notebook cells written out with cwd, env, flags, expected
   wall time and output paths.
5. `--resume` verified.

The agent never launches a long job. Anything expected over ~2 minutes is the
owner's to run.

## 9. Scope discipline

This change is the **heavy counterpart** to the capstone. It owns the
architecture × suppression × synthetic-anomaly comparison on Colab T4. It does
**not** own representation/transfer evaluation, claim semantics, or gates — the
capstone owns those, and no capstone gate depends on this benchmark completing.
Duplication is explicitly prohibited in `design.md` §The Colab notebook
("imports implemented modules; duplication prohibited").

Keep `tasks.md` honest as you go, the way the capstone did: tick only what runs,
leave owner-run items unchecked, and record the count in `todo.md`.

Start with Phase 0.1 and report the inventory before writing any model code.
