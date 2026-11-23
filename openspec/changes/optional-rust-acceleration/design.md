# Design

## Context

`fast-anything` profiles a real command, records calls to a chosen function,
and generates a Rust crate in a project-local `.fast/` folder. Enabling a port
appends a small, reversible block to the original Python module. That block
tries to import a compiled `.so` from `.fast/lib/` and uses it when present;
otherwise it falls back to the original Python function. The Python body is
never removed or edited.

Because the compiled artifacts are untracked, a fresh clone already runs pure
Python. What is missing is an explicit contract: when acceleration is allowed,
how parity is proven, and how a user turns it on or off without touching code.

Measured prototype evidence (synthetic, production-like shapes, conda env
`geospatial`, single-thread BLAS, medians):

| Workflow | Python | Rust | End-to-end |
|---|---:|---:|---:|
| Extraction (`aggregate_bin`) | 9.56 s | 4.66 s | 2.1× |
| Scoring (`ewma` + `cusum`) | 11.23 s | 5.18 s | 2.2× |
| Patch eval | 20.42 s | 20.29 s | 1.0× |
| Training step | 8.36 s | 8.45 s | 1.0× |

Nested profiler shares overlap; the A/B columns are the decision-grade figures.

## Goals / Non-Goals

- Make Rust acceleration strictly opt-in and reversible.
- Preserve the Python implementation as the single source of truth for
  semantics and for every scientific result.
- Give an exact, reproducible way to profile, verify, enable, disable, and
  re-measure.
- Prove parity on real recorded calls before a port can be enabled.

Non-goals:

- No change to research protocol, metrics, thresholds, severity tables, data
  contracts, or platform output.
- No Rust port of training code: the time is in torch `forward` methods and
  autograd, which this tool cannot port. Training acceleration stays a
  `torch.compile` / GPU / Colab question.
- No v1 port of `inject_patch_drop` (eval, ~39% of that workflow): it calls
  several helpers and builds hash-bearing injection records, so it needs a
  dedicated parity suite before it may be considered.

## Decisions

1. **Python is canonical.** A port is an adapter, not a replacement. No Python
   function, signature, default, or call site is deleted or rewritten. Removing
   the appended swap block restores the original behavior exactly.
2. **Opt-in only.** A clean checkout runs the Python path. Acceleration requires
   two explicit actions from `src/`: `fast-anything build` (create the `.so`
   files) and `fast-anything enable <id>` (append or keep the swap block).
   There is no automatic switch based on file presence alone in the documented
   workflow.
3. **Per-run opt-out.** `FAST_ANYTHING_DISABLE=1 <command>` runs the Python path
   even for enabled ports. `fast-anything disable <id>` reverts one port.
   These are the supported ways to compare or to remove acceleration.
4. **Fail open, but never enable unverified code.** A missing or unimportable
   `.so` must fall back to Python silently and keep results identical. A port
   that fails any parity case must never be enabled.
5. **Promotion gate for every port.** Before enablement: (a) `fast-anything
   verify` passes 100% of recorded + derived/edge calls at relative tolerance
   1e-9; (b) the targeted pytest selection passes with acceleration both on and
   off; (c) `fast-anything bench` shows a real end-to-end gain on the committed
   driver, with loop count and machine recorded in the evidence report.
6. **Any change re-opens the gate.** Editing the Python function or the Rust
   crate invalidates the previous verification. Re-record, re-verify, re-test.
7. **Evidence is committed; binaries are not.** The driver and the Markdown
   report are tracked. `.fast/lib/*.so`, recordings, and profiles stay
   untracked, consistent with the repository's text-only Git policy.
8. **Scientific neutrality.** Acceleration changes wall time only. It must not
   change scores, hashes, thresholds, labels, counts, or any reported number.
   The work sits at validation-ladder rung L0 and makes no detection claim.

## Scope of v1

Included: the three already verified ports (`ewma_detector`, `cusum_detector`,
`aggregate_bin`) plus the opt-in contract, parity gate, driver, and evidence.

Deferred to later changes, not implied by this one: `flag_runs` and
`contiguous_runs` (~13% each of the scoring workload after v1), the eval
per-pixel index calls (5.46 M calls at 2.3%; vectorize at the call site first),
and `inject_patch_drop`.

## Risks / Trade-offs

- **Semantic drift.** Either side changing without re-verification could make
  results diverge. Mitigated by decision 5 and 6.
- **Environment coupling.** The `.so` is built for one Python ABI and platform.
  Mitigated by fail-open import and by documenting `fast-anything build` per
  checkout.
- **Confusing shares.** Nested profiler functions are reported separately and
  shares overlap. Only end-to-end A/B numbers justify a port.
- **Driver placement.** The profiling driver must live inside the directory
  passed to `fast-anything`; otherwise module ids gain a `src.` prefix and the
  recorder catches zero calls. It is kept at `src/_fa_bench_driver.py`.
- **Recording constraints.** Keyword arguments must be plain data (use the
  string `"float16"`, not the `np.float16` type) or `fast-anything port` skips
  the calls.
- **No speedup where time sits elsewhere.** Eval and training remain at 1.0×;
  the report says so explicitly rather than hiding it.
