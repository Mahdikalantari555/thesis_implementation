# training-convergence-control Specification

## Purpose

Every trainable arm trains on the owner's laptop CPU until its validation
curve plateaus, is always resumable from a checkpoint, and ends with an
explicit, verbose verdict that says whether more training is needed.

## ADDED Requirements

### Requirement: Plateau-driven stopping on a validation split

The trainer SHALL evaluate every `eval_every` steps on a validation split
disjoint from the test split and SHALL stop when the relative improvement over
the best value stays below `min_rel_delta` for `patience` consecutive
evaluations AND the EMA-smoothed validation slope over the last `window`
evaluations is not steeper than `min_rel_delta / window`, after `min_steps`.

#### Scenario: Plateau stops early
- **WHEN** the validation loss stops improving by more than 0.5 % for 8 evaluations and the smoothed slope is flat
- **THEN** training stops before the step cap with verdict `CONVERGED_PLATEAU`

#### Scenario: Noise does not stop training
- **WHEN** the patience counter is reached but the smoothed slope is still clearly negative
- **THEN** training continues

### Requirement: Budget-exhaustion verdict says whether more training is needed

The trainer SHALL honour `max_steps` and `max_wall_minutes` caps and, when a
cap is reached before a plateau, SHALL classify the run as
`BUDGET_EXHAUSTED_STILL_IMPROVING` (negative slope beyond threshold) or
`BUDGET_EXHAUSTED_FLAT`, print the exact resume command, and for the first case
print an extrapolated number of additional steps.

#### Scenario: Still improving at the cap
- **WHEN** the wall cap is hit while validation loss is still falling
- **THEN** the run prints "needs more training", the `--resume` command and an estimate of extra steps, and records the verdict in `run_manifest.json`

### Requirement: Divergence and collapse verdicts

The trainer SHALL stop with `DIVERGED` on a non-finite loss or a validation
loss above `diverge_factor` × best, and with `COLLAPSED` when the effective rank
or per-dimension embedding std stays below its floor for `patience`
evaluations, keeping `best.pt`.

#### Scenario: Representation collapse
- **WHEN** embedding effective rank falls below the configured floor for the patience window
- **THEN** training stops with `COLLAPSED` and the best non-collapsed checkpoint is kept

### Requirement: Mandatory checkpointing and exact resume

The trainer SHALL write `last.pt` atomically at every evaluation and at least
every `ckpt_every_minutes`, and on SIGINT, containing model, optimizer,
scheduler, step, epoch, RNG states, sampler position, early-stopping state,
EMA/target encoder, config hash, data/split hashes and git SHA; and `best.pt`
on every improvement. `--resume` SHALL continue the same run;
`--init-from` SHALL start a new run from a checkpoint; neither SHALL
re-initialize silently.

#### Scenario: Interrupted run resumes identically
- **WHEN** a run is interrupted and resumed with `--resume`
- **THEN** the remaining loss trajectory matches an uninterrupted run with the same seed within floating-point tolerance

### Requirement: Verbose progress and summary

The trainer SHALL print one line per evaluation (step, epoch, train loss and
components, val loss, best, relative improvement, patience k/P, smoothed slope,
LR, effective rank, steps/s, RAM, ETA) and append the same fields to
`progress.jsonl`, and SHALL print an end-of-run summary with the verdict, best
step, wall time and parameter count.

#### Scenario: Owner reads the verdict
- **WHEN** any run ends
- **THEN** the last printed block names one of the five verdicts and the next action

### Requirement: Fair comparison under unequal stopping

Any comparison between arms SHALL report both the to-plateau result (with
steps, wall time, parameters) and an equal-step cut at the smallest arm's
plateau step, and random-init controls SHALL be evaluated at the same step as
their trained twin.

#### Scenario: Two arms plateau at different steps
- **WHEN** arm A plateaus at 2,000 steps and arm B at 6,000
- **THEN** the report shows both arms at their plateau and both at 2,000 steps
