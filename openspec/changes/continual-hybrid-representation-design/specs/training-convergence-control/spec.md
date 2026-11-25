# training-convergence-control Specification

## Purpose

Every trainable arm trains on the owner's laptop CPU until its validation
curve plateaus, is always resumable from a checkpoint, and ends with an
explicit, verbose verdict that says whether more training is needed.

## ADDED Requirements

### Requirement: Plateau-driven stopping on a multi-signal guard set (r2)

The trainer SHALL evaluate every `eval_every` steps on a validation split
disjoint from the test split and SHALL apply the plateau rule (relative
improvement below `min_rel_delta` for `patience` consecutive evaluations AND an
EMA-smoothed slope over the last `window` evaluations not steeper than
`min_rel_delta / window`, after `min_steps`) to each guard signal separately —
every individual objective's validation loss and the pre-registered primary
probe metric — and SHALL NOT use the weighted total loss as the sole stopping
signal. It SHALL stop when all guard signals have plateaued, or when the
primary probe metric has plateaued and no objective is still improving beyond
threshold.

#### Scenario: Plateau stops early
- **WHEN** every objective and the primary probe metric stop improving by more than 0.5 % for 8 evaluations with flat smoothed slopes
- **THEN** training stops before the step cap with verdict `CONVERGED_PLATEAU`

#### Scenario: Total loss flat, one objective still improving
- **WHEN** the weighted total has plateaued but the forecast loss is still improving beyond threshold
- **THEN** training continues

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

### Requirement: Per-objective and representation-quality monitoring (r2)

At every evaluation the trainer SHALL log each objective's validation loss
separately (reconstruction, forecast, JEPA, Barlow on- and off-diagonal
terms), the effective rank of the embeddings, the fraction of embedding
dimensions with std above a floor and the off-diagonal covariance energy; and
every `probe_every` evaluations it SHALL run a fixed-budget frozen-encoder
evaluation on the validation split (linear probe of season phase and next-
observation NDVI, same-field-same-phase kNN retrieval, gap reconstruction from
the frozen latent) and a stability check (linear CKA to the previous probe
checkpoint, top-K score-rank correlation).

#### Scenario: Probe logged
- **WHEN** a probe evaluation runs
- **THEN** its metrics appear in `progress.jsonl` and in the verbose line with the probe checkpoint step

### Requirement: Objective-conflict guard (r2)

The trainer SHALL flag `OBJECTIVE_CONFLICT` whenever any objective regresses
more than `max_obj_regress` relative to its best value while the total
improves, SHALL print the flag, SHALL make that checkpoint ineligible for
`best.pt`, and SHALL stop for owner review if the conflict persists for
`patience` evaluations.

#### Scenario: Improving total hides a degrading objective
- **WHEN** the total loss falls while the JEPA loss rises 8 % above its best
- **THEN** the evaluation is flagged and its checkpoint cannot be selected as best

### Requirement: Eligibility-filtered checkpoint selection (r2)

`best.pt` SHALL be the checkpoint with the best pre-registered primary probe
metric among checkpoints with no collapse flag, no objective conflict and every
objective within `max_obj_regress` of its best; `best_loss.pt` (lowest total
loss) and `last.pt` SHALL also be kept, and the eligibility log SHALL be
written to the run manifest.

#### Scenario: Lowest-loss checkpoint is collapsed
- **WHEN** the lowest-total-loss checkpoint has effective rank below the floor
- **THEN** it is saved as `best_loss.pt` but not as `best.pt`
