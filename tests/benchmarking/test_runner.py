"""End-to-end runner tests on the fixture (Phase 9 + the AGENTS.md §6.1 local gate).

These deliberately drive the *whole* path — prepare → plan → train → score →
calibrate → manifest → registry transition — because every piece of it already has
a unit test. Only an end-to-end test can catch the ordering and the plumbing: a gate
that is wired but never read, a sidecar nobody writes, a resume that silently
restarts from zero, a budget check that runs after the work was spent.

The fixture is noise plus injected offsets, so nothing here asserts a scientific
value. It asserts that a value was produced, recorded, and traceable to the config
and hashes that produced it. The smoke plan is run once per module and shared: the
local gate has to stay cheap enough that it is actually re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ssl_anomaly.benchmarking import checkpoints as ck
from ssl_anomaly.benchmarking.config import (ARCHITECTURES, load_applicability,
                                  load_benchmark_config)
from ssl_anomaly.benchmarking.matrix import build_matrix, matrix_document, read_matrix, write_matrix
from ssl_anomaly.benchmarking.runner import BenchmarkRunner, RunOptions, RunPlanError

SIDE_CARS = ("run_manifest.json", "metrics.json", "progress.jsonl", "norm_stats.json")
# A1 (input recon) + A3/A7 (latent, for the collapse gate) + A2 (Mamba, for feasibility),
# which is the cheapest plan that touches every gate in the runner.
SMOKE_MODELS = ("A1", "A2", "A3", "A7")


def _build(tmp_path, cfg, app, cells, **opts) -> BenchmarkRunner:
    kw = dict(base=Path(str(tmp_path)), out=Path(tmp_path) / "bench", tier="screening",
              fixture=True, smoke=True, model_ids=SMOKE_MODELS)
    kw.update(opts)
    return BenchmarkRunner(cfg, app, cells, RunOptions(**kw))


@pytest.fixture(scope="module")
def contract():
    cfg = load_benchmark_config()
    app = load_applicability()
    return cfg, app, build_matrix(cfg, app)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory, contract):
    """One shared fixture smoke run for the read-only assertions."""
    cfg, app, cells = contract
    tmp = tmp_path_factory.mktemp("smoke")
    r = _build(tmp, cfg, app, cells)
    info = r.prepare()
    summary = r.run()
    return {"runner": r, "summary": summary, "info": info, "out": tmp / "bench",
            "by_id": {x["experiment_id"]: x for x in summary["results"]}}


def _done(smoke):
    return [x for x in smoke["summary"]["results"] if x["status"] == "Completed"]


def _manifest(smoke, experiment_id):
    row = smoke["by_id"][experiment_id]
    return json.loads((Path(row["run_dir"]) / "run_manifest.json").read_text())


# --------------------------------------------------------------------------- prepare
def test_prepare_verifies_bindings_and_writes_the_matrix(smoke):
    r = smoke["runner"]
    assert (r.out / "matrix.yaml").is_file(), "a run without a committed matrix cannot be cited"
    assert (r.out / "bindings.json").is_file()
    assert len(r.matrix_sha) == 64
    assert read_matrix(r.out / "matrix.yaml")["matrix_sha256"] == r.matrix_sha, (
        "the matrix the run used must be the committed one, not a re-derived surprise")


def test_prepare_audits_objectives_before_any_training(smoke):
    r = smoke["runner"]
    assert r.objective_audit["problems"] == []
    assert len(r.objective_audit["models"]) == 10, "every architecture arm must be audited"
    assert r.registry.counts().get("Running", 0) == 0, "prepare must not claim work"


def test_prepare_hashes_data_splits_and_labels(smoke):
    h = smoke["runner"].hashes
    for key in ("data_sha256", "panel_labels_sha256", "contamination_labels_sha256",
                "train_pool_sha256", "eval_pool_sha256", "panel_generator_sha256"):
        assert h.get(key), f"{key} missing: the manifest could not say what it read"
    assert h["train_pool_sha256"] != h["eval_pool_sha256"], "the splits must be different row sets"
    assert h["panel_labels_sha256"] != h["contamination_labels_sha256"], (
        "the eval panel and the train contamination must be generated separately")


def test_synthetic_pools_are_disjoint_and_the_contract_is_enforced(smoke):
    d = smoke["runner"].disjointness
    assert d["disjoint"], f"train/eval event pools share identities: {d['shared']}"
    assert d["n_eval_events"] > 0 and d["n_train_events"] > 0


def test_capacity_is_resolved_in_band_for_every_arm(smoke):
    cap = smoke["runner"].capacity
    assert cap["all_in_band"], {k: v["note"] for k, v in cap["arms"].items() if not v["in_band"]}
    # capacity is resolved for the whole matrix, not just the arms this run trains: the
    # band has to be comparable across the matrix, and a per-selection resolution would
    # let the same arm land differently in two runs.
    assert cap["n_arms"] == len(ARCHITECTURES)
    for mid, a in cap["arms"].items():
        assert a["trace"], f"{mid}: the search trace is the evidence, not just the winner"
        assert a["chosen"]["params"] > 0


# --------------------------------------------------------------------------- plan
def test_plan_is_priority_ordered_and_filtered(contract, tmp_path):
    cfg, app, cells = contract
    r = _build(tmp_path, cfg, app, cells, dry_run=True)
    r.prepare()
    plan = r.plan()
    assert plan and all(c.model_id in SMOKE_MODELS for c in plan)
    assert [c.priority for c in plan] == sorted(c.priority for c in plan)
    assert all(c.status == "Pending" for c in plan)
    assert all(c.level == "farm" for c in plan), "patch cells are Deferred, never planned"


def test_matrix_drift_is_refused_before_any_work(tmp_path, contract):
    cfg, app, cells = contract
    r = _build(tmp_path, cfg, app, cells)
    r.prepare()
    doc = matrix_document(cfg, cells, applicability_sha="f" * 64, note="tampered")
    write_matrix(doc, r.out / "matrix.yaml")
    with pytest.raises(RunPlanError, match="committed matrix"):
        _build(tmp_path, cfg, app, cells) and BenchmarkRunner(
            cfg, app, cells, RunOptions(base=Path(str(tmp_path)), out=r.out, tier="screening",
                                        fixture=True, smoke=True)).prepare()


# --------------------------------------------------------------------------- bundle
def test_completed_cells_leave_a_complete_bundle(smoke):
    done = _done(smoke)
    assert done, f"nothing completed: {[x['reason'] for x in smoke['summary']['results']]}"
    for x in done:
        rd = Path(x["run_dir"])
        for name in SIDE_CARS:
            assert (rd / name).is_file(), f"{x['experiment_id']}: {name} missing from the bundle"
        assert ck.audit_run_dir(rd, require=SIDE_CARS)["missing"] == []
        m = json.loads((rd / "run_manifest.json").read_text())
        assert m["matrix_sha256"] == smoke["runner"].matrix_sha
        assert m["benchmark_config_sha256"] == smoke["runner"].cfg.document_sha256
        assert m["data"]["data_sha256"] and m["data"]["panel_labels_sha256"]
        assert m["objective"]["objective"], "the objective must be recorded per cell"
        assert m["capacity"]["chosen"]["in_band"]
        assert m["code"], "the code SHA belongs in the manifest"
        assert (rd / "bin_scores.npz").is_file(), "per-bin scores are the re-analysis surface"


def test_every_cell_is_stamped_as_a_fixture(smoke):
    """A plumbing proof must not be mistakable for a result (AGENTS.md §6.1)."""
    assert smoke["summary"]["run"]["stamp"]
    assert smoke["summary"]["run"]["ladder_rung"].startswith("L0")
    for x in smoke["summary"]["results"]:
        if not x["run_dir"]:
            continue
        m = json.loads((Path(x["run_dir"]) / "run_manifest.json").read_text())
        assert "SMOKE FIXTURE" in str(m["stamp"]) and m["ladder_rung"].startswith("L0")


def test_metrics_are_macro_over_cells_with_masks_named(smoke):
    for x in _done(smoke):
        mac = x["metrics"]["macro"]
        assert mac["n_cells_used"] == 15, "the primary metric is the macro over all 15 cells"
        assert mac["cells_missing"] == []
        assert 0.0 <= mac["macro"] <= 1.0 and 0.0 <= mac["event_weighted"] <= 1.0
        assert mac["prevalence_range"], "prevalence is recorded next to the metric, never omitted"
        assert x["metrics"]["level_note"], "patch-level absence must be stated"
        for pc in x["metrics"]["per_cell"]:
            assert pc["n_scored_bins"] > 0 and pc["n_anomaly_bins"] > 0, "masks are named and counted"
            assert not np.isnan(pc["auprc"]), f"a scored cell reported NaN: {pc}"
        assert x["metrics"]["threshold"]["budget"] == float(
            smoke["runner"].cfg.raw["calibration"]["alarm_budget"])


def test_calibration_never_uses_the_scored_partition(smoke):
    for x in _done(smoke):
        m = _manifest(smoke, x["experiment_id"])
        fit = m["calibration"]["fit"]
        # calibration.sources is what the frozen contract pre-registers (train/dev, test
        # forbidden) — "never the scored partition" is NOT the rule, and a test that
        # asserted it would have pushed the runner into an undocumented deviation.
        assert fit["fitted_on"] in fit["allowed_sources"], fit
        assert fit["fitted_on"] in ("train", "dev"), fit
        assert m["calibration"]["threshold"]["contract"]["test_usage"] == "forbidden"
        assert "test" not in m["calibration"]["threshold"]["contract"]["sources"]
        if fit["method"] != "identity":
            # whichever partition the mapping came from, it has to be one the contract
            # lists, and the record must name the *reference distribution* — a bare
            # "fitted_on: train" would not distinguish a train-score reference from a
            # pooled cross-field one, which is the difference the normality rule cares about
            assert fit["params"].get("reference") or fit["n_fit"] > 0, fit
        assert fit["allowed_sources"] == ["train", "dev"]


# --------------------------------------------------------------------------- gates
def test_collapse_gate_disqualifies_rather_than_reporting_a_low_score(smoke):
    """On noise the latent arms collapse within 2 epochs, and the gate must say so.

    A silent pass here would mean the gate is computed but never read — which is how
    a degenerate latent space would reach a ranking table.
    """
    failed = {k: v for k, v in smoke["by_id"].items() if v["status"] == "Failed"}
    assert failed, "no latent arm collapsed on 2 epochs of noise: check the gate is wired"
    assert all("collapse" in str(v["reason"]) for v in failed.values())
    for v in failed.values():
        m = json.loads((Path(v["run_dir"]) / "run_manifest.json").read_text())
        c = m["collapse"]
        assert c["verdict"].startswith("collapsed") and c["reasons"]
        assert c["thresholds"]["effective_rank"]["min"] == float(
            smoke["runner"].cfg.raw["collapse"]["effective_rank_ratio_floor"]), "the gate uses the contract's floor"
        assert c["content_probe"].get("available") is True, (
            "the content probe runs alongside the geometry stats: a low-rank latent whose "
            "probe also failed is an invalid measurement, not a collapsed arm")


def test_reconstruction_arms_are_not_collapse_gated(smoke):
    for k, v in smoke["by_id"].items():
        if not k.startswith(("A1.", "A5.", "A9.")) or not v["run_dir"]:
            continue
        m = json.loads((Path(v["run_dir"]) / "run_manifest.json").read_text())
        assert m["collapse"]["verdict"] == "not_applicable_by_construction", (
            "a reconstruction arm has no latent space; the gate must say that, not pass or fail")


def test_feasibility_gate_is_not_fired_on_the_wrong_device(smoke):
    """The 2× budget describes the T4; a CPU wall-time ratio must not be reported as one."""
    r = smoke["runner"]
    mamba = [v for k, v in smoke["by_id"].items() if k.startswith(SMOKE_MODELS) and ".I3" in k
             and k.split(".")[0] in ("A2", "A4", "A6", "A8", "A10")]
    assert mamba
    if r.device.device != "cuda":
        for v in mamba:
            assert v["status"] != "Not Applicable", f"feasibility verdict on {r.device.device}: {v['reason']}"
        rec = [f for f in Path(smoke["out"]).glob("runs/*/feasibility.json")]
        assert rec, "the gate must still record *why* it abstained"
        feas = json.loads(rec[0].read_text())
        assert feas["not_applicable"] == "wrong_device" and feas["ratio"] is None


def test_registry_state_machine_end_to_end(smoke):
    r = smoke["runner"]
    counts = r.registry.counts()
    total = sum(counts[k] for k in ("Pending", "Running", "Completed", "Failed", "Deferred",
                                   "Not Applicable", "Interrupted") if k in counts)
    assert total == len(r.cells)
    assert counts.get("Running", 0) == 0, "a finished run leaves nothing claimed"
    assert counts["Completed"] >= 1
    assert (r.out / "coverage.md").is_file()
    log = [json.loads(l) for l in open(r.out / "registry.log.jsonl")]
    assert {"Pending", "Running", "Completed"} <= {x["to"] for x in log}, \
        "the append-only log must show the claim, not only the result"
    assert all(x["to"] in {"Pending", "Running", "Completed", "Failed", "Deferred",
                           "Not Applicable", "Interrupted"} for x in log)


# --------------------------------------------------------------------------- failure / resume
def test_failed_cell_carries_the_error_verbatim_and_can_be_requeued(contract, tmp_path):
    class Boom(RuntimeError):
        pass

    cfg, app, cells = contract
    r = _build(tmp_path, cfg, app, cells, model_ids=("A1",), epochs=4)
    orig, state = r._objective_loss, {"n": 0}

    def wrapped(*a, **k):
        v = orig(*a, **k)
        state["n"] += 1
        if state["n"] == 2:
            raise Boom("simulated device error at the epoch boundary")
        return v

    r._objective_loss = wrapped
    r.prepare()
    row = next(iter(r.run()["results"]))
    assert row["status"] == "Failed" and "simulated device error" in row["reason"]
    assert row["run_dir"], "a crashed cell must still point at its partial artefacts"
    cell = next(c for c in r.cells if c.experiment_id == row["experiment_id"])
    entry = r.registry.get(cell.cell_hash)
    assert entry.attempts >= 1 and "Boom" in entry.error
    r.registry.requeue(cell.cell_hash, reason="retry after the simulated device error")
    again = r.registry.get(cell.cell_hash)
    assert again.status == "Pending" and "requeued" in str(again.reason)


def test_resume_continues_from_the_checkpoint_instead_of_restarting(contract, tmp_path):
    """A crashed cell, requeued and resumed, must pick up where its checkpoint stopped.

    The smoke profile caps a cell at two epochs, so the crash is placed *inside* the
    second epoch: the resumed run must then start at epoch 1 instead of re-running
    epoch 0, and the progress log must show both epochs exactly once. That is the whole
    claim — not that resuming is fast.
    """
    class Boom(RuntimeError):
        pass

    cfg, app, cells = contract
    r1 = _build(tmp_path, cfg, app, cells, model_ids=("A1",), limit=1)
    orig, state = r1._objective_loss, {"n": 0}

    def wrapped(*a, **k):
        v = orig(*a, **k)
        state["n"] += 1
        if state["n"] == 2:
            raise Boom("simulated device error during the second epoch")
        return v

    r1._objective_loss = wrapped
    r1.prepare()
    row = next(iter(r1.run()["results"]))
    assert row["status"] == "Failed", row
    rd = Path(row["run_dir"])
    first = [json.loads(line)["epoch"] for line in open(rd / "progress.jsonl")]
    assert first == [0], f"the crashed cell logged {first}: epoch 1 was interrupted, not completed"
    cell = next(c for c in r1.cells if c.experiment_id == row["experiment_id"])
    r1.registry.requeue(cell.cell_hash, reason="resume after the simulated device error")

    r2 = _build(tmp_path, cfg, app, cells, model_ids=("A1",), limit=1, resume=True)
    r2.prepare()
    row2 = next(iter(r2.run()["results"]))
    assert row2["status"] == "Completed", row2["reason"]
    assert any("resumed from epoch 1" in n for n in row2["notes"]), row2["notes"]
    epochs = [json.loads(line)["epoch"] for line in open(Path(row2["run_dir"]) / "progress.jsonl")]
    assert epochs == [0, 1], f"the log must continue, not restart: {epochs}"
    m = json.loads((Path(row2["run_dir"]) / "run_manifest.json").read_text())
    tr = m["training"]
    # one epoch before the crash, one after: the manifest has to state both, or a
    # comparison against an un-crashed cell is a comparison of nothing
    assert tr["resumed_from"] == 1, tr
    assert tr["epochs_this_invocation"] == 1, tr
    assert tr["epochs"] == 2 == tr["epochs_planned"], tr
    assert [h["epoch"] for h in tr["history_tail"]] == [0, 1], tr["history_tail"]
    assert m["training"]["resumed_from"] == 1, "the manifest must record that this cell was resumed"
    assert r2.registry.get(cell.cell_hash).attempts >= 2, "the retry must be visible in the attempt count"


def test_resume_refuses_a_checkpoint_with_different_bindings(smoke):
    r = smoke["runner"]
    row = _done(smoke)[0]
    with pytest.raises(ck.CheckpointError, match="refusing to resume") as exc:
        ck.load_checkpoint(Path(row["run_dir"]), expected_bindings={"matrix_sha256": "0" * 64},
                           strict=True)
    assert r.matrix_sha[:12] in str(exc.value), "the message must name both sides"
    assert "000000000000" in str(exc.value)


# --------------------------------------------------------------------------- budget
def test_budget_exhaustion_defers_the_remaining_plan(contract, tmp_path):
    cfg, app, cells = contract
    r = _build(tmp_path, cfg, app, cells, model_ids=("A1", "A9", "A2"))
    r.prepare()
    r.budget.used_hours = float(cfg.raw["budget"]["t4_hours"])
    s = r.run()
    assert s["results"] == [], "no cell may start once the budget is spent"
    counts = r.registry.counts()
    assert counts.get("Deferred", 0) >= 3, counts
    assert any("budget" in n.lower() for n in s["notes"]), s["notes"]
    reasons = [e.reason for e in r.registry.entries(status="Deferred") if "budget" in str(e.reason)]
    assert reasons, "the deferral must name the budget as its reason"


def test_budget_is_charged_per_cell(contract, tmp_path):
    cfg, app, cells = contract
    r = _build(tmp_path, cfg, app, cells, model_ids=("A1",))
    r.prepare()
    before = r.budget.used_hours
    s = r.run()
    assert s["results"], "the plan must not be empty"
    assert r.budget.used_hours > before, "wall time must be charged against the T4 budget"
    assert r.budget.to_dict()["log"], "the budget log is part of the run record"


def test_dry_run_plans_without_touching_the_registry(contract, tmp_path):
    cfg, app, cells = contract
    r = _build(tmp_path, cfg, app, cells, dry_run=True)
    r.prepare()
    counts = r.registry.counts()
    s = r.run()
    assert s["dry_run"] and s["planned"] == len(s["ids"])
    assert r.registry.counts() == counts, "a dry run must not advance any cell"


# --------------------------------------------------------------------------- §6.2 bundle gate
#
# The bundle is the only thing that survives the trip to the GPU box and back, so its
# completeness rule ("complete or a defect") is tested here, against the shared smoke run:
# a bundle that could not be verified is worse than no bundle, and a *false* "complete" is
# how a missing cell file turns into a missing row in a results table.


def _bundle_tools():
    from ssl_anomaly.benchmarking import bundle as _b

    return _b


def test_a_bundle_refuses_an_unauditable_run(tmp_path):
    """No registry, no cells: a refusal, not an empty bundle that looks published."""
    b = _bundle_tools()
    empty = tmp_path / "nothing"; (empty / "runs").mkdir(parents=True)
    assert b.audit_bundle(empty).complete is False
    plan = b.collect(empty, tmp_path / "dest", name="x")
    assert "refused" in plan and not (tmp_path / "dest" / "x").exists()


def test_completed_cells_produce_a_verified_bundle(smoke, tmp_path):
    """The fixture run must be bundleable as-is, and tampering must be detectable."""
    import hashlib
    b = _bundle_tools()
    assert b.audit_bundle(smoke["out"]).complete, b.audit_bundle(smoke["out"]).to_dict()["defects"]
    plan = b.collect(smoke["out"], tmp_path / "bundles", name="t", notes=["unit test"])
    assert "refused" not in plan, plan.get("refused")
    target = Path(plan["manifest"]).parent
    # the cover page is inside the hashed set, and its cd hint does not double the path
    assert (target / "LATEST_BUNDLE.md").is_file()
    latest = (target / "LATEST_BUNDLE.md").read_text()
    assert f"cd {target}" in latest and f"{target.name}/{target.name}" not in latest
    listed = {f["path"] for f in json.loads((target / "BUNDLE_MANIFEST.json").read_text())["files"]}
    assert "LATEST_BUNDLE.md" in listed, "an unhashed file inside a 'verified' bundle is the hole"
    v = b.verify_bundle(target)
    assert v["ok"] and not v["missing"] and not v["extra"] and not v["corrupt"], v
    victim = next(target.glob("runs/*/metrics.json"))
    digest = hashlib.sha256(victim.read_bytes()).hexdigest()
    victim.write_text(victim.read_text()[:-1] + " ", encoding="utf-8")
    after = b.verify_bundle(target)
    assert not after["ok"] and after["corrupt"] == [victim.relative_to(target).as_posix()], after
    assert digest != hashlib.sha256(victim.read_bytes()).hexdigest()


def test_a_run_dir_is_self_describing_without_the_repo(smoke):
    """The matrix the run bound to is mirrored into the out dir, so `matrix.yaml` +
    `registry.json` + manifests alone answer "which cells were on the menu"."""
    mirrored = Path(smoke["out"]) / "matrix.yaml"
    assert mirrored.is_file(), "the runner must copy the bound matrix into the run dir"
    import yaml
    doc = yaml.safe_load(mirrored.read_text())
    assert doc["matrix_sha256"] == smoke["runner"].matrix_sha
    n = len(list(Path(smoke["out"], "runs").glob("*")))
    assert n and len(doc["cells"]) >= n, "the mirrored matrix cannot describe fewer cells than ran"
