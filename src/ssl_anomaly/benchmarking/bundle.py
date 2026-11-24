"""Result bundles: complete-or-defect, verifiable after transfer (AGENTS.md §6.2).

A Colab run is only usable if the laptop can prove what it contains. The rule this
module enforces is binary on purpose: a bundle is **complete** or it is a **defect**.
There is no third category of "usable for a preliminary look", because in practice a
partial bundle gets read, the missing sidecars get noticed a week later, and the
numbers already written into a draft stay there.

Two failure modes are designed out:

* **A silent missing sidecar.** ``patch_sigma.npz``-class files are the reason an
  earlier bundle could not be re-analysed: the scores existed, the scale they were
  computed on did not. ``REQUIRED`` names every file a cell must have, and
  :func:`audit_bundle` reports per-cell presence *and* the manifest fields that make
  each file interpretable (data hashes, config SHA, matrix SHA).
* **A corrupt or truncated transfer.** `SHA256SUMS.txt` + `BUNDLE_MANIFEST.json`
  let the receiving machine verify byte-for-byte, and the bundle is written into a
  temporary directory and moved into place in one step, so a crash mid-write leaves
  the previous bundle intact rather than a half one with a plausible name.

Nothing here uploads anything: the Drive path convention and the exact commands are
*emitted* for the owner to run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

CELL_REQUIRED = ("run_manifest.json", "meta.json", "last.pt", "progress.jsonl",
                 "norm_stats.json", "metrics.json", "bin_scores.npz")
#: Fields that make the files interpretable. A bundle with the right filenames and an
#: unbound manifest is still useless, so the manifest is audited, not just listed.
MANIFEST_REQUIRED = ("experiment_id", "cell_hash", "seed", "factors", "benchmark_config_sha256",
                     "matrix_sha256", "data", "objective", "metrics", "code")
RUN_REQUIRED = ("matrix.yaml", "registry.json", "registry.log.jsonl", "coverage.md",
                "bindings.json", "capacity.json", "run_summary.json")
#: sidecars whose absence is a *finding*, not a defect (e.g. arms without a latent space)
OPTIONAL_BY_DESIGN = ("collapse.json", "feasibility.json", "classical_fit.json")
SKIP_FOR_HASHES = {"SHA256SUMS.txt", "BUNDLE_MANIFEST.json"}


@dataclass
class CellAudit:
    experiment_id: str
    path: str
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    manifest_missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    usable: bool = False
    status: str = ""

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id, "path": self.path, "status": self.status,
                "present": list(self.present), "missing": list(self.missing),
                "manifest_missing": list(self.manifest_missing), "usable": self.usable,
                "notes": list(self.notes)}

    @property
    def defect(self) -> bool:
        return bool(self.missing or self.manifest_missing)


@dataclass
class BundleAudit:
    root: str
    cells: list[CellAudit]
    run_files: list[str]
    defects: list[CellAudit]
    complete: bool
    counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"root": self.root, "complete": self.complete, "counts": self.counts,
                "run_files": list(self.run_files),
                "defects": [{"experiment_id": d.experiment_id, "missing": d.missing,
                             "manifest_missing": d.manifest_missing, "status": d.status}
                            for d in self.defects],
                "cells": [c.to_dict() for c in self.cells]}


def _sha(p: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def audit_cell(run_dir: Path, *, require: tuple[str, ...] = CELL_REQUIRED) -> CellAudit:
    run_dir = Path(run_dir)
    exp = run_dir.name.rsplit(".seed", 1)[0]
    present = [n for n in require if (run_dir / n).is_file()]
    missing = [n for n in require if not (run_dir / n).is_file()]
    notes: list[str] = []
    for n in OPTIONAL_BY_DESIGN:
        if not (run_dir / n).is_file():
            notes.append(f"{n} absent (by design for this cell shape)")
    man_missing: list[str] = []
    status = ""
    mp = run_dir / "run_manifest.json"
    if mp.is_file():
        try:
            man = json.loads(mp.read_text())
        except json.JSONDecodeError as exc:
            man_missing.append(f"run_manifest.json unparseable: {exc}")
            man = {}
        status = str(man.get("registry_status") or man.get("status") or "")
        for k in MANIFEST_REQUIRED:
            if k not in man or man[k] in (None, "", {}, []):
                man_missing.append(k)
        data = man.get("data") or {}
        for hk in ("data_sha256", "panel_labels_sha256"):
            if not data.get(hk):
                man_missing.append(f"data.{hk}")
        # `meta.json` + `last.pt` are the checkpoint *pair* (checkpoints.SIDECAR_NAME), and a
        # pair only exists where a checkpoint exists. A random-init control or a fitted
        # reference estimator trains nothing, so it has neither: that is a design fact, not a
        # defect. Exactly one of the two, however, is — a sidecar without a checkpoint is a
        # truncated write, and a bundle must not publish one quietly.
        pair = ("meta.json", "last.pt")
        have = [n for n in pair if (run_dir / n).is_file()]
        if len(have) == 1:
            orphan = [n for n in pair if n not in have][0]
            if orphan not in missing:
                missing.append(orphan)
            notes.append(f"checkpoint pair incomplete: {have[0]} exists without {orphan}")
        elif not have and str(man.get("checkpoint") or "saved") != "saved":
            missing = [n for n in missing if n not in pair]
            present_note = (f"no checkpoint recorded: the manifest declares "
                            f"checkpoint={man.get('checkpoint')!r}, so nothing was trained to resume")
            notes.append(present_note)
        if man.get("stamp"):
            notes.append(f"stamped: {man['stamp']}")
        if (man.get("pool_disjointness") or {}).get("disjoint") is False:
            man_missing.append("pool_disjointness.disjoint=false — synthetic train/eval pools overlap")
    else:
        man_missing.append("run_manifest.json")
    usable = not missing and not man_missing
    return CellAudit(exp, str(run_dir), present, missing, man_missing, notes, usable, status)


def audit_bundle(root: Path) -> BundleAudit:
    """Audit a run directory (or an extracted bundle) cell by cell."""
    root = Path(root)
    runs = root / "runs" if (root / "runs").is_dir() else root
    audits = [audit_cell(d) for d in sorted(p for p in runs.glob("*") if p.is_dir())]
    # A run directory does not stamp its own status into the manifest, so the registry —
    # which lives next to it — is joined in by experiment id. A bundle read without it would
    # otherwise report every cell as "unknown" and invite someone to count the directories
    # as if they were results.
    reg = root / "registry.json"
    if reg.is_file():
        try:
            entries = json.loads(reg.read_text())
            rows = entries.get("entries", entries) if isinstance(entries, dict) else entries
            by_id = {}
            for e in rows:
                if isinstance(e, dict) and e.get("experiment_id"):
                    by_id[str(e["experiment_id"])] = str(e.get("status") or "")
            for a_ in audits:
                if not a_.status:
                    a_.status = by_id.get(a_.experiment_id, "not-in-registry")
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            for a_ in audits:
                a_.notes.append(f"registry.json unreadable, statuses unknown: {exc}")
    run_files = [n for n in RUN_REQUIRED if (root / n).is_file()]
    missing_run = [n for n in RUN_REQUIRED if not (root / n).is_file()]
    defects = [a for a in audits if a.defect]
    counts: dict[str, int] = {}
    for a in audits:
        counts[a.status or "unknown"] = counts.get(a.status or "unknown", 0) + 1
    complete = bool(audits) and not defects and not missing_run
    out = BundleAudit(str(root), audits, run_files + [f"MISSING:{n}" for n in missing_run],
                      defects, complete, counts)
    return out


def collect(root: Path, dest: Path, *, name: str | None = None, dry_run: bool = False,
            notes: list[str] | None = None) -> dict:
    """Assemble a verified bundle: audit first, copy, then hash-manifest, then publish.

    The audit runs *before* anything is copied, so a defect is reported while there is
    still time to re-run the cell rather than after a plausible-looking directory has
    been published next to the good ones.
    """
    root = Path(root)
    audit = audit_bundle(root)
    stamp = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    name = name or f"benchmark_{stamp}"
    tmp = Path(dest) / f".{name}.partial"
    target = Path(dest) / name
    plan = {
        "name": name, "source": str(root), "target": str(target), "tmp": str(tmp),
        "complete": audit.complete, "counts": audit.counts,
        "defects": [d.to_dict() for d in audit.defects][:40],
        "n_cells": len(audit.cells), "dry_run": bool(dry_run),
    }
    if not audit.complete:
        plan["refused"] = (
            "bundle is a DEFECT: missing cell files or manifest fields (see defects). "
            "Re-run the affected cells, or record the missing cells as Deferred with a reason; "
            "publishing a partial bundle as if it were complete is the failure this guard exists for"
        )
        return plan
    if not dry_run:
        tmp.mkdir(parents=True, exist_ok=True)
        for f in RUN_REQUIRED:
            if (root / f).is_file():
                shutil.copy2(root / f, tmp / f)
        for extra in ("REPORT.md", "report.json", "matrix.yaml", "capacity.json", "run_summary.json"):
            if (root / extra).is_file():
                shutil.copy2(root / extra, tmp / extra)
        runs_src = root / "runs" if (root / "runs").is_dir() else root
        (tmp / "runs").mkdir(parents=True, exist_ok=True)
        for d in sorted(p for p in runs_src.glob("*") if p.is_dir()):
            shutil.copytree(d, tmp / "runs" / d.name)
        # written before hashing so that the bundle's own cover page is covered by the
        # manifest: an unhashed file inside a "verified" bundle is exactly the hole this
        # module exists to close
        # drive_dir is the folder the bundle sits *in*: write_latest_bundle prints
        # ``cd {drive_dir}/{name}``, so passing the bundle path itself doubles it.
        write_latest_bundle(tmp, name=name, source=str(root), drive_dir=str(Path(dest)),
                            notes=notes or [])
        files = sorted(p for p in tmp.rglob("*") if p.is_file() and p.name not in SKIP_FOR_HASHES)
        sums = [(p.relative_to(tmp).as_posix(), _sha(p), p.stat().st_size) for p in files]
        (tmp / "SHA256SUMS.txt").write_text(
            "".join(f"{h}  {rel}\n" for rel, h, _s in sums), encoding="utf-8")
        man = {
            "bundle": name, "created_utc": stamp, "source": str(root),
            "n_files": len(sums), "bytes": int(sum(s for _r, _h, s in sums)),
            "files": [{"path": r, "sha256": h, "bytes": s} for r, h, s in sums],
            "counts": audit.counts, "cell_audit": [c.to_dict() for c in audit.cells],
            "config_sha256": _read_first(root, "run_summary.json", "hashes", "config_sha256"),
            "matrix_sha256": _read_first(root, "matrix.yaml", "matrix_sha256"),
            "verification": (
                f"cd {target} && sha256sum -c SHA256SUMS.txt"),
            "note": ("complete-or-defect: this manifest exists only because every required cell file "
                     "and manifest field was present at bundle time"),
        }
        (tmp / "BUNDLE_MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        tmp.replace(target)
        plan["manifest"] = str(target / "BUNDLE_MANIFEST.json")
        plan["n_files"] = len(sums)
        plan["bytes"] = man["bytes"]
    return plan


def _read_first(root: Path, name: str, *keys):
    p = Path(root) / name
    if not p.is_file():
        return None
    try:
        if p.suffix == ".json":
            d = json.loads(p.read_text())
        else:
            import yaml
            d = yaml.safe_load(p.read_text()) or {}
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d
    except Exception:  # noqa: BLE001 - a missing provenance field is reported, not fatal
        return None


def verify_bundle(bundle: Path) -> dict:
    """Re-hash every file against the manifest and report the differences."""
    bundle = Path(bundle)
    man_p = bundle / "BUNDLE_MANIFEST.json"
    sums_p = bundle / "SHA256SUMS.txt"
    if not man_p.is_file() and not sums_p.is_file():
        return {"ok": False, "reason": "no BUNDLE_MANIFEST.json or SHA256SUMS.txt — this is not a bundle"}
    listed: dict[str, str] = {}
    meta = None
    if man_p.is_file():
        meta = json.loads(man_p.read_text())
        listed = {f["path"]: f["sha256"] for f in meta.get("files", [])}
    if sums_p.is_file():
        for line in sums_p.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                listed[parts[1].lstrip(" *")] = parts[0]
    have = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
            if p.is_file() and p.name not in SKIP_FOR_HASHES}
    missing = sorted(set(listed) - have)
    extra = sorted(have - set(listed))
    corrupt = [rel for rel in sorted(set(listed) & have) if _sha(bundle / rel) != listed[rel]]
    audit = audit_bundle(bundle)
    return {"ok": not (missing or corrupt) and audit.complete, "bundle": bundle.name,
            "n_listed": len(listed), "n_present": len(have), "missing": missing, "extra": extra,
            "corrupt": corrupt, "counts": audit.counts, "cell_defects": [d.to_dict() for d in audit.defects],
            "manifest_created": (meta or {}).get("created_utc"),
            "note": ("`extra` files are listed rather than deleted: an unlisted file in a verified "
                     "bundle is either a stray or a sign the manifest is stale, and both need a human")}


def write_latest_bundle(dest: Path, *, name: str, source: str, drive_dir: str = "thesis_colab",
                        notes: list[str] | None = None) -> Path:
    """LATEST_BUNDLE.md — the one file that says which bundle is current and how to check it."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Latest benchmark bundle", "",
        f"- bundle: `{name}`", f"- produced from: `{source}`",
        f"- created: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", "",
        "## Verify before use", "", "```bash", f"cd {drive_dir}/{name}",
        "sha256sum -c SHA256SUMS.txt          # any FAILED line means re-download, not re-analyse",
        "python -m ssl_anomaly.benchmarking.bundle .   # cell-level completeness audit", "```", "",
        "## What is in it", "",
        "- `runs/<experiment>/run_manifest.json` — factors, data hashes, objective, capacity, "
        "calibration, collapse verdict, code SHA",
        "- `runs/<experiment>/bin_scores.npz` — per-bin scores, per-channel error + mask, labels, field ids",
        "- `runs/<experiment>/norm_stats.json` — the robust scale the per-channel errors are in "
        "(`patch_sigma.npz` is the same idea; without it a re-analysis is guessing)",
        "- `runs/<experiment>/progress.jsonl`, `metrics.json`, `collapse.json`, `last.pt`, `meta.json`",
        "- `matrix.yaml`, `registry.json`, `registry.log.jsonl`, `coverage.md`, `capacity.json`, "
        "`bindings.json`, `run_summary.json`, `REPORT.md`", "",
        "## Reading rules", "",
        "- A cell absent from `coverage.md` was not run. Do not describe the set as complete.",
        "- A `SMOKE FIXTURE` stamp in any manifest means the numbers are plumbing proof, not results.",
        "- Deferred/Not Applicable rows carry the reason they were excluded; quote the reason, "
        "not an inference.", "",
    ]
    if notes:
        lines += ["## Notes from the run", ""] + [f"- {n}" for n in notes] + [""]
    p = dest / "LATEST_BUNDLE.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def drive_plan(out_dir: Path, *, zip_name: str | None = None) -> dict:
    """The exact commands to zip and upload, in order. Nothing here runs them."""
    out_dir = Path(out_dir)
    zip_name = zip_name or f"{out_dir.name}.zip"
    return {
        "drive_directory": "thesis_colab",
        "steps": [
            f"cd {out_dir.parent}",
            f"zip -r -q {zip_name} {out_dir.name} -x '*/__pycache__/*'",
            f"sha256sum {zip_name} > {zip_name}.sha256",
            f"mv {zip_name} {zip_name}.sha256 ~/GoogleDrive/MyDrive/thesis_colab/",
        ],
        "after_upload": [
            "In Colab: mount Drive, unzip into /content/bench, run "
            "`python -m ssl_anomaly.benchmarking.bundle /content/bench` and paste the JSON output "
            "into the run notes before any figure is drawn.",
        ],
        "note": ("the zip is written next to the run directory and moved, never edited in place: "
                 "an in-place update is how a half-uploaded bundle gets cited"),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Audit, build or verify a benchmark result bundle.")
    ap.add_argument("path", help="run directory (audit/collect) or bundle directory (verify)")
    ap.add_argument("--collect", metavar="DEST", help="assemble a verified bundle under DEST")
    ap.add_argument("--name", default=None)
    ap.add_argument("--verify", action="store_true", help="re-hash every file against the manifest")
    ap.add_argument("--drive-plan", action="store_true", help="print the upload commands")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        res = verify_bundle(Path(a.path))
    elif a.collect:
        res = collect(Path(a.path), Path(a.collect), name=a.name)
        if res.get("complete") and not a.json:
            write_latest_bundle(Path(a.collect), name=res["bundle"], source=res["source"])
    elif a.drive_plan:
        res = drive_plan(Path(a.path), zip_name=a.name)
    else:
        res = audit_bundle(Path(a.path)).to_dict()
    print(json.dumps(res, indent=1, sort_keys=True, default=str))
    if isinstance(res, dict):
        if "complete" in res and not res["complete"]:
            return 2
        if "ok" in res and not res["ok"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
