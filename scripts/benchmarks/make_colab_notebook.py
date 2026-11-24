"""Generate the Colab benchmark notebook (tasks 13.1–13.5) — do not hand-edit the .ipynb.

The notebook is *generated* because its job is to be a faithful index of the CLI: a
hand-copied notebook drifts from the scripts within a week, and the drift is invisible
until a run fails on Colab at 03:00 with a flag that no longer exists. Rebuild it after
any change to ``scripts/benchmarks/*``:

    python scripts/benchmarks/make_colab_notebook.py            # writes the .ipynb
    python scripts/benchmarks/make_colab_notebook.py --check    # fail if it is stale

The section list is the one the change's design fixes (§The Colab notebook); each code
cell calls an implemented module or script, so no training logic is duplicated here
("imports implemented modules only (duplication prohibited)").

The generator parses every code cell before writing the notebook: a SyntaxError in a
generated cell is otherwise invisible until it is run on rented hardware.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "benchmarks" / "architecture_comparison_benchmark.ipynb"
REPO = "Mahdikalantari555/thesis_implementation"
REPO_DIR = "/content/thesis_implementation"
BENCH = "/content/bench"          # fixture/smoke output
RUN = "/content/drive/MyDrive/thesis_colab/bench_full"   # real run output, straight to Drive

MD, CODE = "markdown", "code"


def _cells() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    md = lambda t: out.append((MD, t.strip("\n")))                                   # noqa: E731
    code = lambda t: out.append((CODE, t.strip("\n")))                               # noqa: E731

    md(f"""
# مقایسه معماری‌ها — بنچمارک latent / reconstruction / prediction (Colab T4)

تولیدشده توسط `scripts/benchmarks/make_colab_notebook.py` — **دستی ویرایش نکنید**.
بعد از هر تغییر در `scripts/benchmarks/*`:
`python scripts/benchmarks/make_colab_notebook.py`

**پیش‌نیازها**
1. Runtime → Change runtime type → **T4 GPU**
2. ریپو `{REPO}` public و push شده باشد
3. در Drive: `MyDrive/thesis_colab/thesis_data_full_v2.zip` (~۲۹۴ MB) — همان زیپ، منبع
   رسمی v2؛ نسخهٔ قدیمی/جایگزین خوانده نمی‌شود.

**دو حالت، صریح جدا**
- `--fixture --smoke` (بخش ۹): فقط لوله‌بندی. هر manifest برچسب
  `SMOKE FIXTURE — plumbing evidence only, not a scientific result` می‌گیرد و هیچ ادعایی
  از آن استخراج نمی‌شود.
- حالت واقعی (بخش ۱۰): دیتای v2 روی T4. تخمین زمان از پروفایلِ خودِ اجرا می‌آید؛ سقف
  `budget.t4_hours` است و وقتی تمام شود سلول‌های مانده `Deferred` با ذکر دلیل می‌شوند.

هر منطقی که اینجا دیده می‌شود در `src/ssl_anomaly/benchmarking/*` یا
`scripts/benchmarks/*` زندگی می‌کند؛ نوتبوک فقط صدا می‌زند.
""")

    # 1 ------------------------------------------------------------------
    md("## ۱. محیط — وابستگی‌ها، clone، و مسیرهای کاری")
    code(f"""
import os, sys, json, subprocess
import pathlib

REPO_URL = "https://github.com/{REPO}.git"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy", "scipy", "pandas", "scikit-learn", "pyyaml", "pyarrow"], check=True)
if not pathlib.Path("{REPO_DIR}/.git").is_dir():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "{REPO_DIR}"], check=True)
os.chdir("{REPO_DIR}")
sys.path.insert(0, "{REPO_DIR}/src")
HEAD = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                      text=True).stdout.strip()
BENCH, RUN = "{BENCH}", "{RUN}"
pathlib.Path(BENCH).mkdir(parents=True, exist_ok=True)
pathlib.Path(RUN).mkdir(parents=True, exist_ok=True)
print("HEAD:", HEAD, "| cwd:", os.getcwd())
assert HEAD, "clone failed"
""")

    # 2 ------------------------------------------------------------------
    md("## ۲. سخت‌افزار — device و precision plan")
    code("""
from ssl_anomaly.benchmarking import environment as env
from ssl_anomaly.benchmarking.config import load_benchmark_config

cfg = load_benchmark_config()
dev = env.detect_device(cfg, require=False).to_dict()
print(json.dumps(dev, indent=1, sort_keys=True, default=str))
GPU = bool(dev.get("cuda_available"))
if not GPU:
    print("GPU نیست — فقط بخش ۹ (fixture/smoke) مجاز است؛ بخش ۱۰ را اجرا نکنید.")
""")

    # 3 ------------------------------------------------------------------
    md("## ۳. دیتا — استخراج v2 و pin کردن هشِ همان فایل‌هایی که خوانده می‌شوند")
    code("""
import pathlib, shutil, zipfile
from google.colab import drive

drive.mount("/content/drive")
ZIP = pathlib.Path("/content/drive/MyDrive/thesis_colab/thesis_data_full_v2.zip")
assert ZIP.is_file(), f"zip not found at {ZIP} (put it in MyDrive/thesis_colab/)"
shutil.copy(ZIP, "/content/data_v2.zip")
with zipfile.ZipFile("/content/data_v2.zip") as z:
    names = z.namelist()
    z.extractall(REPO_DIR + "/")
print("extracted entries:", len(names))

# هش‌ها همین‌جا محاسبه و pin می‌شوند — قرارداد مقدار از پیش‌نوشته‌شده را قبول نمی‌کند.
from ssl_anomaly.benchmarking import bindings as bind

rep = bind.verify_bindings(cfg, base=pathlib.Path(REPO_DIR), matrix_sha256=None,
                           applicability_sha256=None, compute_hashes=True,
                           strict=False).to_dict()
print(json.dumps({k: rep[k] for k in rep if k in ("ok", "blocking", "warnings", "artifacts")},
                 indent=1, sort_keys=True, default=str)[:2500])
assert not rep.get("blocking"), "artifact bindings are blocking — fix before any training"
""")

    # 4 ------------------------------------------------------------------
    md("## ۴. پیکربندی — قرارداد منجمد و ماتریس تعهدشده")
    code("""
from ssl_anomaly.benchmarking import matrix as M
from ssl_anomaly.benchmarking.config import applicability_sha256, load_applicability, validate_config

app = load_applicability()
print("config sha:", cfg.document_sha256[:16], "| applicability sha:", applicability_sha256()[:16])
problems = validate_config(cfg.raw)
assert problems == [], problems
cells = M.build_matrix(cfg, app)
v = M.verify_committed_matrix(cfg, app)
print("committed matrix ok:", v["ok"], "| cells:", len(cells), "|", M.status_counts(cells))
assert v["ok"], v["problems"]
""")

    # 5 ------------------------------------------------------------------
    md("## ۵. split — فقط split‌های مجاز؛ test/monitoring هرگز")
    code("""
from ssl_anomaly.benchmarking.splits import TestSplitAccessError, assert_admissible

ev = cfg.raw["evaluation"]
print("allowed:", ev["allowed_splits"], "| forbidden:", ev["forbidden_splits"])
assert_admissible(tuple(ev["allowed_splits"]))
try:
    assert_admissible(("train", "test"))
    raise SystemExit("LEAK GUARD BROKEN: a forbidden split was admitted")
except TestSplitAccessError as exc:
    print("guard holds:", str(exc)[:180])
""")

    # 6 ------------------------------------------------------------------
    md("## ۶. ناهنجاری مصنوعی — پنل ارزیابی و آزمون واقع‌نمایی")
    code(f"""
import numpy as np
from ssl_anomaly.benchmarking import synthetic as syn
from ssl_anomaly.benchmarking.runner import load_clean_data

frame = load_clean_data(cfg, base=pathlib.Path(REPO_DIR), fixture=True)
panel = syn.apply_panel(frame, cfg, seed=int(cfg.raw["synthetic"]["eval_generator"]["seed"]),
                        pool=str(cfg.raw["synthetic"]["eval_generator"].get("pool",
                                                                            "dev_cycles_only")),
                        role="evaluation")
print(json.dumps(syn.audit_realism(panel, cfg), indent=1, sort_keys=True,
                 default=str)[:1800])
""")

    # 7 ------------------------------------------------------------------
    md("## ۷. ظرفیت و objective — باند ±۲۰٪ و ممیزی تفکیک هدف‌ها")
    code("""
from ssl_anomaly.benchmarking.capacity import capacity_table, resolve_capacities
from ssl_anomaly.benchmarking.interface import assert_objectives_distinct

cap = resolve_capacities(cfg, d_channels=int(cfg.raw["training"]["channels"]), d_cov=10)
print("all in band:", cap["all_in_band"], "| spread:", cap["spread"]["max_over_min"])
print(capacity_table(cap))
assert cap["all_in_band"], {k: a["note"] for k, a in cap["arms"].items() if not a["in_band"]}
""")

    # 8 ------------------------------------------------------------------
    md("""## ۸. جستجوی هایپرپارامتر — بودجهٔ برابر، انتخاب فقط روی validation

`d_model` در این جستجو **نیست**: باند ظرفیت را `capacity.tuning` انتخاب می‌کند و اگر HPO
هم همان را تکان می‌داد، برنده می‌توانست فقط «بزرگ‌تر بودن» باشد. برای دیتای واقعی
`--fixture` را بردارید.
""")
    code("""
%%time
r = subprocess.run([sys.executable, "scripts/benchmarks/run_hpo.py", "--trials", "2",
                    "--epochs", "1", "--fixture", "--smoke", "--out", "/content/hpo"],
                   capture_output=True, text=True)
print((r.stdout or "")[-1500:] or (r.stderr or "")[-1500:])
print("rc:", r.returncode, "— selection در /content/hpo/hpo_selection.json؛ پیکربندی برنده با "
      "--selection-file به اجرا اضافه می‌شود و قرارداد را بازنویسی نمی‌کند")
""")

    # 9 ------------------------------------------------------------------
    md("## ۹. غربالگری — fixture/smoke: لوله‌بندی، نه نتیجه")
    code("""
%%time
r = subprocess.run([sys.executable, "scripts/benchmarks/run_benchmark.py", "--execute", "--yes",
                    "--fixture", "--smoke", "--tier", "screening", "--block", "screening",
                    "--out", BENCH], capture_output=True, text=True)
print((r.stdout or "")[-2500:] or (r.stderr or "")[-2500:])
assert r.returncode == 0, f"smoke run failed rc={r.returncode}"
""")

    # 10 -----------------------------------------------------------------
    md("""## ۱۰. بنچمارک کامل — واقعی، resumeدار، با سقف T4

رسمی‌ترین سلول نوتبوک. پیش‌شرط‌ها: سلول ۳ هش‌ها را pin کرده باشد و سلول ۴
`committed matrix ok: True` داده باشد. خروجی مستقیم در Drive نوشته می‌شود تا با قطع
session از دست نرود.
""")
    code("""
%%time
cmd = [sys.executable, "scripts/benchmarks/run_benchmark.py", "--execute", "--yes",
       "--tier", "full", "--block", "full_benchmark", "--seeds", "13,42,2026",
       "--data-root", REPO_DIR, "--out", RUN, "--resume"]
if pathlib.Path("/content/hpo/hpo_selection.json").is_file():
    cmd += ["--selection-file", "/content/hpo/hpo_selection.json"]
print(" ", " ".join(cmd[1:]), flush=True)
with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
print("rc:", rc)
""")

    # 11 -----------------------------------------------------------------
    md("## ۱۱. checkpoint‌ها — هر run dir چه باید داشته باشد")
    code("""
from ssl_anomaly.benchmarking import checkpoints as ck

def audit_run_dir_tree(root):
    rows = []
    for d in sorted(pathlib.Path(root).glob("runs/*")):
        if not d.is_dir():
            continue
        a = ck.audit_run_dir(d)
        rows.append({"dir": d.name, "ok": a["ok"], "missing": a["missing"],
                     "resume_point": ck.has_resume_point(d)})
    return rows

for label, root in (("smoke", BENCH), ("full", RUN)):
    rows = audit_run_dir_tree(root)
    bad = [r for r in rows if not r["ok"]]
    print(label, "dirs:", len(rows), "| incomplete:", len(bad))
    for r in bad[:5]:
        print("   ", r)
print("قاعده: bundle کامل است یا نقص — نقص را ترمیم کن یا Deferred کن؛ آپلود به‌عنوان کامل ممنوع")
""")

    # 12 -----------------------------------------------------------------
    md("## ۱۲. ادامه بعد از قطع — resume با تطبیق bindings")
    code("""
r = subprocess.run([sys.executable, "scripts/benchmarks/run_benchmark.py", "--execute", "--yes",
                    "--fixture", "--smoke", "--tier", "screening", "--block", "screening",
                    "--out", BENCH, "--resume"], capture_output=True, text=True)
print((r.stdout or "")[-1500:] or (r.stderr or "")[-1500:])
print("resume فقط با hash یکسان مجاز است؛ اختلاف ⇒ CheckpointError که هر دو مقدار را چاپ می‌کند")
""")

    # 13 -----------------------------------------------------------------
    md("## ۱۳. ارزیابی روی پنل مصنوعی — family × severity، نه یک عدد تکی")
    code("""
rows = []
for d in sorted(pathlib.Path(BENCH).glob("runs/*"))[:8]:
    man = json.loads((d / "run_manifest.json").read_text())
    macro = (man.get("metrics") or {}).get("macro") or {}
    rows.append({"exp": man["experiment_id"], "rung": man.get("ladder_rung"),
                 "stamp": (man.get("stamp") or "")[:22], "macro_auprc": macro.get("macro"),
                 "n_cells": macro.get("n_cells"), "cells_missing": macro.get("cells_missing")})
print(json.dumps(rows, indent=1)[:2500])
""")

    # 14 -----------------------------------------------------------------
    md("## ۱۴. شاخص‌های محیطی — مرجع، نه ground truth")
    code("""
from ssl_anomaly.benchmarking import indices

n = int(frame.n)
fit_idx, scored = np.arange(0, n * 3 // 4), np.arange(n * 3 // 4, n)
scores = indices.compute_all(frame.x, frame.channel_names, weather=None,
                             fit_idx=fit_idx, scored_idx=scored)
for name, idx in scores.items():
    print(json.dumps({"index": name, "available": idx.available, "missing": list(idx.missing),
                      "note": (idx.note or "")[:120]}, sort_keys=True))
print("CWSI_ET عمداً unavailable است: آرشیو جملهٔ مقاومت آیرودینامیکی را ندارد و جایگزینی‌اش "
      "با CWSI_LST یک شاخص دما را با نام ET معرفی می‌کرد")
""")

    # 15 -----------------------------------------------------------------
    md("## ۱۵. تعمیم — V1 زمانی، V2 مزرعه‌ای، V3 فضایی-زمانی")
    code("""
from ssl_anomaly.benchmarking import protocols

print(json.dumps(protocols.audit_protocols(frame, cfg), indent=1, sort_keys=True,
                 default=str)[:2200])
""")

    # 16 -----------------------------------------------------------------
    md("""## ۱۶. تحلیل آماری — bootstrap خوشه‌ای روی مزرعه، Wilcoxon، Friedman/Nemenyi، Holm

تعداد بوت‌استراپ از کف از پیش ثبت‌شده کمتر نمی‌شود؛ اگر کمتر باشد جدول `—` چاپ می‌کند،
نه برچسب «معنادار».
""")
    code("""
r = subprocess.run([sys.executable, "scripts/benchmarks/report_benchmark.py", "--out", BENCH,
                    "--json"], capture_output=True, text=True)
print((r.stdout or "")[-1800:] or (r.stderr or "")[-1800:])
""")

    # 17 -----------------------------------------------------------------
    md("## ۱۷. پروفایل محاسباتی — ساعت T4، پارامتر، زمان inference")
    code("""
for root in (BENCH, RUN):
    p = pathlib.Path(root) / "run_summary.json"
    if not p.is_file():
        print(root, "— no run_summary.json yet")
        continue
    s = json.loads(p.read_text())
    print(root, json.dumps({"wall": s["run"]["wall_seconds"], "by_status": s["by_status"],
                            "budget": s["run"]["budget"]}, indent=1, sort_keys=True,
                           default=str)[:900])
""")

    # 18 -----------------------------------------------------------------
    md("## ۱۸. سه گزارش برنده — detection / generalization / Pareto")
    code("""
txt = pathlib.Path(BENCH) / "REPORT.md"
if txt.is_file():
    body = txt.read_text()
    i = body.find("## 3.")
    print(body[i:i + 3000])
else:
    print("no REPORT.md yet — run section 16 first")
""")

    # 19 -----------------------------------------------------------------
    md("## ۱۹. گزارش نهایی + evidence pack (CSV/JSON) در Drive")
    code("""
r = subprocess.run([sys.executable, "scripts/benchmarks/report_benchmark.py", "--out", RUN,
                    "--copy-to", RUN + "/report"], capture_output=True, text=True)
print((r.stdout or "")[-1200:] or (r.stderr or "")[-1200:])
""")

    # 20 -----------------------------------------------------------------
    md("""## ۲۰. خروجی — bundle کامل‌یا‌نقص‌دار با SHA256SUMS

`--collect` اول audit می‌کند و در صورت نقص **refuse** می‌کند (exit 2). بعد از آن تنها
راه استفاده، `sha256sum -c SHA256SUMS.txt` است.
""")
    code("""
r = subprocess.run([sys.executable, "scripts/benchmarks/bundle_benchmark.py", RUN, "--collect",
                    "--dest", "/content/drive/MyDrive/thesis_colab/bundles"],
                   capture_output=True, text=True)
print((r.stdout or "")[-2200:] or (r.stderr or "")[-2200:])
assert r.returncode == 0, "bundle is a DEFECT — repair the cells or record them as Deferred"
""")
    return out


def build() -> dict:
    cells = []
    for kind, src in _cells():
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"},
                         "accelerator": "GPU",
                         "generated_by": "scripts/benchmarks/make_colab_notebook.py",
                         "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
            "nbformat": 4, "nbformat_minor": 0}


def strip_stamp(text: str) -> str:
    """Drop the generation timestamp, the only intended difference between two builds."""
    return "\n".join(line for line in text.splitlines() if "generated_utc" not in line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="fail if the committed notebook is stale")
    a = ap.parse_args(argv)
    nb = build()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        body = "".join(cell["source"])
        if body.lstrip().startswith("%%"):
            body = body.lstrip().split("\n", 1)[1]
        try:
            ast.parse(body)
        except SyntaxError as exc:
            print(f"cell {i} does not parse: {exc}\n{body[:400]}", file=sys.stderr)
            return 2
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    p = Path(a.out)
    if a.check:
        if not p.is_file():
            print(f"{p} does not exist", file=sys.stderr)
            return 2
        same = strip_stamp(p.read_text()) == strip_stamp(text)
        print("notebook is current" if same else f"{p} is STALE: re-run the generator",
              file=sys.stderr)
        return 0 if same else 2
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"wrote {p} ({len(nb['cells'])} cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} "
          f"code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
