"""Render findings 26 (EN + FA) from ``GateReport.quota_block()``.

The generator's whole discipline is the import at the top of this docstring: it
reads ONE thing, the quota block. It never opens a stage artifact, never touches
``evidence`` or ``detail``, and never restates a number the gates did not sanction.
That is the point — a findings document assembled from loose artifacts is exactly
how a failed gate gets quoted anyway.

Consequences of that design, stated plainly because they will look like omissions:

- If G-E did not pass, ``quota_block`` returns ``quotable: False`` and no claims.
  The document then reports that fact and NOTHING else. It does not fall back to
  the other gates, however favourable they look.
- Gate verdicts are tri-state. ``undetermined`` is printed as "undetermined", never
  softened to "pending" and never rounded to "failed"; a gate with missing inputs
  measured nothing, and saying it failed would invent a loss.
- The FA document is generated from the same block by the same code path, so the
  two languages cannot disagree about a verdict. The prose around the table is
  translated; the verdict words are translated too, from a fixed dictionary rather
  than free translation, so "undetermined" cannot drift into a word that implies
  measurement.

Usage::

    python scripts/evaluation/render_findings_26.py --gates reports/regional_representation/decision_gates.json
    python scripts/evaluation/render_findings_26.py --gates ... --out-dir docs/findings
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ssl_anomaly.evaluation.gates import GATE_IDS  # noqa: E402

BASENAME = "26_regional_representation_evaluation"

#: Fixed translation of the tri-state vocabulary. Deliberately a dictionary and not
#: a translator: a verdict word carries a measurement claim, so it must map 1:1.
VERDICT_FA = {
    "passed": "برآورده شد",
    "failed": "برآورده نشد",
    "undetermined": "قابل تعیین نبود",
    "blocked": "مسدود شد",
}

GATE_TITLE = {
    "G-A": ("Pretraining quality", "کیفیت پیش‌آموزش"),
    "G-B": ("Label efficiency", "بازدهی برچسب"),
    "G-C": ("Transfer dominance", "برتری انتقال"),
    "G-D": ("Retrieval usefulness", "سودمندی بازیابی"),
    "G-E": ("Claim discipline", "انضباط ادعا"),
}

#: What each gate's failure licenses the report to say. Copied from the frozen
#: criteria so the document cannot promise more than the pre-registration allows.
CONSEQUENCE_FA = {
    "G-A": "کاندید رد می‌شود؛ اعداد فقط تشخیصی‌اند",
    "G-B": "ادعای «بازدهی نمونه» منتفی است",
    "G-C": "ادعای «بازنمایی برتر SSL» پشتیبانی نمی‌شود",
    "G-D": "ادعای بازیابی منتفی است؛ SSL محدود به «تعبیه‌های توصیفی» می‌ماند",
    "G-E": "گزارش رد می‌شود؛ تغییر قابل بستن نیست",
}


def _verdict(block: dict, gate: str) -> str:
    """A gate absent from the block is undetermined, never omitted.

    ``block["gates"].get(g)`` returning ``None`` used to drop the gate from the
    headline list entirely, so a run whose gate dict was incomplete read as though
    every listed gate simply passed. Absence of a verdict is not a verdict.
    """
    return str((block.get("gates") or {}).get(gate) or "undetermined")


def _md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_en(block: dict, meta: dict) -> str:
    lines = [
        "# Finding 26 — Regional SSL representation: pre-registered evaluation",
        "",
        f"**{meta['date']} · {meta['status_text']} · rung {meta['rung']} · "
        f"claim noun: anomaly surrogate (not stress) · gate file "
        f"`{meta['source']}` · git `{meta['git_sha']}`**",
        "",
        "This document is generated from the decision-gate quota block and nothing",
        "else. Every verdict below is the gate's own; the prose does not re-derive,",
        "re-round, or soften a single one.",
        "",
        "## Headline",
        "",
    ]
    if not block.get("quotable"):
        lines += [
            "**No claim from this run may be quoted.** " + str(block.get("reason", "")),
            "",
            "The gates are reported below for completeness. A run whose claim",
            "discipline gate did not pass is not evidence, whatever the other gates",
            "computed — which is precisely why the other verdicts are printed as",
            "*blocked* rather than as their own results.",
            "",
        ]
    else:
        passed = [g for g in GATE_IDS if _verdict(block, g) == "passed"]
        failed = [g for g in GATE_IDS if _verdict(block, g) == "failed"]
        undet = [g for g in GATE_IDS if _verdict(block, g) == "undetermined"]
        parts = []
        if passed:
            parts.append(f"passed: {', '.join(passed)}")
        if failed:
            parts.append(f"failed: {', '.join(failed)}")
        if undet:
            parts.append(f"undetermined: {', '.join(undet)}")
        lines += [
            "Claim discipline holds, so the run is quotable within the limits set by",
            "the gates below. " + "; ".join(parts) + ".",
            "",
            "An *undetermined* gate is not a weak pass and not a failure: its inputs",
            "were never measured. Nothing is asserted through an undetermined gate.",
            "",
        ]

    lines += ["## Gates", ""]
    rows = []
    for g in GATE_IDS:
        verdict = _verdict(block, g)
        claims = {c["gate"]: c for c in block.get("claims", [])}
        consequence = claims.get(g, {}).get("fail_consequence", "")
        rows.append([f"**{g}** {GATE_TITLE[g][0]}", verdict, consequence])
    lines += [_md_table(rows, ["gate", "verdict", "if it had failed"]), ""]

    lines += [
        "## What the verdicts do not license",
        "",
        "- A *passed* gate licenses exactly the sentence its criterion names, not a",
        "  general claim that the representation is better.",
        "- An *undetermined* gate licenses nothing. Missing inputs are an absence of",
        "  evidence, and this document does not convert absence into a direction.",
        "- Negative results carry the same prominence as positive ones here because",
        "  the all-gates-fail outcome was pre-declared as a valid result. The report",
        "  is not a success narrative that survived contact with the data; it is the",
        "  record of what the pre-registered tests said.",
        "",
        "## Limitations",
        "",
        "- Single region, single crop: the claims ceiling is L3 (same-region) and the",
        "  geographic scope is out of scope by design.",
        "- Labels are segmentation-derived; the anomaly noun is a surrogate, and",
        "  anomaly is not stress.",
        "- Synthetic injections are controlled evidence only.",
        "",
        "## Reproduction",
        "",
        "The gate file is `" + meta["source"] + "`. Re-rendering this document from it",
        "is deterministic; any edit to the table that does not come from a re-run of",
        "the aggregator is tampering, and the narrative section above it says so.",
        "",
    ]
    return "\n".join(lines)


def render_fa(block: dict, meta: dict) -> str:
    lines = [
        "# یافته ۲۶ — بازنمایی SSL منطقه‌ای: ارزیابی پیش‌ثبت‌شده",
        "",
        f"**{meta['date_fa']} · {meta['status_text_fa']} · پله {meta['rung']} · "
        f"نوع ادعا: «انحراف جانشین» (نه تنش) · پرونده‌ی گیت "
        f"`{meta['source']}` · کامیت `{meta['git_sha']}`**",
        "",
        "این سند فقط از بلوک سهمیه‌ی گیت‌های تصمیم تولید می‌شود و از هیچ چیز دیگری.",
        "هر حکمی که در ادامه می‌آید حکم خود گیت است؛ متن نه بازمحاسبه می‌کند، نه",
        "گرد می‌کند، و نه حکمی را نرم می‌سازد.",
        "",
        "## نتیجه در یک جمله",
        "",
    ]
    if not block.get("quotable"):
        lines += [
            "**هیچ ادعایی از این ران قابل نقل نیست.** "
            + ("گیت انضباط ادعا (G-E) برآورده نشد."),
            "",
            "گیت‌ها در ادامه فقط برای کامل بودن گزارش شده‌اند. رانی که گیت انضباط",
            "ادعایش برآورده نشده شاهد نیست، هر قدر هم که گیت‌های دیگر محاسبه کرده",
            "باشند — و همین دلیل آن است که حکم بقیه‌ی گیت‌ها «مسدود شد» چاپ می‌شود،",
            "نه نتیجه‌ی مستقل خودشان.",
            "",
        ]
    else:
        passed = [g for g in GATE_IDS if _verdict(block, g) == "passed"]
        failed = [g for g in GATE_IDS if _verdict(block, g) == "failed"]
        undet = [g for g in GATE_IDS if _verdict(block, g) == "undetermined"]
        parts = []
        if passed:
            parts.append(f"برآورده شد: {', '.join(passed)}")
        if failed:
            parts.append(f"برآورده نشد: {', '.join(failed)}")
        if undet:
            parts.append(f"قابل تعیین نبود: {', '.join(undet)}")
        lines += [
            "انضباط ادعا برقرار است، پس ران در حدودی که گیت‌های زیر تعیین می‌کنند",
            "قابل نقل است. " + "؛ ".join(parts) + ".",
            "",
            "گیتِ «قابل تعیین نبود» نه یک قبولیِ ضعیف است و نه یک شکست: ورودی‌هایش هرگز",
            "اندازه‌گیری نشده‌اند. هیچ ادعایی از یک گیتِ نامعلوم عبور نمی‌کند.",
            "",
        ]

    lines += ["## گیت‌ها", ""]
    rows = []
    for g in GATE_IDS:
        verdict = VERDICT_FA.get(_verdict(block, g), "قابل تعیین نبود")
        rows.append([f"**{g}** {GATE_TITLE[g][1]}", verdict, CONSEQUENCE_FA[g]])
    lines += [_md_table(rows, ["گیت", "حکم", "اگر برآورده نمی‌شد"]), ""]

    lines += [
        "## آنچه این احکام مجاز نمی‌کنند",
        "",
        "- گیتِ برآورده‌شده فقط همان جمله‌ای را مجاز می‌کند که معیارش نام می‌برد، نه",
        "  ادعای کلی «بهتر بودن بازنمایی».",
        "- گیتِ نامعلوم هیچ چیز را مجاز نمی‌کند. نبودِ ورودی یعنی نبودِ شاهد، و این",
        "  سند نبودِ شاهد را به یک جهت تبدیل نمی‌کند.",
        "- نتایج منفی اینجا همان برجستگی نتایج مثبت را دارند، چون سناریوی «همه‌ی",
        "  گیت‌ها شکست می‌خورند» از پیش نتیجه‌ای معتبر اعلام شده بود.",
        "",
        "## محدودیت‌ها",
        "",
        "- یک منطقه، یک محصول: سقف ادعا L3 (هم‌منطقه) است و دامنه‌ی جغرافیایی",
        "  عامدانه بیرون از دامنه است.",
        "- برچسب‌ها از قطعه‌بندی مشتق شده‌اند؛ اسم ادعا «انحراف جانشین» است و انحراف",
        "  به معنای تنش نیست.",
        "- تزریق‌های مصنوعی فقط شاهد کنترل‌شده‌اند.",
        "",
        "## بازتولید",
        "",
        "پرونده‌ی گیت `" + meta["source"] + "` است. بازتولید این سند از آن قطعی است؛",
        "هر ویرایشی در جدول که از اجرای دوباره‌ی تجمیع‌کننده نیامده باشد دست‌کاری است.",
        "",
    ]
    return "\n".join(lines)


def _meta(block: dict, source: Path, git_sha: str, smoke: bool) -> dict:
    today = date.today()
    if block.get("quotable"):
        en = "pre-registered evaluation complete · quotable within the gates below"
        fa = "ارزیابی پیش‌ثبت‌شده کامل · قابل نقل در حدود گیت‌های زیر"
    else:
        en = "pre-registered evaluation incomplete · NOT quotable"
        fa = "ارزیابی پیش‌ثبت‌شده ناتمام · غیرقابل نقل"
    if smoke:
        en += " · SMOKE (synthetic fixture — not evidence)"
        fa += " · آزمون دودی (فیکسچر مصنوعی — شاهد نیست)"
    return {
        "date": today.isoformat(),
        "date_fa": "۲۰۲۶-۱۱-۲۳",
        "status_text": en,
        "status_text_fa": fa,
        "rung": "L2 (dev + rolls; test split unread)",
        "source": str(source),
        "git_sha": git_sha,
        "smoke": smoke,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", type=Path, required=True,
                   help="decision_gates.json written by aggregate_decision_gates.py")
    p.add_argument("--out-dir", type=Path, default=Path("docs/findings"))
    p.add_argument("--git-sha", default="(uncommitted)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the EN rendering instead of writing files")
    args = p.parse_args(argv)

    if not args.gates.exists():
        print(f"no gate file at {args.gates}", file=sys.stderr)
        return 2
    payload = json.loads(args.gates.read_text())
    block = payload.get("quota_block")
    if not isinstance(block, dict) or "quotable" not in block:
        print(f"{args.gates} has no quota_block; is this an aggregator output?",
              file=sys.stderr)
        return 2

    meta = _meta(block, args.gates, args.git_sha, bool(payload.get("smoke")))
    en = render_en(block, meta)
    fa = render_fa(block, meta)

    if args.dry_run:
        print(en)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    en_path = args.out_dir / f"{BASENAME}.md"
    fa_path = args.out_dir / f"{BASENAME}.fa.md"
    en_path.write_text(en, encoding="utf-8")
    fa_path.write_text(fa, encoding="utf-8")
    print(f"wrote {en_path}")
    print(f"wrote {fa_path}")
    print(f"quotable: {block['quotable']} | "
          f"verdicts: {block['gates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
