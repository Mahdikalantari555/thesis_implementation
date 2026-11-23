"""Shared plumbing for ``scripts/evaluation/*``.

Every capstone script has the same skeleton — parse args, choose smoke vs real,
load data or a fixture, run the module, write the artifact, self-check — and the
same three ways to go wrong:

1. writing synthetic smoke numbers into ``reports/regional_representation/``
   where a reader would take them for evidence;
2. silently substituting a fixture when the real data is missing;
3. reporting success when a stage produced only ``absent`` results.

The helpers here make those mistakes hard: :func:`resolve_run` owns the path and
data decision in one place, and :func:`finish_smoke` refuses to exit 0 unless the
script's own checks passed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def add_common_args(parser) -> None:
    """The flags every evaluation script accepts."""
    parser.add_argument(
        "--smoke", action="store_true",
        help="run on the synthetic fixture; writes to reports/_smoke/ and is never evidence",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output directory (default: reports/regional_representation, or the "
             "smoke tree under reports/_smoke/)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[13, 42, 2026],
        help="seeds; the pre-registered set is 13 42 2026",
    )


@dataclass
class RunContext:
    """Resolved run: where to write, what data to use, and whether it counts."""

    smoke: bool
    paths: object                 # ssl_anomaly.evaluation.runner.RunPaths
    tag: str

    @property
    def is_evidence(self) -> bool:
        """Only a non-smoke run can produce a quotable number."""
        return not self.smoke


def resolve_run(args, *, stage: str, default_tag: str | None = None):
    """Decide smoke-vs-real and the output directory from parsed args.

    A ``--smoke`` run ALWAYS writes under the smoke tree, even if ``--out`` points
    somewhere else: the flag is a promise that the output is not evidence, and a
    custom ``--out`` must not be able to break that promise.
    """
    from ssl_anomaly.evaluation.runner import RunPaths

    if args.smoke:
        tag = default_tag or f"smoke-{stage}"
        if args.out is not None:
            out = Path(args.out)
            if "reports/_smoke" not in str(out).replace("\\", "/"):
                print(
                    f"note: --smoke overrides --out; writing to the smoke tree instead "
                    f"of {out} so synthetic numbers cannot land in reports/",
                    file=sys.stderr,
                )
            else:
                return RunContext(smoke=True, paths=RunPaths(out, smoke=True), tag=tag)
        return RunContext(smoke=True, paths=RunPaths.for_run(tag), tag=tag)
    return RunContext(smoke=False, paths=RunPaths(args.out) if args.out else RunPaths.for_run(),
                      tag="real")


def require_real_data(path: Path, what: str) -> Path:
    """Fail loudly with the owner's next step, rather than substituting a fixture."""
    if not path.exists():
        raise SystemExit(
            f"missing {what}: {path}\n"
            f"Real runs need the frozen v2 chain (see AGENTS.md §6.2 and "
            f"openspec/changes/regional-ssl-representation-evaluation/design.md).\n"
            f"Run with --smoke to exercise the pipeline on the synthetic fixture.\n"
            f"This is a data-availability stop, not a code defect."
        )
    return path


def finish_smoke(expectation, paths, stage: str) -> int:
    """Print the smoke report and return the process exit code."""
    checked = expectation.to_dict()
    for check in checked["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        detail = f"  [{check['detail']}]" if check["detail"] else ""
        print(f"  {mark}  {check['label']}{detail}")
    if not checked["ok"]:
        print(f"SMOKE FAILED for {stage}", file=sys.stderr)
        return 1
    print(f"smoke ok for {stage} — wrote {paths.out_dir}")
    return 0
