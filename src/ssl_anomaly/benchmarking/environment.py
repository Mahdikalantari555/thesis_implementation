"""Hardware reality: device, VRAM, precision (tasks 12.3, 12.4; spec T4 compatibility).

The spec scenario is explicit: *"WHEN no CUDA device is available THEN the
benchmark stops with a diagnostic before any training (classical baselines and
preprocessing MAY run on CPU, clearly labelled)"*. So this module answers three
questions and records the answers in every manifest:

1. What device is actually present? (never assumed, never "cuda if available"
   buried inside a training loop — the decision is made once, up front)
2. Is mixed precision numerically appropriate here? (bf16 on Ampere+, fp16 with
   GradScaler below that, fp32 otherwise; the reason is recorded, not inferred
   from a flag)
3. How much VRAM is in use, and when is the cycle over? (peak allocated for the
   cell + a running budget check)

``CpuNotAllowedError`` is the fail-closed path; ``allow_cpu`` is only ever set by
the runner for the CPU-permitted work the spec names (classical baselines,
preprocessing, unit tests), and the permission is written into the manifest so a
CPU-labelled number can never masquerade as a T4 number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CpuNotAllowedError(RuntimeError):
    """Training was requested on a host without the contracted accelerator."""


def torch_available() -> bool:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return True


@dataclass
class DeviceReport:
    """What the machine actually has, as measured at run start."""

    device: str = "cpu"
    accelerator_expected: str = ""
    cuda_available: bool = False
    device_name: str | None = None
    capability: str | None = None
    total_vram_mb: float | None = None
    supports_bf16: bool = False
    supports_fp16: bool = False
    allow_cpu_for: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def cpu_reason(self) -> str:
        """Why a component ran on the CPU, in the manifest's words.

        Recorded rather than inferred: "classical baselines on CPU" is a legitimate
        choice, and "the GPU silently disappeared" is a defect — the two must not
        look alike in the run record.
        """
        if self.cuda_available:
            return "not_on_cpu"
        return ("no CUDA device visible: reference/classical components are CPU-eligible by "
                "policy, neural training is not (it is gated off above)")

    @property
    def is_accelerator(self) -> bool:
        return self.device.startswith("cuda")

    def dtype_for(self, policy: str) -> str:
        """Resolve the config's mixed-precision policy to a concrete dtype name.

        ``auto`` prefers bf16 (no scaler, wider exponent range — the safe choice
        for GRU/SSM recurrences), falls back to fp16 + GradScaler, and to fp32 if
        neither is appropriate. The chosen path and *why* are both returned by
        :meth:`precision_plan`.
        """
        if policy in ("", None):
            policy = "auto"
        if policy == "fp32":
            return "float32"
        if policy == "bf16":
            return "bfloat16" if self.supports_bf16 else "float32"
        if policy == "fp16":
            return "float16" if self.supports_fp16 else "float32"
        if policy == "auto":
            if self.supports_bf16:
                return "bfloat16"
            if self.supports_fp16 and self.is_accelerator:
                return "float16"
            return "float32"
        raise ValueError(f"unknown mixed-precision policy {policy!r}")

    def precision_plan(self, policy: str) -> dict:
        dtype = self.dtype_for(policy)
        return {
            "requested": policy,
            "dtype": dtype,
            "autocast": dtype in ("bfloat16", "float16") and self.is_accelerator,
            "grad_scaler": dtype == "float16" and self.is_accelerator,
            "reason": _precision_reason(dtype, self),
        }

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "accelerator_expected": self.accelerator_expected,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "capability": self.capability,
            "total_vram_mb": self.total_vram_mb,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "allow_cpu_for": list(self.allow_cpu_for),
            "notes": self.notes,
        }


def _precision_reason(dtype: str, rep: DeviceReport) -> str:
    if dtype == "float32":
        if not rep.is_accelerator:
            return "fp32: no accelerator present, autocast would only add rounding risk"
        return "fp32: neither bf16 nor fp16 is available on this device"
    if dtype == "bfloat16":
        return "bf16: device reports Ampere-equivalent or newer bf16 support"
    return "fp16: bf16 unavailable; fp16 needs the GradScaler for GRU/SSM recurrences"


def detect_device(cfg=None, *, require: bool | None = None) -> DeviceReport:
    """Measure the device. With training required and no CUDA ⇒ raise (spec scenario)."""
    expected = ""
    allow: tuple[str, ...] = ()
    if cfg is not None:
        hw = cfg.raw.get("hardware", {}) or {}
        expected = str(hw.get("accelerator", ""))
        allow = tuple(hw.get("allow_cpu_for", ()) or ())
    rep = DeviceReport(accelerator_expected=expected, allow_cpu_for=allow)
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        rep.notes.append("torch is not importable in this environment")
        rep.device = "cpu"
        return rep
    rep.cuda_available = bool(torch.cuda.is_available())
    if rep.cuda_available:
        rep.device = "cuda"
        try:
            rep.device_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            rep.capability = f"{major}.{minor}"
            free, total = torch.cuda.mem_get_info()
            rep.total_vram_mb = round(total / 2**20, 1)
            rep.supports_bf16 = bool(major >= 8)
            rep.supports_fp16 = True
            del free
        except Exception as exc:  # pragma: no cover - driver dependent
            rep.notes.append(f"cuda introspection incomplete: {type(exc).__name__}: {exc}")
    else:
        rep.device = "cpu"
        rep.notes.append("torch.cuda.is_available() is False")
    need = bool(require) if require is not None else True
    if need and rep.device != "cuda":
        raise CpuNotAllowedError(
            "training on a non-CUDA host is refused by the contract (spec "
            "reproducible-benchmark-execution / T4 absence). The CPU-permitted "
            f"work for this change is {list(allow) or ['preprocessing', 'classical_baselines', 'unit_tests', 'smoke']}. "
            f"Expected accelerator: {expected or 'NVIDIA T4'}. On Colab: "
            "Runtime > Change runtime type > T4 GPU, then re-run this cell."
        )
    return rep


class VramMonitor:
    """Peak-VRAM accounting per cell, with an optional hard cap.

    On CPU the monitor degrades to RSS when psutil is present and to ``None``
    otherwise — a missing measurement is recorded as absent, never as zero,
    because a 0 would read as "used no memory".
    """

    def __init__(self, device: str, *, cap_mb: float | None = None) -> None:
        self.device = device
        self.cap_mb = cap_mb
        self._cuda = device.startswith("cuda")
        self._torch = None
        self.peak_mb: float | None = None
        self.violations: list[str] = []
        if self._cuda:
            try:
                import torch  # noqa: PLC0415

                self._torch = torch
                torch.cuda.reset_peak_memory_stats()
            except Exception:  # pragma: no cover
                self._cuda = False

    def sample(self, where: str = "") -> float | None:
        mb = self._current_peak_mb()
        self.peak_mb = mb if self.peak_mb is None else max(self.peak_mb, mb or 0.0)
        if self.cap_mb and (self.peak_mb or 0.0) > self.cap_mb:
            self.violations.append(
                f"peak {self.peak_mb:.0f} MB exceeds the cap {self.cap_mb:.0f} MB {where}".strip()
            )
        return self.peak_mb

    def _current_peak_mb(self) -> float | None:
        if self._cuda and self._torch is not None:
            try:
                return round(float(self._torch.cuda.max_memory_allocated()) / 2**20, 1)
            except Exception:  # pragma: no cover
                return None
        try:
            import psutil  # noqa: PLC0415
            import os

            rss = psutil.Process(os.getpid()).memory_info().rss
            return round(float(rss) / 2**20, 1)
        except Exception:
            return None

    def release(self) -> None:
        """Explicit GPU-memory release between cells (spec requirement)."""
        if self._cuda and self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "peak_vram_mb": self.peak_mb,
            "cap_mb": self.cap_mb,
            "violations": list(self.violations),
            "measure": (
                "torch.cuda.max_memory_allocated" if self._cuda else "process RSS (CPU host)"
            ),
        }


class BudgetMeter:
    """Wall-clock budget accounting for one cycle (task 12.7).

    The meter is deliberately independent of the registry: the scheduler asks it
    "is there room for the next cell?" before claiming, so a cell that cannot fit
    is never marked ``Running`` just to be marked ``Deferred`` afterwards.
    """

    def __init__(self, t4_hours: float, *, estimates: dict[str, float] | None = None) -> None:
        self.t4_hours = float(t4_hours)
        self.used_hours = 0.0
        #: per-model hours-per-estimate from the smoke profile
        self.estimates: dict[str, float] = dict(estimates or {})
        self.log: list[dict] = []

    def estimate(self, model_id: str, *, tier: str = "full") -> float | None:
        base = self.estimates.get(model_id)
        if base is None:
            return None
        return float(base) if tier == "full" else float(base) / 6.0

    def has_room(self, model_id: str, *, tier: str = "full") -> bool:
        # The hard cap first, and independently of the profile. Without this line the
        # missing-profile branch would make ``t4_hours`` advisory: an unprofiled model
        # (the state of every model at the start of the first run) would be admitted
        # forever, and the contract's "budget exhaustion ⇒ defer the remainder" could
        # never fire. The estimate only decides how much *more* a cell may cost.
        if self.exhausted:
            self.log.append({"model": model_id, "decision": "refuse_budget_spent",
                             "used_hours": round(self.used_hours, 4),
                             "t4_hours": self.t4_hours})
            return False
        est = self.estimate(model_id, tier=tier)
        if est is None:
            # No profile yet ⇒ admit the cell (the screening tier exists to build
            # the profile) but say so, because "budget-aware" cannot mean "blind".
            self.log.append({"model": model_id, "decision": "admit_unprofiled"})
            return True
        return (self.used_hours + est) <= self.t4_hours

    def charge(self, model_id: str, seconds: float, *, tier: str = "full", cell: str = "") -> None:
        hours = float(seconds) / 3600.0
        self.used_hours += hours
        self.estimates.setdefault(model_id, hours)
        if hours > 0:
            self.estimates[model_id] = max(self.estimates[model_id], hours) if tier == "full" \
                else max(self.estimates[model_id], hours * 6.0)
        self.log.append(
            {"model": model_id, "cell": cell, "hours": round(hours, 5),
             "used_hours": round(self.used_hours, 4), "remaining_hours": self.remaining_hours}
        )

    @property
    def remaining_hours(self) -> float:
        return round(max(0.0, self.t4_hours - self.used_hours), 4)

    @property
    def exhausted(self) -> bool:
        return self.used_hours >= self.t4_hours

    def to_dict(self) -> dict:
        return {
            "t4_hours": self.t4_hours,
            "used_hours": round(self.used_hours, 4),
            "remaining_hours": self.remaining_hours,
            "exhausted": self.exhausted,
            "estimates_hours": {k: round(float(v), 5) for k, v in self.estimates.items()},
            "log": self.log[-200:],
        }
