"""Sequence backbones for the benchmark arms (design §Backbone notes).

Both trunks expose the *same* causal contract so a paradigm comparison is never
confounded with a "the other backbone quietly saw the future" problem:

``CausalStackBackbone.forward(x, valid)`` → ``hidden[b, t]`` is a function of
``x[b, 0..t]`` only. For the GRU that is a unidirectional stack; for Mamba it is
the pure-PyTorch selective scan of :class:`ssl_anomaly.models.cnn_mamba.SelectiveSSM`
with a **causal** depthwise convolution (left-padded only) in front of it.

Two decisions worth stating in a thesis:

1. **The Mamba trunk is the repository's existing pure-PyTorch selective scan,
   not the CUDA ``selective_scan_kernel``.** The design permits a CPU fallback for
   unit tests only; the practical consequence is that the Mamba arms are slower
   per step than a fused kernel would be, so the feasibility gate measures their
   wall time against the GRU twin *on the same device and code path* and records
   ``Not Applicable`` with the numbers if the ratio breaks. Substituting a fused
   kernel only for Mamba would also be a comparison nobody could reproduce.
2. **The non-causal variants exist but the benchmark does not use them.** A
   bidirectional trunk can see bin ``t+1`` while scoring bin ``t``, which is exactly
   the temporal-leakage failure AGENTS.md §7 names for an early-warning claim. The
   factory refuses ``causal=False`` for any prediction or JEPA arm and the arms
   themselves are constructed causal-only, so it is not a matter of remembering to
   pass a flag.

``NumericalInstability`` is raised the moment a trunk produces a non-finite value:
the design requires the Mamba arms to be marked ``Not Applicable`` *with recorded
evidence*, and evidence is only available if instability stops the run instead of
propagating NaNs into a metric that then reports 0.5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from ssl_anomaly.models.cnn_mamba import SelectiveSSM


class NumericalInstability(RuntimeError):
    """A trunk produced non-finite activations — the feasibility gate's evidence."""

    def __init__(self, message: str, *, evidence: dict) -> None:
        super().__init__(message)
        self.evidence = evidence


@dataclass
class BackboneOutput:
    """Per-bin causal states + the pooled end state."""

    hidden: torch.Tensor  # (B, T, D)
    pooled: torch.Tensor  # (B, D)

    def __post_init__(self) -> None:
        if self.hidden.ndim != 3:
            raise ValueError(f"hidden must be (B, T, D), got {tuple(self.hidden.shape)}")
        if self.pooled.shape != (self.hidden.shape[0], self.hidden.shape[2]):
            raise ValueError(
                f"pooled {tuple(self.pooled.shape)} must be (B, D) with hidden "
                f"{tuple(self.hidden.shape)}"
            )


class SequenceBackbone(ABC, nn.Module):
    """Common contract: (B, T, C) → causal per-bin states."""

    causal: bool = True

    @property
    @abstractmethod
    def d_model(self) -> int:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> BackboneOutput:
        ...

    # ---- shared guards ---------------------------------------------------
    def _check_finite(self, t: torch.Tensor, where: str) -> None:
        if bool(torch.isfinite(t).all()):
            return
        bad = (~torch.isfinite(t)).sum().item()
        raise NumericalInstability(
            f"{type(self).__name__} produced {int(bad)} non-finite value(s) in {where}",
            evidence={
                "where": where,
                "n_nonfinite": int(bad),
                "shape": list(t.shape),
                "max_abs_finite": (
                    float(t[torch.isfinite(t)].abs().max().item())
                    if bool(torch.isfinite(t).any())
                    else None
                ),
            },
        )


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise conv with left-only padding — no bin sees a later bin."""

    def __init__(self, d_model: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel = int(kernel_size)
        self.pad = self.kernel - 1
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=self.kernel, padding=0, groups=d_model, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                       # (B, D, T)
        h = nn.functional.pad(h, (self.pad, 0))     # causal: pad the past only
        return self.conv(h).transpose(1, 2)


class CausalMambaBlock(nn.Module):
    """Gate + causal depthwise conv + selective scan, one residual block."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = CausalDepthwiseConv1d(d_inner, kernel_size=4)
        self.ssm = SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        v, z = self.in_proj(h).chunk(2, dim=-1)
        v = nn.functional.silu(self.conv(v))
        # forward-only scan ⇒ h_t depends on x[..t]; the SSM's own parameters are
        # input-dependent (selective), which is the whole point of the arm.
        y = self.ssm(v, reverse=False)
        y = y * nn.functional.silu(z)
        y = self.drop(self.out_proj(y))
        if valid is not None:
            y = y.masked_fill(~valid.unsqueeze(-1), 0.0)
        return x + y


class CausalStackBackbone(SequenceBackbone):
    """``kind`` ∈ {gru, mamba}: projection → causal stack → LayerNorm.

    Parameters
    ----------
    d_in:
        Token width (channels + mask flags + covariates, see the arm).
    d_model:
        Hidden width; the capacity tuner adjusts this to hit the ±20 % band.
    n_layers:
        GRU layers / number of Mamba blocks.
    causal:
        ``False`` is only legal for reconstruction arms, and even there the
        benchmark keeps it ``True`` so that every arm in a comparison shares the
        "no future information" contract. Passing ``False`` requires the caller to
        name why, which :meth:`build_backbone` enforces.
    """

    def __init__(
        self,
        d_in: int,
        *,
        kind: str = "gru",
        d_model: int = 96,
        n_layers: int = 1,
        dropout: float = 0.1,
        d_state: int = 16,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if kind not in ("gru", "mamba"):
            raise ValueError(f"unknown backbone kind {kind!r} (gru|mamba)")
        self.kind = kind
        self.causal = bool(causal)
        self._d_model = int(d_model)
        self.proj = nn.Linear(int(d_in), int(d_model))
        if kind == "gru":
            self.stack = nn.GRU(
                input_size=int(d_model),
                hidden_size=int(d_model),
                num_layers=int(n_layers),
                batch_first=True,
                dropout=float(dropout) if int(n_layers) > 1 else 0.0,
                bidirectional=not self.causal,
            )
        else:
            blocks = nn.ModuleList(
                [
                    CausalMambaBlock(int(d_model), d_state=int(d_state), dropout=float(dropout))
                    for _ in range(int(n_layers))
                ]
            )
            self.stack = blocks
        self.norm = nn.LayerNorm(int(d_model))

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def d_out(self) -> int:
        """Hidden width of the emitted states.

        Equal to ``d_model`` because every trunk here is causal — a bidirectional
        stack would emit ``2 × d_model`` and every head would have to know that,
        which is the kind of asymmetry that quietly makes two arms unequal.
        """
        return self._d_model

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> BackboneOutput:
        h = self.proj(x)
        if valid is not None:
            h = h.masked_fill(~valid.unsqueeze(-1), 0.0)
        if self.kind == "gru":
            if valid is not None and not bool(valid.all()):
                lengths = valid.sum(dim=1).clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    h, lengths, batch_first=True, enforce_sorted=False
                )
                out, hn = self.stack(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            else:
                out, hn = self.stack(h)
            hidden = self.norm(out)
            pooled = hn[-1] if not self.causal else hn[-1]
            if not self.causal:  # bidirectional: concatenate the two final states
                pooled = torch.cat([hn[-2], hn[-1]], dim=-1)[:, : self._d_model]
            self._check_finite(hidden, "gru.hidden")
        else:
            for block in self.stack:
                h = block(h, valid)
            hidden = self.norm(h)
            self._check_finite(hidden, "mamba.hidden")
            pooled = _last_valid(hidden, valid)
        return BackboneOutput(hidden=hidden, pooled=pooled.contiguous())


def _last_valid(hidden: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """Last valid per-bin state (a padded tail must not become the summary)."""
    if valid is None:
        return hidden[:, -1]
    idx = valid.to(torch.long).cumsum(dim=1)
    last = (idx[:, -1:] - 1).clamp(min=0)                    # (B, 1)
    return hidden.gather(1, last.unsqueeze(-1).expand(-1, 1, hidden.shape[-1])).squeeze(1)


#: ``build_backbone`` refuses a non-causal trunk unless the caller names the arm it
#: is for; that keeps "let me just make it bidirectional" from being silent.
NONCAUSAL_ALLOWED_FOR = ("A1", "A2", "A3", "A4")


def build_backbone(
    d_in: int,
    *,
    kind: str,
    d_model: int,
    n_layers: int = 1,
    dropout: float = 0.1,
    d_state: int = 16,
    causal: bool = True,
    for_arm: str = "",
) -> CausalStackBackbone:
    if not causal and for_arm not in NONCAUSAL_ALLOWED_FOR:
        raise ValueError(
            f"causal=False is not permitted for arm {for_arm or '<unnamed>'!r}: prediction and "
            f"JEPA objectives must not see future bins. Non-causal is allowed only for "
            f"{list(NONCAUSAL_ALLOWED_FOR)}."
        )
    return CausalStackBackbone(
        d_in,
        kind=kind,
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        d_state=d_state,
        causal=causal,
    )


class CausalMambaBackbone(CausalStackBackbone):
    """Named alias so tests and reports can ask for the Mamba trunk explicitly."""

    def __init__(self, d_in: int, **kw) -> None:
        kw["kind"] = "mamba"
        super().__init__(d_in, **kw)
