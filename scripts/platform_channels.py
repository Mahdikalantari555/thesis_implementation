"""Declarative registry of platform data channels.

Single source of truth (Python side) for the channel contract between
``scripts/export_platform_data.py`` and the platform web app. The TypeScript
mirror lives at ``platform/src/data/channels.ts``; parity is enforced by
``tests/unit/test_platform_data_contract.py::test_channel_registry_parity``.

Each entry declares:
  - ``name``: channel id (matches the platform ``IndicatorId``)
  - ``unit``: display unit
  - ``physical_range``: (min, max) inclusive physical envelope
  - ``provenance_default``: ``observed`` | ``modelled`` | ``proxy`` | ``absent``
  - ``required``: whether every observation must carry the channel
  - ``kind``: spectral | thermal | water | weather
  - ``derived_from``: optional source channel when this is a documented proxy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

ProvenanceState = Literal["observed", "modelled", "proxy", "absent"]
ChannelKind = Literal["spectral", "thermal", "water", "weather"]


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    unit: str
    physical_range: Tuple[float, float]
    provenance_default: ProvenanceState = "observed"
    required: bool = True
    kind: ChannelKind = "spectral"
    derived_from: Optional[str] = None


# Order mirrors the platform INDICATORS list.
CHANNEL_REGISTRY: tuple[ChannelSpec, ...] = (
    ChannelSpec("ndvi", "index", (0.0, 1.0), "observed", True, "spectral"),
    ChannelSpec("evi", "index", (0.0, 1.0), "observed", True, "spectral"),  # computed from blue band
    ChannelSpec("savi", "index", (0.0, 1.0), "observed", True, "spectral"),
    ChannelSpec("ndmi", "index", (-1.0, 1.0), "observed", True, "water"),
    ChannelSpec("tvdi", "index", (0.0, 1.0), "proxy", True, "thermal", "cwsi"),
    ChannelSpec("cwsi", "index", (0.0, 1.0), "observed", True, "thermal"),
    ChannelSpec("eta", "mm/d", (0.0, 8.0), "proxy", True, "water", "ndvi"),
    ChannelSpec("lst", "\u00b0C", (15.0, 55.0), "observed", False, "thermal"),  # conditional
    ChannelSpec("precip", "mm", (0.0, 300.0), "modelled", True, "weather"),
    ChannelSpec("temp", "\u00b0C", (10.0, 48.0), "modelled", True, "weather"),
)

CHANNEL_NAMES: tuple[str, ...] = tuple(c.name for c in CHANNEL_REGISTRY)


def by_name(name: str) -> Optional[ChannelSpec]:
    for c in CHANNEL_REGISTRY:
        if c.name == name:
            return c
    return None


def as_dict() -> list[dict]:
    out = []
    for c in CHANNEL_REGISTRY:
        out.append(
            {
                "name": c.name,
                "unit": c.unit,
                "physical_range": list(c.physical_range),
                "provenance_default": c.provenance_default,
                "required": c.required,
                "kind": c.kind,
                "derived_from": c.derived_from,
            }
        )
    return out


__all__ = ["ChannelSpec", "CHANNEL_REGISTRY", "CHANNEL_NAMES", "by_name", "as_dict"]
