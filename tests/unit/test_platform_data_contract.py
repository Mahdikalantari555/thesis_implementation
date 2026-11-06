"""Platform data contract parity test.

Verifies that the exported JSON payloads satisfy the channel registry contract:
every channel channel name in platform_channels.py matches channels.ts,
every observation carries the contract v2 fields, and ranges are valid.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from platform_channels import CHANNEL_REGISTRY, by_name  # type: ignore

DATA_DIR = ROOT / "platform" / "src" / "data"
FARMS_PATH = DATA_DIR / "farms_khuzestan.json"
SERIES_PATH = DATA_DIR / "series_khuzestan.json"


def test_farms_contract_version():
    farms = json.loads(FARMS_PATH.read_text(encoding="utf-8"))
    assert len(farms) == 490
    for f in farms:
        assert f["contractVersion"] == 2
        assert f["planted"] is None
        assert f["harvest"] is None


def test_series_has_contract_v2_fields():
    series = json.loads(SERIES_PATH.read_text(encoding="utf-8"))
    obs = next(iter(series.values()))[0]
    expected = {
        "date", "sampleDate", "idx", "acquisitionDate", "missing",
        "stalenessReason", "ndvi", "evi", "savi", "ndmi", "tvdi", "cwsi",
        "eta", "lst", "precip", "temp", "scores", "provenance",
        "scoreCollinearity",
    }
    assert expected.issubset(obs.keys()), f"missing: {expected - set(obs.keys())}"
    assert obs["scoreCollinearity"] == "cwsi"


def test_provenance_matches_registry():
    series = json.loads(SERIES_PATH.read_text(encoding="utf-8"))
    obs = next(iter(series.values()))[0]
    prov = obs["provenance"]
    for ch in ["ndvi", "evi", "savi", "ndmi", "tvdi", "cwsi", "eta", "lst", "precip", "temp"]:
        spec = by_name(ch)
        assert prov[ch] in ("observed", "modelled", "proxy", "absent", "legacy")


def test_channel_range_validation():
    series = json.loads(SERIES_PATH.read_text(encoding="utf-8"))
    for fid, obs_list in series.items():
        for obs in obs_list:
            for ch in ["ndvi", "evi", "savi", "ndmi", "tvdi", "cwsi", "eta", "lst"]:
                v = obs[ch]
                if v is not None:
                    lo, hi = by_name(ch).physical_range
                    assert lo <= v <= hi, f"{fid} {ch} {v} not in [{lo},{hi}]"
            if obs["precip"] is not None:
                lo, hi = by_name("precip").physical_range
                assert lo <= obs["precip"] <= hi
            if obs["temp"] is not None:
                lo, hi = by_name("temp").physical_range
                assert lo <= obs["temp"] <= hi


def test_staleness_guard():
    series = json.loads(SERIES_PATH.read_text(encoding="utf-8"))
    obs_list = next(iter(series.values()))
    missing = [o for o in obs_list if o["missing"]]
    assert len(missing) > 0
    for o in missing:
        assert o["stalenessReason"] is not None


def test_neighbors_ranked():
    farms = json.loads(FARMS_PATH.read_text(encoding="utf-8"))
    for f in farms:
        assert len(f["neighbors"]) <= 4
        assert all(n != f["id"] for n in f["neighbors"])


def test_registry_parity_ts():
    """Python registry has 10 channels; check names match expected set."""
    names = [c.name for c in CHANNEL_REGISTRY]
    assert names == ["ndvi", "evi", "savi", "ndmi", "tvdi", "cwsi", "eta", "lst", "precip", "temp"]
    assert len(names) == 10
    assert by_name("lst").required is False
    assert by_name("precip").provenance_default == "modelled"
    assert by_name("temp").provenance_default == "modelled"
    assert by_name("tvdi").derived_from == "cwsi"
    assert by_name("eta").derived_from == "ndvi"
