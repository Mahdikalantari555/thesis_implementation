"""Farm selection for the mini dataset: 3farms_dev layer + Farms.geojson id map."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")
pytest.importorskip("zarr")
from affine import Affine  # noqa: E402
from shapely.geometry import box  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_mini_dataset", ROOT / "scripts" / "build_mini_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_mini_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load_builder()
CRS = "EPSG:32639"


def _reference():
    # Five strip farms, each 300 m wide, ids 10..14 from west to east.
    names = ["A-01", "A-02", "B-07", "C-03", "C-04"]
    geoms = [box(237000 + 300 * i, 3434700, 237270 + 300 * i, 3435600) for i in range(5)]
    return gpd.GeoDataFrame(
        {"Farm": names, "OGC_FID": [10, 11, 12, 13, 14], "area": [24.3] * 5, "Age": ["P", "R01", "R02", "R03", "P"]},
        geometry=geoms, crs=CRS,
    )


def _layer(names, reindex=True):
    ref = _reference()
    sub = ref[ref["Farm"].isin(names)].copy().reset_index(drop=True)
    if reindex:  # exported layers often carry re-indexed OGC_FIDs
        sub["OGC_FID"] = range(1, len(sub) + 1)
    return sub


def test_farm_set_comes_from_layer_and_ids_from_reference():
    farms = B.resolve_farms(_layer(["A-01", "B-07", "C-04"]), _reference())
    assert [f["farm_id"] for f in farms] == ["A-01", "B-07", "C-04"]
    assert [f["ogc_fid"] for f in farms] == [10, 12, 14]  # reference ids, not layer ids
    assert [f["layer_ogc_fid"] for f in farms] == [1, 2, 3]
    assert all(f["layer_vs_reference_centroid_m"] == 0.0 for f in farms)
    assert farms[0]["source_attributes"]["Age"] == "P"


def test_wrong_count_unknown_name_and_duplicates_are_refused():
    with pytest.raises(RuntimeError, match="expected 3"):
        B.resolve_farms(_layer(["A-01", "B-07"]), _reference())
    bad = _layer(["A-01", "B-07", "C-04"])
    bad.loc[0, "Farm"] = "ZZ-99"
    with pytest.raises(RuntimeError, match="ID reference"):
        B.resolve_farms(bad, _reference())
    dup = _layer(["A-01", "B-07", "C-04"])
    dup.loc[1, "Farm"] = "A-01"
    with pytest.raises(RuntimeError, match="Duplicate"):
        B.resolve_farms(dup, _reference())
    with pytest.raises(RuntimeError, match="not in selection layer"):
        B.resolve_farms(_layer(["A-01", "B-07", "C-04"]), _reference(), names=("A-02",))


def test_subset_names_narrow_without_count_check():
    farms = B.resolve_farms(_layer(["A-01", "B-07", "C-04"]), _reference(), names=("B-07",))
    assert [f["farm_id"] for f in farms] == ["B-07"]


def _mask_for(geom, transform, shape):
    from rasterio.features import rasterize

    return rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")


def test_alignment_detects_own_fid_and_off_by_one():
    ref = _reference()
    transform = Affine(30.0, 0, 236700.0, 0, -30.0, 3435900.0)
    shape = (50, 60)
    own = ref.loc[ref["OGC_FID"] == 12].geometry.iloc[0]
    ok = B.archive_alignment(_mask_for(own, transform, shape), transform, ref, 12, own)
    assert ok["best_is_own_fid"] and ok["best_reference_fid"] == 12
    assert ok["by_reference_fid"]["12"]["iou"] > 0.9
    assert ok["selection_layer_iou"] > 0.9
    # Archive pixels that actually belong to the eastern neighbour (fid 13):
    east = ref.loc[ref["OGC_FID"] == 13].geometry.iloc[0]
    off = B.archive_alignment(_mask_for(east, transform, shape), transform, ref, 12, own)
    assert not off["best_is_own_fid"] and off["best_reference_fid"] == 13
    assert off["selection_layer_iou"] < 0.2


def test_label_crop_kind_uses_age_attribute_not_farm_name():
    spec = importlib.util.spec_from_file_location("label_mini", ROOT / "scripts" / "label_mini_suspected_stress.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # heavy optional deps on the owner laptop only
        pytest.skip(f"label script deps unavailable: {exc}")
    assert mod.crop_kind({"Age": "P"})[0] == "plant"
    assert mod.crop_kind({"Age": "R03"})[0] == "ratoon"
    assert mod.crop_kind({})[1].startswith("fallback")
