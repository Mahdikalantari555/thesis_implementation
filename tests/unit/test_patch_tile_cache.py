import numpy as np
import rasterio

from scripts import extract_patches as E


def test_three_fields_byte_identical(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    for name in [*E.SR_FILE_MAP.values(), *E.ST_FILE_MAP.values(), 'qa_pixel.tif']:
        values = rng.integers(0, 20000, (149, 150), dtype=np.uint16)
        with rasterio.open(tmp_path / name, 'w', driver='GTiff', width=150, height=149,
                           count=1, dtype='uint16', tiled=True, compress='deflate') as dst:
            dst.write(values, 1)
    windows = [E.P.FieldWindow(r, c, 16, 19, 48, 3, 4)
               for r, c in [(-2, 6), (7, 23), (137, 139)]]
    expected = [E.read_scene_patch(tmp_path, w, band_scale='auto') for w in windows]
    cache = E.TileCache(2)
    real_open = rasterio.open
    opens = []

    def counted(path):
        opens.append(path)
        return real_open(path)

    monkeypatch.setattr(rasterio, 'open', counted)
    for w, reference in zip(windows, expected):
        actual = E.read_scene_patch(tmp_path, w, band_scale='auto', cache=cache)
        assert [x.tobytes() for x in actual] == [x.tobytes() for x in reference]
    assert len(opens) == 16  # second field reuses all eight tiles


def test_store_parity(tmp_path):
    import pandas as pd
    from tests.unit.test_extract_patches import _make_world

    mask, scenes, cycles = _make_world(tmp_path)
    from shutil import copytree
    copytree(next(scenes.iterdir()), scenes / "landsat_20240112_LC08_L2")
    with rasterio.open(mask, 'r+') as dst:
        ids = dst.read(1)
        ids[11:18, 32:41] = 103
        dst.write(ids, 1)
    df = pd.read_parquet(cycles)
    pd.concat([df, df.iloc[:1].assign(field=103)]).to_parquet(cycles, index=False)
    roots = []
    hashes = []
    for budget, batch, workers in [(0, 1, 1), (2, 1, 1), (0, 4, 1), (2, 4, 4)]:
        store = tmp_path / f'{budget}-{batch}.zarr'
        assert E.main(['--mask', str(mask), '--scenes-root', str(scenes),
                       '--cycles', str(cycles), '--tile-cache-mib', str(budget),
                       '--workers', str(workers), '--field-batch-size', str(batch), '--profile-io', '--patch-size', '48', '--out', str(store)]) == 0
        roots.append(E.P.open_store(store))
        hashes.append([r['sha256'] for r in E.read_manifest(store)])
    assert all(h == hashes[0] for h in hashes) and len(hashes[0]) == 3
    for key in roots[0].array_keys():
        assert all(roots[0][key][:].tobytes() == r[key][:].tobytes() for r in roots[1:]), key


def test_shared_handles_closed_at_scene_boundary(tmp_path, monkeypatch):
    from tests.unit.test_extract_patches import _make_world

    _, scenes, _ = _make_world(tmp_path)
    path = next(scenes.iterdir())
    opened = []
    real_open = rasterio.open

    def track(path):
        src = real_open(path)
        opened.append(src)
        return src

    monkeypatch.setattr(rasterio, 'open', track)
    jobs = [(i, i, E.P.FieldWindow(i, i, 16, 16), [(0, 0, 20)]) for i in range(4)]
    E.prefetch_batch(jobs, {0: [path]}, E.TileCache(0), 'auto')
    assert len(opened) == 8
    assert all(src.closed for src in opened)
