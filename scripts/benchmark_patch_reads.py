"""Short synthetic end-to-end benchmark; uses only temporary stores and rasters."""
from contextlib import redirect_stdout
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import random
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import extract_patches as E


def make_fixture(root):
    rng = np.random.default_rng(490)
    shape = (512, 1024)
    profile = dict(driver='GTiff', count=1, width=shape[1], height=shape[0],
                   dtype='uint16', tiled=True, compress='deflate', blockxsize=512,
                   blockysize=512, transform=from_origin(300000, 4000000, 30, 30))
    ids = range(1, 9)
    mask = np.zeros(shape, 'uint16')
    # Two rows of neighbouring fields; include tile and GTiff block boundaries.
    for fid in ids:
        y, x = 30 + ((fid-1) // 4)*50, [40, 90, 470, 520][(fid-1) % 4]
        mask[y:y+48, x:x+48] = fid
    with rasterio.open(root/'mask.tif', 'w', **profile) as dst:
        dst.write(mask, 1)
    pd.DataFrame(dict(field=ids, t_start=0, t_end=130)).to_parquet(root/'cycles.parquet')
    for i in range(8):
        day = date(2020, 1, 1) + timedelta(days=16*i)
        scene = root / f'landsat_{day:%Y%m%d}_LC08'
        scene.mkdir()
        for name in [*E.SR_FILE_MAP.values(), *E.ST_FILE_MAP.values(), 'qa_pixel.tif']:
            data = (rng.choice([0, 8], shape, p=[.9, .1]).astype('uint16') if name == 'qa_pixel.tif'
                    else rng.integers(9000, 16000, shape, dtype='uint16'))
            with rasterio.open(scene/name, 'w', **profile) as dst:
                dst.write(data, 1)


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fixture(root)
        configs = [(1, 0, 1), (4, 32, 1), (4, 32, 2), (4, 32, 4)]
        times = {key: [] for key in configs}
        oracle = None
        for repeat in range(3):
            order = configs.copy()
            random.Random(490+repeat).shuffle(order)
            for batch, cache, workers in order:
                with TemporaryDirectory(dir=root) as run:
                    store = Path(run)/'patches.zarr'
                    args = ['--mask', str(root/'mask.tif'), '--cycles', str(root/'cycles.parquet'),
                            '--scenes-root', str(root), '--out', str(store), '--patch-size', '48',
                            '--workers', str(workers), '--field-batch-size', str(batch), '--tile-cache-mib', str(cache)]
                    start = perf_counter()
                    with redirect_stdout(StringIO()):
                        code = E.main(args)
                    elapsed = perf_counter() - start
                    assert code == 0
                    records = E.read_manifest(store)
                    assert len(records) == 8 and all(r['n_scenes'] == 8 for r in records)
                    hashes = [(r['field_id'], r['sha256']) for r in records]
                    if oracle is None:
                        oracle = hashes
                    assert hashes == oracle
                    times[batch, cache, workers].append(elapsed)
        for key, samples in times.items():
            print(f'workers={key[2]} batch={key[0]} cache={key[1]} MiB: {samples}, median={median(samples):.3f}s')


if __name__ == '__main__':
    main()
