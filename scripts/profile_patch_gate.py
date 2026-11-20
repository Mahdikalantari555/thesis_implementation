"""Short A/B gate. Only writes disposable stores under a temporary directory.

Without --synthetic, the owner must stop extraction and confirm --quiet-machine.
Keep the production archive and the process environment unchanged between trials.
"""
import argparse
import builtins
import os
from unittest.mock import patch
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import time

import rasterio

import extract_patches as E
from benchmark_patch_reads import make_fixture

ANCHORS = {
    1: (455, '8a5e116665bcf8282a22982d4cba1694fef3aabd676157f9384802700f82dd75'),
    2: (326, '13bdd036cf595d26776cf509c434c3c414c21ac1c988309667ba8cf1cd40c910'),
}


def fingerprint(root):
    # Manifest SHAs canonicalize NaNs and cover only values; also compare raw
    # logical bytes of every array (one field at a time, outside measurement).
    digest = hashlib.sha256()
    for key in sorted(root.array_keys()):
        arr = root[key]
        digest.update(str((key, arr.shape, arr.dtype.str)).encode())
        for fi in range(arr.shape[0]):
            digest.update(arr[fi].tobytes())
    return digest.hexdigest()


def probe_codec(root):
    """Offline encode/decode of one actual cycle. Never alter the store codec."""
    from numcodecs import Blosc

    cycle = root['values'][0, 0]
    for bins in (18, 36):
        for name, level in [('lz4', 5), ('lz4', 1), ('zstd', 1)]:
            codec = Blosc(cname=name, clevel=level, shuffle=Blosc.SHUFFLE)
            chunks = [cycle[i:i+bins] for i in range(0, len(cycle), bins)]
            times = []
            for _ in range(3):
                tick = time.perf_counter()
                encoded = [codec.encode(chunk) for chunk in chunks]
                decoded = [codec.decode(buf) for buf in encoded]
                times.append(time.perf_counter() - tick)
                if decoded != [chunk.tobytes() for chunk in chunks]:
                    raise ValueError('codec probe did not round trip')
            print('codec probe', bins, name, level, sum(map(len, encoded)), median(times))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--quiet-machine', action='store_true')
    ap.add_argument('--sweep', action='store_true', help='also measure individual levers')
    ap.add_argument('--temp-parent', type=Path, help='use same drive as production output')
    args = ap.parse_args()
    if not args.synthetic and not args.quiet_machine:
        ap.error('Stop the live reader yourself before confirming --quiet-machine')
    with TemporaryDirectory(dir=args.temp_parent) as tmp:
        folder = Path(tmp)
        if args.synthetic:
            make_fixture(folder)
            inputs = ['--scenes-root', str(folder), '--mask', str(folder/'mask.tif'),
                      '--cycles', str(folder/'cycles.parquet')]
        else:
            inputs = ['--fields', '1,2', '--splits', 'data/processed/splits_locked_v2.json',
                      '--cycles', 'data/processed/cycles_490_v2.parquet',
                      '--mask', 'D:/RS/Thesis/Data/Processed/interim/field_mask_full.tif']
        configs = [('baseline', [], 2, {}), ('candidate', ['--native-block-cache', '--short-bin-median'], 2, {})]
        if args.sweep:
            configs += [('cpu', ['--short-bin-median'], 2, {}), ('blocks', ['--native-block-cache'], 2, {}),
                        ('4 workers', configs[1][1], 4, {}),
                        ('GDAL cache', configs[1][1], 2, {'GDAL_CACHEMAX': 512*1024**2}),
                        ('GDAL decode', configs[1][1], 2, {'GDAL_NUM_THREADS': '2'})]
        oracle = None
        samples = defaultdict(list)
        for trial in range(3):
            order = configs[trial % len(configs):] + configs[:trial % len(configs)]
            for name, extra, workers, env in order:
                with TemporaryDirectory(dir=folder) as run, rasterio.Env(**env):
                    store = Path(run)/'patches.zarr'
                    cmd = inputs + ['--out', str(store), '--patch-size', '48', '--workers', str(workers),
                                    '--field-batch-size', '4', '--tile-cache-mib', '1024', '--profile-io'] + extra
                    capture = StringIO()
                    tick = time.perf_counter()
                    # Undo the CLI Windows retry shim after each in-process trial.
                    with redirect_stdout(capture), patch.object(os, "replace", os.replace), \
                            patch.object(builtins, "open", builtins.open):
                        code = E.main(cmd)
                    wall = time.perf_counter() - tick
                    if code:
                        raise RuntimeError(f'{name}: exit {code}')
                    metrics = Counter()
                    for line in capture.getvalue().splitlines():
                        if line.startswith('io_profile '):
                            profile = json.loads(line.partition(' ')[2])
                            metrics.update(profile)
                            print('profile', name, profile, flush=True)
                    records = {r['field_id']: (r['n_scenes'], r['sha256']) for r in E.read_manifest(store)}
                    root = E.P.open_store(store)
                    proof = fingerprint(root), records
                    if (not args.synthetic and records != ANCHORS) or (oracle is not None and proof != oracle):
                        raise RuntimeError(f'Candidate does not pass byte-identity; discard it. {name}: {proof}')
                    if oracle is None:
                        oracle = proof
                        print('chunks/metadata', root['values'].chunks, root['values'].info)
                        if args.sweep:
                            probe_codec(root)
                    samples[name].append([wall, metrics['batch_wall_s'], metrics['scene_read_s'],
                                          metrics['aggregation_s'], metrics['write_readback_s']])
                    print('PASS', name, trial+1, proof, flush=True)
        print('Medians (s): method, end-to-end, batch wall, scene_read, aggregation, write/readback')
        for name, rows in samples.items():
            print(name, *(round(median(column), 4) for column in zip(*rows)))
        ratio = median(r[0] for r in samples['baseline']) / median(r[0] for r in samples['candidate'])
        print(f'End-to-end baseline/candidate: {ratio:.3f}x')
        if not args.synthetic and ratio < 2:
            raise SystemExit('Speed gate failed: do not switch.')


if __name__ == '__main__':
    main()
