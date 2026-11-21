"""Small, explicitly synthetic locked split provenance for patch tests."""

from ssl_anomaly.io import patches as P
from ssl_anomaly.monitor import holdout as H
from ssl_anomaly.monitor import splits as S


def bind_patch_store(
    root,
    *,
    spatial=(103,),
    train_keys=None,
    dev_keys=(),
    known=(101, 102, 103),
    cutoff="2024-01-01",
    parquet_path=None,
):
    """Lock a synthetic split manifest around the fixture store.

    With ``parquet_path`` the fixture also writes the manifest-pinned cycles
    table (``field, cycle_id, t_start, t_end, censored``) from the store's
    own windows and records its real sha256, exercising the v2 authority
    path (``eval_dev._patch_dev_panel``) exactly like the production chain.
    """
    if train_keys is None:
        train_keys = [
            c.key
            for c in P.patch_cycle_index(root).values()
            if c.field_id in known and c.field_id not in spatial and c.key not in dev_keys
        ]
    sha = "a" * 64
    cycle_files = None
    if parquet_path is not None:
        import hashlib

        import pandas as pd

        rows = [
            {"field": c.field_id, "cycle_id": c.cycle_id,
             "t_start": int(c.start_day), "t_end": int(c.end_day) + 1,
             "censored": False}
            for c in P.patch_cycle_index(root).values()
        ]
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        cycle_files = {"parquet": str(parquet_path)}
    splits = {
        "base_date": "2013-03-30",
        "test_period_start": cutoff,
        "cycle_file_hashes": {"cycles_parquet_sha256": sha},
        "kmeans": {"test_block_fields": list(spatial)},
        "train_fields": sorted(set(known) - set(spatial)),
        "test_fields": list(spatial),
        "cycles": {"train": list(train_keys), "dev": list(dev_keys), "test": []},
    }
    if cycle_files is not None:
        splits["cycle_files"] = cycle_files
    splits["manifest_sha256"] = S.manifest_hash(splits)
    root.attrs.update(
        {
            "cycle_epoch": splits["base_date"],
            "cycle_source_sha256": sha,
            "splits_manifest_sha256": splits["manifest_sha256"],
        }
    )
    return splits, H.build_holdout_manifest(splits)
