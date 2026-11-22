"""partition_calib_eval: the field-level calib/eval split (audit item 13).

The per-cycle hash partition let one field's cycles straddle the calibration /
evaluation boundary — within-dev spatial bleed at threshold-calibration
stakes. The split is now FIELD-level: every field lands entirely on one side.
"""

import pytest

from scripts.eval_dev import partition_calib_eval


def _keys(n_fields: int, cycles_per_field: int = 3) -> list[str]:
    return [f"{f}:{c}" for f in range(100, 100 + n_fields)
            for c in range(cycles_per_field)]


def test_no_field_straddles_the_boundary():
    keys = _keys(10)
    calib, ev, _ = partition_calib_eval(keys)
    calib_fields = {k.split(":", 1)[0] for k in (keys[i] for i in calib)}
    eval_fields = {k.split(":", 1)[0] for k in (keys[i] for i in ev)}
    assert not (calib_fields & eval_fields), "a field's cycles straddle the split"


def test_partition_is_deterministic_and_covers_everything():
    keys = _keys(12)
    c1, e1, h1 = partition_calib_eval(keys)
    c2, e2, h2 = partition_calib_eval(keys)
    assert (c1, e1, h1) == (c2, e2, h2)
    assert sorted(c1 + e1) == list(range(len(keys)))
    assert not set(c1) & set(e1)


def test_both_sides_are_nonempty_and_sized():
    keys = _keys(10, cycles_per_field=2)
    calib, ev, _ = partition_calib_eval(keys, frac_eval=0.5)
    assert calib and ev
    assert len(ev) == 10  # 5 of 10 fields × 2 cycles


def test_fewer_than_two_fields_is_a_hard_error():
    with pytest.raises(ValueError, match="≥2 distinct fields"):
        partition_calib_eval([f"7:{c}" for c in range(8)])


def test_eval_hash_is_stable_and_key_derived():
    keys = _keys(8)
    _, _, h1 = partition_calib_eval(keys)
    _, _, h2 = partition_calib_eval(keys)
    assert h1 == h2 and len(h1) == 16
    _, _, h3 = partition_calib_eval(keys, salt="different-salt")
    assert h3 != h1
