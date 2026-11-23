"""Opt-in fingerprinted disk cache for deterministic eval intermediates.

Validation-ladder rung **L0** (infrastructure only). A cache entry contains
bytes a cold run already produced; it never creates, changes, labels or
fabricates a research value. With no flag set, this module is never imported by
the eval path and no file is read or written.

Three layers, promoted in the order they are consumed:

* ``l1`` - normalized patch batch (the store -> in-memory boundary).
* ``l2`` - clean calibration reference paths for the empirical p-value.
* ``l3`` - per-cell injected tensor ``Xinj`` plus its hash-bearing records.

Safety rules, all enforced here:

* **Content fingerprints, not paths.** A key is a SHA-256 over a canonical
  serialization of every input that can change the value, plus a format
  version. Changing any input changes the key.
* **Fail open.** A miss, corrupt file, missing key, shape/dtype mismatch or
  hash mismatch returns ``None`` (recompute). A cache problem never fails an
  eval run and never returns a partial artifact.
* **Atomic writes.** temp file + ``os.replace``; a reader sees a complete
  entry or none.
* **Bounded.** a cell above ``max_cell_bytes`` is not written and the cold
  path runs with a diagnostic.
* **Untracked.** the default root is ``data/cache/eval/`` (Git ignores
  ``/data/``).

Environment switches (read at construction time):

    EVAL_CACHE_READ=1            reuse existing valid entries
    EVAL_CACHE_WRITE=1           create entries (a separate, explicit action)
    EVAL_CACHE_DIR=<path>        override the cache root
    EVAL_CACHE_MAX_CELL_MB=<int> per-cell byte cap (default 512)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

CACHE_FORMAT = 1

ENV_READ = "EVAL_CACHE_READ"
ENV_WRITE = "EVAL_CACHE_WRITE"
ENV_DIR = "EVAL_CACHE_DIR"
ENV_MAX_MB = "EVAL_CACHE_MAX_CELL_MB"

DEFAULT_DIR = "data/cache/eval"
DEFAULT_MAX_CELL_MB = 512

_LAYERS = ("l1", "l2", "l3")
_NAN_SENTINEL = -9.87654321e30

# Private keys written onto a batch dict to avoid re-hashing large arrays for
# every severity cell. They are cache metadata only and are never scored.
_BATCH_FP_KEY = "_eval_cache_batch_fp"


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------
def _array_digest(value: np.ndarray) -> dict[str, str]:
    """Shape/dtype-tagged, NaN-normalized digest of one array.

    NaN payloads are not stable across reads, so NaN entries are mapped to a
    fixed sentinel before hashing. This is a *hashing* normalization only: the
    cached payloads themselves are stored bit-exact and never altered.
    """
    a = np.asarray(value)
    if a.dtype.kind == "f":
        a = np.where(np.isnan(a), np.asarray(_NAN_SENTINEL, dtype=a.dtype), a)
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii"))
    h.update(str(a.dtype.str).encode("ascii"))
    h.update(a.tobytes())
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": h.hexdigest()}


def _canonical(value: Any) -> Any:
    """Recursively reduce an input to a JSON-serializable canonical form."""
    if isinstance(value, np.ndarray):
        return _array_digest(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.dtype):
        return value.str
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def fingerprint(*parts: Any) -> str:
    """Stable SHA-256 over the canonical serialization of ``parts``."""
    payload = json.dumps(_canonical(list(parts)), sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def batch_fingerprint(batch: dict) -> str:
    """One fingerprint over every batch field that affects injection.

    Memoized on the batch dict itself: ``patch_arm_scores`` is called once per
    severity cell with the same batch, so re-hashing ~tens of MB per cell would
    erase part of the speedup. The memo key is cache metadata, never scored.
    """
    hit = batch.get(_BATCH_FP_KEY)
    if isinstance(hit, str):
        return hit
    fields = ("raw", "obs", "in_field", "sensor_id", "norm_stats")
    parts = {name: batch[name] for name in fields if name in batch}
    parts["X_dtype"] = str(batch["X"].dtype) if "X" in batch else "float32"
    parts["cache_format"] = CACHE_FORMAT
    fp = fingerprint(parts)
    batch[_BATCH_FP_KEY] = fp
    return fp


def l3_key(batch: dict, *, level_id: str, family: str, index: str, round_idx: int,
           basis: str, delta: float, duration: int) -> str:
    """Key for one severity cell's injected tensor + hash records."""
    return fingerprint(
        CACHE_FORMAT, "l3", batch_fingerprint(batch),
        level_id, family, index, int(round_idx), str(basis),
        float(delta), int(duration),
        "rng=default_rng(round_idx)",  # a caller-supplied RNG bypasses the cache
    )


def l2_key(*, model_id: str, sigma_sha256: str, alignment: str,
           idx: np.ndarray, field_row: np.ndarray | None, chunk: int,
           x: np.ndarray, pixel_valid: np.ndarray,
           scoring_inputs: dict[str, np.ndarray]) -> str:
    """Key for the clean reference paths used by the empirical p-value."""
    return fingerprint(
        CACHE_FORMAT, "l2", str(model_id), str(sigma_sha256), str(alignment),
        idx, field_row, int(chunk), x, pixel_valid, scoring_inputs,
    )


def store_fingerprint(root: Any) -> str:
    """Cheap content digest of the patch store metadata an L1 entry depends on.

    Reads only the small provenance arrays and ``attrs``; value chunks are NOT
    read (that is what the cache exists to avoid).
    """
    h = hashlib.sha256()
    attrs = dict(getattr(root, "attrs", {}) or {})
    h.update(json.dumps(_canonical(attrs), sort_keys=True, default=str).encode("utf-8"))
    for name in ("field_ids", "cycle_ids", "cycle_start_day", "cycle_end_day", "cycle_valid"):
        a = np.ascontiguousarray(np.asarray(root[name][:]))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def l1_key(root: Any, field_ids: np.ndarray, cycle_ids: np.ndarray, norm: dict) -> str:
    """Key for a normalized batch built from a frozen patch store."""
    return fingerprint(
        CACHE_FORMAT, "l1", store_fingerprint(root),
        np.asarray(field_ids, dtype=np.int64), np.asarray(cycle_ids, dtype=np.int64),
        norm,
    )


# ---------------------------------------------------------------------------
# Cache handle
# ---------------------------------------------------------------------------
class EvalArtifactCache:
    """Opt-in, fail-open handle to the on-disk eval cache.

    ``from_env()`` is the normal constructor. Passing explicit flags is useful
    in tests and for a single command. A handle with neither read nor write
    enabled reports ``enabled == False`` and is never wired in by callers.
    """

    def __init__(self, root: str | Path | None = None, *, read: bool = False,
                 write: bool = False, max_cell_bytes: int | None = None) -> None:
        self.root = Path(root) if root is not None else self._default_root()
        self.read = bool(read)
        self.write = bool(write)
        self.max_cell_bytes = int(max_cell_bytes) if max_cell_bytes is not None else (
            DEFAULT_MAX_CELL_MB * 1024 * 1024
        )
        if self.max_cell_bytes <= 0:
            raise ValueError("eval cache max_cell_bytes must be positive")
        self.stats: dict[str, int] = {"hit": 0, "miss": 0, "write": 0,
                                      "skip_size": 0, "invalidated": 0, "error": 0}
        self.events: list[str] = []

    @staticmethod
    def _default_root() -> Path:
        # Resolve relative to the repository root (two levels above this file's
        # package: monitor -> ssl_anomaly -> src -> repo root).
        repo = Path(__file__).resolve().parents[3]
        raw = os.environ.get(ENV_DIR, DEFAULT_DIR)
        path = Path(raw)
        return path if path.is_absolute() else repo / path

    @classmethod
    def from_env(cls) -> "EvalArtifactCache":
        read = os.environ.get(ENV_READ, "") == "1"
        write = os.environ.get(ENV_WRITE, "") == "1"
        mb = os.environ.get(ENV_MAX_MB, "")
        max_bytes = int(float(mb) * 1024 * 1024) if mb else None
        return cls(os.environ.get(ENV_DIR), read=read, write=write,
                   max_cell_bytes=max_bytes)

    @property
    def enabled(self) -> bool:
        return self.read or self.write

    def _note(self, message: str) -> None:
        self.events.append(message)

    def _dir(self, layer: str, key: str) -> Path:
        if layer not in _LAYERS:
            raise ValueError(f"unknown cache layer {layer!r}")
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("cache key must be a 64-char lowercase sha256")
        return self.root / layer / key

    # -- generic read / write ---------------------------------------------
    def load(self, layer: str, key: str,
             arrays: tuple[str, ...] | None = None) -> tuple[dict[str, np.ndarray], dict] | None:
        """Return ``(arrays, meta)`` on a valid hit, else ``None`` (fail open).

        ``arrays=None`` reads the array names from the entry's own metadata.
        This lets L1 tolerate a batch that legitimately omits an optional field.
        """
        if not self.read:
            return None
        entry = self._dir(layer, key)
        try:
            meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
            if meta.get("cache_format") != CACHE_FORMAT:
                raise ValueError("cache format mismatch")
            if meta.get("fingerprint") != key:
                raise ValueError("fingerprint mismatch")
            if arrays is None:
                arrays = tuple(meta.get("arrays", []))
            if set(meta.get("arrays", [])) != set(arrays):
                raise ValueError("array set mismatch")
            with np.load(entry / "data.npz", allow_pickle=False) as data:
                out = {name: np.asarray(data[name]) for name in arrays}
            expected = meta.get("shapes", {})
            for name, arr in out.items():
                if list(arr.shape) != expected.get(name):
                    raise ValueError(f"shape mismatch for {name}")
            self.stats["hit"] += 1
            return out, meta
        except Exception as exc:  # noqa: BLE001 - fail open is the contract
            self.stats["invalidated" if entry.exists() else "miss"] += 1
            self._note(f"eval cache {layer}/{key[:12]} invalidated: {type(exc).__name__}: {exc}")
            return None

    def save(self, layer: str, key: str, arrays: dict[str, np.ndarray],
             meta_extra: dict | None = None) -> bool:
        """Atomically write an entry. Never raise; return True only on success."""
        if not self.write:
            return False
        total = sum(int(np.asarray(a).nbytes) for a in arrays.values())
        if total > self.max_cell_bytes:
            self.stats["skip_size"] += 1
            self._note(
                f"eval cache {layer}/{key[:12]} skipped: {total} bytes exceeds "
                f"cap {self.max_cell_bytes}"
            )
            return False
        entry = self._dir(layer, key)
        try:
            entry.mkdir(parents=True, exist_ok=True)
            meta = {
                "cache_format": CACHE_FORMAT,
                "fingerprint": key,
                "arrays": sorted(arrays),
                "shapes": {name: list(np.asarray(a).shape) for name, a in arrays.items()},
                "dtypes": {name: str(np.asarray(a).dtype.str) for name, a in arrays.items()},
            }
            if meta_extra:
                meta.update(meta_extra)
            # Two temp files in the same directory, then atomic renames.
            with tempfile.NamedTemporaryFile(dir=entry, suffix=".npz.tmp",
                                             delete=False) as tmp:
                npz_path = Path(tmp.name)
            with tempfile.NamedTemporaryFile(dir=entry, suffix=".json.tmp", mode="w",
                                             encoding="utf-8", delete=False) as tmp:
                json_path = Path(tmp.name)
                json.dump(meta, tmp, sort_keys=True, allow_nan=False)
            # Open a binary handle: passing the Path would make numpy append a
            # second '.npz' suffix, leaving an empty temp file as the entry.
            with npz_path.open("wb") as fh:
                np.savez(fh, **{k: np.asarray(v) for k, v in arrays.items()})
            os.replace(npz_path, entry / "data.npz")
            os.replace(json_path, entry / "meta.json")
            self.stats["write"] += 1
            return True
        except Exception as exc:  # noqa: BLE001 - a cache failure is not an eval failure
            self.stats["error"] += 1
            self._note(f"eval cache {layer}/{key[:12]} write failed: {type(exc).__name__}: {exc}")
            for p in (entry / "data.npz", entry / "meta.json"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            return False

    # -- typed layer helpers ----------------------------------------------
    def load_l1(self, key: str) -> dict | None:
        got = self.load("l1", key, None)
        if got is None:
            return None
        arrays, meta = got
        out = dict(arrays)
        out["unnormalized_channels"] = list(meta.get("unnormalized_channels", []))
        out["cycles_found"] = int(meta.get("cycles_found", 0))
        out["norm_stats"] = meta.get("norm_stats", {})
        return out

    def save_l1(self, key: str, batch: dict) -> bool:
        names = ("X", "raw", "obs", "in_field", "sensor_id", "covariates", "pixel_valid")
        arrays = {name: batch[name] for name in names if name in batch}
        return self.save("l1", key, arrays, {
            "unnormalized_channels": list(batch.get("unnormalized_channels", [])),
            "cycles_found": int(batch.get("cycles_found", 0)),
            "norm_stats": batch.get("norm_stats", {}),
        })

    def load_l2(self, key: str) -> np.ndarray | None:
        got = self.load("l2", key, ("paths",))
        return None if got is None else got[0]["paths"]

    def save_l2(self, key: str, paths: np.ndarray) -> bool:
        return self.save("l2", key, {"paths": np.asarray(paths)})

    def load_l3(self, key: str) -> tuple[np.ndarray, list[dict]] | None:
        got = self.load("l3", key, ("Xinj",))
        if got is None:
            return None
        arrays, meta = got
        try:
            records = json.loads((self._dir("l3", key) / "records.json").read_text(
                encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError("injection records must be a list")
        except Exception as exc:  # noqa: BLE001
            self.stats["invalidated"] += 1
            self._note(f"eval cache l3/{key[:12]} records invalidated: {exc}")
            return None
        return arrays["Xinj"], records

    def save_l3(self, key: str, Xinj: np.ndarray, records: list[dict]) -> bool:
        ok = self.save("l3", key, {"Xinj": np.asarray(Xinj)})
        if not ok:
            return False
        try:
            entry = self._dir("l3", key)
            with tempfile.NamedTemporaryFile(dir=entry, suffix=".json.tmp", mode="w",
                                             encoding="utf-8", delete=False) as tmp:
                json_path = Path(tmp.name)
                json.dump(records, tmp, sort_keys=True, allow_nan=False)
            os.replace(json_path, entry / "records.json")
            return True
        except Exception as exc:  # noqa: BLE001
            self.stats["error"] += 1
            self._note(f"eval cache l3/{key[:12]} records write failed: {exc}")
            return False


def resolve_cache(cache: Any) -> EvalArtifactCache | None:
    """Resolve the ``cache`` argument of a scoring function.

    * ``None``  -> build from environment; return ``None`` when no flag is set.
    * ``False`` -> explicitly off, even if environment flags are set.
    * an :class:`EvalArtifactCache` -> used as-is.
    """
    if cache is False:
        return None
    if cache is None:
        handle = EvalArtifactCache.from_env()
        return handle if handle.enabled else None
    return cache


__all__ = [
    "CACHE_FORMAT", "DEFAULT_DIR", "DEFAULT_MAX_CELL_MB",
    "ENV_DIR", "ENV_MAX_MB", "ENV_READ", "ENV_WRITE",
    "EvalArtifactCache", "batch_fingerprint", "fingerprint", "l1_key", "l2_key",
    "l3_key", "resolve_cache", "store_fingerprint",
]
