"""Cross-platform resolution of the thesis data archive.

The canonical archive is ``D:\\RS\\Thesis`` on the owner's Windows machine.
On the Linux mirror it is mounted at ``/home/mahdi/RS/Thesis``. Scripts keep
their Windows-style defaults (so the owner's commands stay unchanged); this
helper maps those paths to the local mount when running on non-Windows, and is
a no-op on Windows. The local root can be overridden with ``THESIS_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

_WINDOWS_ROOT = "d:/rs/thesis"


def localize(path: str | Path) -> Path:
    """Return ``path`` as a Path, translating the Windows archive root on Linux.

    Only paths under ``D:/RS/Thesis`` are translated; everything else is
    returned untouched. On Windows this is the identity function.
    """
    p = Path(path)
    if os.name == "nt":
        return p
    text = str(p).replace("\\", "/")
    if not text.lower().startswith(_WINDOWS_ROOT):
        return p
    local_root = Path(os.environ.get("THESIS_ROOT", "/home/mahdi/RS/Thesis"))
    rest = text[len(_WINDOWS_ROOT) :].lstrip("/")
    return local_root / rest if rest else local_root
