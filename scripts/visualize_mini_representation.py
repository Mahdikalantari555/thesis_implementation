"""Supervisor-facing representation figures from the real pilot.

UMAP and t-SNE both crash natively in this conda environment (matplotlib 3.11.1
+ numpy 2.5.2 + threadpoolctl on Windows, exit 0xC06D007F), so we fall back
to a **deterministic, pure-Python 2-D projection**: for each embedding, the two
dimensions with the highest Fisher ratio for the grouping variable (farm or
stage). This is a stable, honest substitute — it shows whether the embedding's
most informative axes separate farms/phases, but it is NOT UMAP/t-SNE and must
not be labelled as such.

Every value plotted comes from `representation_first_pilot_embeddings.npz`
(frozen SSL embeddings, matched random-init embeddings, raw features, metadata)
and `representation_first_pilot.json`.

Outputs (reports/mini_dataset/figures/):
  representation_structure.png     — 2×3 grid: SSL / raw / random, coloured by
                                    farm (top) and stage (bottom).
  representation_cross_year.png   — SSL top-2 dims with train/test overlay +
                                    metric bars.
  representation_readme.md        — how to read, provenance, caveats.

Usage::

    python scripts/visualize_mini_representation.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "reports" / "mini_dataset" / "representation_first_pilot_embeddings.npz"
DEFAULT_REPORT = ROOT / "reports" / "mini_dataset" / "representation_first_pilot.json"
DEFAULT_DIR = ROOT / "reports" / "mini_dataset" / "figures"

FARM_COLORS = {"R12-06": (31, 119, 180), "R12-08": (255, 127, 14), "R12-10": (44, 160, 44)}
STAGE_COLORS = {"S1": (78, 121, 167), "S2": (242, 142, 43), "S3": (225, 87, 89), "S4": (89, 161, 79)}

FONT_CANDIDATES = [
    Path("D:/Dev/Miniforge3/envs/geo/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]
FONT_PATH = next((p for p in FONT_CANDIDATES if p.exists()), None)
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = (
            ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH else ImageFont.load_default()
        )
    return _FONT_CACHE[size]


def _top_dims(x: np.ndarray, labels: np.ndarray, k: int = 2) -> np.ndarray:
    """Two dimensions with highest Fisher ratio for `labels` (pure Python)."""
    classes = np.unique(labels)
    scores = np.zeros(x.shape[1], dtype=np.float64)
    for d in range(x.shape[1]):
        col = x[:, d]
        grand = float(col.mean())
        between = 0.0
        within = 0.0
        for c in classes:
            m = labels == c
            if m.sum() < 2:
                continue
            mu = float(col[m].mean())
            between += float(m.sum()) * (mu - grand) ** 2
            within += float(np.sum((col[m] - mu) ** 2))
        scores[d] = between / max(within, 1e-12)
    return np.argsort(scores)[::-1][:k]


def _blend(fg, bg, a):
    return tuple(int(round(a * f + (1 - a) * b)) for f, b in zip(fg, bg))


class Canvas:
    def __init__(self, w, h, title, footer):
        self.im = Image.new("RGB", (w, h), "white")
        self.d = ImageDraw.Draw(self.im)
        self.text((w // 2, 18), title, 22, (20, 30, 45), anchor="mm", bold=True)
        self.text((w // 2, h - 22), footer, 15, (90, 50, 10), anchor="mm")

    def text(self, xy, s, size=14, fill=(30, 30, 30), anchor="la", bold=False):
        f = _font(size)
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor,
                    stroke_width=1 if bold else 0, stroke_fill=fill if bold else None)

    def circle(self, x, y, r, fill):
        self.d.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def rect(self, box, fill=None, outline=None):
        self.d.rectangle(box, fill=fill, outline=outline)

    def save(self, path):
        self.im.save(path)


def _scatter_panel(c: Canvas, box, x2d, color_groups, title):
    x0, y0, x1, y1 = box
    pad = 34
    xL, xR, yT, yB = x0 + pad, x1 - 12, y0 + 34, y1 - 12
    c.rect((x0, y0, x1, y1), fill=(255, 255, 255), outline=(215, 220, 226))
    for i in range(5):
        xx = xL + i * (xR - xL) / 4
        yy = yT + i * (yB - yT) / 4
        c.d.line([(xx, yT), (xx, yB)], fill=(240, 240, 240))
        c.d.line([(xL, yy), (xR, yy)], fill=(240, 240, 240))
    xs, ys = x2d[:, 0], x2d[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)
    for g, col in color_groups["map"].items():
        sel = color_groups["groups"] == g
        for xi, yi in zip(xs[sel], ys[sel]):
            px = xL + (xi - xmin) / xr * (xR - xL)
            py = yB - (yi - ymin) / yr * (yB - yT)
            c.circle(px, py, 2, col)
    c.text((x0 + 8, y0 + 6), title, 15, (25, 35, 50), bold=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    z, z_rand, raw = d["z"], d["z_rand"], d["raw"]
    farms, stages = d["farms"], d["stages"]
    years = d["years"]

    # --- Figure 1: structure, farm (top) + stage (bottom) -----------------
    c = Canvas(1820, 1040,
               "What did the SSL encoder actually learn? — mini dataset, L1 screening",
               "Each panel uses its two most discriminative dimensions (Fisher ratio). "
               "NOT a stress claim. UMAP/t-SNE unavailable in this environment (native crash).")
    reps = [("Frozen SSL", z), ("Raw spectral + weather", raw), ("Random-init encoder", z_rand)]
    for j, (name, x) in enumerate(reps):
        dims = _top_dims(x, farms, 2)
        _scatter_panel(c, (40 + j * 600, 60, 40 + (j + 1) * 600 - 20, 500),
                      x[:, dims], {"map": FARM_COLORS, "groups": farms},
                      f"{name} — by FARM")
        dims = _top_dims(x, stages, 2)
        _scatter_panel(c, (40 + j * 600, 520, 40 + (j + 1) * 600 - 20, 960),
                      x[:, dims], {"map": STAGE_COLORS, "groups": stages},
                      f"{name} — by STAGE")
    # legends
    lx, ly = 50, 975
    for k, col in FARM_COLORS.items():
        c.circle(lx, ly, 5, col); c.text((lx + 10, ly - 6), k, 12, (60, 60, 60)); lx += 110
    lx = 500
    for k, col in STAGE_COLORS.items():
        c.circle(lx, ly, 5, col); c.text((lx + 10, ly - 6), k, 12, (60, 60, 60)); lx += 90
    p1 = args.out / "representation_structure.png"
    c.save(p1)
    print(f"wrote {p1}", flush=True)

    # --- Figure 2: cross-year overlap + metric bars ------------------------
    c2 = Canvas(1320, 620, "Representation quality ≠ suspected-stress discrimination",
                "Caveat: near-duplicate windows remain; shuffled-year control did NOT collapse (R@1≈0.999); "
                "effective rank 1.9/16. Treat the R@1 gap as preliminary.")
    # left: SSL top dims, train/test overlay for one farm/stage
    dims = _top_dims(z, farms, 2)
    x2 = z[:, dims]
    xmin, xmax = float(x2[:, 0].min()), float(x2[:, 0].max())
    ymin, ymax = float(x2[:, 1].min()), float(x2[:, 1].max())
    xr, yr = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    box = (40, 60, 620, 560)
    x0, y0, x1, y1 = box
    xL, xR, yT, yB = x0 + 30, x1 - 12, y0 + 30, y1 - 12
    c2.rect(box, fill=(255, 255, 255), outline=(215, 220, 226))
    target_farm, target_stage = "R12-08", "S3"
    sel = (farms == target_farm) & (stages == target_stage)
    for is_test, color in ((False, (31, 119, 180)), (True, (225, 87, 89))):
        m = sel & ((years >= 2023) if is_test else (years < 2023))
        for xi, yi in zip(x2[m, 0], x2[m, 1]):
            px = xL + (xi - xmin) / xr * (xR - xL)
            py = yB - (yi - ymin) / yr * (yB - yT)
            c2.circle(px, py, 4, color)
    c2.text((x0 + 8, y0 + 6), f"Frozen SSL — {target_farm}, stage {target_stage}\n"
                                f"blue = train years, red = test years", 14, (25, 35, 50), bold=True)
    # right: metric bars
    bx0, by0, bx1, by1 = 660, 60, 1280, 560
    c2.rect((bx0, by0, bx1, by1), fill=(255, 255, 255), outline=(215, 220, 226))
    labels = ["SSL", "Raw", "Random"]
    r1 = [report["cysp_rg"]["ssl"]["r_at_1"], report["cysp_rg"]["raw"]["r_at_1"],
          report["cysp_rg_random_init"]["ssl"]["r_at_1"]]
    aprc = [report["probe_ssl_embedding"]["auprc"], report["probe_raw_features"]["auprc"],
            report["probe_random_embedding"]["auprc"]]
    ymaxv = max(max(r1), max(aprc)) * 1.25
    barw = 55
    for i in range(3):
        cx = bx0 + 120 + i * 160
        h1 = r1[i] / ymaxv * (by1 - by0 - 80)
        h2 = aprc[i] / ymaxv * (by1 - by0 - 80)
        c2.rect((cx - barw, by1 - 60 - h1, cx - 4, by1 - 60), fill=(78, 121, 167))
        c2.rect((cx + 4, by1 - 60 - h2, cx + barw, by1 - 60), fill=(225, 87, 89))
        c2.text((cx, by1 - 46), labels[i], 13, (60, 60, 60), anchor="mm")
        c2.text((cx - barw // 2, by1 - 60 - h1 - 12), f"{r1[i]:.3f}", 11, (40, 40, 40), anchor="mm")
        c2.text((cx + barw // 2, by1 - 60 - h2 - 12), f"{aprc[i]:.3f}", 11, (40, 40, 40), anchor="mm")
    c2.text((bx0 + 8, by0 + 6), "blue = cross-year R@1, red = suspected AUPRC", 14,
            (25, 35, 50), bold=True)
    p2 = args.out / "representation_cross_year.png"
    c2.save(p2)
    print(f"wrote {p2}", flush=True)

    (args.out / "representation_readme.md").write_text(
        "# How to read these figures\n\n"
        "- `representation_structure.png`: each panel uses its two most discriminative dimensions "
        "(Fisher ratio for the colour variable) of the frozen SSL / raw / random-init embeddings. "
        "Top row coloured by farm, bottom row by stage. Tighter clusters = the representation carries "
        "that structure better than the others.\n"
        "- `representation_cross_year.png`: left, one farm/stage with train (blue) vs test (red) years "
        "overlaid on the SSL top-2 dims; right, the two metric families side by side.\n\n"
        "**Renderer note:** UMAP and t-SNE crash natively in this conda environment, so a deterministic "
        "pure-Python 2-D projection (top-2 Fisher-ratio dimensions) is used instead. Do not label it UMAP/t-SNE.\n\n"
        "**Do not claim:** stress detection, early warning, geographic generalization, or SSL superiority. "
        "L1/L2 screening on 3 adjacent farms with weak suspicion labels; CWSI shares inputs with the model. "
        "The retrieval metric has a failed negative control and the embedding has low effective rank.\n",
        encoding="utf-8",
    )
    print("wrote representation_readme.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
