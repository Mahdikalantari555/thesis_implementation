"""Supervisor-facing TIMELINE figures from the real mini dataset.

Rendered with Pillow because matplotlib's native drawing backend crashes in
this conda environment (matplotlib 3.11.1 + numpy 2.5.2 on Windows); Pillow
drawing was verified working. Every plotted value comes from real artifacts
under Data/Processed/:

* NDVI and LST: field-mean of mini satellite.zarr pixels (red/nir/lst bands),
  masked by pixel_valid AND in_field. LST Kelvin -> Celsius (273.15).
* Crop cycles: phenology/cycles_mini.parquet (start_date / end_date / status).
* Flagged dates: suspected_stress/suspected_patches.parquet (weak suspicion
  from re-masked METRIC CWSI_ET / ETaDaily / ETrF / LAI / TVDI products).
* Anomaly score: a DERIVED, phase-conditioned robust deviation — for each
  farm × stage S1–S4, median/MAD of the field-mean value over TRAIN years only
  (< 2023), then (value - median) / (1.4826*MAD + floor). NOT a validated
  anomaly, NOT model inference, NOT stress ground truth.

Outputs (reports/mini_dataset/figures/):
  timeline_seasonal.png      — NDVI & LST curves with cycle start/end and flags.
  timeline_anomaly_score.png — phase-conditioned deviation of NDVI & LST.
  timeline_flagged_dates.png — which dates are flagged + year x month heatmap.
  timeline_readme.md         — how to read, provenance, caveats.

Usage::

    python scripts/visualize_mini_timeline.py
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import zarr
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from ssl_anomaly.utils.paths import localize  # noqa: E402

MINI = localize("D:/RS/Thesis/Data/Processed/mini_research_dataset")
STORE = MINI / "satellite.zarr"
CYCLES = MINI / "phenology" / "cycles_mini.parquet"
PATCHES = MINI / "suspected_stress" / "suspected_patches.parquet"
DEFAULT_OUT = ROOT / "reports" / "mini_dataset" / "figures"

BAND_ORDER = ["blue", "green", "red", "nir", "swir1", "swir2", "lst"]
RED_I, NIR_I, LST_I = BAND_ORDER.index("red"), BAND_ORDER.index("nir"), BAND_ORDER.index("lst")
STAGE_EDGES = (0.0, 0.25, 0.60, 0.85, 1.0001)
STAGE_NAMES = ("S1", "S2", "S3", "S4")
TRAIN_CUTOFF = np.datetime64("2023-01-01", "D")
TRAIN_DAY = int(TRAIN_CUTOFF.astype("datetime64[D]").astype(int))
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
# Farm names come from the store (selection layer 3farms_dev), never hard-coded.
FARMS: tuple[str, ...] = ()
FARM_COLORS: dict[str, tuple[int, int, int]] = {}


def set_farms(names: list[str]) -> None:
    global FARMS, FARM_COLORS
    FARMS = tuple(sorted(names))
    FARM_COLORS = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(FARMS)}

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


# ----------------------------- data ----------------------------------------

def _stage_of(date: np.datetime64, cycles: pd.DataFrame) -> str | None:
    for _, c in cycles.iterrows():
        start = np.datetime64(str(c["start_date"]), "D")
        end = np.datetime64(str(c["end_date"]), "D")
        if start <= date <= end:
            progress = float((date - start).astype(int)) / max(1, int((end - start).astype(int)))
            for i in range(4):
                if STAGE_EDGES[i] <= progress < STAGE_EDGES[i + 1]:
                    return STAGE_NAMES[i]
    return None


def _farm_series(farm: str) -> dict:
    g = zarr.open_group(str(STORE / farm), mode="r")
    dates = np.array([np.datetime64(str(x), "D") for x in g["dates"][:]])
    data = np.asarray(g["data"], dtype=np.float32)
    valid = np.asarray(g["pixel_valid"], dtype=bool)
    in_field = np.asarray(g["in_field"], dtype=bool)
    mask = valid & in_field[None, :, :]
    red = data[:, RED_I].astype(np.float64)
    nir = data[:, NIR_I].astype(np.float64)
    lst_c = data[:, LST_I].astype(np.float64) - 273.15
    den = nir + red
    ndvi = np.where(np.abs(den) > 1e-8, (nir - red) / den, np.nan)
    ndvi = np.where(np.isfinite(ndvi) & mask, ndvi, np.nan)
    lst_c = np.where(np.isfinite(lst_c) & mask, lst_c, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        n_mean = np.nanmean(ndvi, axis=(1, 2))
        l_mean = np.nanmean(lst_c, axis=(1, 2))
    return {"dates": dates, "ndvi": n_mean, "lst_c": l_mean}


def _phase_score(values: np.ndarray, dates: np.ndarray, stages: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for stage in STAGE_NAMES:
        sm = stages == stage
        base = sm & (dates < TRAIN_CUTOFF)
        if base.sum() < 5:
            continue
        med = float(np.median(values[base]))
        mad = float(np.median(np.abs(values[base] - med)))
        floor = max(1.4826 * mad, 0.05 * max(1.0, abs(med)))
        out[sm] = (values[sm] - med) / floor
    return out


# -------------------------- drawing helpers --------------------------------

def _blend(fg: tuple[int, int, int], bg: tuple[int, int, int], a: float) -> tuple[int, int, int]:
    return tuple(int(round(a * f + (1 - a) * b)) for f, b in zip(fg, bg))


class Canvas:
    def __init__(self, w: int, h: int, title: str, footer: str):
        self.im = Image.new("RGB", (w, h), "white")
        self.d = ImageDraw.Draw(self.im)
        self.text((w // 2, 16), title, 24, (20, 30, 45), anchor="mm", bold=True)
        self.text((w // 2, h - 26), footer, 17, (90, 50, 10), anchor="mm")

    def text(self, xy, s, size=15, fill=(30, 30, 30), anchor="la", bold=False):
        f = _font(size)
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor,
                    stroke_width=1 if bold else 0,
                    stroke_fill=fill if bold else None)

    def line(self, pts, fill, width=2):
        self.d.line(pts, fill=fill, width=width, joint="curve")

    def rect(self, box, fill=None, outline=None, width=1):
        self.d.rectangle(box, fill=fill, outline=outline, width=width)

    def circle(self, x, y, r, fill):
        self.d.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def save(self, path: Path):
        self.im.save(path)


def _panel(c: Canvas, box, dates, values, title, color, cycles, flags,
           hlines=(), show_x=True):
    """Draw one time-series panel into `box` = (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    xL, xR = x0 + 66, x1 - 18
    yT, yB = y0 + 38, y1 - (38 if show_x else 14)
    days = dates.astype("datetime64[D]").astype(int)
    dmin, dmax = int(days.min()), int(days.max())
    finite = np.isfinite(values)
    ymin, ymax = float(values[finite].min()), float(values[finite].max())
    for h in hlines:
        ymin, ymax = min(ymin, h), max(ymax, h)
    pad = max(0.06 * (ymax - ymin), 0.05)
    ymin, ymax = ymin - pad, ymax + pad

    def px(day):
        return xL + (day - dmin) / max(1, dmax - dmin) * (xR - xL)

    def py(v):
        return yB - (v - ymin) / max(1e-9, ymax - ymin) * (yB - yT)

    c.rect((x0, y0, x1, y1), fill=(255, 255, 255), outline=(215, 220, 226))
    # cycle shading (draw before lines)
    for _, cyc in cycles.iterrows():
        a = int(np.datetime64(str(cyc["start_date"]), "D").astype("datetime64[D]").astype(int))
        b = int(np.datetime64(str(cyc["end_date"]), "D").astype("datetime64[D]").astype(int))
        status = str(cyc.get("status", "normal"))
        fill = (
            (255, 236, 236) if status == "overlong_unresolved"
            else (255, 248, 220) if status == "short_unresolved"
            else (232, 242, 252)
        )
        c.rect((int(px(a)), yT, int(px(b)), yB), fill=fill)
        c.line([(px(a), yT), (px(a), yB)], (61, 126, 166), 1)
        c.line([(px(b), yT), (px(b), yB)], (183, 119, 45), 1)
        label = (
            "U" if status == "overlong_unresolved"
            else "s" if status == "short_unresolved"
            else str(cyc.get("cycle_id", ""))
        )
        c.text(((px(a) + px(b)) / 2, yT + 2), label, 12, (110, 110, 110), anchor="ma")
    # y grid + labels
    for k in range(5):
        v = ymin + k * (ymax - ymin) / 4
        yy = py(v)
        c.line([(xL, yy), (xR, yy)], (232, 232, 232), 1)
        c.text((xL - 6, yy), f"{v:.2f}" if abs(v) < 20 else f"{v:.0f}", 12,
               (110, 110, 110), anchor="ra")
    # train/test boundary
    c.line([(px(TRAIN_DAY), yT), (px(TRAIN_DAY), yB)], (40, 40, 40), 1)
    # x ticks (years)
    for year in range(2013, 2027):
        day = int(np.datetime64(f"{year}-01-01", "D").astype(int))
        if dmin <= day <= dmax:
            c.line([(px(day), yB), (px(day), yB + 4)], (120, 120, 120), 1)
            if show_x:
                c.text((px(day), yB + 6), str(year), 12, (90, 90, 90), anchor="mt")
    # threshold lines
    for h in hlines:
        c.line([(xL, py(h)), (xR, py(h))], (192, 57, 43), 1)
    # series, broken at NaN
    seg = []
    for day, v in zip(days, values):
        if np.isfinite(v):
            seg.append((px(int(day)), py(v)))
        elif seg:
            c.line(seg, color, 2)
            seg = []
    if seg:
        c.line(seg, color, 2)
    # flagged dates
    for day, v, n in zip(days, values, flags):
        if n > 0 and np.isfinite(v):
            c.circle(px(int(day)), py(v), 3, (142, 0, 0))
    c.text((x0 + 8, y0 + 8), title, 16, (25, 35, 50), bold=True)


# ------------------------------ figures ------------------------------------

def _figure_seasonal(out, series, cycles_map, summaries):
    c = Canvas(1900, 1270,
               "Seasonal signal on the real mini grid — crop cycles (start | end) and weak-suspicion flags",
               "Blue band = plant cycle; orange band = ratoon cycle. Thin lines = cycle START / END. "
               "Dashed black = train/test boundary (2023-01-01). Red dots = flagged acquisition dates.")
    for i, farm in enumerate(FARMS):
        s = series[farm]
        for j, (key, title, color) in enumerate((("ndvi", "NDVI (-)", (31, 119, 180)),
                                                 ("lst_c", "LST (°C)", (192, 57, 43)))):
            x0 = 45 + j * 935
            y0 = 70 + i * 375
            _panel(c, (x0, y0, x0 + 885, y0 + 350), s["dates"], s[key],
                   f"{farm} — {title}", color, cycles_map[farm], s["n_flag"],
                   show_x=(i == 2))
    c.line([(52, 56), (80, 56)], (142, 0, 0), 3)
    c.text((88, 56), "flagged date (weak suspicion)", 14, (60, 60, 60), anchor="lm")
    c.save(out / "timeline_seasonal.png")


def _figure_scores(out, series, cycles_map):
    c = Canvas(1900, 1270,
               "Phase-conditioned deviation (DERIVED proxy, not validated anomaly): train-year median/MAD per stage",
               "Dotted red = ±2 robust z. Stage-conditioned (S1–S4) on TRAIN years only; most of the seasonal cycle is removed. "
               "This is a derived deviation — not a validated anomaly, not stress, not model inference.")
    for i, farm in enumerate(FARMS):
        s = series[farm]
        for j, (key, title, color) in enumerate((("ndvi_z", "NDVI deviation (robust z)", (31, 119, 180)),
                                                 ("lst_z", "LST deviation (robust z)", (192, 57, 43)))):
            x0 = 45 + j * 935
            y0 = 70 + i * 375
            _panel(c, (x0, y0, x0 + 885, y0 + 350), s["dates"], s[key],
                   f"{farm} — {title}", color, cycles_map[farm], s["n_flag"],
                   hlines=(-2, 2), show_x=(i == 2))
    c.save(out / "timeline_anomaly_score.png")


def _figure_flags(out, series):
    c = Canvas(1900, 1150,
               "Which acquisition dates are flagged?  —  weak suspicion on 3 farms, 2013–2026",
               "Flags come from a weak-suspicion rule on re-masked METRIC ET/CWSI/LAI/TVDI products; "
               "they are NOT confirmed stress and carry no water/salinity typing.")
    box = (45, 70, 1855, 600)
    x0, y0, x1, y1 = box
    xL, xR, yT, yB = x0 + 66, x1 - 18, y0 + 38, y1 - 38
    days_all = np.concatenate([s["dates"].astype("datetime64[D]").astype(int) for s in series.values()])
    dmin, dmax = int(days_all.min()), int(days_all.max())
    vmax = max(int(s["n_flag"].max()) for s in series.values()) + 1
    c.rect(box, fill=(255, 255, 255), outline=(215, 220, 226))
    for k in range(5):
        v = k * vmax / 4
        yy = yB - v / vmax * (yB - yT)
        c.line([(xL, yy), (xR, yy)], (232, 232, 232), 1)
        c.text((xL - 6, yy), f"{v:.0f}", 12, (110, 110, 110), anchor="ra")
    for year in range(2013, 2027):
        day = int(np.datetime64(f"{year}-01-01", "D").astype(int))
        xx = xL + (day - dmin) / max(1, dmax - dmin) * (xR - xL)
        c.line([(xx, yB), (xx, yB + 4)], (120, 120, 120), 1)
        c.text((xx, yB + 6), str(year), 12, (90, 90, 90), anchor="mt")
    for farm in FARMS:
        s = series[farm]
        pts = [(xL + (int(d) - dmin) / max(1, dmax - dmin) * (xR - xL),
                yB - n / vmax * (yB - yT))
               for d, n in zip(s["dates"].astype("datetime64[D]").astype(int), s["n_flag"])]
        c.line(pts, FARM_COLORS[farm], 2)
        for (xx, yy), n in zip(pts, s["n_flag"]):
            if n > 0:
                c.circle(xx, yy, 3, FARM_COLORS[farm])
    c.text((x0 + 8, y0 + 8), "Count of suspected 5×5 patches per acquisition date", 16,
           (25, 35, 50), bold=True)
    for i, farm in enumerate(FARMS):
        c.line([(x0 + 12 + i * 150, y0 + 36), (x0 + 42 + i * 150, y0 + 36)], FARM_COLORS[farm], 3)
        c.text((x0 + 48 + i * 150, y0 + 36), farm, 13, (60, 60, 60), anchor="lm")

    # year x month heatmap
    hy0, hy1 = 650, 1090
    years = list(range(2013, 2027))
    heat = np.zeros((len(years), 12), dtype=float)
    for farm in FARMS:
        s = series[farm]
        for d, n in zip(s["dates"], s["n_flag"]):
            y, m = int(str(d)[:4]), int(str(d)[5:7]) - 1
            if y in years and n > 0:
                heat[years.index(y), m] += n
    hmax = max(float(heat.max()), 1.0)
    gx0, gx1 = 150, 1500
    gy0, gy1 = hy0 + 30, hy1 - 10
    cw, ch = (gx1 - gx0) / 12, (gy1 - gy0) / len(years)
    for r, y in enumerate(years):
        c.text((gx0 - 12, gy0 + r * ch + ch / 2), str(y), 13, (80, 80, 80), anchor="rm")
        for m in range(12):
            v = heat[r, m] / hmax
            col = _blend((142, 0, 0), (255, 250, 245), v)
            c.rect((gx0 + m * cw, gy0 + r * ch, gx0 + (m + 1) * cw - 2, gy0 + (r + 1) * ch - 2),
                   fill=col)
            if heat[r, m] > 0:
                c.text((gx0 + (m + 0.5) * cw, gy0 + (r + 0.5) * ch), str(int(heat[r, m])),
                       11, (255, 255, 255) if v > 0.55 else (120, 20, 20), anchor="mm")
    for m, name in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
        c.text((gx0 + (m + 0.5) * cw, gy0 - 8), name, 12, (80, 80, 80), anchor="mb")
    c.text((gx0, hy0 + 2), "Flagged patches by year × month (all three farms)", 16,
           (25, 35, 50), bold=True)
    for k in range(5):
        v = hmax * k / 4
        col = _blend((142, 0, 0), (255, 250, 245), v / hmax)
        c.rect((gx1 + 20, gy1 - (k + 1) * 22, gx1 + 44, gy1 - k * 22), fill=col)
        c.text((gx1 + 50, gy1 - (k + 0.5) * 22), f"{v:.0f}", 12, (80, 80, 80), anchor="lm")
    c.save(out / "timeline_flagged_dates.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cycles", type=Path, default=CYCLES,
                    help="cycle parquet to plot (defaults to the canonical mini output)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    set_farms([p.name for p in STORE.iterdir() if p.is_dir() and (p / ".zgroup").exists()])
    if not FARMS:
        raise SystemExit(f"No farm groups found in {STORE}")

    cycles = pd.read_parquet(args.cycles)
    patches = pd.read_parquet(PATCHES)
    flags = patches.groupby(["farm", "date"], as_index=False)["n_suspected"].sum()
    flag_map = {(r.farm, str(r.date)[:10]): int(r.n_suspected) for r in flags.itertuples()}

    series, cycles_map, summaries = {}, {}, []
    for farm in FARMS:
        s = _farm_series(farm)
        fc = cycles[cycles.farm == farm]
        stages = np.array([_stage_of(d, fc) for d in s["dates"]])
        s["stages"] = stages
        s["ndvi_z"] = _phase_score(s["ndvi"], s["dates"], stages)
        s["lst_z"] = _phase_score(s["lst_c"], s["dates"], stages)
        s["n_flag"] = np.array([flag_map.get((farm, str(d)[:10]), 0) for d in s["dates"]], dtype=int)
        series[farm], cycles_map[farm] = s, fc
        summaries.append(
            f"| {farm} | {len(s['dates'])} | {int((s['n_flag'] > 0).sum())} | "
            f"{np.nanmean(s['ndvi']):.3f} | {np.nanmean(s['lst_c']):.2f} | "
            f"{int((np.abs(s['ndvi_z']) >= 2).sum())} | {int((np.abs(s['lst_z']) >= 2).sum())} |"
        )
        print(f"{farm}: dates={len(s['dates'])} flagged={int((s['n_flag'] > 0).sum())} "
              f"NDVI={np.nanmean(s['ndvi']):.3f} LST={np.nanmean(s['lst_c']):.2f}C", flush=True)

    _figure_seasonal(args.out, series, cycles_map, summaries)
    print("wrote timeline_seasonal.png", flush=True)
    _figure_scores(args.out, series, cycles_map)
    print("wrote timeline_anomaly_score.png", flush=True)
    _figure_flags(args.out, series)
    print("wrote timeline_flagged_dates.png", flush=True)

    table = (
        "| Farm | Dates | Flagged dates | Mean NDVI | Mean LST (°C) | NDVI \\|z\\|≥2 | LST \\|z\\|≥2 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(summaries) + "\n"
    )
    (args.out / "timeline_readme.md").write_text(
        "# How to read the timeline figures\n\n"
        "- `timeline_seasonal.png`: field-mean NDVI and LST (°C) per farm, 2013–2026. "
        "Blue/orange bands are sugarcane cycles (plant/ratoon) with their START and END lines; "
        "the dashed black line is the 2023 train/test boundary. Red dots are flagged acquisition dates.\n"
        "- `timeline_anomaly_score.png`: the same dates as a *phase-conditioned deviation* — for each "
        "farm and stage S1–S4, the robust z against the train-year median/MAD. Dotted lines mark ±2.\n"
        "- `timeline_flagged_dates.png`: top, the count of suspected 5×5 patches on every acquisition "
        "date; bottom, the same flags aggregated by year × month.\n\n"
        "**Provenance:** NDVI = (nir−red)/(nir+red) and LST = band `lst`−273.15 from "
        "`Data/Processed/mini_research_dataset/satellite.zarr`, masked by `pixel_valid` and `in_field`. "
        "Cycles from `phenology/cycles_mini.parquet`. Flags from "
        "`suspected_stress/suspected_patches.parquet` (weak suspicion from re-masked METRIC products).\n\n"
        "**Do not claim:** confirmed stress, water/salinity typing, early warning, or validation of the "
        "SSL model. The flags derive from METRIC products that share LST/NIR inputs with the spectral "
        "signal, so agreement with the NDVI/LST deviation is not independent evidence. "
        "Claim rung: L1/L2 screening only.\n\n"
        "**Renderer note:** figures are drawn with Pillow because matplotlib's native draw/save crashes "
        "in this conda environment (matplotlib 3.11.1 + numpy 2.5.2, Windows).\n\n"
        "## Run summary\n\n" + table,
        encoding="utf-8",
    )
    print("wrote timeline_readme.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
