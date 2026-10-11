"""
Fluorescence / mask alignment check.

Reads data/processed/cohort_v2.csv and data/processed/segmentation_log.csv and
asks a single question: does the fluorescence signal in CH1 (green plane) and
CH2 (red plane) actually fall inside the CH4-derived organoid mask?

For each confirmed well the mask is loaded and reduced to its largest connected
component (exactly as the extractor does). Signal pixels for a channel are those
above Otsu's threshold computed over the whole single-band image, at native
resolution and bit depth. Per channel we record:

  inside_frac        fraction of all signal pixels that fall inside the mask
  outside_frac       fraction that fall outside it
  mean_in / mean_out mean channel intensity inside vs outside the mask
  centroid_dist      distance (px) between the signal centroid and mask centroid
  signal_area_ratio  signal area divided by mask area

Per-well rows go to data/processed/fluor_alignment.csv and a summary is printed.
This is statistics only: no image is written, modified, or resized. Failures
(missing channel, empty mask, degenerate/constant channel) are recorded as NaN.

This script writes only data/processed/fluor_alignment.csv; it modifies nothing
else.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage import measure
from skimage.filters import threshold_otsu

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
COHORT_CSV = REPO_ROOT / "data" / "processed" / "cohort_v2.csv"
LOG_CSV = REPO_ROOT / "data" / "processed" / "segmentation_log.csv"
OUT_CSV = REPO_ROOT / "data" / "processed" / "fluor_alignment.csv"

# Channel -> (cohort path column, RGB plane index). CH1 green, CH2 red.
CHANNELS = [("CH1", "ch1_path", 1), ("CH2", "ch2_path", 0)]

PER_CHANNEL_METRICS = [
    "inside_frac", "outside_frac", "mean_in", "mean_out",
    "centroid_dist", "signal_area_ratio",
]
ID_COLUMNS = ["dataset", "drug", "cmax", "well", "selected_day"]
FEATURE_COLUMNS = [f"{ch}_{m}" for ch, _, _ in CHANNELS for m in PER_CHANNEL_METRICS]


# ---------------------------------------------------------------------------
# Inputs (same conventions as extract_features.py)
# ---------------------------------------------------------------------------
def read_cohort():
    with COHORT_CSV.open(newline="", encoding="utf-8") as fh:
        return {(r["dataset"], r["drug"], r["cmax"], r["well"]): r
                for r in csv.DictReader(fh)}


def read_confirmed_from_log():
    latest, order = {}, []
    with LOG_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], row["drug"], row["cmax"], row["well"])
            if key not in latest:
                order.append(key)
            latest[key] = row
    return [latest[k] for k in order if latest[k]["status"] == "confirmed"]


def load_largest_component(mask_path: Path):
    if not mask_path.is_file():
        return None
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if not mask.any():
        return None
    labels = measure.label(mask, connectivity=2)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def load_channel_band(rec, path_col, plane):
    rel = (rec.get(path_col) or "").strip()
    if not rel:
        return None
    abs_path = REPO_ROOT / rel
    if not abs_path.is_file():
        return None
    try:
        arr = tifffile.imread(str(abs_path))
    except Exception:
        return None
    if arr.ndim < 3 or arr.shape[-1] <= plane:
        return None
    return np.ascontiguousarray(arr[..., plane])


def centroid(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return None
    return np.array([ys.mean(), xs.mean()])


# ---------------------------------------------------------------------------
# Per-channel alignment metrics
# ---------------------------------------------------------------------------
def nan_channel():
    return {m: np.nan for m in PER_CHANNEL_METRICS}


def channel_alignment(band, comp):
    feats = nan_channel()
    if band is None:
        return feats

    # Otsu over the whole single-band image; needs >1 distinct value.
    if band.min() == band.max():
        return feats
    thresh = threshold_otsu(band)
    signal = band > thresh
    n_signal = int(signal.sum())

    mask_area = int(comp.sum())
    # mean intensity inside vs outside the mask (over all pixels).
    in_px = band[comp].astype(np.float64)
    out_px = band[~comp].astype(np.float64)
    feats["mean_in"] = float(in_px.mean()) if in_px.size else np.nan
    feats["mean_out"] = float(out_px.mean()) if out_px.size else np.nan
    feats["signal_area_ratio"] = (n_signal / mask_area) if mask_area > 0 else np.nan

    if n_signal == 0:
        # No signal pixels: inside/outside fractions and centroid undefined.
        return feats

    inside = int((signal & comp).sum())
    feats["inside_frac"] = inside / n_signal
    feats["outside_frac"] = (n_signal - inside) / n_signal

    sig_c = centroid(signal)
    mask_c = centroid(comp)
    if sig_c is not None and mask_c is not None:
        feats["centroid_dist"] = float(np.hypot(*(sig_c - mask_c)))
    return feats


def build_row(log_row, cohort):
    key = (log_row["dataset"], log_row["drug"], log_row["cmax"], log_row["well"])
    rec = cohort.get(key, {})
    row = {
        "dataset": log_row["dataset"],
        "drug": log_row["drug"],
        "cmax": log_row["cmax"],
        "well": log_row["well"],
        "selected_day": log_row.get("selected_day", ""),
    }

    mask_path = REPO_ROOT / log_row["mask_path"] if log_row.get("mask_path") else None
    comp = load_largest_component(mask_path) if mask_path is not None else None

    for ch, path_col, plane in CHANNELS:
        if comp is None:
            feats = nan_channel()
        else:
            band = load_channel_band(rec, path_col, plane)
            feats = channel_alignment(band, comp)
        for m in PER_CHANNEL_METRICS:
            row[f"{ch}_{m}"] = feats[m]
    return row


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(df):
    print("\n" + "=" * 70)
    print("FLUORESCENCE / MASK ALIGNMENT SUMMARY")
    print("=" * 70)
    print(f"\nWells: {len(df)}")

    for ch, _, _ in CHANNELS:
        inside = df[f"{ch}_inside_frac"].dropna()
        print(f"\n{ch} inside_frac (fraction of signal inside the mask):")
        if inside.empty:
            print("    no valid wells")
        else:
            q1, med, q3 = inside.quantile([0.25, 0.50, 0.75])
            print(f"    median = {med:.3f}   IQR = {q1:.3f} - {q3:.3f}   (n={len(inside)})")
            n_below = int((inside < 0.5).sum())
            print(f"    wells with < 50% of signal inside the mask: {n_below}")
        ratio = df[f"{ch}_signal_area_ratio"].dropna()
        if not ratio.empty:
            rel = "larger" if ratio.median() > 1 else "smaller"
            print(f"    median signal_area / mask_area = {ratio.median():.3f} "
                  f"(fluorescent region is {rel} than the brightfield body)")

    # Ten lowest inside fractions per channel, to reveal any drug pattern.
    for ch, _, _ in CHANNELS:
        col = f"{ch}_inside_frac"
        valid = df.dropna(subset=[col])
        print(f"\nTen wells with lowest {ch} inside_frac (drug / cmax):")
        if valid.empty:
            print("    no valid wells")
            continue
        for _, r in valid.nsmallest(10, col).iterrows():
            print(f"    {r[col]:.3f}   {r['drug']:5s} cmax={str(r['cmax']):4s}  "
                  f"{r['dataset']} / {r['well']}")


def main():
    cohort = read_cohort()
    confirmed = read_confirmed_from_log()
    print(f"Confirmed wells to check: {len(confirmed)}")

    rows = []
    for i, log_row in enumerate(confirmed, 1):
        rows.append(build_row(log_row, cohort))
        if i % 20 == 0 or i == len(confirmed):
            print(f"  processed {i}/{len(confirmed)}")

    df = pd.DataFrame(rows, columns=ID_COLUMNS + FEATURE_COLUMNS)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV.relative_to(REPO_ROOT).as_posix()}")

    print_report(df)


if __name__ == "__main__":
    main()
