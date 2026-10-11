"""
Feature extraction.

Reads data/processed/cohort_v2.csv and data/processed/segmentation_log.csv,
loads each confirmed well's saved mask, keeps the single largest connected
component, and computes 23 morphology/intensity/texture features per channel
(CH1, CH2, CH4) over that one component. The result is written to
data/processed/features_v2.csv and a diagnostic report is printed.

Design decisions that matter for correctness:

  * Only wells whose LATEST log status is "confirmed" are processed (the log is
    append-only; a later row supersedes an earlier one).

  * One mask -> one largest component -> used for geometry AND intensity on all
    three channels, so every channel describes the same physical object.

  * Channel bands are read at native resolution and native 8-bit, no resizing:
      CH1 (stored RGB) -> green plane
      CH2 (stored RGB) -> red plane
      CH4 (greyscale)  -> as is

  * The GLCM is computed INSIDE the mask, by hand. graycomatrix has no mask
    argument, so for each angle offset we form the shifted image/mask and count
    intensity pairs only where both the pixel and its offset neighbour fall in
    the largest component. Background is never zero-filled and fed to
    graycomatrix (that would count zeros as real intensities).

  * Any failure (missing channel file, empty mask, component too small for a
    statistic, degenerate/zero-variance texture) yields np.nan, never 0 -- a
    zero is a real measurement and must not double as a failure sentinel.

This script only writes data/processed/features_v2.csv; it modifies nothing else.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure

# Make stdout tolerant of non-ASCII glyphs when redirected.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
COHORT_CSV = REPO_ROOT / "data" / "processed" / "cohort_v2.csv"
BASELINE_CSV = REPO_ROOT / "data" / "processed" / "baseline_v2.csv"
LOG_CSV = REPO_ROOT / "data" / "processed" / "segmentation_log.csv"
OUT_CSV = REPO_ROOT / "data" / "processed" / "features_v2.csv"

IMG_H, IMG_W = 720, 960
IMG_PIXELS = IMG_H * IMG_W
GLCM_LEVELS = 256

# Channel -> (cohort path column, RGB plane index or None for greyscale).
# CH1 uses the green plane (index 1); CH2 uses the red plane (index 0).
CHANNELS = [
    ("CH1", "ch1_path", 1),
    ("CH2", "ch2_path", 0),
    ("CH4", "ch4_path", None),
]

# Angle offsets (drow, dcol) for GLCM at distance 1. Because we symmetrise each
# matrix (P + P.T), the sign of the offset is irrelevant, so one offset per
# direction covers 0/45/90/135 degrees.
GLCM_OFFSETS = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]

GEOMETRY_METRICS = [
    "Area", "Perc_Covered", "Perimeter", "Eccentricity", "Solidity",
    "Aspect_Ratio", "Circularity", "Equiv_Diameter", "Extent",
]
INTENSITY_METRICS = [
    "Total_Intensity", "Intensity_per_Area", "Mean_Intensity", "Std_Intensity",
    "Median_Intensity", "P25_Intensity", "P75_Intensity", "IQR_Intensity",
]
TEXTURE_METRICS = [
    "GLCM_Contrast", "GLCM_Homogeneity", "GLCM_Energy", "GLCM_Correlation",
    "GLCM_Dissimilarity", "GLCM_ASM",
]
ALL_METRICS = GEOMETRY_METRICS + INTENSITY_METRICS + TEXTURE_METRICS  # 23

ID_COLUMNS = [
    "dataset", "drug", "cmax", "well", "selected_day", "role", "label_drug",
    "n_connected_components", "largest_component_fraction",
]
FEATURE_COLUMNS = [f"{ch}_{m}" for ch, _, _ in CHANNELS for m in ALL_METRICS]  # 69


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _read_wells(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["dataset"], r["drug"], r["cmax"], r["well"]): r
                for r in csv.DictReader(fh)}


def read_well_index():
    """Merge cohort_v2.csv and baseline_v2.csv into one key -> (record, role)
    lookup. Cohort and baseline are disjoint by construction (the cohort audit
    cross-checks this), so a well belongs to exactly one role."""
    index = {}
    for key, rec in _read_wells(COHORT_CSV).items():
        index[key] = (rec, "cohort")
    for key, rec in _read_wells(BASELINE_CSV).items():
        index[key] = (rec, "baseline")  # disjoint from cohort by construction
    return index


def read_confirmed_from_log():
    """Return an ordered list holding the LATEST CONFIRMED row per well.

    The log is append-only, so a re-segmented well has several confirmed rows;
    keeping the last one processes the well once, using its most recent mask.
    Rows with a non-confirmed status are ignored (they cannot supersede an
    earlier confirmation here -- only a later confirmation does)."""
    latest_confirmed = {}  # key -> most recent confirmed row
    order = []
    with LOG_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "confirmed":
                continue
            key = (row["dataset"], row["drug"], row["cmax"], row["well"])
            if key not in latest_confirmed:
                order.append(key)
            latest_confirmed[key] = row  # later confirmed row wins
    return [latest_confirmed[k] for k in order]


# ---------------------------------------------------------------------------
# Mask + channel loading
# ---------------------------------------------------------------------------
def load_largest_component(mask_path: Path):
    """Return (component_bool_mask, largest_component_fraction) or (None, nan)
    when the mask is missing or empty."""
    if not mask_path.is_file():
        return None, np.nan
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    total = int(mask.sum())
    if total == 0:
        return None, np.nan
    labels = measure.label(mask, connectivity=2)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background
    largest = int(counts.argmax())
    comp = labels == largest
    frac = float(counts[largest]) / float(total)
    return comp, frac


def load_channel_band(rec, path_col, plane):
    """Read a single-band uint8 image for a channel at native resolution, or
    None if the file is missing/unreadable. No resizing or rescaling."""
    import tifffile

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
    if plane is None:
        band = arr if arr.ndim == 2 else arr[..., 0]
    else:
        if arr.ndim < 3 or arr.shape[-1] <= plane:
            return None
        band = arr[..., plane]
    return np.ascontiguousarray(band)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def nan_metrics():
    return {m: np.nan for m in ALL_METRICS}


def geometry_features(comp: np.ndarray):
    """Geometry from regionprops on the largest component. Depends only on the
    mask, so it is identical across channels."""
    feats = {m: np.nan for m in GEOMETRY_METRICS}
    props = measure.regionprops(comp.astype(int))
    if not props:
        return feats
    p = props[0]
    area = float(p.area)
    perim = float(p.perimeter)
    feats["Area"] = area
    feats["Perc_Covered"] = area / IMG_PIXELS
    feats["Perimeter"] = perim if perim > 0 else np.nan
    feats["Eccentricity"] = float(p.eccentricity)
    feats["Solidity"] = float(p.solidity)
    maj = float(p.axis_major_length)
    minr = float(p.axis_minor_length)
    feats["Aspect_Ratio"] = maj / minr if minr > 0 else np.nan
    feats["Circularity"] = (4.0 * np.pi * area) / (perim ** 2) if perim > 0 else np.nan
    feats["Equiv_Diameter"] = float(p.equivalent_diameter_area)
    feats["Extent"] = float(p.extent)
    return feats


def intensity_features(band: np.ndarray, comp: np.ndarray):
    feats = {m: np.nan for m in INTENSITY_METRICS}
    px = band[comp].astype(np.float64)
    if px.size == 0:
        return feats
    area = float(px.size)
    total = float(px.sum())
    feats["Total_Intensity"] = total
    feats["Intensity_per_Area"] = total / area
    feats["Mean_Intensity"] = float(px.mean())
    feats["Std_Intensity"] = float(px.std())
    feats["Median_Intensity"] = float(np.median(px))
    p25 = float(np.percentile(px, 25))
    p75 = float(np.percentile(px, 75))
    feats["P25_Intensity"] = p25
    feats["P75_Intensity"] = p75
    feats["IQR_Intensity"] = p75 - p25
    return feats


def _masked_glcm(band: np.ndarray, comp: np.ndarray, offset):
    """Symmetrised, normalised GLCM (levels x levels) for one offset, counting
    only pairs where both the pixel and its neighbour lie in the component.
    Returns None if no valid pair exists."""
    dr, dc = offset
    rows, cols = band.shape
    r0 = max(0, -dr); r1 = rows - max(0, dr)
    c0 = max(0, -dc); c1 = cols - max(0, dc)
    if r1 <= r0 or c1 <= c0:
        return None

    base_img = band[r0:r1, c0:c1]
    base_msk = comp[r0:r1, c0:c1]
    shift_img = band[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    shift_msk = comp[r0 + dr:r1 + dr, c0 + dc:c1 + dc]

    both = base_msk & shift_msk
    if not both.any():
        return None

    i = base_img[both].astype(np.intp)
    j = shift_img[both].astype(np.intp)
    flat = np.bincount(i * GLCM_LEVELS + j, minlength=GLCM_LEVELS * GLCM_LEVELS)
    P = flat.reshape(GLCM_LEVELS, GLCM_LEVELS).astype(np.float64)
    P = P + P.T  # symmetrise
    s = P.sum()
    if s == 0:
        return None
    return P / s


def _glcm_props(P: np.ndarray):
    """Six Haralick properties from a normalised symmetric GLCM."""
    idx = np.arange(GLCM_LEVELS)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    diff = I - J
    contrast = float((P * (diff ** 2)).sum())
    dissimilarity = float((P * np.abs(diff)).sum())
    homogeneity = float((P / (1.0 + diff ** 2)).sum())
    asm = float((P * P).sum())
    energy = float(np.sqrt(asm))

    mu_i = float((I * P).sum())
    mu_j = float((J * P).sum())
    var_i = float((P * (I - mu_i) ** 2).sum())
    var_j = float((P * (J - mu_j) ** 2).sum())
    sig_i = np.sqrt(var_i)
    sig_j = np.sqrt(var_j)
    if sig_i < 1e-15 or sig_j < 1e-15:
        correlation = np.nan  # zero-variance region: correlation undefined
    else:
        correlation = float((P * (I - mu_i) * (J - mu_j)).sum() / (sig_i * sig_j))

    return {
        "GLCM_Contrast": contrast,
        "GLCM_Homogeneity": homogeneity,
        "GLCM_Energy": energy,
        "GLCM_Correlation": correlation,
        "GLCM_Dissimilarity": dissimilarity,
        "GLCM_ASM": asm,
    }


def texture_features(band: np.ndarray, comp: np.ndarray):
    """GLCM properties averaged over the four angles, computed inside the mask."""
    per_angle = []
    for offset in GLCM_OFFSETS:
        P = _masked_glcm(band, comp, offset)
        per_angle.append(_glcm_props(P) if P is not None else
                         {m: np.nan for m in TEXTURE_METRICS})

    feats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN mean
        for m in TEXTURE_METRICS:
            vals = np.array([a[m] for a in per_angle], dtype=np.float64)
            feats[m] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan
    return feats


def bg_correct_band(band, comp):
    """Flat-field + background-normalize one channel band.

    1. dilate the organoid mask and 'inpaint' it by filling with the background
       median (cv2 is unavailable; a constant fill is enough to keep the organoid
       from biasing the illumination estimate),
    2. estimate the flat field as a heavy gaussian blur (sigma ~ width/8) of the
       filled image and divide it out (rescaled to preserve mean level),
    3. background-normalize: (pixel - bg_median) / bg_mad, where bg stats come
       from pixels OUTSIDE the dilated mask of the flat-fielded image.

    Returns (band_corr_float, band_u8) -- the float background-relative band for
    intensity stats, and a robust 8-bit flat-fielded band for GLCM texture.
    """
    import scipy.ndimage as ndi

    band = band.astype(np.float32)
    H, W = band.shape
    mask_dil = ndi.binary_dilation(comp, iterations=15)
    outside = ~mask_dil
    bg_raw = band[outside]
    if bg_raw.size < 50:
        outside = ~comp
        bg_raw = band[outside]
    bg_med_raw = float(np.median(bg_raw)) if bg_raw.size else 0.0

    filled = band.copy()
    filled[mask_dil] = bg_med_raw
    sigma = max(4.0, W / 8.0)
    flat = ndi.gaussian_filter(filled, sigma=sigma)
    flat = np.where(flat < 1e-3, 1e-3, flat)
    band_ff = band / flat * float(flat.mean())

    bg = band_ff[outside]
    bg_median = float(np.median(bg)) if bg.size else 0.0
    mad = float(np.median(np.abs(bg - bg_median))) * 1.4826 if bg.size else 0.0
    if mad < 1e-6:
        mad = float(bg.std()) if bg.size and bg.std() > 0 else 1.0
    band_corr = (band_ff - bg_median) / mad

    lo, hi = np.percentile(band_ff, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    band_u8 = np.clip((band_ff - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return band_corr, band_u8


def channel_features(rec, comp, geom, path_col, plane, bg_correct=False):
    """23 metrics for one channel. Geometry is mask-derived (always available
    when the component is valid); intensity/texture need the channel band."""
    band = load_channel_band(rec, path_col, plane)
    feats = dict(geom)  # geometry duplicated per channel (same object)
    if band is None:
        feats.update({m: np.nan for m in INTENSITY_METRICS})
        feats.update({m: np.nan for m in TEXTURE_METRICS})
        return feats
    if bg_correct:
        band_corr, band_u8 = bg_correct_band(band, comp)
        feats.update(intensity_features(band_corr, comp))
        feats.update(texture_features(band_u8, comp))
    else:
        feats.update(intensity_features(band, comp))
        feats.update(texture_features(band, comp))
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_row(log_row, well_index, bg_correct=False):
    """Build one feature row, or return None if the confirmed well is not part of
    either cohort_v2 or baseline_v2 (out of scope for the feature table)."""
    key = (log_row["dataset"], log_row["drug"], log_row["cmax"], log_row["well"])
    entry = well_index.get(key)
    if entry is None:
        return None
    rec, role = entry

    try:
        n_comp = int(log_row["n_connected_components"])
    except (TypeError, ValueError):
        n_comp = np.nan

    row = {
        "dataset": log_row["dataset"],
        "drug": log_row["drug"],
        "cmax": log_row["cmax"],
        "well": log_row["well"],
        "selected_day": log_row.get("selected_day", ""),
        "role": role,
        "label_drug": log_row["drug"],  # cohort drug is the class label
        "n_connected_components": n_comp,
    }

    mask_path = REPO_ROOT / log_row["mask_path"] if log_row.get("mask_path") else None
    comp, frac = (None, np.nan)
    if mask_path is not None:
        comp, frac = load_largest_component(mask_path)
    row["largest_component_fraction"] = frac

    if comp is None:
        for ch, _, _ in CHANNELS:
            for m, v in nan_metrics().items():
                row[f"{ch}_{m}"] = v
        return row

    geom = geometry_features(comp)
    for ch, path_col, plane in CHANNELS:
        feats = channel_features(rec, comp, geom, path_col, plane, bg_correct)
        for m in ALL_METRICS:
            row[f"{ch}_{m}"] = feats[m]
    return row


def print_report(df, expected):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"\nRows written: {len(df)}   Wells expected (confirmed in log): {expected}")
    if len(df) != expected:
        print(f"  ! mismatch: {expected - len(df)} confirmed well(s) not in output")

    # Per-feature-column stats.
    print("\nPer-feature column stats (NaN / zeros / distinct / std):")
    print(f"  {'column':28s} {'NaN':>5s} {'zeros':>6s} {'distinct':>9s} {'std':>14s}  flags")
    n = len(df)
    for col in FEATURE_COLUMNS:
        s = df[col]
        n_nan = int(s.isna().sum())
        n_zero = int((s == 0).sum())
        distinct = int(s.dropna().nunique())
        std = s.std(skipna=True)
        flags = []
        if distinct <= 1:
            flags.append("CONSTANT")
        if n and n_nan / n > 0.20:
            flags.append(f">20%NaN({100*n_nan//n}%)")
        std_str = "nan" if pd.isna(std) else f"{std:.4g}"
        flag_str = "  ".join(flags)
        if flags:
            print(f"  {col:28s} {n_nan:5d} {n_zero:6d} {distinct:9d} {std_str:>14s}  {flag_str}")
    # Also emit a compact all-columns table so nothing is hidden.
    print("\n  (columns with no flags omitted above; full table below)")
    for col in FEATURE_COLUMNS:
        s = df[col]
        n_nan = int(s.isna().sum())
        n_zero = int((s == 0).sum())
        distinct = int(s.dropna().nunique())
        std = s.std(skipna=True)
        std_str = "nan" if pd.isna(std) else f"{std:.4g}"
        print(f"    {col:28s} NaN={n_nan:<4d} zeros={n_zero:<4d} distinct={distinct:<5d} std={std_str}")

    # Lowest largest_component_fraction (shattered masks).
    print("\nFive wells with lowest largest_component_fraction (shattered masks):")
    low = df.nsmallest(5, "largest_component_fraction")
    for _, r in low.iterrows():
        print(f"    {r['largest_component_fraction']:.4f}  "
              f"{r['dataset']} / {r['drug']} / {r['cmax']} / {r['well']}  "
              f"(n_components={r['n_connected_components']})")

    # Per-channel Area vs Total_Intensity correlation.
    print("\nPer-channel corr(Area, Total_Intensity) -- intensity should track mask:")
    for ch, _, _ in CHANNELS:
        a = df[f"{ch}_Area"]
        t = df[f"{ch}_Total_Intensity"]
        pair = pd.concat([a, t], axis=1).dropna()
        if len(pair) < 2:
            print(f"    {ch}: n<2, correlation undefined")
        else:
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            print(f"    {ch}: r = {corr:.4f}  (n={len(pair)})")


def main():
    parser = argparse.ArgumentParser(description="Feature extraction.")
    parser.add_argument(
        "--bg-correct", action="store_true",
        help="Flat-field + background-normalize each channel before computing "
        "intensity/texture (shape features unchanged). Writes to "
        "features_v2_bgcorr.csv by default.")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default features_v2.csv, or "
                        "features_v2_bgcorr.csv with --bg-correct).")
    args = parser.parse_args()

    out_csv = (Path(args.out) if args.out else
               (REPO_ROOT / "data" / "processed" /
                ("features_v2_bgcorr.csv" if args.bg_correct else "features_v2.csv")))

    well_index = read_well_index()
    confirmed = read_confirmed_from_log()   # unique wells: latest confirmed each
    n_unique_confirmed = len(confirmed)
    print(f"Unique confirmed wells in log: {n_unique_confirmed}")
    print(f"Background correction: {'ON' if args.bg_correct else 'off'}; "
          f"output -> {out_csv.name}")

    rows = []
    orphans = []
    for i, log_row in enumerate(confirmed, 1):
        row = build_row(log_row, well_index, bg_correct=args.bg_correct)
        if row is None:
            orphans.append((log_row["dataset"], log_row["drug"],
                            log_row["cmax"], log_row["well"]))
        else:
            rows.append(row)
        if i % 20 == 0 or i == n_unique_confirmed:
            print(f"  processed {i}/{n_unique_confirmed}")

    df = pd.DataFrame(rows, columns=ID_COLUMNS + FEATURE_COLUMNS)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows x {len(df.columns)} cols to "
          f"{out_csv.relative_to(REPO_ROOT).as_posix()}")

    # Row count must equal the number of unique confirmed wells (in scope).
    in_scope = n_unique_confirmed - len(orphans)
    print("\nRow-count check:")
    print(f"  unique confirmed wells         = {n_unique_confirmed}")
    print(f"  confirmed but not in cohort/baseline (skipped) = {len(orphans)}")
    print(f"  rows written                   = {len(df)}")
    ok = len(df) == in_scope
    print(f"  rows == in-scope confirmed wells -> {'PASS' if ok else 'FAIL'}")
    if orphans:
        for o in orphans:
            print(f"    orphan (not in cohort/baseline): {o}")

    counts = Counter(df["role"])
    print("\nBreakdown by role:")
    for role in ("cohort", "baseline"):
        print(f"  {role:9s} = {counts.get(role, 0)}")
    print(f"  {'TOTAL':9s} = {len(df)}")

    print_report(df, in_scope)


if __name__ == "__main__":
    main()
