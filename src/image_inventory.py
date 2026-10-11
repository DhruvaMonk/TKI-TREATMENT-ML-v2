"""
Image inventory.

Reads data/processed/cohort_v2.csv and, for every well, opens each available
channel image (CH1/CH2/CH4) with tifffile and records metadata only. Nothing is
written back to the images and no pixel data is resized or modified -- images are
opened read-only purely to read their shape, dtype, and pixel-value statistics.

Per-image rows are written to data/processed/image_inventory.csv with columns:
  dataset, drug, cmax, well, selected_day, channel,
  width, height, dtype, min_value, max_value, distinct_values

A summary is then printed to stdout (counts only, no per-file spam):
  * images grouped by (dataset, channel, width, height)
  * images grouped by (dataset, channel, dtype)
  * per (dataset, channel), the min and max of the per-image max pixel value
    (reveals whether 16-bit files use the full range or only the low 8 bits)
  * wells whose three channels do not share identical width and height
  * count of images with fewer than 10 distinct pixel values (blank/degenerate)
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
COHORT_CSV = REPO_ROOT / "data" / "processed" / "cohort_v2.csv"
OUT_CSV = REPO_ROOT / "data" / "processed" / "image_inventory.csv"

# Cohort channel columns -> channel label.
CHANNEL_COLUMNS = [("ch1_path", "CH1"), ("ch2_path", "CH2"), ("ch4_path", "CH4")]

DEGENERATE_DISTINCT_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
def image_metadata(path: Path):
    """Return (width, height, dtype, min_value, max_value, distinct_values).

    The image is opened read-only; no pixels are written or resized. Width and
    height are taken from the file's declared X and Y axes (via tifffile's axis
    labels) rather than from raw array positions, so that RGB files stored as
    YXS (height, width, sample) are not confused with grayscale YX files. Pixel
    statistics are computed over the whole array (all colour samples included).
    """
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        axes = series.axes
        shape = series.shape
        arr = series.asarray()

    # Map axis label -> size using the series' declared axes (e.g. "YXS", "YX").
    axis_size = dict(zip(axes, shape))
    if "X" in axis_size and "Y" in axis_size:
        width = int(axis_size["X"])
        height = int(axis_size["Y"])
    else:
        # Fallback for any file without labelled X/Y: use the last two axes.
        height, width = int(shape[-2]), int(shape[-1])

    min_value = arr.min().item()
    max_value = arr.max().item()
    distinct_values = int(np.unique(arr).size)
    return width, height, str(arr.dtype), min_value, max_value, distinct_values


# ---------------------------------------------------------------------------
# Inventory build
# ---------------------------------------------------------------------------
def build_rows():
    rows = []
    missing = []

    with COHORT_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            for col, channel in CHANNEL_COLUMNS:
                rel = (rec.get(col) or "").strip()
                if not rel:
                    continue
                path = REPO_ROOT / rel
                if not path.is_file():
                    missing.append(rel)
                    continue
                width, height, dtype, min_v, max_v, distinct = image_metadata(path)
                rows.append(
                    {
                        "dataset": rec["dataset"],
                        "drug": rec["drug"],
                        "cmax": rec["cmax"],
                        "well": rec["well"],
                        "selected_day": rec["selected_day"],
                        "channel": channel,
                        "width": width,
                        "height": height,
                        "dtype": dtype,
                        "min_value": min_v,
                        "max_value": max_v,
                        "distinct_values": distinct,
                    }
                )
    return rows, missing


def write_rows(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "drug",
        "cmax",
        "well",
        "selected_day",
        "channel",
        "width",
        "height",
        "dtype",
        "min_value",
        "max_value",
        "distinct_values",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def print_dimension_counts(rows):
    print("\n=== Images by (dataset, channel, width, height) ===")
    counts = Counter(
        (r["dataset"], r["channel"], r["width"], r["height"]) for r in rows
    )
    for key in sorted(counts):
        dataset, channel, width, height = key
        print(f"  {dataset:32s} {channel}  {width}x{height}  n={counts[key]}")


def print_dtype_counts(rows):
    print("\n=== Images by (dataset, channel, dtype) ===")
    counts = Counter((r["dataset"], r["channel"], r["dtype"]) for r in rows)
    for key in sorted(counts):
        dataset, channel, dtype = key
        print(f"  {dataset:32s} {channel}  {dtype:8s} n={counts[key]}")


def print_max_value_range(rows):
    print("\n=== Per-image max pixel value: min/max per (dataset, channel) ===")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["channel"])].append(r["max_value"])
    for key in sorted(grouped):
        dataset, channel = key
        vals = grouped[key]
        print(
            f"  {dataset:32s} {channel}  "
            f"min_of_max={min(vals):<8} max_of_max={max(vals):<8} n={len(vals)}"
        )


def print_channel_dim_mismatches(rows):
    print("\n=== Wells whose channels differ in width/height ===")
    dims = defaultdict(dict)  # (dataset, drug, cmax, well) -> {channel: (w, h)}
    for r in rows:
        key = (r["dataset"], r["drug"], r["cmax"], r["well"])
        dims[key][r["channel"]] = (r["width"], r["height"])

    mismatches = 0
    for key in sorted(dims):
        sizes = set(dims[key].values())
        if len(sizes) > 1:
            mismatches += 1
            dataset, drug, cmax, well = key
            detail = ", ".join(
                f"{ch}={w}x{h}" for ch, (w, h) in sorted(dims[key].items())
            )
            print(f"  {dataset} / {drug} / {cmax} / {well}: {detail}")
    if mismatches == 0:
        print("  (none -- all wells have channel-consistent dimensions)")
    print(f"  mismatched wells: {mismatches}")


def print_degenerate_count(rows):
    print("\n=== Blank/degenerate images (distinct pixel values < "
          f"{DEGENERATE_DISTINCT_THRESHOLD}) ===")
    degenerate = [r for r in rows if r["distinct_values"] < DEGENERATE_DISTINCT_THRESHOLD]
    print(f"  count: {len(degenerate)}")
    by_channel = Counter((r["dataset"], r["channel"]) for r in degenerate)
    for key in sorted(by_channel):
        dataset, channel = key
        print(f"    {dataset:32s} {channel}  n={by_channel[key]}")


def main():
    rows, missing = build_rows()
    write_rows(rows)

    rel_out = OUT_CSV.relative_to(REPO_ROOT).as_posix()
    print(f"Inventoried {len(rows)} images from {COHORT_CSV.relative_to(REPO_ROOT).as_posix()}")
    print(f"Wrote per-image rows to {rel_out}")
    if missing:
        print(f"\n*** {len(missing)} channel paths in the cohort csv were not found on disk:")
        for rel in missing:
            print(f"    {rel}")

    print_dimension_counts(rows)
    print_dtype_counts(rows)
    print_max_value_range(rows)
    print_channel_dim_mismatches(rows)
    print_degenerate_count(rows)


if __name__ == "__main__":
    main()
