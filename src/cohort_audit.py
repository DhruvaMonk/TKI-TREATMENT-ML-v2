"""
Cohort audit.

Walks the four REORGANIZED dataset trees under data/raw/, selects one timepoint
per well (the highest day present), applies the cohort filters, and writes the
surviving wells to data/processed/cohort_v2.csv.

Filters (counts are printed after each step):
  1. Timepoint selection: for each well keep the highest day parsed from its
     filenames. Wells with no day token use their unsuffixed files as a single
     timepoint (day 8 is assigned for the TKI DAY 8(E.P) plate).
  2. Drop the IMA drug class entirely.
  3. Drop the CTRL class from TKI_DRUGS_OCT_REORGANIZED only (CTRL from
     TKI_data_REORGANIZED is kept as a normal class).
  4. Drop treated wells with Cmax < 0.1. Controls (CTRL class) are exempt.
  5. Drop any well without a CH1 image on its selected day.

Files with "Overlay" in the name are ignored throughout.
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter, OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_CSV = REPO_ROOT / "data" / "processed" / "cohort_v2.csv"
BASELINE_CSV = REPO_ROOT / "data" / "processed" / "baseline_v2.csv"

# Dataset folder name -> default day for wells that carry no day token.
# Only the DAY 8(E.P) plate has an assigned single-timepoint day (8); the other
# plates leave it blank when a well has no day-tokened files.
DATASETS = OrderedDict(
    [
        ("TKI DAY 8(E.P)_REORGANIZED", 8),
        ("TKI_data_REORGANIZED", None),
        ("TKI_DRUGS_OCT_REORGANIZED", None),
        ("TKI_PLATE 2_REORGANIZED", None),
    ]
)

# Dataset from which the CTRL class must be dropped (step 3).
OCT_DATASET = "TKI_DRUGS_OCT_REORGANIZED"

CHANNELS = (1, 2, 4)

DAY_RE = re.compile(r"day\s*0*(\d+)", re.IGNORECASE)
CH_RE = re.compile(r"ch\s*(\d+)", re.IGNORECASE)

EXPECTED_TOTAL = 256
EXPECTED_PER_CLASS = {
    "AXI": 59,
    "OSI": 59,
    "IBRU": 53,
    "SORA": 43,
    "DOX": 29,
    "CTRL": 13,
}


# ---------------------------------------------------------------------------
# Well discovery
# ---------------------------------------------------------------------------
def iter_well_dirs(dataset_root: Path):
    """Yield every leaf directory that directly contains .tif files."""
    for dirpath, _dirnames, filenames in os.walk(dataset_root):
        if any(f.lower().endswith((".tif", ".tiff")) for f in filenames):
            yield Path(dirpath)


def parse_well_path(dataset_root: Path, well_dir: Path):
    """Return (drug, cmax, well) from a well directory's relative path.

    Normal layout : <Drug>/<Cmax>/<Well>
    CTRL layout   : <Drug=CTRL>/<Well>   (no Cmax level)
    """
    parts = well_dir.relative_to(dataset_root).parts
    if len(parts) >= 3:
        return parts[0], parts[1], parts[-1]
    if len(parts) == 2:
        return parts[0], "", parts[1]
    # Unexpected shallow layout; treat the single component as the well.
    return parts[0] if parts else "", "", parts[-1] if parts else ""


def select_timepoint(well_dir: Path, default_day):
    """Pick the highest-day timepoint for a well.

    Returns (selected_day, channel_map) where channel_map maps channel number
    -> repo-relative posix path for the files on the selected day. Overlay files
    are ignored.
    """
    tifs = [
        f
        for f in os.listdir(well_dir)
        if f.lower().endswith((".tif", ".tiff")) and "overlay" not in f.lower()
    ]

    parsed = []  # (filename, day_or_None, channel_or_None)
    for f in tifs:
        dm = DAY_RE.search(f)
        cm = CH_RE.search(f)
        day = int(dm.group(1)) if dm else None
        ch = int(cm.group(1)) if cm else None
        parsed.append((f, day, ch))

    day_vals = [d for _, d, _ in parsed if d is not None]
    if day_vals:
        selected_day = max(day_vals)
        selected = [p for p in parsed if p[1] == selected_day]
    else:
        # No day tokens anywhere: single timepoint from the unsuffixed files.
        selected_day = default_day
        selected = [p for p in parsed if p[1] is None]

    channel_map = {}
    for fname, _day, ch in sorted(selected):
        if ch in CHANNELS and ch not in channel_map:
            rel = (well_dir / fname).relative_to(REPO_ROOT).as_posix()
            channel_map[ch] = rel

    return selected_day, channel_map


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def build_records():
    records = []
    for dataset_name, default_day in DATASETS.items():
        dataset_root = RAW_DIR / dataset_name
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset_root}")
        for well_dir in iter_well_dirs(dataset_root):
            drug, cmax, well = parse_well_path(dataset_root, well_dir)
            selected_day, channel_map = select_timepoint(well_dir, default_day)
            records.append(
                {
                    "dataset": dataset_name,
                    "drug": drug,
                    "cmax": cmax,
                    "well": well,
                    "selected_day": selected_day,
                    "channel_map": channel_map,
                }
            )
    return records


def cmax_value(rec):
    try:
        return float(rec["cmax"])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Baseline well list (for delta normalization)
# ---------------------------------------------------------------------------
# Baseline wells are the untreated / zero-dose condition each dataset's delta
# normalization subtracts against. They are deliberately NOT part of the 256-well
# training cohort, so they are selected from the full discovered set (before the
# cohort filters run), using the same timepoint and CH1 rules as the cohort.
#
# Selection, per dataset:
#   * Primary: wells with Cmax exactly 0, excluding IMA  -> source "zero_dose".
#   * Fallback: only if a dataset has NO Cmax=0 wells at all, use that dataset's
#     CTRL wells -> source "ctrl_fallback".
# CTRL must NOT serve as both a training class and a baseline for the same
# dataset: subtracting a group's own mean from itself makes that class trivially
# separable. So a dataset that has zero-dose wells never contributes its CTRL to
# the baseline (e.g. TKI_data's CTRL stays out; only TKI_DRUGS_OCT, which has no
# Cmax=0 wells, falls back to its CTRL).
def zero_dose_wells(all_records, dataset):
    """Cmax==0 wells for a dataset (excluding IMA) that have CH1 on their day."""
    out = []
    for rec in all_records:
        if rec["dataset"] != dataset or rec["drug"] == "IMA":
            continue
        cm = cmax_value(rec)
        if cm is not None and cm == 0.0 and 1 in rec["channel_map"]:
            out.append(rec)
    return out


def ctrl_wells(all_records, dataset):
    """CTRL wells for a dataset that have CH1 on their selected day."""
    return [
        rec
        for rec in all_records
        if rec["dataset"] == dataset
        and rec["drug"] == "CTRL"
        and 1 in rec["channel_map"]
    ]


def build_baseline_records(all_records):
    """Select baseline wells per dataset: zero-dose if any exist, else CTRL
    fallback. Returns a list of (record, source) pairs, deduplicated by key."""
    baseline, seen = [], set()
    datasets = list(DATASETS.keys())
    # Include any dataset present in the records but not listed in DATASETS.
    for ds in {r["dataset"] for r in all_records}:
        if ds not in datasets:
            datasets.append(ds)

    for ds in datasets:
        zeros = zero_dose_wells(all_records, ds)
        if zeros:
            chosen = [(r, "zero_dose") for r in zeros]
        else:
            chosen = [(r, "ctrl_fallback") for r in ctrl_wells(all_records, ds)]
        for rec, source in chosen:
            key = (rec["dataset"], rec["drug"], rec["cmax"], rec["well"])
            if key in seen:
                continue
            seen.add(key)
            baseline.append((rec, source))

    baseline.sort(key=lambda x: (x[0]["dataset"], x[0]["drug"], x[0]["cmax"], x[0]["well"]))
    return baseline


def write_baseline(baseline):
    fieldnames = [
        "dataset",
        "drug",
        "cmax",
        "well",
        "selected_day",
        "channels_present",
        "ch1_path",
        "ch2_path",
        "ch4_path",
        "role",
        "baseline_source",
    ]
    BASELINE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r, source in baseline:
            cm = r["channel_map"]
            present = ";".join(f"CH{c}" for c in CHANNELS if c in cm)
            writer.writerow(
                {
                    "dataset": r["dataset"],
                    "drug": r["drug"],
                    "cmax": r["cmax"],
                    "well": r["well"],
                    "selected_day": "" if r["selected_day"] is None else r["selected_day"],
                    "channels_present": present,
                    "ch1_path": cm.get(1, ""),
                    "ch2_path": cm.get(2, ""),
                    "ch4_path": cm.get(4, ""),
                    "role": "baseline",
                    "baseline_source": source,
                }
            )


def report_baseline(all_records, cohort_records):
    baseline = build_baseline_records(all_records)
    write_baseline(baseline)
    print("\n" + "-" * 60)
    print(f"Baseline well list -> {BASELINE_CSV.relative_to(REPO_ROOT).as_posix()}")
    print("Baseline counts by dataset, drug, source:")
    counts = Counter((r["dataset"], r["drug"], source) for r, source in baseline)
    for dataset, drug, source in sorted(counts):
        print(f"  {dataset:32s} {drug:5s} {source:13s} = {counts[(dataset, drug, source)]}")
    print(f"  TOTAL baseline wells = {len(baseline)}  (no assertion on this total)")

    # Cross-check: no well may appear in both the cohort and the baseline.
    cohort_keys = {(r["dataset"], r["drug"], r["cmax"], r["well"]) for r in cohort_records}
    baseline_keys = {(r["dataset"], r["drug"], r["cmax"], r["well"]) for r, _ in baseline}
    overlap = cohort_keys & baseline_keys
    print(f"\nCross-check: wells in BOTH cohort_v2 and baseline_v2 = {len(overlap)} "
          f"(must be 0) -> {'PASS' if not overlap else 'FAIL'}")
    if overlap:
        for key in sorted(overlap):
            print(f"    overlap: {key}")


def main():
    steps = []  # (label, remaining_count)

    records = build_records()
    # Full discovered set, kept for the baseline selection (which must include
    # wells the cohort filters below deliberately drop). The filter steps rebind
    # `records` to new lists, so this reference stays intact.
    all_records = records
    steps.append(("1. Timepoint selected (all wells discovered)", len(records)))
    print(f"[step 1] wells after timepoint selection: {len(records)}")

    # Step 2: drop IMA entirely.
    records = [r for r in records if r["drug"] != "IMA"]
    steps.append(("2. Dropped IMA class", len(records)))
    print(f"[step 2] after dropping IMA: {len(records)}")

    # Step 3: drop CTRL from the OCT dataset only.
    records = [
        r
        for r in records
        if not (r["dataset"] == OCT_DATASET and r["drug"] == "CTRL")
    ]
    steps.append(("3. Dropped CTRL from OCT dataset", len(records)))
    print(f"[step 3] after dropping OCT CTRL: {len(records)}")

    # Step 4: drop treated wells with Cmax < 0.1 (controls exempt).
    def keep_cmax(r):
        if r["drug"] == "CTRL":
            return True
        cm = cmax_value(r)
        return cm is not None and cm >= 0.1

    records = [r for r in records if keep_cmax(r)]
    steps.append(("4. Dropped treated Cmax < 0.1 (controls exempt)", len(records)))
    print(f"[step 4] after dropping Cmax < 0.1: {len(records)}")

    # Step 5: drop wells without a CH1 image on the selected day.
    records = [r for r in records if 1 in r["channel_map"]]
    steps.append(("5. Dropped wells with no CH1 on selected day", len(records)))
    print(f"[step 5] after dropping wells with no CH1: {len(records)}")

    # Deterministic ordering for the output.
    records.sort(
        key=lambda r: (r["dataset"], r["drug"], r["cmax"], r["well"])
    )

    # Write survivors.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "drug",
        "cmax",
        "well",
        "selected_day",
        "channels_present",
        "ch1_path",
        "ch2_path",
        "ch4_path",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            cm = r["channel_map"]
            present = ";".join(f"CH{c}" for c in CHANNELS if c in cm)
            writer.writerow(
                {
                    "dataset": r["dataset"],
                    "drug": r["drug"],
                    "cmax": r["cmax"],
                    "well": r["well"],
                    "selected_day": "" if r["selected_day"] is None else r["selected_day"],
                    "channels_present": present,
                    "ch1_path": cm.get(1, ""),
                    "ch2_path": cm.get(2, ""),
                    "ch4_path": cm.get(4, ""),
                }
            )
    print(f"\nWrote {len(records)} wells to {OUT_CSV.relative_to(REPO_ROOT).as_posix()}")

    # -------------------------------------------------------------------
    # Verification (assert, but report instead of raising on failure).
    # -------------------------------------------------------------------
    counts = Counter(r["drug"] for r in records)
    total = len(records)

    try:
        assert total == EXPECTED_TOTAL, (
            f"total {total} != expected {EXPECTED_TOTAL}"
        )
        for cls, exp in EXPECTED_PER_CLASS.items():
            assert counts.get(cls, 0) == exp, (
                f"{cls} count {counts.get(cls, 0)} != expected {exp}"
            )
        print("\nAll cohort assertions passed.")
        print(f"  total = {total}")
        for cls in EXPECTED_PER_CLASS:
            print(f"  {cls:5s} = {counts.get(cls, 0)}")
    except AssertionError as err:
        print(f"\n*** COHORT ASSERTION FAILED: {err}")
        print("\nActual per-class counts:")
        for cls in sorted(set(EXPECTED_PER_CLASS) | set(counts)):
            exp = EXPECTED_PER_CLASS.get(cls, "-")
            print(f"  {cls:6s} actual={counts.get(cls, 0):4d}  expected={exp}")
        print(f"  {'TOTAL':6s} actual={total:4d}  expected={EXPECTED_TOTAL}")
        print("\nPer-step drop table:")
        prev = None
        for label, remaining in steps:
            drop = "" if prev is None else f"(-{prev - remaining})"
            print(f"  {remaining:4d}  {drop:8s} {label}")
            prev = remaining

    # -------------------------------------------------------------------
    # Baseline well list (additive; does not touch the cohort output above).
    # -------------------------------------------------------------------
    report_baseline(all_records, records)

    cohort_ok = total == EXPECTED_TOTAL
    print(
        f"\ncohort_v2.csv unchanged: {len(records)} wells written, "
        f"256-well assertion {'holds' if cohort_ok else 'FAILED'}. "
        f"Baseline is written to a separate file and does not affect it."
    )


if __name__ == "__main__":
    main()
