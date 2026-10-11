"""
Interactive SAM2 cohort segmentation.

Reads data/processed/cohort_v2.csv and produces exactly one binary mask per well,
saved permanently to disk. Segmentation is done on the CH4 brightfield image only
(single-channel greyscale, the organoid a dark blob on a lighter field). The one
mask per well is meant to be applied later to all three channels, so CH1/CH2 are
never segmented here.

Controls:
  Left click   : positive prompt (organoid)
  Right click  : negative prompt (exclude this area)
  D            : toggle freehand draw mode (click+drag to draw mask)
  Y / Enter    : confirm mask -> save + next well
  R            : reset the current well
  S            : skip well
  Q            : quit and save progress

On confirm each well writes:
  * a binary mask PNG at data/masks/<dataset>/<drug>/<cmax>/<well>.png,
    single-channel uint8, values 0 and 255, the exact height/width of the source
    CH4 image (no resizing), and
  * an append-only row in data/processed/segmentation_log.csv.

Resumability: on startup the log is read and any well whose latest status is
"confirmed" is skipped. An existing mask file is never overwritten unless the
well is deliberately re-segmented via --redo.

Flags:
  --only-dataset "<dataset>"  restrict the queue to one dataset
  --redo "<well identifier>"  re-segment a single well (overwrites its mask)

This tool writes only under data/masks/ and appends to
data/processed/segmentation_log.csv. It never modifies or rewrites any source
image and never resizes pixel data.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage import measure

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402

# Make stdout tolerant of non-ASCII glyphs when redirected.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
COHORT_CSV = REPO_ROOT / "data" / "processed" / "cohort_v2.csv"
BASELINE_CSV = REPO_ROOT / "data" / "processed" / "baseline_v2.csv"
MASKS_DIR = REPO_ROOT / "data" / "masks"
LOG_CSV = REPO_ROOT / "data" / "processed" / "segmentation_log.csv"
CKPT = REPO_ROOT / "sam2_checkpoint.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

LOG_FIELDNAMES = [
    "dataset",
    "drug",
    "cmax",
    "well",
    "selected_day",
    "mask_path",
    "mask_pixel_area",
    "n_connected_components",
    "timestamp",
    "status",
]


# ---------------------------------------------------------------------------
# Cohort / queue
# ---------------------------------------------------------------------------
def read_cohort(queue_csv: Path = COHORT_CSV):
    with queue_csv.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cmax_component(cmax: str) -> str:
    """Directory component for a well's cmax. Controls have a blank cmax in the
    cohort csv; use "NA" so the path never collapses a level."""
    cmax = (cmax or "").strip()
    return cmax if cmax else "NA"


def mask_path_for(rec) -> Path:
    return (
        MASKS_DIR
        / rec["dataset"]
        / rec["drug"]
        / cmax_component(rec["cmax"])
        / f"{rec['well']}.png"
    )


def well_key(rec) -> tuple:
    return (rec["dataset"], rec["drug"], rec["cmax"], rec["well"])


# ---------------------------------------------------------------------------
# Log (append-only; latest row per well wins)
# ---------------------------------------------------------------------------
def read_log_status():
    """Return {well_key: latest_status} from the append-only log."""
    status = {}
    if not LOG_CSV.exists() or LOG_CSV.stat().st_size == 0:
        return status
    with LOG_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], row["drug"], row["cmax"], row["well"])
            status[key] = row["status"]  # later rows overwrite -> latest wins
    return status


def append_log_row(rec, mask_path, area, n_components, status):
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_CSV.exists() or LOG_CSV.stat().st_size == 0
    with LOG_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "dataset": rec["dataset"],
                "drug": rec["drug"],
                "cmax": rec["cmax"],
                "well": rec["well"],
                "selected_day": rec["selected_day"],
                "mask_path": mask_path,
                "mask_pixel_area": area,
                "n_connected_components": n_components,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": status,
            }
        )


# ---------------------------------------------------------------------------
# SAM2
# ---------------------------------------------------------------------------
def load_sam2():
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(SAM2_CONFIG, str(CKPT), device=device)
    print(f"SAM2 loaded on {device}.")
    return SAM2ImagePredictor(model)


def run_sam2(predictor, pos_points, neg_points):
    all_points = pos_points + neg_points
    all_labels = [1] * len(pos_points) + [0] * len(neg_points)
    if not all_points:
        return None
    masks, scores, _ = predictor.predict(
        point_coords=np.array(all_points),
        point_labels=np.array(all_labels),
        multimask_output=True,
    )
    return masks[np.argmax(scores)].astype(bool)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def ensure_2d(img):
    """CH4 is greyscale; drop any trailing sample axis defensively."""
    if img.ndim > 2:
        img = img[..., 0]
    return img


def norm_display(img):
    img = ensure_2d(img).astype(np.float32)
    mn, mx = img.min(), img.max()
    return ((img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)


def save_mask_png(path: Path, mask: np.ndarray):
    """Write a single-channel uint8 PNG with values 0/255, no resizing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask.astype(bool).astype(np.uint8)) * 255
    Image.fromarray(out, mode="L").save(str(path))


# ---------------------------------------------------------------------------
# Interactive segmentation of a single well
# ---------------------------------------------------------------------------
def segment_well(predictor, rec, position, total):
    """Show the GUI for one well. Returns one of:
    ('confirmed', mask) | ('skipped', None) | ('quit', None) | ('missing', None)
    """
    ch4_path = (rec.get("ch4_path") or "").strip()
    if not ch4_path:
        print(f"  ! no CH4 path for {well_key(rec)} -- cannot segment, skipping")
        return "missing", None
    abs_ch4 = REPO_ROOT / ch4_path
    if not abs_ch4.is_file():
        print(f"  ! CH4 file missing on disk: {ch4_path} -- skipping")
        return "missing", None

    ch4_2d = ensure_2d(tifffile.imread(str(abs_ch4)))
    ch4_disp = norm_display(ch4_2d)
    predictor.set_image(np.stack([ch4_disp] * 3, axis=-1))

    H, W = ch4_2d.shape
    state = {
        "mask": None,
        "result": None,  # 'confirmed' | 'skipped' | 'quit'
        "draw_mode": False,
        "drawing": False,
        "draw_pts": [],
        "pos_points": [],
        "neg_points": [],
    }
    overlay_im = [None]
    pos_plots = []
    neg_plots = []
    draw_line = [None]
    mode_text = [None]

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.08)
    ax.imshow(ch4_disp, cmap="gray")
    ax.set_title(
        f"{rec['dataset']}   {rec['drug']}   cmax={rec['cmax'] or 'NA'}   "
        f"well {rec['well']}   day {rec['selected_day'] or 'NA'}\n"
        f"well {position} of {total}   |   "
        f"Left=pos  Right=neg  D=draw  Y/Enter=confirm  R=reset  S=skip  Q=quit",
        fontsize=9,
    )
    ax.axis("off")
    mode_text[0] = fig.text(0.75, 0.01, "Mode: SAM2", fontsize=9, color="blue")

    def redraw_overlay(mask):
        if overlay_im[0]:
            overlay_im[0].remove()
        rgba = np.zeros((H, W, 4), dtype=float)
        if mask is not None:
            rgba[mask] = [0, 1, 0, 0.4]
        overlay_im[0] = ax.imshow(rgba)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        x, y = int(event.xdata), int(event.ydata)

        if state["draw_mode"]:
            if event.button == 1:
                state["drawing"] = True
                state["draw_pts"] = [(x, y)]
            return

        if event.button == 1:
            state["pos_points"].append([x, y])
            p, = ax.plot(x, y, "g+", markersize=14, markeredgewidth=2)
            pos_plots.append(p)
        elif event.button == 3:
            state["neg_points"].append([x, y])
            p, = ax.plot(x, y, "rx", markersize=14, markeredgewidth=2)
            neg_plots.append(p)
        else:
            return

        state["mask"] = run_sam2(predictor, state["pos_points"], state["neg_points"])
        redraw_overlay(state["mask"])

    def on_motion(event):
        if not state["draw_mode"] or not state["drawing"]:
            return
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        state["draw_pts"].append((x, y))
        if draw_line[0]:
            draw_line[0].remove()
        xs = [p[0] for p in state["draw_pts"]]
        ys = [p[1] for p in state["draw_pts"]]
        draw_line[0], = ax.plot(xs, ys, "g-", linewidth=2)
        fig.canvas.draw_idle()

    def on_release(event):
        if not state["draw_mode"] or not state["drawing"]:
            return
        state["drawing"] = False
        pts = state["draw_pts"]
        if len(pts) < 3:
            return
        from matplotlib.path import Path as MplPath

        path = MplPath(np.array(pts))
        yy, xx = np.mgrid[0:H, 0:W]
        points = np.column_stack([xx.ravel(), yy.ravel()])
        inside = path.contains_points(points).reshape(H, W)
        state["mask"] = inside
        redraw_overlay(inside)

    def on_key(event):
        if event.key in ("y", "enter"):
            if state["mask"] is not None and state["mask"].any():
                state["result"] = "confirmed"
                plt.close(fig)
        elif event.key == "r":
            state["mask"] = None
            state["pos_points"].clear()
            state["neg_points"].clear()
            state["draw_pts"] = []
            state["drawing"] = False
            for p in pos_plots:
                p.remove()
            for p in neg_plots:
                p.remove()
            pos_plots.clear()
            neg_plots.clear()
            if draw_line[0]:
                draw_line[0].remove()
                draw_line[0] = None
            redraw_overlay(None)
        elif event.key == "d":
            state["draw_mode"] = not state["draw_mode"]
            state["pos_points"].clear()
            state["neg_points"].clear()
            mode = "DRAW" if state["draw_mode"] else "SAM2"
            mode_text[0].set_text(f"Mode: {mode}")
            mode_text[0].set_color("red" if state["draw_mode"] else "blue")
            fig.canvas.draw_idle()
        elif event.key == "s":
            state["result"] = "skipped"
            plt.close(fig)
        elif event.key == "q":
            state["result"] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if state["result"] == "confirmed":
        return "confirmed", state["mask"]
    if state["result"] == "quit":
        return "quit", None
    return "skipped", None


# ---------------------------------------------------------------------------
# Redo matching
# ---------------------------------------------------------------------------
def find_redo_target(records, identifier):
    """Match a --redo identifier against the cohort. Accepts a bare well name or
    any trailing slash-qualified path like 'drug/cmax/well' or the full
    'dataset/drug/cmax/well'. Returns the single matching record or None; prints
    candidates when the match is ambiguous."""
    ident = identifier.strip().strip("/")

    def qualified(rec):
        return "/".join(
            [rec["dataset"], rec["drug"], cmax_component(rec["cmax"]), rec["well"]]
        )

    matches = [
        rec
        for rec in records
        if rec["well"] == ident or qualified(rec).endswith(ident)
    ]
    if not matches:
        print(f"--redo: no well matched '{identifier}'.")
        return None
    if len(matches) > 1:
        print(f"--redo: '{identifier}' is ambiguous; matched {len(matches)} wells:")
        for rec in matches:
            print(f"    {qualified(rec)}")
        print("Qualify further, e.g. --redo 'drug/cmax/well'.")
        return None
    return matches[0]


# ---------------------------------------------------------------------------
# Non-GUI listing of remaining (unsegmented) wells
# ---------------------------------------------------------------------------
def list_remaining(queues, only_dataset=None):
    """Print remaining unsegmented wells per dataset (latest log status !=
    'confirmed'), sorted smallest-remaining first, with a per-role breakdown.

    `queues` is a list of (role, csv_path) pairs. Reads the append-only log once
    and never touches SAM2 or the GUI.
    """
    status = read_log_status()

    # per dataset -> {"cohort": n, "baseline": n, "total": n}
    remaining = {}
    present_queues = []
    for role, csv_path in queues:
        if not csv_path.is_file():
            print(f"  ! queue CSV not found, skipping: {csv_path}")
            continue
        present_queues.append(role)
        for rec in read_cohort(csv_path):
            ds = rec["dataset"]
            if only_dataset and ds != only_dataset:
                continue
            if status.get(well_key(rec)) == "confirmed":
                continue
            bucket = remaining.setdefault(ds, {"cohort": 0, "baseline": 0, "total": 0})
            bucket[role] = bucket.get(role, 0) + 1
            bucket["total"] += 1

    label = " + ".join(present_queues) if present_queues else "(none)"
    print(f"\nRemaining unsegmented wells (queues: {label}), smallest first:")
    if not remaining:
        print("  (nothing remaining)")
        return

    show_roles = len(present_queues) > 1
    for ds, b in sorted(remaining.items(), key=lambda kv: kv[1]["total"]):
        if show_roles:
            print(f"  {b['total']:4d}  {ds:34s}  "
                  f"(cohort {b['cohort']}, baseline {b['baseline']})")
        else:
            print(f"  {b['total']:4d}  {ds}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive SAM2 cohort segmentation.")
    parser.add_argument("--only-dataset", default=None, help="Restrict to one dataset.")
    parser.add_argument("--redo", default=None, help="Re-segment one well (identifier).")
    parser.add_argument(
        "--list", action="store_true", dest="list_only",
        help="Print the remaining unsegmented wells per dataset and exit. No GUI "
        "and no SAM2 load. Without --queue it reports cohort_v2.csv and "
        "baseline_v2.csv combined (with a per-role breakdown); with --queue it "
        "reports only that queue. Datasets are sorted smallest-remaining first.",
    )
    parser.add_argument(
        "--queue",
        default=None,
        help="Well-list CSV to segment (same columns as cohort_v2.csv). "
        "Defaults to data/processed/cohort_v2.csv. Use e.g. baseline_v2.csv to "
        "segment baseline wells; masks and the log are keyed by "
        "dataset/drug/cmax/well either way, so the two never collide.",
    )
    args = parser.parse_args()

    # ---- List mode: report remaining wells and exit (no GUI, no SAM2) ----
    if args.list_only:
        if args.queue:
            queue_csv = Path(args.queue)
            if not queue_csv.is_file():
                queue_csv = REPO_ROOT / args.queue
            queues = [("queue", queue_csv)]
        else:
            queues = [("cohort", COHORT_CSV), ("baseline", BASELINE_CSV)]
        list_remaining(queues, only_dataset=args.only_dataset)
        return

    if args.queue:
        queue_csv = Path(args.queue)
        if not queue_csv.is_file():
            queue_csv = REPO_ROOT / args.queue  # allow a repo-relative path
        if not queue_csv.is_file():
            print(f"Queue CSV not found: {args.queue}")
            sys.exit(1)
    else:
        queue_csv = COHORT_CSV
    print(f"Queue source: {queue_csv.relative_to(REPO_ROOT).as_posix() if queue_csv.is_relative_to(REPO_ROOT) else queue_csv}")

    records = read_cohort(queue_csv)

    # ---- Redo path: single well, overwrite allowed ----------------------
    if args.redo:
        target = find_redo_target(records, args.redo)
        if target is None:
            sys.exit(1)
        print(f"Re-segmenting {'/'.join(well_key(target))}")
        predictor = load_sam2()
        result, mask = segment_well(predictor, target, 1, 1)
        if result == "confirmed":
            path = mask_path_for(target)
            save_mask_png(path, mask)
            area = int(mask.sum())
            n_comp = int(measure.label(mask, connectivity=2).max())
            rel = path.relative_to(REPO_ROOT).as_posix()
            append_log_row(target, rel, area, n_comp, "confirmed")
            if n_comp > 1:
                print(
                    f"  ! WARNING: {target['well']} mask has {n_comp} connected "
                    f"components -- saved as drawn (largest-blob choice left to extractor)."
                )
            print(f"  saved {rel}  area={area}  components={n_comp}")
        else:
            print(f"  {result}; mask not changed.")
        return

    # ---- Normal queue ----------------------------------------------------
    queue = records
    if args.only_dataset:
        queue = [r for r in records if r["dataset"] == args.only_dataset]
        if not queue:
            datasets = sorted({r["dataset"] for r in records})
            print(f"No wells for dataset '{args.only_dataset}'. Available:")
            for d in datasets:
                print(f"    {d}")
            sys.exit(1)

    status = read_log_status()
    total = len(queue)
    todo = [r for r in queue if status.get(well_key(r)) != "confirmed"]
    already = total - len(todo)
    print(f"Queue: {total} wells" + (f" (dataset '{args.only_dataset}')" if args.only_dataset else ""))
    print(f"Already confirmed: {already}   Remaining: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    predictor = load_sam2()

    confirmed = skipped = 0
    for rec in queue:
        position = queue.index(rec) + 1
        if status.get(well_key(rec)) == "confirmed":
            continue

        result, mask = segment_well(predictor, rec, position, total)

        if result == "quit":
            print(f"\nQuit. Confirmed {confirmed}, skipped {skipped} this session.")
            print("Progress is saved (log is append-only).")
            return

        if result == "missing":
            continue

        if result == "skipped":
            append_log_row(rec, "", 0, 0, "skipped")
            skipped += 1
            print(f"  skipped {rec['well']}")
            continue

        # confirmed
        path = mask_path_for(rec)
        if path.exists():
            print(
                f"  ! mask already exists for {rec['well']} ({path.relative_to(REPO_ROOT).as_posix()}). "
                f"Not overwriting -- use --redo '{rec['well']}' to re-segment. Treated as skip."
            )
            continue

        save_mask_png(path, mask)
        area = int(mask.sum())
        n_comp = int(measure.label(mask, connectivity=2).max())
        rel = path.relative_to(REPO_ROOT).as_posix()
        append_log_row(rec, rel, area, n_comp, "confirmed")
        status[well_key(rec)] = "confirmed"
        confirmed += 1
        if n_comp > 1:
            print(
                f"  ! WARNING: {rec['well']} mask has {n_comp} connected components -- "
                f"saved as drawn (largest-blob choice left to the extractor)."
            )
        print(f"  OK {rec['well']}  area={area}  components={n_comp}  -> {rel}")

    print(f"\nDataset complete. Confirmed {confirmed}, skipped {skipped} this session.")


if __name__ == "__main__":
    main()
