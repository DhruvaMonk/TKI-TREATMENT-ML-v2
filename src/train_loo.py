"""
Leave-one-out modeling pipeline.

Reads data/processed/features_v2.csv. The label is `label_drug`. Only rows with
role == cohort are trained/tested; role == baseline rows are never trained or
tested on -- they exist solely to compute the per-dataset delta baseline.

Normalization order:
  1. Delta normalization: each cohort row has its own dataset's baseline mean
     (per feature, over that dataset's baseline rows) subtracted, so the model
     sees change relative to untreated. The baseline mean depends only on
     baseline rows (disjoint from the cohort), so it is computed once up front --
     it never sees a held-out cohort sample.
  2. Intra-dataset z-scoring: each feature is standardized within its dataset
     group, using means/stds computed from TRAINING-fold rows only (inside LOO).

Evaluation is leave-one-out over the cohort rows. Every fitted step below lives
inside the LOO loop, fitted on the training fold and applied to the held-out
sample; nothing is fitted on the full dataset:
  * median imputation of NaN features (train-fold medians),
  * intra-dataset z-score statistics (train-fold rows),
  * 4x gaussian-noise augmentation (sigma 0.05) of TRAINING rows only,
  * balanced class weights, with training samples from the held-out sample's
    dataset given double weight.

Models: XGBoost, Random Forest, and a soft-voting ensemble of the two, each
reported separately. Fixed random seeds throughout.

A confound check reuses the same normalized features and LOO procedure but with
`dataset` as the label, reported over all cohort rows and over CTRL rows only,
against the prior values of 81% (all) and 87.5% (CTRL).

All outputs go to results/. No file outside that directory is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_v2.csv"
RESULTS_DIR = REPO_ROOT / "results"

# Suffix appended to every output filename. Set from CLI (e.g. "_dropCH2") so an
# ablation run never clobbers the baseline artifacts in results/.
OUT_SUFFIX = ""

# Normalization mode. "zscore" = delta + per-fold intra-dataset z-score (default).
# "baseline" = (x - baseline_mean) / baseline_sd per dataset from untreated wells,
# with the per-fold z-score disabled. Set from --norm.
INFOLD_ZSCORE = True

SEED = 42
AUG_FACTOR = 4          # training fold becomes 4x (original + 3 noisy copies)
AUG_SIGMA = 0.05        # gaussian noise sigma on normalized features
DATASET_WEIGHT_FACTOR = 2.0  # extra weight for same-dataset-as-held-out rows

ID_COLUMNS = [
    "dataset", "drug", "cmax", "well", "selected_day", "role", "label_drug",
    "n_connected_components", "largest_component_fraction",
]

# Prior confound-probe accuracies to print for comparison.
PRIOR_CONFOUND_ALL = 0.81
PRIOR_CONFOUND_CTRL = 0.875


def make_xgb():
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def delta_normalize(df, feature_cols):
    """Subtract each cohort row's own-dataset baseline mean (per feature). The
    baseline mean is over that dataset's role==baseline rows only. Returns the
    delta-normalized cohort frame (baseline rows are dropped)."""
    baseline = df[df["role"] == "baseline"]
    cohort = df[df["role"] == "cohort"].copy()
    base_means = baseline.groupby("dataset")[feature_cols].mean()  # NaN-aware

    missing = []
    for ds, idx in cohort.groupby("dataset").groups.items():
        if ds in base_means.index:
            cohort.loc[idx, feature_cols] = (
                cohort.loc[idx, feature_cols].to_numpy()
                - base_means.loc[ds, feature_cols].to_numpy()
            )
        else:
            missing.append(ds)  # no baseline: leave absolute values, warn
    if missing:
        print(f"  ! WARNING: no baseline rows for datasets {missing}; "
              f"their cohort rows were NOT delta-normalized.")
    return cohort


def impute_fold(X_train, X_test):
    """Median imputation using train-fold medians; all-NaN feature -> 0."""
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(X_train), med, X_train)
    Xte = np.where(np.isnan(X_test), med, X_test)
    return Xtr, Xte


def zscore_fold(X_train, ds_train, X_test, ds_test):
    """Intra-dataset z-score using train-fold rows of each dataset. Zero-variance
    features get std=1. A dataset with no training rows falls back to the pooled
    training statistics."""
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for ds in np.unique(np.concatenate([ds_train, ds_test])):
        tr_mask = ds_train == ds
        if tr_mask.sum() == 0:
            mu = X_train.mean(axis=0)
            sd = X_train.std(axis=0)
        else:
            mu = X_train[tr_mask].mean(axis=0)
            sd = X_train[tr_mask].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        te_mask = ds_test == ds
        if tr_mask.any():
            Xtr[tr_mask] = (X_train[tr_mask] - mu) / sd
        if te_mask.any():
            Xte[te_mask] = (X_test[te_mask] - mu) / sd
    return Xtr, Xte


def augment(X, y, w, rng, factor=AUG_FACTOR, sigma=AUG_SIGMA):
    """Return the training set grown to `factor`x: the original plus factor-1
    gaussian-noise copies. Labels and weights are replicated."""
    Xs, ys, ws = [X], [y], [w]
    for _ in range(factor - 1):
        Xs.append(X + rng.normal(0.0, sigma, size=X.shape))
        ys.append(y)
        ws.append(w)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def fold_sample_weights(y_train, ds_train, ds_held):
    """Balanced class weights, doubled for rows from the held-out sample's
    dataset."""
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_map = dict(zip(classes, cw))
    w = np.array([cw_map[c] for c in y_train], dtype=float)
    w *= np.where(ds_train == ds_held, DATASET_WEIGHT_FACTOR, 1.0)
    return w


def proba_aligned(model, X, n_classes):
    """predict_proba aligned to global class indices 0..n_classes-1."""
    p = model.predict_proba(X)
    out = np.zeros((X.shape[0], n_classes))
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = p[:, j]
    return out


# ---------------------------------------------------------------------------
# LOO core
# ---------------------------------------------------------------------------
def loo_predict(X, y, datasets, n_classes, collect_importance=False):
    """Run the full per-fold pipeline. Returns dict of per-model predictions and,
    optionally, summed tree importances."""
    n = len(y)
    preds = {"xgb": np.empty(n, int), "rf": np.empty(n, int),
             "ensemble": np.empty(n, int)}
    imp = {"xgb": np.zeros(X.shape[1]), "rf": np.zeros(X.shape[1])}

    for i in range(n):
        test_idx = np.array([i])
        train_idx = np.delete(np.arange(n), i)

        Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
        ytr = y[train_idx]
        ds_tr, ds_te = datasets[train_idx], datasets[test_idx]
        ds_held = ds_te[0]

        Xtr_i, Xte_i = impute_fold(Xtr_raw, Xte_raw)
        if INFOLD_ZSCORE:
            Xtr_z, Xte_z = zscore_fold(Xtr_i, ds_tr, Xte_i, ds_te)
        else:
            Xtr_z, Xte_z = Xtr_i, Xte_i

        w = fold_sample_weights(ytr, ds_tr, ds_held)
        rng = np.random.default_rng(SEED + i)
        Xtr_a, ytr_a, w_a = augment(Xtr_z, ytr, w, rng)

        xgb = make_xgb()
        xgb.fit(Xtr_a, ytr_a, sample_weight=w_a)
        rf = make_rf()
        rf.fit(Xtr_a, ytr_a, sample_weight=w_a)

        p_xgb = proba_aligned(xgb, Xte_z, n_classes)
        p_rf = proba_aligned(rf, Xte_z, n_classes)
        p_ens = (p_xgb + p_rf) / 2.0

        preds["xgb"][i] = int(p_xgb.argmax(1)[0])
        preds["rf"][i] = int(p_rf.argmax(1)[0])
        preds["ensemble"][i] = int(p_ens.argmax(1)[0])

        if collect_importance:
            imp["xgb"] += xgb.feature_importances_
            imp["rf"] += rf.feature_importances_

        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"    fold {i + 1}/{n}")

    if collect_importance:
        imp = {k: v / n for k, v in imp.items()}
        return preds, imp
    return preds


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def per_dataset_accuracy(y_true, y_pred, datasets):
    out = {}
    for ds in np.unique(datasets):
        m = datasets == ds
        out[ds] = float(accuracy_score(y_true[m], y_pred[m]))
    return out


def model_report(name, y_true, y_pred, datasets, class_names, lines, results):
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"\n### {name}")
    lines.append(f"Overall LOO accuracy: {acc:.4f}  ({int(round(acc*len(y_true)))}/{len(y_true)})")

    rep = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0,
        output_dict=True,
    )
    lines.append("\nPer-class precision / recall / F1:")
    lines.append(f"  {'class':10s} {'prec':>8s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for c in class_names:
        r = rep[c]
        lines.append(f"  {c:10s} {r['precision']:8.4f} {r['recall']:8.4f} "
                     f"{r['f1-score']:8.4f} {int(r['support']):8d}")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    lines.append("\nConfusion matrix (rows = true, cols = pred):")
    header = "  " + " ".join(f"{c[:6]:>7s}" for c in class_names)
    lines.append("          " + header)
    for i, c in enumerate(class_names):
        row = " ".join(f"{v:7d}" for v in cm[i])
        lines.append(f"  {c:8s} {row}")

    ds_acc = per_dataset_accuracy(y_true, y_pred, datasets)
    lines.append("\nAccuracy by dataset:")
    for ds, a in ds_acc.items():
        lines.append(f"  {ds:34s} {a:.4f}  (n={int((datasets==ds).sum())})")

    # Persist artifacts.
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        RESULTS_DIR / f"confusion_{name}{OUT_SUFFIX}.csv")
    pd.DataFrame(rep).transpose().to_csv(
        RESULTS_DIR / f"classification_report_{name}{OUT_SUFFIX}.csv")
    results[name] = {"accuracy": acc, "per_dataset": ds_acc,
                     "classification_report": rep}


def importance_report(imp, feature_names, lines, results):
    for model_key in ("xgb", "rf"):
        order = np.argsort(imp[model_key])[::-1][:20]
        lines.append(f"\nTop 20 features by importance ({model_key.upper()}):")
        rows = []
        for rank, idx in enumerate(order, 1):
            fname = feature_names[idx]
            val = float(imp[model_key][idx])
            lines.append(f"  {rank:2d}. {fname:28s} {val:.5f}")
            rows.append({"rank": rank, "feature": fname, "importance": val})
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f"feature_importance_{model_key}{OUT_SUFFIX}.csv", index=False)
        results.setdefault("feature_importance", {})[model_key] = rows


# ---------------------------------------------------------------------------
# Confound check
# ---------------------------------------------------------------------------
def confound_accuracy(sub, feature_cols):
    """LOO accuracy of predicting `dataset` from the same normalized features.
    Returns (accuracy or None, reason). A single dataset (or <2 samples) is not
    computable."""
    ds_labels = sub["dataset"].to_numpy()
    if len(np.unique(ds_labels)) < 2:
        return None, f"only {len(np.unique(ds_labels))} dataset present"
    if len(sub) < 2:
        return None, "fewer than 2 rows"

    X = sub[feature_cols].to_numpy(dtype=float)
    datasets = ds_labels
    le = LabelEncoder().fit(ds_labels)
    y = le.transform(ds_labels)
    n = len(y)
    preds = np.empty(n, int)

    for i in range(n):
        train_idx = np.delete(np.arange(n), i)
        test_idx = np.array([i])
        Xtr_i, Xte_i = impute_fold(X[train_idx], X[test_idx])
        if INFOLD_ZSCORE:
            Xtr_z, Xte_z = zscore_fold(Xtr_i, datasets[train_idx], Xte_i, datasets[test_idx])
        else:
            Xtr_z, Xte_z = Xtr_i, Xte_i
        w = fold_sample_weights(y[train_idx], datasets[train_idx], datasets[test_idx][0])
        rng = np.random.default_rng(SEED + i)
        Xtr_a, ytr_a, w_a = augment(Xtr_z, y[train_idx], w, rng)
        clf = make_rf()
        clf.fit(Xtr_a, ytr_a, sample_weight=w_a)
        preds[i] = int(clf.predict(Xte_z)[0])

    return float(accuracy_score(y, preds)), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave-one-out drug-label modeling.")
    parser.add_argument(
        "--drop-channel", nargs="+", default=[], metavar="CH",
        help="One or more channel prefixes (e.g. CH2) whose feature columns are "
        "excluded before modeling. Outputs are written with a '_drop<CH..>' "
        "suffix so the baseline artifacts are not overwritten.")
    parser.add_argument(
        "--drop-class", nargs="+", default=[], metavar="LABEL",
        help="One or more drug labels (e.g. DOX) whose cohort rows are removed "
        "before modeling. Baseline rows are unaffected (delta baseline is "
        "drug-agnostic).")
    parser.add_argument(
        "--only-dataset", nargs="+", default=None, metavar="DS",
        help="Restrict cohort rows to these dataset(s). Baseline rows for the "
        "same datasets are kept so delta normalization still works.")
    parser.add_argument(
        "--norm", choices=["zscore", "baseline"], default="zscore",
        help="Normalization: 'zscore' = delta + per-fold intra-dataset z-score "
        "(default); 'baseline' = (x - baseline_mean)/baseline_sd per dataset from "
        "untreated wells, per-fold z-score off.")
    parser.add_argument(
        "--features", default=None,
        help="Override the input features CSV (default features_v2.csv).")
    parser.add_argument(
        "--out-suffix", default=None,
        help="Override the output filename suffix (default derives from "
        "--drop-channel / --drop-class / --norm, else empty).")
    return parser.parse_args()


def main():
    global OUT_SUFFIX, INFOLD_ZSCORE
    args = parse_args()

    # Normalize dropped-channel prefixes to the 'CH2_' form used by the columns.
    drop_channels = [c.strip().rstrip("_").upper() for c in args.drop_channel]
    drop_classes = [c.strip().upper() for c in args.drop_class]
    if args.out_suffix is not None:
        OUT_SUFFIX = args.out_suffix
    else:
        parts = []
        if drop_channels:
            parts.append("drop" + "".join(drop_channels))
        if drop_classes:
            parts.append("drop" + "".join(drop_classes))
        if args.only_dataset:
            parts.append("only" + "".join(
                "".join(w[0] for w in d.split()) for d in args.only_dataset))
        if args.norm != "zscore":
            parts.append(f"norm{args.norm}")
        if parts:
            OUT_SUFFIX = "_" + "_".join(parts)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features_csv = Path(args.features) if args.features else FEATURES_CSV
    df = pd.read_csv(features_csv)

    if args.only_dataset:
        keep = set(args.only_dataset)
        before = len(df)
        df = df[df["dataset"].isin(keep)].reset_index(drop=True)
        print(f"--only-dataset {args.only_dataset}: kept {len(df)}/{before} rows.")

    feature_cols = [c for c in df.columns if c not in ID_COLUMNS]

    if drop_channels:
        prefixes = tuple(f"{c}_" for c in drop_channels)
        dropped = [c for c in feature_cols if c.startswith(prefixes)]
        feature_cols = [c for c in feature_cols if c not in dropped]
        if not dropped:
            print(f"  ! WARNING: --drop-channel {drop_channels} matched no "
                  f"feature columns; nothing dropped.")
        else:
            print(f"Dropping {len(dropped)} feature columns for channel(s) "
                  f"{drop_channels}: kept {len(feature_cols)} features.")

    print(f"Loaded {len(df)} rows, {len(feature_cols)} feature columns.")
    print(f"  roles: {Counter(df['role'])}")

    # Delta normalization (uses baseline rows only; safe outside the LOO loop).
    cohort = delta_normalize(df, feature_cols)
    cohort = cohort.reset_index(drop=True)

    if args.norm == "baseline":
        # Divide the already-delta'd cohort by each dataset's untreated-well sd,
        # and disable the per-fold z-score. Baseline stats come only from
        # untreated wells (disjoint from the cohort), so this is leak-free.
        INFOLD_ZSCORE = False
        base = df[df["role"] == "baseline"]
        bstd = base.groupby("dataset")[feature_cols].std().replace(0, 1.0)
        for ds, idx in cohort.groupby("dataset").groups.items():
            if ds in bstd.index:
                cohort.loc[idx, feature_cols] = (
                    cohort.loc[idx, feature_cols] / bstd.loc[ds]).to_numpy()
        print("--norm baseline: divided by per-dataset untreated sd; "
              "per-fold z-score disabled.")

    if drop_classes:
        before = len(cohort)
        cohort = cohort[~cohort["label_drug"].str.upper().isin(drop_classes)]
        cohort = cohort.reset_index(drop=True)
        print(f"--drop-class {drop_classes}: kept {len(cohort)}/{before} "
              f"cohort rows; classes now {sorted(cohort['label_drug'].unique())}.")
    print(f"Cohort rows to model: {len(cohort)}  "
          f"datasets: {cohort['dataset'].nunique()}  "
          f"classes: {cohort['label_drug'].nunique()}")

    X = cohort[feature_cols].to_numpy(dtype=float)
    datasets = cohort["dataset"].to_numpy()
    le = LabelEncoder().fit(cohort["label_drug"])
    y = le.transform(cohort["label_drug"])
    class_names = list(le.classes_)
    n_classes = len(class_names)

    lines = ["=" * 72, "LOO MODELING REPORT", "=" * 72]
    lines.append(f"\nFeatures CSV : {features_csv.relative_to(REPO_ROOT).as_posix()}  (norm={args.norm})")
    if drop_channels:
        lines.append(f"Dropped chan : {', '.join(drop_channels)}  "
                     f"({len(feature_cols)} feature columns kept)")
    lines.append(f"Cohort rows  : {len(cohort)}   classes: {n_classes} ({', '.join(class_names)})")
    lines.append(f"Datasets     : {cohort['dataset'].nunique()}")
    lines.append(f"Class counts : {dict(Counter(cohort['label_drug']))}")
    lines.append(f"Seeds fixed  : SEED={SEED}   augment={AUG_FACTOR}x sigma={AUG_SIGMA}")

    results = {"n_cohort": len(cohort), "classes": class_names,
               "n_datasets": int(cohort["dataset"].nunique())}

    print("\nRunning drug-label LOO...")
    preds, imp = loo_predict(X, y, datasets, n_classes, collect_importance=True)

    for name in ("xgb", "rf", "ensemble"):
        model_report(name, y, preds[name], datasets, class_names, lines, results)

    importance_report(imp, feature_cols, lines, results)

    # ---- Confound check -------------------------------------------------
    lines.append("\n" + "=" * 72)
    lines.append("CONFOUND CHECK -- predicting `dataset` from the same features")
    lines.append("=" * 72)
    print("\nRunning confound check...")

    acc_all, reason_all = confound_accuracy(cohort, feature_cols)
    ctrl = cohort[cohort["label_drug"] == "CTRL"]
    acc_ctrl, reason_ctrl = confound_accuracy(ctrl, feature_cols)

    def fmt(acc, reason):
        return f"{acc:.4f}" if acc is not None else f"N/A ({reason})"

    lines.append(f"\nAll cohort rows (n={len(cohort)}):")
    lines.append(f"  dataset-probe accuracy : {fmt(acc_all, reason_all)}")
    lines.append(f"  prior value            : {PRIOR_CONFOUND_ALL:.4f}")
    lines.append(f"\nCTRL rows only (n={len(ctrl)}):")
    lines.append(f"  dataset-probe accuracy : {fmt(acc_ctrl, reason_ctrl)}")
    lines.append(f"  prior value            : {PRIOR_CONFOUND_CTRL:.4f}")
    lines.append("\nLower is better: it means the normalization removed dataset "
                 "identity, so the drug signal is not a dataset confound.")

    results["confound"] = {
        "all": {"accuracy": acc_all, "reason": reason_all, "n": len(cohort),
                "prior": PRIOR_CONFOUND_ALL},
        "ctrl": {"accuracy": acc_ctrl, "reason": reason_ctrl, "n": len(ctrl),
                 "prior": PRIOR_CONFOUND_CTRL},
    }

    # ---- Persist --------------------------------------------------------
    pred_df = cohort[["dataset", "drug", "cmax", "well", "label_drug"]].copy()
    pred_df["true"] = [class_names[i] for i in y]
    for name in ("xgb", "rf", "ensemble"):
        pred_df[f"pred_{name}"] = [class_names[i] for i in preds[name]]
    pred_df.to_csv(RESULTS_DIR / f"predictions{OUT_SUFFIX}.csv", index=False)

    report_txt = "\n".join(lines)
    (RESULTS_DIR / f"report{OUT_SUFFIX}.txt").write_text(report_txt, encoding="utf-8")
    (RESULTS_DIR / f"metrics{OUT_SUFFIX}.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")

    print("\n" + report_txt)
    print(f"\nResults written to {RESULTS_DIR.relative_to(REPO_ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
