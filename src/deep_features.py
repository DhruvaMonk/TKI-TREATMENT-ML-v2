"""
Frozen deep embeddings per well.

For every confirmed well we crop the largest-blob bounding box out of each
channel band (CH1 green plane and CH4 greyscale -- CH2 is deliberately skipped),
push each crop through a FROZEN backbone, mean-pool the token/spatial features,
and concatenate the per-channel vectors into one embedding per well.

Backbone: DINOv2 ViT-S/14 via torch.hub (facebookresearch/dinov2). If that
cannot be fetched (offline, hub error) we fall back to an ImageNet ResNet50 from
torchvision. If neither can be loaded the caller catches the exception and skips
step F.

The embeddings + aligned metadata are cached to
data/processed/deep_features_v2.npz so re-runs are instant. Nothing else is
modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_features as ef  # reuse mask/channel loaders  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_NPZ = REPO_ROOT / "data" / "processed" / "deep_features_v2.npz"
CACHE_META = REPO_ROOT / "data" / "processed" / "deep_features_v2_meta.csv"

# Channels to embed (skip CH2 per the step-F spec).
DEEP_CHANNELS = [("CH1", "ch1_path", 1), ("CH4", "ch4_path", None)]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_SIZE = 224


def _load_backbone():
    """Return (model, forward_fn, name). forward_fn(batch)->[B,D] pooled feats."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Try DINOv2 ViT-S/14 ------------------------------------------------
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                               verbose=False)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        def fwd(batch):
            out = model.forward_features(batch.to(device))
            tok = out["x_norm_patchtokens"]      # [B, N, 384]
            return tok.mean(dim=1)               # mean-pool patch tokens
        print("Backbone: DINOv2 ViT-S/14 (dim 384/channel).")
        return model, fwd, "dinov2_vits14", device
    except Exception as e:  # noqa: BLE001
        print(f"  ! DINOv2 unavailable ({type(e).__name__}: {e}); "
              f"falling back to ResNet50.")

    # --- Fallback: ImageNet ResNet50 ---------------------------------------
    import torchvision
    from torchvision.models import ResNet50_Weights

    net = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()                 # 2048-d global-avg-pooled feats
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    def fwd(batch):
        return net(batch.to(device))
    print("Backbone: ResNet50 IMAGENET1K_V2 (dim 2048/channel).")
    return net, fwd, "resnet50_imagenet", device


def _crop_channel(rec, comp, path_col, plane):
    """Largest-blob bbox crop for one channel, as a float32 [3,H,W] normalized
    tensor-ready array, or None if the band is unreadable."""
    band = ef.load_channel_band(rec, path_col, plane)
    if band is None:
        return None
    ys, xs = np.where(comp)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = band[y0:y1, x0:x1].astype(np.float32)
    # Per-crop min-max to [0,1] (robust to differing channel dynamic range).
    mn, mx = float(crop.min()), float(crop.max())
    crop = (crop - mn) / (mx - mn + 1e-8)
    rgb = np.stack([crop, crop, crop], axis=-1)   # grey -> 3 channels
    return rgb


def _to_batch_tensor(rgb_list):
    import torch
    import torch.nn.functional as F

    tens = []
    for rgb in rgb_list:
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,h,w]
        t = F.interpolate(t, size=(CROP_SIZE, CROP_SIZE), mode="bilinear",
                          align_corners=False)
        tens.append(t)
    x = torch.cat(tens, dim=0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def build_embeddings(batch_size=32):
    """Build (meta_df, X) where X rows align to meta_df; also writes the cache."""
    import torch

    well_index = ef.read_well_index()
    confirmed = ef.read_confirmed_from_log()

    model, fwd, name, device = _load_backbone()

    # Gather per-well crops first (so we know the embedding dim from a probe).
    metas, ch_crops = [], []      # ch_crops[i] = {"CH1": rgb or None, "CH4": ...}
    for log_row in confirmed:
        key = (log_row["dataset"], log_row["drug"], log_row["cmax"], log_row["well"])
        entry = well_index.get(key)
        if entry is None:
            continue
        rec, role = entry
        mask_path = REPO_ROOT / log_row["mask_path"] if log_row.get("mask_path") else None
        if mask_path is None:
            continue
        comp, _ = ef.load_largest_component(mask_path)
        if comp is None:
            continue
        crops = {}
        for ch, col, plane in DEEP_CHANNELS:
            crops[ch] = _crop_channel(rec, comp, col, plane)
        metas.append({
            "dataset": rec["dataset"], "drug": rec["drug"], "cmax": rec["cmax"],
            "well": rec["well"], "role": role, "label_drug": rec["drug"],
        })
        ch_crops.append(crops)

    # Probe embedding dim per channel with a dummy black crop.
    dummy = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
    with torch.no_grad():
        d = fwd(_to_batch_tensor([dummy])).shape[1]

    def embed_channel(ch):
        """Return [n_wells, d] embeddings for one channel; zero-vector if a crop
        is missing (rare)."""
        out = np.zeros((len(ch_crops), d), dtype=np.float32)
        idxs = [i for i, c in enumerate(ch_crops) if c[ch] is not None]
        with torch.no_grad():
            for s in range(0, len(idxs), batch_size):
                chunk = idxs[s:s + batch_size]
                batch = _to_batch_tensor([ch_crops[i][ch] for i in chunk])
                feats = fwd(batch).float().cpu().numpy()
                for k, i in enumerate(chunk):
                    out[i] = feats[k]
        return out

    per_ch = [embed_channel(ch) for ch, _, _ in DEEP_CHANNELS]
    X = np.concatenate(per_ch, axis=1)   # concat CH1 | CH4
    meta_df = pd.DataFrame(metas)

    CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_NPZ, X=X, backbone=name)
    meta_df.to_csv(CACHE_META, index=False)
    print(f"Deep embeddings: {X.shape[0]} wells x {X.shape[1]} dims "
          f"({name}); cached to {CACHE_NPZ.name}.")
    return meta_df, X, name


def load_or_build():
    if CACHE_NPZ.is_file() and CACHE_META.is_file():
        data = np.load(CACHE_NPZ, allow_pickle=True)
        meta_df = pd.read_csv(CACHE_META)
        name = str(data["backbone"])
        print(f"Deep embeddings loaded from cache: {data['X'].shape} ({name}).")
        return meta_df, data["X"], name
    return build_embeddings()


if __name__ == "__main__":
    load_or_build()
