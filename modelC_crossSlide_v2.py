"""
Model C — Cross-Slide v2  (v1 + 5 izboljsav za R3/R4 in kalibracijo)
=========================================================================
Osnova: modelC_crossSlide_v1.py (pravi train/test TMA split iz clanka,
gnezdena Faza A/B validacija, RAM-varcno bbox branje, disk cache za
spec_dense, T rekalibracija usklajena s final_epochs).

PET NOVOSTI glede na v1:

1. Uravnotezena izbira inner-val crop-a (namesto "najmanjsi z vsemi razredi").
   Ugotovili smo, da je prejsnja izbira (train_crop_09) imela za R3 samo
   119 pikslov (0.8%) -- signal za best-of-N/T/sigma je bil za R3 skoraj
   slep. Nova funkcija `select_inner_val_candidates` oceni vsak kandidat po
   NAJMANJSEM razrednem stevilu (min count med prisotnimi razredi) -- crop
   z visjo to vrednostjo ima bolj uravnotezeno zastopanost VSEH razredov.

2. Rotacija cez vec inner-val kandidatov (--rotate-inner-val).
   Namesto enega inner-val crop-a se Faza A ponovi za VSAK crop, ki ima
   vseh 6 razredov (tipicno 2: crop_08, crop_09). Rezultati (best_epochs,
   kalibracijski logits) se zdruzijo cez rotacije -- bistveno bolj robusten
   signal za redke razrede (R3 dobi glas iz OBEH crop-ov, ne le enega).

3. Per-class temperature scaling (--per-class-temperature).
   Namesto ene skalarne T (ki predpostavi, da so vsi razredi enako
   razkalibrirani) se optimizira vektor T_c, en na razred. R4 je bil
   opazno bolj razkalibriran (ll=4.7) kot ostali razredi -- ena T tega
   ne more popraviti.

4. Focal loss (--balance-strategy focal).
   Namesto (ali poleg) class weights eksplicitno poveca gradientni
   prispevek tezkih/napacno klasificiranih primerov: -alpha*(1-p_t)^gamma*log(p_t).

5. Oversampling (--balance-strategy oversample).
   Zdaj imamo ~125,853 train pikslov (4-5x vec kot na starem majhnem
   datasetu, kjer je oversampling skodil zaradi premajhne raznolikosti) --
   vreden ponovnega testa.

`--balance-strategy` je EN izbor (weights/focal/oversample), ne kombinacija --
weights in oversampling skupaj bi se dvakrat "popravila" v isto smer.

`--calib-ensemble` privzeto 12 (=n-ensemble) — boljsi rezultat po testu.

Vse ostalo (RAM-varcno bbox branje, disk cache, Faza A/B gnezdena struktura,
T usklajena s final_epochs) ostane enako v1.
"""

import argparse
import glob
import os
import random
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.optimize import minimize, minimize_scalar
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES  = 6
PATCH_SIZE   = 17
N_PCA        = 16
NEIGH_SCALES = [3, 5, 7]
RESULTS_FILE = "rezultati_report.txt"
NUM_WORKERS  = 0

BBOX_MARGIN = 12

SPEC_DIM_RAW = None
SPEC_DIM     = None

SIGMA_CHOICES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_SOFTEN = 0.5

METODOLOGIJA_OPOMBA = (
    "modelC_crossSlide_v2: v1 + (1) uravnotezena izbira inner-val (min-razredno "
    "stevilo namesto velikosti), (2) opcijska rotacija cez vec inner-val "
    "kandidatov z zdruzeno kalibracijo, (3) opcijska per-class temperature, "
    "(4) opcijski focal loss, (5) opcijski oversampling. T rekalibrirana na "
    "final_epochs-usklajenem modelu (kot v1). Ref. clanek: SVM=56.41%, "
    "CNN=79.45%+/-1.25 (isti train/test TMA split)."
)


# ---------------------------------------------------------------------------
# Nalaganje — SAMO bounding box okoli anotiranih pikslov (RAM-varcno, kot v1)
# ---------------------------------------------------------------------------
def peek_classes_and_wns(path):
    with h5py.File(path, 'r') as f:
        classes = np.array(f['classes'])
        wns     = np.array(f['wns'])
    return classes, wns


def get_annotated_bbox(classes, margin=BBOX_MARGIN):
    H, W = classes.shape
    coords = np.argwhere(classes != -1)
    r0 = max(0, int(coords[:, 0].min()) - margin)
    r1 = min(H, int(coords[:, 0].max()) + margin + 1)
    c0 = max(0, int(coords[:, 1].min()) - margin)
    c1 = min(W, int(coords[:, 1].max()) + margin + 1)
    return r0, r1, c0, c1


def read_crop_bbox(path, margin=BBOX_MARGIN):
    with h5py.File(path, 'r') as f:
        classes_full = np.array(f['classes'])
        r0, r1, c0, c1 = get_annotated_bbox(classes_full, margin)
        data_bbox        = np.array(f['data'][r0:r1, c0:c1, :], dtype=np.float32)
        tissue_mask_bbox = np.array(f['tissue_mask'][r0:r1, c0:c1])
    classes_bbox = classes_full[r0:r1, c0:c1]
    return data_bbox, tissue_mask_bbox, classes_bbox, r0, c0


def select_inner_val_candidates(crop_paths, verbose=True):
    """
    Prebere samo 'classes' iz vseh crop-ov. Oceni vsakega kandidata (ki ima
    vseh 6 razredov) po NAJMANJSEM razrednem stevilu med prisotnimi razredi
    -- visja vrednost = bolj uravnotezena zastopanost vseh razredov, ne le
    "ima vse razrede prisotne". Vrne SEZNAM kandidatov, razvrscenih od
    najbolj do najmanj uravnotezenega.
    """
    info = []
    if verbose:
        print("  Pregled train crop-ov (samo 'classes', poceni branje):")
    for p in crop_paths:
        classes, _ = peek_classes_and_wns(p)
        ann = (classes != -1)
        n = int(ann.sum())
        if n == 0:
            continue
        vals, counts = np.unique(classes[ann], return_counts=True)
        has_all = len(vals) == NUM_CLASSES
        min_count = int(counts.min()) if has_all else 0
        info.append({"path": p, "n": n, "has_all": has_all, "min_count": min_count,
                     "classes": vals})
        marker = "*" if has_all else " "
        if verbose:
            mc_str = str(min_count) if has_all else "-"
            print(f"    {marker} {os.path.basename(p)}: {n:,} anotiranih, "
                  f"min_razred={mc_str}, razredi={list(vals)}")

    with_all = [x for x in info if x["has_all"]]
    if with_all:
        with_all.sort(key=lambda x: x["min_count"], reverse=True)
        candidates = [x["path"] for x in with_all]
    else:
        info_sorted = sorted(info, key=lambda x: x["n"], reverse=True)
        candidates = [info_sorted[0]["path"]]
        if verbose:
            print("  OPOZORILO: noben crop nima vseh 6 razredov. Vzet najvecji.")

    if verbose:
        print(f"\n  Kandidati za inner-val (razvrsceni po uravnotezenosti, "
              f"najboljsi prvi):")
        for i, p in enumerate(candidates):
            print(f"    {i+1}. {os.path.basename(p)}")

    return candidates


# ---------------------------------------------------------------------------
# Cache za PCA-NEODVISEN del (spec_dense) — identicno v1
# ---------------------------------------------------------------------------
def _crop_cache_key(path, margin, scales):
    import hashlib
    key_str = f"{os.path.abspath(path)}|margin={margin}|scales={list(scales)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def get_spec_dense_cached(path, scales=NEIGH_SCALES, margin=BBOX_MARGIN,
                          cache_dir=None, label=""):
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _crop_cache_key(path, margin, scales)
        cache_path = os.path.join(cache_dir, f"specdense_{key}.npz")
        if os.path.exists(cache_path):
            t0 = time.time()
            d = np.load(cache_path)
            print(f"  [{label}] Cache HIT: {os.path.basename(path)}  "
                  f"({time.time()-t0:.1f}s namesto uniform_filter)")
            return (d["coords"], d["y"], d["spec_dense"], d["tissue_mask"],
                    int(d["H"]), int(d["W"]), int(d["r0"]), int(d["c0"]))

    t0 = time.time()
    data, tissue_mask, classes, r0, c0 = read_crop_bbox(path, margin)
    H, W, D = data.shape
    flat = data.reshape(-1, D)
    ann    = (classes != -1)
    coords = np.argwhere(ann)
    y      = classes[ann].astype(np.int64)

    spec_map = flat.reshape(H, W, D)
    r, c = coords[:, 0], coords[:, 1]
    parts = [flat[r * W + c].copy()]
    for scale in scales:
        mean_map = uniform_filter(spec_map, size=[scale, scale, 1], mode='reflect')
        parts.append(mean_map[r, c].copy())
        del mean_map
    spec_dense = np.concatenate(parts, axis=1).astype(np.float32)
    del data, flat, spec_map

    print(f"  [{label}] Neighbourhood ekstrakcija {os.path.basename(path)}: "
          f"bbox={H}x{W} ({100*H*W/(800*1200):.1f}% od 800x1200), "
          f"{len(y):,} anotiranih  ({time.time()-t0:.1f}s)")

    if cache_path:
        np.savez_compressed(cache_path, coords=coords, y=y, spec_dense=spec_dense,
                            tissue_mask=tissue_mask, H=H, W=W, r0=r0, c0=c0)
        print(f"  [{label}] Cache shranjen (~{spec_dense.nbytes/1e6:.0f} MB): "
              f"{os.path.basename(cache_path)}")

    return coords, y, spec_dense, tissue_mask, H, W, r0, c0


def fit_pca_pooled(crop_paths, n_pca=N_PCA, seed=42, label="", cache_dir=None,
                    scales=NEIGH_SCALES):
    print(f"  PCA fit ({label}): zbiranje anotiranih spektrov iz {len(crop_paths)} crop-ov...")
    pooled = []
    for p in crop_paths:
        coords, y, spec_dense, tissue_mask, H, W, r0, c0 = get_spec_dense_cached(
            p, scales, BBOX_MARGIN, cache_dir, label=label)
        X_ann = spec_dense[:, :SPEC_DIM_RAW].copy()
        pooled.append(X_ann)
        print(f"    {os.path.basename(p)}: {len(X_ann):,} spektrov")
    X_pool = np.concatenate(pooled, axis=0)
    del pooled
    n_bad = int((~np.isfinite(X_pool)).any(axis=1).sum())
    if n_bad:
        print(f"  OPOZORILO: {n_bad} anotiranih spektrov z NaN/Inf -- sanitiziram na 0.")
        X_pool = np.nan_to_num(X_pool, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Skupaj za PCA fit: {len(X_pool):,} spektrov, dim={X_pool.shape[1]}")
    t0 = time.time()
    pca = PCA(n_components=n_pca, random_state=seed)
    pca.fit(X_pool)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%  "
          f"({time.time()-t0:.1f}s)")
    del X_pool
    return pca


def process_crop(path, pca, scales=NEIGH_SCALES, label="", cache_dir=None):
    coords, y, spec_dense, tissue_mask, H, W, r0, c0 = get_spec_dense_cached(
        path, scales, BBOX_MARGIN, cache_dir, label=label)

    t0 = time.time()
    print(f"  [{label}] PCA transform ({H*W:,} pikslov v bbox)...")
    data, _, _, _, _ = read_crop_bbox(path, BBOX_MARGIN)
    flat = data.reshape(-1, data.shape[-1])
    n_bad = int((~np.isfinite(flat)).any(axis=1).sum())
    if n_bad:
        print(f"  [{label}] OPOZORILO: {n_bad} pikslov z NaN/Inf v surovih podatkih "
              f"(verjetno pokvarjen senzorski piksel) -- sanitiziram na 0.")
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat_pca = pca.transform(flat).astype(np.float32)
    data_pca = flat_pca.reshape(H, W, N_PCA)
    del data, flat, flat_pca

    print(f"  [{label}] -> data_pca={data_pca.shape}, spec_dense={spec_dense.shape}, "
          f"anotiranih={len(y):,}  ({time.time()-t0:.1f}s)")
    return {
        "data_pca": data_pca, "spec_dense": spec_dense, "coords": coords,
        "y": y, "H": H, "W": W, "tissue_mask": tissue_mask,
        "bbox_offset": (r0, c0),
    }


def pad_pca(data_pca, patch_size=PATCH_SIZE):
    pad = patch_size // 2
    return np.pad(data_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect').astype(np.float32)


def stack_crops(crop_results, crop_ids):
    samples_parts, spec_parts, y_parts = [], [], []
    for cid, res in zip(crop_ids, crop_results):
        n = len(res["y"])
        crop_col = np.full((n, 1), cid, dtype=np.int64)
        samples_parts.append(np.concatenate([crop_col, res["coords"]], axis=1))
        spec_parts.append(res["spec_dense"])
        y_parts.append(res["y"])
    samples = np.concatenate(samples_parts, axis=0)
    spec    = np.concatenate(spec_parts, axis=0)
    y       = np.concatenate(y_parts, axis=0)
    return samples, spec, y


# ---------------------------------------------------------------------------
# Balansiranje razredov — oversampling (namesto/poleg class weights)
# ---------------------------------------------------------------------------
def oversample_pool(samples, spec, y, seed=42, verbose=True, label="train", target=None):
    rng = np.random.RandomState(seed)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    target = int(counts.max()) if target is None else int(target)
    if verbose:
        print(f"  Oversampling ({label}): {list(counts)} -> {target}/razred")
    idx_parts = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(y == c)[0]
        if len(cls_idx) == 0:
            continue
        if len(cls_idx) < target:
            extra = rng.choice(cls_idx, size=target - len(cls_idx), replace=True)
            idx_parts.append(np.concatenate([cls_idx, extra]))
        else:
            idx_parts.append(cls_idx)
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    if verbose:
        print(f"  Skupaj po oversamplingu ({label}): {len(idx):,} (prej {len(y):,})")
    return samples[idx], spec[idx], y[idx]


# ---------------------------------------------------------------------------
# Dataset — identicno v1
# ---------------------------------------------------------------------------
_AUGMENTS = [
    lambda p: p,
    lambda p: torch.flip(p, [1]),
    lambda p: torch.flip(p, [2]),
    lambda p: torch.rot90(p, 1, [1, 2]),
    lambda p: torch.rot90(p, 2, [1, 2]),
    lambda p: torch.rot90(p, 3, [1, 2]),
    lambda p: torch.flip(torch.rot90(p, 1, [1, 2]), [1]),
    lambda p: torch.flip(torch.rot90(p, 1, [1, 2]), [2]),
]


class MultiCropDataset(Dataset):
    def __init__(self, padded_pca_by_crop, spec_dense, samples, labels,
                 patch_size=PATCH_SIZE, augment=False, tta_idx=-1, spec_noise=0.0):
        self.pad = patch_size // 2
        self.padded_pca_by_crop = padded_pca_by_crop
        self.spec_dense = spec_dense
        self.samples = samples
        self.labels = labels
        self.augment = augment
        self.tta_idx = tta_idx
        self.spec_noise = spec_noise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        crop_id, r, c = self.samples[idx]
        pad = self.pad
        padded = self.padded_pca_by_crop[int(crop_id)]
        rp, cp = r + pad, c + pad
        patch = padded[rp-pad:rp+pad+1, cp-pad:cp+pad+1]
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1).copy())
        if self.tta_idx >= 0:
            patch_t = _AUGMENTS[self.tta_idx](patch_t)
        elif self.augment:
            patch_t = _AUGMENTS[random.randint(0, 7)](patch_t)
        spectrum_t = torch.from_numpy(self.spec_dense[idx].copy())
        if self.augment and self.spec_noise > 0:
            noise = torch.zeros_like(spectrum_t)
            noise[:SPEC_DIM_RAW] = torch.randn(SPEC_DIM_RAW) * self.spec_noise
            spectrum_t = spectrum_t + noise
        return patch_t, spectrum_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Arhitektura: Dual-Stream CNN — identicno v1
# ---------------------------------------------------------------------------
class DualStreamCNN(nn.Module):
    def __init__(self, n_channels=N_PCA, spec_dim=None,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        if spec_dim is None:
            spec_dim = SPEC_DIM
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spectral_mlp = nn.Sequential(
            nn.Linear(spec_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout + 0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def get_spatial_features(self, patch):
        return self.spatial_pool(self.spatial_cnn(patch)).squeeze(-1).squeeze(-1)

    def get_spectral_features(self, spectrum):
        return self.spectral_mlp(spectrum)

    def get_logits(self, patch, spectrum):
        f = torch.cat([self.get_spatial_features(patch),
                       self.get_spectral_features(spectrum)], dim=1)
        return self.fusion(f)

    def forward(self, patch, spectrum):
        return self.log_softmax(self.get_logits(patch, spectrum))


# ---------------------------------------------------------------------------
# Focal loss (Lin et al. 2017): -alpha_c * (1-p_t)^gamma * log(p_t)
# log_probs mora biti LogSoftmax izhod (kot NLLLoss pricakuje).
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight  # ze na pravi napravi, ali None
        self.gamma = gamma

    def forward(self, log_probs, target):
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = -focal_term * log_pt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def build_criterion(strategy, weights, device, focal_gamma=2.0):
    """weights = ze izracunane (softened) class weights, NE na napravi."""
    if strategy == "focal":
        return FocalLoss(weight=weights.to(device), gamma=focal_gamma)
    elif strategy == "oversample":
        return nn.NLLLoss()  # podatki so ze uravnotezeni z oversamplingom
    else:  # "weights"
        return nn.NLLLoss(weight=weights.to(device))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda"); print(f"  Naprava: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


def compute_class_weights(y, soften=WEIGHT_SOFTEN):
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    raw     = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    weights = raw ** soften
    print(f"  Class weights (soften={soften}): {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def get_logits_array(model, dataset, device, batch_size=512):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    all_l  = []
    for patches, spectra, _ in loader:
        all_l.append(
            model.get_logits(patches.to(device), spectra.to(device)).cpu().numpy()
        )
    return np.concatenate(all_l)


@torch.no_grad()
def get_logits_tta(model, padded_pca_by_crop, spec_dense, samples, labels, device, batch_size=512):
    logits_sum = None
    for aug_idx in range(8):
        ds = MultiCropDataset(padded_pca_by_crop, spec_dense, samples, labels,
                              augment=False, tta_idx=aug_idx)
        logits = get_logits_array(model, ds, device, batch_size)
        logits_sum = logits.copy() if logits_sum is None else logits_sum + logits
    return logits_sum / 8


def find_temperature(logits, y):
    def neg_ll(T):
        s = logits / T
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-9, 1.0)
        return -np.mean(np.log(p[np.arange(len(y)), y]))
    res = minimize_scalar(neg_ll, bounds=(0.1, 10.0), method='bounded')
    T   = res.x
    print(f"  Temperature (skalarna): T={T:.4f}  {neg_ll(1.0):.5f} -> {neg_ll(T):.5f}")
    return T


def apply_temperature(logits, T):
    s = logits / T
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def find_temperature_per_class(logits, y, n_classes=NUM_CLASSES):
    """Vektor T_c, en na razred ('vector scaling') — bolj prilagodljivo kot
    ena skalarna T, ce so razredi razlicno razkalibrirani (npr. R4 huje kot
    ostali)."""
    def neg_ll(T_vec):
        T_vec = np.clip(T_vec, 0.05, 20.0)
        s = logits / T_vec[None, :]
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-9, 1.0)
        return -np.mean(np.log(p[np.arange(len(y)), y]))
    x0 = np.ones(n_classes)
    bounds = [(0.1, 10.0)] * n_classes
    res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
    T_vec = res.x
    print(f"  Temperature (per-class): " +
          ", ".join(f"R{c}={t:.2f}" for c, t in enumerate(T_vec)))
    print(f"  ll: {neg_ll(np.ones(n_classes)):.5f} -> {neg_ll(T_vec):.5f}")
    return T_vec


def apply_temperature_per_class(logits, T_vec):
    s = logits / T_vec[None, :]
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def format_T(T_opt):
    if np.isscalar(T_opt):
        return f"{float(T_opt):.4f}"
    return "[" + ",".join(f"{t:.3f}" for t in T_opt) + "]"


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h {m}min {s}s"
    if m: return f"{m}min {s}s"
    return f"{s}s"


def gaussian_smooth_probs(probs, tissue_mask, sigma):
    if sigma <= 0: return probs
    smoothed = np.zeros_like(probs)
    mf = tissue_mask.astype(np.float32)
    for c in range(probs.shape[-1]):
        num = gaussian_filter(probs[:, :, c] * mf, sigma=sigma)
        den = gaussian_filter(mf, sigma=sigma)
        smoothed[:, :, c] = num / np.where(den < 1e-8, 1e-8, den)
    smoothed = np.clip(smoothed, 1e-7, 1.0)
    smoothed /= smoothed.sum(axis=-1, keepdims=True)
    return smoothed.astype(np.float32)


def build_full_canvas_prob_map(H, W, coords, probs, fallback_prior):
    prob_map = np.tile(fallback_prior, (H * W, 1)).reshape(H, W, NUM_CLASSES)
    prob_map[coords[:, 0], coords[:, 1]] = probs
    return prob_map


def find_best_sigma(H, W, coords, probs, y_true, tissue_mask, fallback_prior,
                    sigma_choices=SIGMA_CHOICES):
    prob_map = build_full_canvas_prob_map(H, W, coords, probs, fallback_prior)
    print(f"  {'sigma':>6}  {'log-loss':>10}")
    best_sigma, best_ll = 0.0, None
    for sigma in sigma_choices:
        if sigma > 0:
            smoothed = gaussian_smooth_probs(prob_map, tissue_mask, sigma)
            m = prob_map.copy()
            m[tissue_mask] = smoothed[tissue_mask]
        else:
            m = prob_map
        probs_at_coords = m[coords[:, 0], coords[:, 1]]
        probs_at_coords = np.clip(probs_at_coords, 1e-7, 1.0)
        probs_at_coords /= probs_at_coords.sum(axis=1, keepdims=True)
        ll = log_loss(y_true, probs_at_coords, labels=np.arange(NUM_CLASSES))
        marker = ""
        if best_ll is None or ll < best_ll:
            best_ll = ll; best_sigma = sigma; marker = " *"
        print(f"  {sigma:>6.1f}  {ll:>10.5f}{marker}")
    print(f"  Najboljsa sigma: {best_sigma:.1f} (ll={best_ll:.5f})")
    return best_sigma


def print_per_class_table(y_true, y_pred, probs, title="Per-class rezultati"):
    print(f"\n  {title}")
    print(f"  {'Razred':>8}  {'N':>6}  {'OA':>8}  {'Log-loss':>10}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*10}")
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"  {c:>8}  {'-':>6}  {'-':>8}  {'-':>10}")
            continue
        oa_c = accuracy_score(y_true[mask], y_pred[mask])
        ll_c = log_loss(y_true[mask], probs[mask], labels=np.arange(NUM_CLASSES))
        print(f"  {c:>8}  {mask.sum():>6}  {oa_c*100:>7.2f}%  {ll_c:>10.5f}")
    oa_tot = accuracy_score(y_true, y_pred)
    ll_tot = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))
    print(f"  {'SKUPAJ':>8}  {len(y_true):>6}  {oa_tot*100:>7.2f}%  {ll_tot:>10.5f}")


def write_results_report(model_name, innerval_oa, innerval_ll, test_oa, test_ll,
                          output_path, t_opt_str, sigma, max_epochs, best_epoch,
                          final_epochs, n_ensemble, extra_note="", total_duration_str=None):
    """
    NIKOLI ne sme vreci izjeme navzven -- do te tocke je ze vso drago racunanje
    (ure treninga) uspesno koncano, zapis v log je le pripomocek. Ce pisanje
    v datoteko spodleti (npr. prehoden TimeoutError zaradi I/O konkurence),
    poskusimo se dvakrat, sicer izpisemo vsebino na stdout, da se NIC ne
    izgubi, in se vrnemo brez crasha.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dur_part = f"  cas={total_duration_str}" if total_duration_str else ""
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"TEST_OA={test_oa*100:6.2f}%  TEST_ll={test_ll:.5f}  "
        f"T={t_opt_str}  sigma={sigma:.1f}  n_ensemble={n_ensemble}  "
        f"max_ep={max_epochs}(best={best_epoch})  final_ep={final_epochs}{dur_part}"
        f"  -> {output_path}\n"
    ]
    if extra_note:
        lines.append(f"{'':>19}  Opomba: {extra_note}\n")

    for attempt in range(3):
        try:
            if not os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, "w") as f:
                    f.write("# Rezultati modelov — FTIR klasifikacija tkiva\n")
                    f.write(f"# {'-'*90}\n")
                print(f"  -> {RESULTS_FILE} (ustvarjena nova)")
            else:
                print(f"  -> {RESULTS_FILE} (dodana vrstica)")
            with open(RESULTS_FILE, "a") as f:
                f.writelines(lines)
            return
        except OSError as e:
            print(f"  OPOZORILO: pisanje v {RESULTS_FILE} ni uspelo "
                  f"(poskus {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2)

    print(f"  OPOZORILO: {RESULTS_FILE} po 3 poskusih se vedno ni dosegljiv. "
          f"Vsebina (dodaj rocno, ce zelis):")
    print("  " + "".join(lines).replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Trening — Faza A (best-of-N) in Faza B (blind). Sprejmeta ze zgrajen
# `criterion` (weights / focal / oversample+neutezen), ne le utezi.
# ---------------------------------------------------------------------------
def train_single(padded_pca_train, spec_train, samples_train, y_train,
                 val_ds, y_val, device, criterion,
                 max_epochs, batch_size, lr, seed, spec_noise):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = MultiCropDataset(padded_pca_train, spec_train, samples_train, y_train,
                                augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    model     = DualStreamCNN(spec_dim=SPEC_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_loss, best_state, best_epoch = float('inf'), None, 1
    print(f"  {'Ep':>4}  {'Train ll':>10}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for patches, spectra, labels in loader:
            patches = patches.to(device); spectra = spectra.to(device)
            labels  = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(patches, spectra), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        train_loss = total / len(train_ds)
        scheduler.step()

        val_logits = get_logits_array(model, val_ds, device)
        s = val_logits - val_logits.max(axis=1, keepdims=True)
        e = np.exp(s); val_probs = e / e.sum(axis=1, keepdims=True)
        val_oa = accuracy_score(y_val, np.argmax(val_probs, axis=1))
        val_ll = log_loss(y_val, val_probs, labels=np.arange(NUM_CLASSES))
        marker = " *" if val_ll < best_val_loss else "  "
        print(f"  {epoch:>4}  {train_loss:>10.5f}  {val_oa*100:>8.2f}%  "
              f"{val_ll:>9.5f}{marker}  {optimizer.param_groups[0]['lr']:>9.2e}")

        if val_ll < best_val_loss:
            best_val_loss = val_ll
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch

    model.load_state_dict(best_state)
    print(f"  Treniran v {time.time()-t0:.1f}s | best ep={best_epoch}, val_ll={best_val_loss:.5f}")
    return model, best_epoch


def train_blind(padded_pca_train, spec_train, samples_train, y_train, device, criterion,
                final_epochs, batch_size, lr, seed, spec_noise):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = MultiCropDataset(padded_pca_train, spec_train, samples_train, y_train,
                                augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    model     = DualStreamCNN(spec_dim=SPEC_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs, eta_min=1e-6)

    print(f"  {'Ep':>4}  {'Train ll':>10}  {'LR':>9}")
    t0 = time.time()
    for epoch in range(1, final_epochs + 1):
        model.train()
        total = 0.0
        for patches, spectra, labels in loader:
            patches = patches.to(device); spectra = spectra.to(device)
            labels  = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(patches, spectra), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        train_loss = total / len(train_ds)
        scheduler.step()
        if epoch % 3 == 0 or epoch in (1, final_epochs):
            print(f"  {epoch:>4}  {train_loss:>10.5f}  "
                  f"{optimizer.param_groups[0]['lr']:>9.2e}")
    print(f"  Treniran v {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global SPEC_DIM_RAW, SPEC_DIM, BBOX_MARGIN, NUM_WORKERS
    run_start = time.time()

    parser = argparse.ArgumentParser(
        description="Model C Cross-Slide v2: v1 + uravnotezen inner-val/rotacija + "
                    "per-class T + focal loss + oversampling"
    )
    parser.add_argument("--train-dir",  default="FTIR-data/train_preprocessed")
    parser.add_argument("--test-file",  default="FTIR-data/test_preprocessed/test_expanded_crop_preprocessed.hdf5")
    parser.add_argument("--output",     default="modelC_crossSlide_v2_test.npy")
    parser.add_argument("--max-epochs", type=int,   default=15,
                        help="Zgornja meja epoh za Fazo A (best-of-N na inner-val).")
    parser.add_argument("--final-epochs", type=int, default=None,
                        help="Rocna preglasitev epoh za Fazo B (privzeto: mediana "
                             "best_epochs, zdruzena cez vse rotacije).")
    parser.add_argument("--calib-ensemble", type=int, default=12,
                        help="Stevilo modelov za rekalibracijski ensemble (T na "
                             "final_epochs-usklajenem modelu). Privzeto=12 (isto "
                             "kot n-ensemble) — boljsi rezultat po testu.")
    parser.add_argument("--rotate-inner-val", action="store_true",
                        help="Ponovi Fazo A za VSE crop-e z vsemi 6 razredi kot "
                             "inner-val (ne le najbolj uravnotezenega), zdruzi "
                             "rezultate (pooled T-kalibracija). Draje, robustnejse.")
    parser.add_argument("--per-class-temperature", action="store_true",
                        help="Vektor T (en na razred) namesto ene skalarne T.")
    parser.add_argument("--balance-strategy", choices=["weights", "focal", "oversample"],
                        default="weights",
                        help="weights=softened class weights (privzeto v10-v13 stil), "
                             "focal=focal loss (+weights kot alpha), "
                             "oversample=oversampling do najvecjega razreda + neutezen loss.")
    parser.add_argument("--weight-soften", type=float, default=WEIGHT_SOFTEN)
    parser.add_argument("--oversample-target", type=int, default=None,
                        help="Fiksno stevilo pikslov/razred pri --balance-strategy "
                             "oversample (privzeto: velikost najvecjega razreda). "
                             "Clanek uporablja 100,000/razred za CNN.")
    parser.add_argument("--focal-gamma",   type=float, default=2.0)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--n-ensemble", type=int,   default=4,
                        help="Privzeto 4 za hiter sanity-check; koncni tek naj bo 12.")
    parser.add_argument("--spec-noise", type=float, default=0.005)
    parser.add_argument("--neigh-scales", type=str, default="3,5,7",
                        help="Vejica-loceni seznam velikosti oken za neighbourhood-mean "
                             "znacilke v spektralni veji (privzeto '3,5,7', kot v1/v2). "
                             "Prazen niz '' = BREZ neighbourhood znacilk, samo raw "
                             "spekter v spektralni veji (ablacija za locitev ucinka "
                             "spektralne veje od ucinka prostorskega glajenja).")
    parser.add_argument("--bbox-margin", type=int, default=BBOX_MARGIN)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers. Privzeto 0 (varno na macOS/MPS). "
                             "Na Linux/CUDA strezniku z vec jedri nastavi npr. 4-8 za "
                             "vzporedno nalaganje podatkov (odpravi CPU-vezano ozko grlo).")
    parser.add_argument("--cache-dir",  default="FTIR-data/_cache")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    args.neigh_scales = ([int(s) for s in args.neigh_scales.split(",") if s.strip()]
                         if args.neigh_scales.strip() else [])
    NUM_WORKERS = args.num_workers
    BBOX_MARGIN = args.bbox_margin
    cache_dir = args.cache_dir if args.cache_dir else None

    # ------------------------------------------------------------------
    print("\n=== 1. Odkrivanje train crop-ov in izbira inner-val kandidatov ===")
    train_crop_paths = sorted(glob.glob(os.path.join(args.train_dir, "train_crop_*.hdf5")))
    print(f"  Najdenih train crop-ov: {len(train_crop_paths)}")
    for p in train_crop_paths:
        print(f"    {p}")

    _, wns_ref = peek_classes_and_wns(train_crop_paths[0])
    SPEC_DIM_RAW = len(wns_ref)
    SPEC_DIM     = SPEC_DIM_RAW * (1 + len(args.neigh_scales))
    print(f"\n  Spektralnih kanalov (SPEC_DIM_RAW): {SPEC_DIM_RAW}  ->  SPEC_DIM={SPEC_DIM}")
    print(f"  Neighbourhood scales: {args.neigh_scales if args.neigh_scales else '(brez — samo raw spekter)'}")
    print(f"  Bbox margina (RAM varcevanje): {BBOX_MARGIN} pikslov")

    candidates = select_inner_val_candidates(train_crop_paths)
    rotation_candidates = candidates if args.rotate_inner_val else candidates[:1]
    print(f"\n  Nacin inner-val: " +
          (f"ROTACIJA cez {len(rotation_candidates)} kandidatov"
           if args.rotate_inner_val else
           f"En kandidat (najbolj uravnotezen): {os.path.basename(rotation_candidates[0])}"))
    print(f"  Balance strategy: {args.balance_strategy}"
          + (f" (gamma={args.focal_gamma})" if args.balance_strategy == "focal" else ""))
    print(f"  Temperature: {'per-class' if args.per_class_temperature else 'skalarna'}")

    device = get_device()
    n_param = sum(p.numel() for p in DualStreamCNN(spec_dim=SPEC_DIM).parameters()
                 if p.requires_grad)
    print(f"\n  Konfiguracija: Patch {PATCH_SIZE}x{PATCH_SIZE} | PCA({N_PCA}) | TTA 8 | "
          f"Ensemble {args.n_ensemble} | DualStreamCNN ({n_param:,} param)")
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    seeds = list(range(args.seed, args.seed + args.n_ensemble))

    # ==================================================================
    # FAZA A — priprava podatkov za vsako rotacijo (PCA fit + procesiranje)
    # ==================================================================
    rotations = []
    for r_idx, inner_val_path in enumerate(rotation_candidates):
        inner_train_paths = [p for p in train_crop_paths if p != inner_val_path]
        rlabel = f"rot{r_idx+1}/{len(rotation_candidates)}:{os.path.basename(inner_val_path)}"

        print(f"\n=== 2.{r_idx+1} Faza A [{rlabel}] — PCA fit na {len(inner_train_paths)} crop-ih ===")
        pca_A = fit_pca_pooled(inner_train_paths, N_PCA, args.seed, label=rlabel,
                              cache_dir=cache_dir, scales=args.neigh_scales)

        print(f"\n=== 3.{r_idx+1} Faza A [{rlabel}] — procesiranje crop-ov ===")
        it_results, it_ids = [], []
        for i, p in enumerate(inner_train_paths):
            res = process_crop(p, pca_A, scales=args.neigh_scales,
                              label=f"{rlabel}/it{i}", cache_dir=cache_dir)
            it_results.append(res); it_ids.append(i)
        iv_res = process_crop(inner_val_path, pca_A, scales=args.neigh_scales,
                              label=f"{rlabel}/iv", cache_dir=cache_dir)

        samples_it, spec_it, y_it_true = stack_crops(it_results, it_ids)
        padded_it  = {cid: pad_pca(res["data_pca"]) for cid, res in zip(it_ids, it_results)}
        padded_iv  = {0: pad_pca(iv_res["data_pca"])}
        samples_iv = np.concatenate([
            np.zeros((len(iv_res["y"]), 1), dtype=np.int64), iv_res["coords"]
        ], axis=1)
        iv_ds = MultiCropDataset(padded_iv, iv_res["spec_dense"], samples_iv,
                                 iv_res["y"], augment=False)
        del it_results

        print(f"\n  [{rlabel}] inner-train={len(y_it_true):,}px, inner-val={len(iv_res['y']):,}px")
        for c in range(NUM_CLASSES):
            nt = (y_it_true == c).sum(); nv = (iv_res["y"] == c).sum()
            print(f"    R{c}: it={nt:6d} ({100*nt/len(y_it_true):.1f}%)  "
                  f"iv={nv:6d} ({100*nv/max(len(iv_res['y']),1):.1f}%)")

        weights_A = compute_class_weights(y_it_true, soften=args.weight_soften)
        criterion_A = build_criterion(args.balance_strategy, weights_A, device, args.focal_gamma)

        if args.balance_strategy == "oversample":
            samples_it, spec_it, y_it = oversample_pool(
                samples_it, spec_it, y_it_true, seed=args.seed, label=rlabel,
                target=args.oversample_target)
        else:
            y_it = y_it_true

        rotations.append(dict(
            label=rlabel, inner_val_path=inner_val_path,
            padded_it=padded_it, spec_it=spec_it, samples_it=samples_it, y_it=y_it,
            y_it_true=y_it_true,
            padded_iv=padded_iv, samples_iv=samples_iv, iv_res=iv_res, iv_ds=iv_ds,
            criterion=criterion_A,
        ))

    # ==================================================================
    # PASS 1 — iskanje hiperparametrov (best-of-N) na vsaki rotaciji
    # ==================================================================
    all_best_epochs = []
    for rd in rotations:
        print(f"\n=== 4. Faza A [{rd['label']}] — Ensemble ({args.n_ensemble} x do {args.max_epochs} ep) ===")
        best_epochs_r = []
        for i, seed in enumerate(seeds):
            print(f"\n  -- {rd['label']} model {i+1}/{args.n_ensemble} (seed={seed}) --")
            _, best_epoch = train_single(
                rd['padded_it'], rd['spec_it'], rd['samples_it'], rd['y_it'],
                rd['iv_ds'], rd['iv_res']['y'], device, rd['criterion'],
                max_epochs=args.max_epochs, batch_size=args.batch_size,
                lr=args.lr, seed=seed, spec_noise=args.spec_noise,
            )
            best_epochs_r.append(best_epoch)
        rd['best_epochs'] = best_epochs_r
        all_best_epochs.extend(best_epochs_r)
        print(f"  [{rd['label']}] best_epochs={best_epochs_r}")

    median_epochs = int(round(float(np.median(all_best_epochs))))
    print(f"\n  Vsi best_epochs (zdruzeni cez {len(rotations)} rotacij): {all_best_epochs}")
    print(f"  Mediana: {median_epochs}  (povprecje bi bilo: {round(np.mean(all_best_epochs))})")

    if args.final_epochs is not None:
        final_epochs = args.final_epochs
        print(f"  final_epochs ROCNO PREGLASEN: {final_epochs} "
              f"(namesto mediane {median_epochs})")
    else:
        final_epochs = median_epochs

    # ==================================================================
    # PASS 2 — rekalibracijski ensemble (usklajen final_epochs) na vsaki
    # rotaciji, zdruzen za skupno T-kalibracijo
    # ==================================================================
    all_calib_logits, all_calib_y = [], []
    for rd in rotations:
        n_calib = max(1, min(args.calib_ensemble, args.n_ensemble))
        print(f"\n=== 4b. Rekalibracijski ensemble [{rd['label']}] "
              f"({n_calib} x {final_epochs} ep, blind, usklajen s final_epochs) ===")
        calib_seeds = seeds[:n_calib]
        calib_logits_list = []
        for i, seed in enumerate(calib_seeds):
            print(f"\n  -- {rd['label']} rekalibracija {i+1}/{n_calib} (seed={seed}) --")
            model_c = train_blind(
                rd['padded_it'], rd['spec_it'], rd['samples_it'], rd['y_it'],
                device, rd['criterion'],
                final_epochs=final_epochs, batch_size=args.batch_size,
                lr=args.lr, seed=seed, spec_noise=args.spec_noise,
            )
            calib_logits_list.append(
                get_logits_tta(model_c, rd['padded_iv'], rd['iv_res']['spec_dense'],
                               rd['samples_iv'], rd['iv_res']['y'], device)
            )
        calib_logits_avg_r = np.mean(calib_logits_list, axis=0)
        rd['calib_logits'] = calib_logits_avg_r
        all_calib_logits.append(calib_logits_avg_r)
        all_calib_y.append(rd['iv_res']['y'])

    joint_logits = np.concatenate(all_calib_logits, axis=0)
    joint_y      = np.concatenate(all_calib_y, axis=0)

    print(f"\n=== 5. Temperature scaling (zdruzen inner-val, {len(joint_y):,} pikslov "
          f"cez {len(rotations)} rotacij) ===")
    if args.per_class_temperature:
        T_opt = find_temperature_per_class(joint_logits, joint_y)
        apply_T = lambda logits: apply_temperature_per_class(logits, T_opt)
    else:
        T_opt = find_temperature(joint_logits, joint_y)
        apply_T = lambda logits: apply_temperature(logits, T_opt)
    T_opt_str = format_T(T_opt)

    joint_probs = apply_T(joint_logits)
    joint_pred  = np.argmax(joint_probs, axis=1)
    innerval_oa = accuracy_score(joint_y, joint_pred)
    innerval_ll = log_loss(joint_y, joint_probs, labels=np.arange(NUM_CLASSES))
    print(f"  INNERVAL (zdruzen) OA: {innerval_oa*100:.2f}%  |  ll: {innerval_ll:.5f}")
    print_per_class_table(joint_y, joint_pred, joint_probs,
                          "Per-class OA in log-loss (zdruzen inner-val, vse rotacije):")

    print("\n=== 6. Sigma sweep (per rotacija, kombinirano z mediano) ===")
    sigma_candidates = []
    for rd in rotations:
        probs_r = apply_T(rd['calib_logits'])
        prior_r = np.bincount(rd['y_it_true'], minlength=NUM_CLASSES).astype(np.float32)
        prior_r /= prior_r.sum()
        sigma_r = find_best_sigma(
            rd['iv_res']['H'], rd['iv_res']['W'], rd['iv_res']['coords'],
            probs_r, rd['iv_res']['y'], rd['iv_res']['tissue_mask'], prior_r,
            sigma_choices=SIGMA_CHOICES)
        sigma_candidates.append(sigma_r)
    sigma_opt = float(np.median(sigma_candidates))
    print(f"  Sigma po rotacijah: {sigma_candidates}  -> mediana: {sigma_opt:.1f}")

    # sprosti Fazo A pomnilnik pred Fazo B
    for rd in rotations:
        del rd['padded_it'], rd['padded_iv'], rd['samples_it'], rd['spec_it']
    del rotations, all_calib_logits, all_calib_y

    # ==================================================================
    # FAZA B — vseh 8 crop-ov, blind trening, en dotik s pravim TEST filom
    # ==================================================================
    print(f"\n=== 7. Faza B — PCA fit na vseh {len(train_crop_paths)} train crop-ih ===")
    pca_B = fit_pca_pooled(train_crop_paths, N_PCA, args.seed, label="Faza B",
                          cache_dir=cache_dir, scales=args.neigh_scales)

    print(f"\n=== 8. Faza B — procesiranje vseh train crop-ov (samo bbox) ===")
    outer_train_results, outer_train_ids = [], []
    for i, p in enumerate(train_crop_paths):
        res = process_crop(p, pca_B, scales=args.neigh_scales,
                          label=f"outer-train {i}", cache_dir=cache_dir)
        outer_train_results.append(res)
        outer_train_ids.append(i)

    samples_ot, spec_ot, y_ot_true = stack_crops(outer_train_results, outer_train_ids)
    padded_ot = {cid: pad_pca(res["data_pca"]) for cid, res in zip(outer_train_ids, outer_train_results)}
    print(f"\n  Skupaj outer-train (Faza B): {len(y_ot_true):,} pikslov iz {len(train_crop_paths)} crop-ov")

    weights_B = compute_class_weights(y_ot_true, soften=args.weight_soften)
    criterion_B = build_criterion(args.balance_strategy, weights_B, device, args.focal_gamma)
    if args.balance_strategy == "oversample":
        samples_ot, spec_ot, y_ot = oversample_pool(
            samples_ot, spec_ot, y_ot_true, seed=args.seed, label="Faza B",
            target=args.oversample_target)
    else:
        y_ot = y_ot_true
    del outer_train_results

    print(f"\n=== 9. Faza B — procesiranje TEST datoteke (locen fizicni slajd, samo bbox) ===")
    test_res = process_crop(args.test_file, pca_B, scales=args.neigh_scales,
                            label="TEST", cache_dir=cache_dir)
    padded_test = {0: pad_pca(test_res["data_pca"])}
    samples_test = np.concatenate([
        np.zeros((len(test_res["y"]), 1), dtype=np.int64), test_res["coords"]
    ], axis=1)

    print(f"\n=== 10. Faza B — Ensemble ({args.n_ensemble} modelov x {final_epochs} epoh, brez peeka) ===")
    print(f"  TEST se prvic dotakne SELE po koncanem treningu (TTA napoved).")
    test_logits_sum = np.zeros((len(test_res["y"]), NUM_CLASSES), dtype=np.float64)
    for i, seed in enumerate(seeds):
        print(f"\n  -- Faza B model {i+1}/{args.n_ensemble} (seed={seed}) --")
        model_b = train_blind(
            padded_ot, spec_ot, samples_ot, y_ot, device, criterion_B,
            final_epochs=final_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, spec_noise=args.spec_noise,
        )
        print(f"  TTA na TEST (prvic in edinkrat)...")
        test_logits_sum += get_logits_tta(
            model_b, padded_test, test_res["spec_dense"], samples_test,
            test_res["y"], device
        ).astype(np.float64)

    # ------------------------------------------------------------------
    print("\n=== 11. KONCNA evaluacija na TEST (edini dotik, pravi cross-slide) ===")
    test_probs = apply_T((test_logits_sum / args.n_ensemble).astype(np.float32))
    test_pred = np.argmax(test_probs, axis=1)
    test_oa   = accuracy_score(test_res["y"], test_pred)
    test_ll   = log_loss(test_res["y"], test_probs, labels=np.arange(NUM_CLASSES))
    print(f"  TEST OA (pred smoothing): {test_oa*100:.2f}%")
    print(f"  TEST ll (pred smoothing): {test_ll:.5f}")
    print(f"  Primerjava inner-val (zdruzen): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Ref clanek CNN (SD, isti split): OA=79.45% +/- 1.25")
    print(f"  Ref clanek SVM (SD, isti split): OA=56.41%")
    print_per_class_table(test_res["y"], test_pred, test_probs,
                          "Per-class OA in log-loss (Faza B, TEST — koncni test):")

    prior_B = np.bincount(y_ot_true, minlength=NUM_CLASSES).astype(np.float32)
    prior_B /= prior_B.sum()
    prob_map = build_full_canvas_prob_map(
        test_res["H"], test_res["W"], test_res["coords"], test_probs, prior_B)

    if sigma_opt > 0:
        smoothed  = gaussian_smooth_probs(prob_map, test_res["tissue_mask"], sigma_opt)
        final_map = prob_map.copy()
        final_map[test_res["tissue_mask"]] = smoothed[test_res["tissue_mask"]]
    else:
        final_map = prob_map

    final_map = np.clip(final_map, 1e-7, 1.0)
    final_map /= final_map.sum(axis=-1, keepdims=True)

    smoothed_at_coords = final_map[test_res["coords"][:, 0], test_res["coords"][:, 1]]
    smoothed_at_coords = np.clip(smoothed_at_coords, 1e-7, 1.0)
    smoothed_at_coords /= smoothed_at_coords.sum(axis=1, keepdims=True)
    test_pred_sm = np.argmax(smoothed_at_coords, axis=1)
    test_oa_sm   = accuracy_score(test_res["y"], test_pred_sm)
    test_ll_sm   = log_loss(test_res["y"], smoothed_at_coords, labels=np.arange(NUM_CLASSES))
    print(f"\n  TEST OA (po smoothing, sigma={sigma_opt:.1f}): {test_oa_sm*100:.2f}%")
    print(f"  TEST ll (po smoothing, sigma={sigma_opt:.1f}): {test_ll_sm:.5f}")

    np.save(args.output, final_map.astype(np.float32))
    r0_test, c0_test = test_res["bbox_offset"]
    print(f"\n  Shranjeno: {args.output}  shape={final_map.shape}")
    print(f"  (bbox znotraj izvornega TEST platna: offset r0={r0_test}, c0={c0_test}, "
          f"velikost {test_res['H']}x{test_res['W']} od 800x1200)")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (crossSlide v2) ===")
    print(f"  SKUPAJ CAS TEKA: {format_duration(time.time() - run_start)}")
    print(f"  Train: {len(train_crop_paths)} crop-ov (br1003-br2085b), {len(y_ot_true):,} pikslov")
    print(f"  Test:  1 datoteka (brc961-br1001), {len(test_res['y']):,} pikslov")
    print(f"  Inner-val nacin: {'rotacija x' + str(len(rotation_candidates)) if args.rotate_inner_val else 'en kandidat'}")
    print(f"  Balance strategy: {args.balance_strategy} | Temperature: "
          f"{'per-class' if args.per_class_temperature else 'skalarna'} ({T_opt_str})")
    print(f"  Faza A: max_epochs={args.max_epochs}, best_epochs (vse rotacije)={all_best_epochs}, "
          f"mediana={median_epochs}")
    if args.final_epochs is not None:
        print(f"  Faza B: final_epochs={final_epochs} (ROCNO PREGLASEN), ensemble={args.n_ensemble}")
    else:
        print(f"  Faza B: final_epochs={final_epochs} (=mediana), ensemble={args.n_ensemble}")
    print(f"  sigma={sigma_opt:.1f} | weight_soften={args.weight_soften} | bbox_margin={BBOX_MARGIN}")
    print(f"\n  Inner-val (zdruzen):    OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (TEST, KONCNI): OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")
    print(f"\n  Primerjava:")
    print(f"    Clanek SVM (isti split):  OA=56.41%")
    print(f"    Clanek CNN (isti split):  OA=79.45% +/- 1.25")
    print(f"    crossSlide v2 (ta tek, n_ensemble={args.n_ensemble}): OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")

    # ------------------------------------------------------------------
    print(f"\n=== 12. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelC_crossSlide_v2",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa_sm, test_ll=test_ll_sm,
        output_path=args.output,
        t_opt_str=T_opt_str, sigma=sigma_opt,
        max_epochs=args.max_epochs, best_epoch=final_epochs,
        final_epochs=final_epochs, n_ensemble=args.n_ensemble,
        total_duration_str=format_duration(time.time() - run_start),
        extra_note=(METODOLOGIJA_OPOMBA +
                   f" [strategy={args.balance_strategy}, "
                   f"rotate={args.rotate_inner_val}, "
                   f"per_class_T={args.per_class_temperature}]")
    )


if __name__ == "__main__":
    main()
