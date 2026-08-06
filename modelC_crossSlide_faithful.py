"""
Model C — Cross-Slide Faithful  (single-stream CNN po Mayerich et al. 2018, Fig. 3/2.5)
=========================================================================================
Namen: kar se da zvesta replika clanka CNN arhitekture za SD podatke, testirana na
PRAVEM cross-slide train/test splitu (isto ogrodje kot modelC_crossSlide_v1/v2), da
locimo ali je vrzel do clanka (79.45%+/-1.25% OA) posledica arhitekture/treninga ali
lastnosti podatkov (npr. myofibroblasts/R4 porazdelitveni premik med slajdoma).

ZVESTO CLANKU (2.4-2.5.2, Fig. 3, "SD = ista arhitektura brez BN, input 17x17x16"):
  - Single-stream CNN (BREZ spektralne MLP veje iz v1/v2 — samo prostorski patch)
  - Conv(32,3x3) -> MaxPool(2x2) -> Conv(64,3x3) -> Conv(64,3x3) -> MaxPool(2x2)
    -> FC(128) -> Softmax(6)
  - BREZ BatchNorm (clanek: BN samo za HD, SD = "ista arhitektura brez BN")
  - Softplus namesto ReLU (clanek: boljsa konvergenca, gladek pri 0)
  - Dropout 0.5 (keep_prob 0.5 v treningu, clanek tocka 2.5.2 #2)
  - Utezi: normalna porazdelitev, mean=0, std=0.02 (clanek tocka 2.5.2 #6)
  - Patch 17x17x16 PCA (SD velikost po clanku, glej Fig.3 opombo)
  - FIKSNIH 8 epoh (clanek tocka 2.5.2 #8: "train for 8 epochs, terminating when
    validation accuracy began to decline" — mi to beremo kot a-priori fiksno stevilo,
    NE kot sumno best-of-N iskanje na majhnem inner-val kot v1/v2, ker se je slednje
    izkazalo nestabilno: best_epochs so nihali med 1 in 12 na istih podatkih)
  - Oversampling do fiksnega cilja/razred namesto class weights (clanek: "stack
    copies of underrepresented classes", CNN cilj 100,000/razred — mi privzeto
    manjse stevilo, glej --oversample-target, da omejimo pretirano podvajanje na
    manjsem naboru kot v clanku)

MODERNIZIRANO (namerna odstopanja od clanka, po dogovoru):
  - Adam namesto Adadelta(lr=0.1) — slednji je danes nenavadna izbira in slabse
    konvergira na nasih podatkih/frameworku; Adam je uporabljen v vseh nasih
    ostalih modelih, kar olajsa primerjavo.
  - Cosine LR decay + gradient clipping + majhen weight_decay (1e-4) — standardna
    sodobna stabilizacija, ne spreminja arhitekture.
  - BREZ Local Response Normalization (LRN) — clanek je dvoumen ali je LRN del SD
    variante (Fig.3 prikazuje LRN za HD shemo); izpuscena zaradi enostavnosti,
    danes redko uporabljana.

RAZLIKA OD v1/v2:
  - BREZ spektralne MLP veje, BREZ neighbourhood mean [3,5,7] znacilk (spec_dense) —
    samo prostorski PCA patch. To pomeni LAZJI in HITREJSI cache (ni potreben
    uniform_filter), manj RAM-a.
  - BREZ Faze A best-of-N iskanja epoh (Pass 1 iz v1/v2) — epohe so fiksne
    (--final-epochs, privzeto 8), zato gremo direktno v rekalibracijski ensemble.
  - Inner-val ostane (rotacija opcijska) SAMO za kalibracijo temperature/sigma in
    poznejso primerjavo z v1/v2/transformerjem — ne za iskanje hiperparametrov.

Vse ostalo (RAM-varcno bbox branje, disk cache, gnezdena Faza A/B struktura, T
rekalibracija, sigma sweep, write_results_report) je podedovano iz v1/v2.
"""

import argparse
import glob
import hashlib
import os
import random
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.optimize import minimize, minimize_scalar
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES  = 6
PATCH_SIZE   = 17
N_PCA        = 16
RESULTS_FILE = "rezultati_report.txt"
NUM_WORKERS  = 0

BBOX_MARGIN = 12

SIGMA_CHOICES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_SOFTEN = 0.5

METODOLOGIJA_OPOMBA = (
    "modelC_crossSlide_faithful: kar se da zvesta replika clanka SD CNN arhitekture "
    "(single-stream Conv32-Pool-Conv64-Conv64-Pool-FC128-Softmax6, brez BN, softplus, "
    "dropout 0.5, init N(0,0.02), fiksnih --final-epochs namesto sumnega best-of-N) "
    "na PRAVEM cross-slide splitu. Modernizirano: Adam namesto Adadelta, brez LRN. "
    "Ref. clanek: SVM=56.41%, CNN=79.45%+/-1.25 (isti train/test TMA split)."
)


# ---------------------------------------------------------------------------
# Nalaganje — SAMO bounding box okoli anotiranih pikslov (RAM-varcno, kot v1/v2)
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
    """Enako kot v1/v2: razvrsti crop-e z vsemi 6 razredi po NAJMANJSEM razrednem
    stevilu (bolj uravnotezeni najprej)."""
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
# Lahek cache: SAMO coords/y/tissue_mask (BREZ spec_dense/uniform_filter —
# faithful model ne uporablja spektralne veje niti neighbourhood znacilk)
# ---------------------------------------------------------------------------
def _crop_cache_key(path, margin):
    key_str = f"{os.path.abspath(path)}|margin={margin}|faithful-coords"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def get_coords_cached(path, margin=BBOX_MARGIN, cache_dir=None, label=""):
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _crop_cache_key(path, margin)
        cache_path = os.path.join(cache_dir, f"coordsonly_{key}.npz")
        if os.path.exists(cache_path):
            t0 = time.time()
            d = np.load(cache_path)
            print(f"  [{label}] Cache HIT: {os.path.basename(path)}  "
                  f"({time.time()-t0:.1f}s)")
            return (d["coords"], d["y"], d["tissue_mask"],
                    int(d["H"]), int(d["W"]), int(d["r0"]), int(d["c0"]))

    t0 = time.time()
    data, tissue_mask, classes, r0, c0 = read_crop_bbox(path, margin)
    H, W, D = data.shape
    ann    = (classes != -1)
    coords = np.argwhere(ann)
    y      = classes[ann].astype(np.int64)
    del data

    print(f"  [{label}] Coords ekstrakcija {os.path.basename(path)}: "
          f"bbox={H}x{W}, {len(y):,} anotiranih  ({time.time()-t0:.1f}s)")

    if cache_path:
        np.savez_compressed(cache_path, coords=coords, y=y,
                            tissue_mask=tissue_mask, H=H, W=W, r0=r0, c0=c0)
        print(f"  [{label}] Cache shranjen: {os.path.basename(cache_path)}")

    return coords, y, tissue_mask, H, W, r0, c0


def fit_pca_pooled(crop_paths, n_pca=N_PCA, seed=42, label="", margin=BBOX_MARGIN):
    print(f"  PCA fit ({label}): zbiranje anotiranih spektrov iz {len(crop_paths)} crop-ov...")
    pooled = []
    for p in crop_paths:
        data, _, classes, _, _ = read_crop_bbox(p, margin)
        ann = (classes != -1)
        pooled.append(data[ann].copy())
        del data
        print(f"    {os.path.basename(p)}: {int(ann.sum()):,} spektrov")
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


def process_crop(path, pca, cache_dir=None, label="", margin=BBOX_MARGIN,
                 extra_smooth_scales=()):
    coords, y, tissue_mask, H, W, r0, c0 = get_coords_cached(
        path, margin, cache_dir, label=label)

    t0 = time.time()
    print(f"  [{label}] PCA transform ({H*W:,} pikslov v bbox)...")
    data, _, _, _, _ = read_crop_bbox(path, margin)
    flat = data.reshape(-1, data.shape[-1])
    n_bad = int((~np.isfinite(flat)).any(axis=1).sum())
    if n_bad:
        print(f"  [{label}] OPOZORILO: {n_bad} pikslov z NaN/Inf v surovih podatkih "
              f"(verjetno pokvarjen senzorski piksel) -- sanitiziram na 0.")
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat_pca = pca.transform(flat).astype(np.float32)
    data_pca = flat_pca.reshape(H, W, N_PCA)
    del data, flat, flat_pca

    if extra_smooth_scales:
        # Dodatni glajeni PCA kanali-seti (namesto cele spektralne veje kot v v1/v2)
        # -- lahek nacin za dodati prostorski kontekst na vec skalah hkrati, ne da
        # bi reintroducirali celotno neighbourhood-mean spektralno vejo, ki je
        # skodila R4/myofibroblastom v dual-stream arhitekturi.
        parts = [data_pca]
        for scale in extra_smooth_scales:
            smooth = uniform_filter(data_pca, size=[scale, scale, 1], mode='reflect')
            parts.append(smooth)
        data_pca = np.concatenate(parts, axis=-1).astype(np.float32)
        del parts

    print(f"  [{label}] -> data_pca={data_pca.shape}, anotiranih={len(y):,}  "
          f"({time.time()-t0:.1f}s)")
    return {
        "data_pca": data_pca, "coords": coords, "y": y, "H": H, "W": W,
        "tissue_mask": tissue_mask, "bbox_offset": (r0, c0),
    }


def pad_pca(data_pca, patch_size=PATCH_SIZE):
    pad = patch_size // 2
    return np.pad(data_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect').astype(np.float32)


def stack_crops(crop_results, crop_ids):
    samples_parts, y_parts = [], []
    for cid, res in zip(crop_ids, crop_results):
        n = len(res["y"])
        crop_col = np.full((n, 1), cid, dtype=np.int64)
        samples_parts.append(np.concatenate([crop_col, res["coords"]], axis=1))
        y_parts.append(res["y"])
    samples = np.concatenate(samples_parts, axis=0)
    y       = np.concatenate(y_parts, axis=0)
    return samples, y


# ---------------------------------------------------------------------------
# Balansiranje razredov — oversampling do fiksnega cilja (clanek: "stack
# copies of underrepresented classes")
# ---------------------------------------------------------------------------
def oversample_pool(samples, y, seed=42, verbose=True, label="train", target=None):
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
            idx_parts.append(rng.choice(cls_idx, size=target, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    if verbose:
        print(f"  Skupaj po oversamplingu ({label}): {len(idx):,} (prej {len(y):,})")
    return samples[idx], y[idx]


# ---------------------------------------------------------------------------
# Dataset — SAMO prostorski patch (brez spektralne veje)
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


class PatchDataset(Dataset):
    def __init__(self, padded_pca_by_crop, samples, labels,
                 patch_size=PATCH_SIZE, augment=False, tta_idx=-1):
        self.pad = patch_size // 2
        self.padded_pca_by_crop = padded_pca_by_crop
        self.samples = samples
        self.labels = labels
        self.augment = augment
        self.tta_idx = tta_idx

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
        return patch_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Arhitektura: Single-Stream CNN (clanek Fig.3, SD varianta = brez BN)
# ---------------------------------------------------------------------------
class SingleStreamCNN(nn.Module):
    def __init__(self, n_channels=N_PCA, num_classes=NUM_CLASSES, patch_size=PATCH_SIZE,
                dropout=0.5, use_lrn=False):
        super().__init__()
        # Clanek Fig. 3: Conv32->BN->LRN->MP, Conv64->Conv64->BN->LRN->MP.
        # Za SD se BN izpusti, a LRN OSTANE (locena arhitekturna odlocitev v
        # clanku) -- doslej nisva tega implementirala (--use-lrn flag).
        lrn = (lambda: nn.LocalResponseNorm(size=5)) if use_lrn else (lambda: nn.Identity())
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.Softplus(), nn.Dropout2d(dropout),
            lrn(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Softplus(), nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Softplus(), nn.Dropout2d(dropout),
            lrn(),
            nn.MaxPool2d(2),
        )
        # Po dveh MaxPool(2) (conv plasti ohranjajo prostorsko dimenzijo zaradi
        # padding=1) je prostorska dimenzija patch_size//2//2 (npr. 17->4, 25->6).
        pooled = patch_size // 2 // 2
        self.flatten_dim = 64 * pooled * pooled
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        # Clanek 2.5.2 #6: normal(mean=0, std=0.02) za vse utezi.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_logits(self, patch):
        f = self.features(patch)
        f = f.reshape(f.size(0), -1)
        if f.shape[1] != self.flatten_dim:
            # Varnostna past, ce se PATCH_SIZE spremeni.
            raise RuntimeError(
                f"Nepricakovana dimenzija po conv slojih: {f.shape[1]} "
                f"(pricakovano {self.flatten_dim}). Preveri PATCH_SIZE."
            )
        f = F.softplus(self.fc1(f))
        f = self.dropout(f)
        return self.fc2(f)

    def forward(self, patch):
        return self.log_softmax(self.get_logits(patch))


class SmoothedNLLLoss(nn.Module):
    """Label smoothing za NLLLoss (vhod = log-probs iz LogSoftmax). Cilja
    prekomerno samozavestne napovedi (visok log-loss pri R2/R3 kljub temu,
    da OA ostane priblizno enak) -- mehcanje ze med treningom, ne le naknadno
    prek temperature kalibracije."""
    def __init__(self, weight=None, smoothing=0.1, n_classes=NUM_CLASSES):
        super().__init__()
        self.weight = weight
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, log_probs, target):
        nll = F.nll_loss(log_probs, target, weight=self.weight, reduction='none')
        smooth = -log_probs.mean(dim=1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()


def build_criterion(strategy, weights, device, label_smoothing=0.0):
    """weights = ze izracunane (softened) class weights, NE na napravi."""
    w = None if strategy == "oversample" else weights.to(device)
    if label_smoothing > 0:
        return SmoothedNLLLoss(weight=w, smoothing=label_smoothing)
    if strategy == "oversample":
        return nn.NLLLoss()  # podatki so ze uravnotezeni z oversamplingom
    else:  # "weights"
        return nn.NLLLoss(weight=w)


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
    for patches, _ in loader:
        all_l.append(model.get_logits(patches.to(device)).cpu().numpy())
    return np.concatenate(all_l)


@torch.no_grad()
def get_logits_tta(model, padded_pca_by_crop, samples, labels, device, batch_size=512,
                   patch_size=PATCH_SIZE):
    logits_sum = None
    for aug_idx in range(8):
        ds = PatchDataset(padded_pca_by_crop, samples, labels, patch_size=patch_size,
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
    res = minimize_scalar(neg_ll, bounds=(0.1, 50.0), method='bounded')
    T   = res.x
    print(f"  Temperature (skalarna): T={T:.4f}  {neg_ll(1.0):.5f} -> {neg_ll(T):.5f}")
    return T


def apply_temperature(logits, T):
    s = logits / T
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def find_temperature_per_class(logits, y, n_classes=NUM_CLASSES):
    def neg_ll(T_vec):
        T_vec = np.clip(T_vec, 0.05, 60.0)
        s = logits / T_vec[None, :]
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-9, 1.0)
        return -np.mean(np.log(p[np.arange(len(y)), y]))
    x0 = np.ones(n_classes)
    bounds = [(0.1, 50.0)] * n_classes
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
                          output_path, t_opt_str, sigma, final_epochs, n_ensemble,
                          extra_note="", total_duration_str=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dur_part = f"  cas={total_duration_str}" if total_duration_str else ""
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"TEST_OA={test_oa*100:6.2f}%  TEST_ll={test_ll:.5f}  "
        f"T={t_opt_str}  sigma={sigma:.1f}  n_ensemble={n_ensemble}  "
        f"final_ep={final_epochs}{dur_part}"
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
# Trening — fiksne epohe (BREZ best-of-N iskanja, glej opombo na vrhu datoteke)
# ---------------------------------------------------------------------------
def train_blind(padded_pca, samples, y, device, criterion,
                final_epochs, batch_size, lr, seed, n_channels=N_PCA,
                patch_size=PATCH_SIZE, use_lrn=False):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = PatchDataset(padded_pca, samples, y, patch_size=patch_size, augment=True)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    model     = SingleStreamCNN(n_channels=n_channels, patch_size=patch_size, use_lrn=use_lrn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs, eta_min=1e-6)

    print(f"  {'Ep':>4}  {'Train ll':>10}  {'LR':>9}")
    t0 = time.time()
    for epoch in range(1, final_epochs + 1):
        model.train()
        total = 0.0
        for patches, labels in loader:
            patches = patches.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(patches), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        train_loss = total / len(train_ds)
        scheduler.step()
        print(f"  {epoch:>4}  {train_loss:>10.5f}  "
              f"{optimizer.param_groups[0]['lr']:>9.2e}")
    print(f"  Treniran v {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global BBOX_MARGIN, NUM_WORKERS
    run_start = time.time()

    parser = argparse.ArgumentParser(
        description="Model C Cross-Slide Faithful: single-stream CNN po clanku, "
                    "fiksne epohe, brez best-of-N iskanja"
    )
    parser.add_argument("--train-dir",  default="FTIR-data/train_preprocessed")
    parser.add_argument("--test-file",  default="FTIR-data/test_preprocessed/test_expanded_crop_preprocessed.hdf5",
                        help="En sam TEST file (star nacin, ignoriran ce je podan --test-dir).")
    parser.add_argument("--test-dir",   default=None,
                        help="Mapa z vec test_crop_*.hdf5 (razsirjen test set, glej "
                             "find_test_crops.py/create_test_crops.py). Ce podano, "
                             "preglasi --test-file -- evalvacija je zdruzena (pooled) "
                             "cez VSE test crop-e, glajenje pa je per-crop (locena "
                             "prostorska obmocja).")
    parser.add_argument("--output",     default="modelC_crossSlide_faithful_test.npy")
    parser.add_argument("--final-epochs", type=int, default=8,
                        help="Fiksno stevilo epoh (clanek: 8, tocka 2.5.2 #8).")
    parser.add_argument("--calib-ensemble", type=int, default=12)
    parser.add_argument("--rotate-inner-val", action="store_true")
    parser.add_argument("--per-class-temperature", action="store_true")
    parser.add_argument("--balance-strategy", choices=["weights", "oversample"],
                        default="oversample",
                        help="clanek uporablja oversampling ('stack copies of "
                             "underrepresented classes'), zato je privzeto tukaj.")
    parser.add_argument("--weight-soften", type=float, default=WEIGHT_SOFTEN)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing (0=izklopljeno). Cilja pretirano "
                             "samozavestne napovedi (visok log-loss pri R2/R3) "
                             "z regularizacijo med treningom, ne le naknadno "
                             "prek temperature.")
    parser.add_argument("--oversample-target", type=int, default=None,
                        help="Fiksno stevilo pikslov/razred (clanek: 100,000 za CNN; "
                             "privzeto brez vrednosti = velikost najvecjega razreda).")
    parser.add_argument("--extra-smooth-scale", type=str, default="",
                        help="Vejica-loceni seznam prostorskih skal za dodatne glajene "
                             "PCA kanale-sete (uniform_filter), poleg raw PCA kanalov v "
                             "CNN vhod (npr. '3' -> 32 kanalov, '3,7' -> 48 kanalov pri "
                             "N_PCA=16). Lahek nacin za dodati prostorski kontekst brez "
                             "cele neighbourhood-mean spektralne veje (ki je skodila R4 "
                             "v dual-stream v1/v2). Privzeto prazno = izklopljeno (cisto "
                             "zvesta clanku, 16 kanalov).")
    parser.add_argument("--batch-size", type=int,   default=128,
                        help="clanek: batch=128 (tocka 2.5.2 #7).")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Adam lr (modernizirano, clanek: Adadelta lr=0.1).")
    parser.add_argument("--n-ensemble", type=int,   default=4)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE,
                        help="Velikost prostorskega patch-a (privzeto 17, clanek-zvesto "
                             "za SD). Vecji patch (npr. 25) da vec konteksta CNN-ju za "
                             "morebitno izboljsavo R2/R3 -- NAMERNO odstopanje od clanka, "
                             "ne 'faithful' vec pri neprivzeti vrednosti.")
    parser.add_argument("--use-lrn", action="store_true",
                        help="Dodaj Local Response Normalization po Conv32 in po "
                             "drugem Conv64 (pred MaxPool), kot v clanku Fig. 3 -- "
                             "za SD se BN izpusti, LRN pa OSTANE v clanku. Privzeto "
                             "izklopljeno (doslejsnje 'faithful' obnasanje), da "
                             "obstojeci rezultati ostanejo primerljivi.")
    parser.add_argument("--bbox-margin", type=int, default=BBOX_MARGIN)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers. Privzeto 0 (varno na macOS/MPS). "
                             "Na Linux/CUDA strezniku z vec jedri nastavi npr. 4-8 za "
                             "vzporedno nalaganje podatkov (odpravi CPU-vezano ozko grlo).")
    parser.add_argument("--cache-dir",  default="FTIR-data/_cache")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    args.extra_smooth_scale = ([int(s) for s in args.extra_smooth_scale.split(",") if s.strip()]
                               if args.extra_smooth_scale.strip() else [])
    BBOX_MARGIN = args.bbox_margin
    NUM_WORKERS = args.num_workers
    patch_size = args.patch_size
    cache_dir = args.cache_dir if args.cache_dir else None

    # ------------------------------------------------------------------
    print("\n=== 1. Odkrivanje train crop-ov in izbira inner-val kandidatov ===")
    train_crop_paths = sorted(glob.glob(os.path.join(args.train_dir, "train_crop_*.hdf5")))
    print(f"  Najdenih train crop-ov: {len(train_crop_paths)}")
    for p in train_crop_paths:
        print(f"    {p}")

    candidates = select_inner_val_candidates(train_crop_paths)
    rotation_candidates = candidates if args.rotate_inner_val else candidates[:1]
    print(f"\n  Nacin inner-val: " +
          (f"ROTACIJA cez {len(rotation_candidates)} kandidatov"
           if args.rotate_inner_val else
           f"En kandidat (najbolj uravnotezen): {os.path.basename(rotation_candidates[0])}"))
    print(f"  Balance strategy: {args.balance_strategy}"
          + (f" (target={args.oversample_target})" if args.balance_strategy == "oversample" else ""))
    print(f"  Temperature: {'per-class' if args.per_class_temperature else 'skalarna'}")
    print(f"  Fiksne epohe (brez iskanja): {args.final_epochs}")

    device = get_device()
    n_channels = N_PCA * (1 + len(args.extra_smooth_scale))
    n_param = sum(p.numel() for p in SingleStreamCNN(n_channels=n_channels, patch_size=patch_size,
                                                      use_lrn=args.use_lrn).parameters()
                 if p.requires_grad)
    print(f"\n  Konfiguracija: Patch {patch_size}x{patch_size} | PCA({N_PCA}) | "
          f"kanalov={n_channels}" +
          (f" (+glajeni scale={args.extra_smooth_scale})" if args.extra_smooth_scale else "") +
          f" | TTA 8 | Ensemble {args.n_ensemble} | SingleStreamCNN ({n_param:,} param)")
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    seeds = list(range(args.seed, args.seed + args.n_ensemble))
    final_epochs = args.final_epochs

    # ==================================================================
    # FAZA A — priprava podatkov + rekalibracijski ensemble (BREZ best-of-N)
    # ==================================================================
    rotations = []
    for r_idx, inner_val_path in enumerate(rotation_candidates):
        inner_train_paths = [p for p in train_crop_paths if p != inner_val_path]
        rlabel = f"rot{r_idx+1}/{len(rotation_candidates)}:{os.path.basename(inner_val_path)}"

        print(f"\n=== 2.{r_idx+1} Faza A [{rlabel}] — PCA fit na {len(inner_train_paths)} crop-ih ===")
        pca_A = fit_pca_pooled(inner_train_paths, N_PCA, args.seed, label=rlabel)

        print(f"\n=== 3.{r_idx+1} Faza A [{rlabel}] — procesiranje crop-ov ===")
        it_results, it_ids = [], []
        for i, p in enumerate(inner_train_paths):
            res = process_crop(p, pca_A, cache_dir=cache_dir, label=f"{rlabel}/it{i}",
                              extra_smooth_scales=args.extra_smooth_scale)
            it_results.append(res); it_ids.append(i)
        iv_res = process_crop(inner_val_path, pca_A, cache_dir=cache_dir, label=f"{rlabel}/iv",
                              extra_smooth_scales=args.extra_smooth_scale)

        samples_it, y_it_true = stack_crops(it_results, it_ids)
        padded_it  = {cid: pad_pca(res["data_pca"], patch_size) for cid, res in zip(it_ids, it_results)}
        padded_iv  = {0: pad_pca(iv_res["data_pca"], patch_size)}
        samples_iv = np.concatenate([
            np.zeros((len(iv_res["y"]), 1), dtype=np.int64), iv_res["coords"]
        ], axis=1)
        iv_ds = PatchDataset(padded_iv, samples_iv, iv_res["y"], patch_size=patch_size, augment=False)
        del it_results

        print(f"\n  [{rlabel}] inner-train={len(y_it_true):,}px, inner-val={len(iv_res['y']):,}px")
        for c in range(NUM_CLASSES):
            nt = (y_it_true == c).sum(); nv = (iv_res["y"] == c).sum()
            print(f"    R{c}: it={nt:6d} ({100*nt/len(y_it_true):.1f}%)  "
                  f"iv={nv:6d} ({100*nv/max(len(iv_res['y']),1):.1f}%)")

        weights_A = compute_class_weights(y_it_true, soften=args.weight_soften)
        criterion_A = build_criterion(args.balance_strategy, weights_A, device,
                                      label_smoothing=args.label_smoothing)

        if args.balance_strategy == "oversample":
            samples_it, y_it = oversample_pool(
                samples_it, y_it_true, seed=args.seed, label=rlabel,
                target=args.oversample_target)
        else:
            y_it = y_it_true

        n_calib = max(1, min(args.calib_ensemble, args.n_ensemble))
        print(f"\n=== 4.{r_idx+1} Rekalibracijski ensemble [{rlabel}] "
              f"({n_calib} x {final_epochs} ep, blind) ===")
        calib_seeds = seeds[:n_calib]
        calib_logits_list = []
        for i, seed in enumerate(calib_seeds):
            print(f"\n  -- {rlabel} rekalibracija {i+1}/{n_calib} (seed={seed}) --")
            model_c = train_blind(
                padded_it, samples_it, y_it, device, criterion_A,
                final_epochs=final_epochs, batch_size=args.batch_size,
                lr=args.lr, seed=seed, n_channels=n_channels, patch_size=patch_size,
                use_lrn=args.use_lrn,
            )
            calib_logits_list.append(
                get_logits_tta(model_c, padded_iv, samples_iv, iv_res["y"], device,
                              patch_size=patch_size)
            )
        calib_logits_avg_r = np.mean(calib_logits_list, axis=0)

        rotations.append(dict(
            label=rlabel, iv_res=iv_res, y_it_true=y_it_true,
            calib_logits=calib_logits_avg_r,
        ))
        del padded_it, padded_iv, samples_it, samples_iv

    all_calib_logits = [rd["calib_logits"] for rd in rotations]
    all_calib_y      = [rd["iv_res"]["y"] for rd in rotations]
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

    del rotations, all_calib_logits, all_calib_y

    # ==================================================================
    # FAZA B — vsi train crop-i, blind trening, en dotik s pravim TEST filom
    # ==================================================================
    print(f"\n=== 7. Faza B — PCA fit na vseh {len(train_crop_paths)} train crop-ih ===")
    pca_B = fit_pca_pooled(train_crop_paths, N_PCA, args.seed, label="Faza B")

    print(f"\n=== 8. Faza B — procesiranje vseh train crop-ov (samo bbox) ===")
    outer_train_results, outer_train_ids = [], []
    for i, p in enumerate(train_crop_paths):
        res = process_crop(p, pca_B, cache_dir=cache_dir, label=f"outer-train {i}",
                          extra_smooth_scales=args.extra_smooth_scale)
        outer_train_results.append(res)
        outer_train_ids.append(i)

    samples_ot, y_ot_true = stack_crops(outer_train_results, outer_train_ids)
    padded_ot = {cid: pad_pca(res["data_pca"], patch_size) for cid, res in zip(outer_train_ids, outer_train_results)}
    print(f"\n  Skupaj outer-train (Faza B): {len(y_ot_true):,} pikslov iz {len(train_crop_paths)} crop-ov")

    weights_B = compute_class_weights(y_ot_true, soften=args.weight_soften)
    criterion_B = build_criterion(args.balance_strategy, weights_B, device,
                                  label_smoothing=args.label_smoothing)
    if args.balance_strategy == "oversample":
        samples_ot, y_ot = oversample_pool(
            samples_ot, y_ot_true, seed=args.seed, label="Faza B",
            target=args.oversample_target)
    else:
        y_ot = y_ot_true
    del outer_train_results

    print(f"\n=== 9. Faza B — procesiranje TEST podatkov (locen fizicni slajd, samo bbox) ===")
    if args.test_dir:
        test_paths = sorted(glob.glob(os.path.join(args.test_dir, "test_crop_*.hdf5")))
        print(f"  --test-dir podan: najdenih {len(test_paths)} test crop-ov v {args.test_dir}")
    else:
        test_paths = [args.test_file]
        print(f"  En sam TEST file: {args.test_file}")

    test_results, test_ids = [], []
    for i, p in enumerate(test_paths):
        res = process_crop(p, pca_B, cache_dir=cache_dir, label=f"TEST-{i}",
                          extra_smooth_scales=args.extra_smooth_scale)
        test_results.append(res); test_ids.append(i)

    samples_test, y_test = stack_crops(test_results, test_ids)
    padded_test = {cid: pad_pca(res["data_pca"], patch_size) for cid, res in zip(test_ids, test_results)}
    print(f"\n  Skupaj TEST: {len(y_test):,} pikslov iz {len(test_paths)} crop-ov "
          f"(prej: 1 crop, 86,993 pikslov)")

    print(f"\n=== 10. Faza B — Ensemble ({args.n_ensemble} modelov x {final_epochs} epoh, brez peeka) ===")
    print(f"  TEST se prvic dotakne SELE po koncanem treningu (TTA napoved).")
    test_logits_sum = np.zeros((len(y_test), NUM_CLASSES), dtype=np.float64)
    for i, seed in enumerate(seeds):
        print(f"\n  -- Faza B model {i+1}/{args.n_ensemble} (seed={seed}) --")
        model_b = train_blind(
            padded_ot, samples_ot, y_ot, device, criterion_B,
            final_epochs=final_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, n_channels=n_channels, patch_size=patch_size,
            use_lrn=args.use_lrn,
        )
        print(f"  TTA na TEST (prvic in edinkrat)...")
        test_logits_sum += get_logits_tta(
            model_b, padded_test, samples_test, y_test, device, patch_size=patch_size
        ).astype(np.float64)

    # ------------------------------------------------------------------
    print("\n=== 11. KONCNA evaluacija na TEST (edini dotik, pravi cross-slide) ===")
    test_probs = apply_T((test_logits_sum / args.n_ensemble).astype(np.float32))
    test_pred = np.argmax(test_probs, axis=1)
    test_oa   = accuracy_score(y_test, test_pred)
    test_ll   = log_loss(y_test, test_probs, labels=np.arange(NUM_CLASSES))
    print(f"  TEST OA (pred smoothing): {test_oa*100:.2f}%")
    print(f"  TEST ll (pred smoothing): {test_ll:.5f}")
    print(f"  Ref clanek CNN (SD, isti split): OA=79.45% +/- 1.25")
    print(f"  Ref clanek SVM (SD, isti split): OA=56.41%")
    print_per_class_table(y_test, test_pred, test_probs,
                          "Per-class OA in log-loss (Faza B, TEST — koncni test):")

    prior_B = np.bincount(y_ot_true, minlength=NUM_CLASSES).astype(np.float32)
    prior_B /= prior_B.sum()

    # Glajenje je PER-CROP (vsak test crop je svoje lokalno prostorsko obmocje --
    # glajenje cez disjunktne, prostorsko oddaljene crop-e nima smisla), koncna
    # OA/ll pa je zdruzena (pooled) cez vse crop-e skupaj.
    offset = 0
    smoothed_parts, y_parts = [], []
    saved_maps = {}
    for cid, res in zip(test_ids, test_results):
        n = len(res["y"])
        probs_c = test_probs[offset:offset + n]
        prob_map = build_full_canvas_prob_map(res["H"], res["W"], res["coords"],
                                              probs_c, prior_B)
        if sigma_opt > 0:
            smoothed = gaussian_smooth_probs(prob_map, res["tissue_mask"], sigma_opt)
            final_map = prob_map.copy()
            final_map[res["tissue_mask"]] = smoothed[res["tissue_mask"]]
        else:
            final_map = prob_map
        final_map = np.clip(final_map, 1e-7, 1.0)
        final_map /= final_map.sum(axis=-1, keepdims=True)

        at_coords = final_map[res["coords"][:, 0], res["coords"][:, 1]]
        at_coords = np.clip(at_coords, 1e-7, 1.0)
        at_coords /= at_coords.sum(axis=1, keepdims=True)
        smoothed_parts.append(at_coords)
        y_parts.append(res["y"])

        r0c, c0c = res["bbox_offset"]
        saved_maps[f"crop{cid:02d}_probs"] = final_map.astype(np.float32)
        saved_maps[f"crop{cid:02d}_offset"] = np.array([r0c, c0c], dtype=np.int64)
        offset += n

    smoothed_at_coords = np.concatenate(smoothed_parts, axis=0)
    y_test_sm = np.concatenate(y_parts, axis=0)
    test_pred_sm = np.argmax(smoothed_at_coords, axis=1)
    test_oa_sm   = accuracy_score(y_test_sm, test_pred_sm)
    test_ll_sm   = log_loss(y_test_sm, smoothed_at_coords, labels=np.arange(NUM_CLASSES))
    print(f"\n  TEST OA (po smoothing, sigma={sigma_opt:.1f}, zdruzeno cez "
          f"{len(test_paths)} crop-ov): {test_oa_sm*100:.2f}%")
    print(f"  TEST ll (po smoothing, sigma={sigma_opt:.1f}, zdruzeno): {test_ll_sm:.5f}")

    np.savez_compressed(args.output.replace(".npy", ".npz") if args.test_dir else args.output,
                        **saved_maps) if args.test_dir else \
        np.save(args.output, saved_maps["crop00_probs"])
    print(f"\n  Shranjeno: {args.output.replace('.npy', '.npz') if args.test_dir else args.output} "
          f"({len(test_paths)} crop map(e))")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (crossSlide faithful) ===")
    print(f"  SKUPAJ CAS TEKA: {format_duration(time.time() - run_start)}")
    print(f"  Train: {len(train_crop_paths)} crop-ov (br1003-br2085b), {len(y_ot_true):,} pikslov")
    print(f"  Test:  {len(test_paths)} crop-ov (brc961-br1001), {len(y_test):,} pikslov")
    print(f"  Arhitektura: SingleStreamCNN (brez BN, softplus, dropout 0.5, N(0,0.02))")
    print(f"  Balance strategy: {args.balance_strategy} | Temperature: "
          f"{'per-class' if args.per_class_temperature else 'skalarna'} ({T_opt_str})")
    print(f"  Fiksne epohe: {final_epochs} | ensemble={args.n_ensemble}")
    print(f"  sigma={sigma_opt:.1f} | bbox_margin={BBOX_MARGIN}")
    print(f"\n  Inner-val (zdruzen):    OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (TEST, KONCNI): OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")
    print(f"\n  Primerjava:")
    print(f"    Clanek SVM (isti split):  OA=56.41%")
    print(f"    Clanek CNN (isti split):  OA=79.45% +/- 1.25")
    print(f"    crossSlide faithful (ta tek, n_ensemble={args.n_ensemble}): "
          f"OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")

    # ------------------------------------------------------------------
    print(f"\n=== 12. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelC_crossSlide_faithful",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa_sm, test_ll=test_ll_sm,
        output_path=args.output,
        t_opt_str=T_opt_str, sigma=sigma_opt,
        final_epochs=final_epochs, n_ensemble=args.n_ensemble,
        total_duration_str=format_duration(time.time() - run_start),
        extra_note=(METODOLOGIJA_OPOMBA +
                   f" [strategy={args.balance_strategy}, "
                   f"oversample_target={args.oversample_target}, "
                   f"rotate={args.rotate_inner_val}, "
                   f"per_class_T={args.per_class_temperature}, "
                   f"patch_size={patch_size}, use_lrn={args.use_lrn}, "
                   f"extra_smooth_scale={args.extra_smooth_scale}, "
                   f"weight_soften={args.weight_soften}]")
    )


if __name__ == "__main__":
    main()
