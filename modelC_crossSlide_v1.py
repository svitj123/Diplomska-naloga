"""
Model C — Cross-Slide v1  (pravi train/test TMA split iz clanka, RAM-varcen)
=================================================================================
Prva verzija, prilagojena novim podatkom:
  Train: FTIR-data/train_preprocessed/train_crop_*.hdf5  (8 crop-ov, br1003-br2085b)
  Test:  FTIR-data/test_preprocessed/test_expanded_crop_preprocessed.hdf5  (brc961-br1001)

To je NATANKO isti train/test TMA split kot v clanku (Tabela 2):
  "BR1003 and BR2005b are used for training while BR961 and BR1001 are used
  for testing."

RAZLIKA OD modelC_crossSlide_v1_RAM.py (prejsnja verzija, propadla zaradi
pomanjkanja RAM-a — stroj ima samo 8GB!):
  Prejsnja verzija je nalozila CEL surov crop (800x1200x813 float32 = ~2.9GB)
  v pomnilnik, nato pa se dodaten ~2.9GB zacasni array za uniform_filter —
  skupaj ~5.8GB SAMO za en crop, kar je na 8GB stroju povzrocilo hudo
  pomnilnisko stisko (swap thrashing), ki je posredno povzrocila nakljucne
  napake (npr. OSError pri torch._dynamo lazy importu).

  Ta verzija bere iz vsakega crop-a SAMO BOUNDING BOX okoli anotiranih
  pikslov (+ varnostna margina), NE celotnega 800x1200 platna:
    - h5py omogoca branje SAMO izbranega izseka neposredno z diska
      (f['data'][r0:r1, c0:c1, :]) — celoten canvas se NIKOLI ne materializira
    - margina (BBOX_MARGIN=12) je vecja od najvecjega potrebnega radija
      (patch_size//2=8 za patch-e, max(NEIGH_SCALES)//2=3 za filtre)
      -> rezultat je NUMERICNO IDENTICEN polnemu branju, ni aproksimacija,
         samo manj porabljenega pomnilnika
    - ce so anotirani piksli prostorsko strnjeni (kar tipicno so — gre za
      konkreten tkivni izsek, ne razprsene tocke), je prihranek lahko
      velikostnega reda (npr. 10x manj pomnilnika za manjse crop-e)

Metodologija (Faza A/B nested split) ostaja ista logika kot v10-v13 in
prejsnja verzija te skripte — spremenjeno je SAMO nacin branja podatkov:
  Faza A: 7 inner-train crop-ov -> best-of-N na 1 inner-val crop-u
          (brez early stopping, best-of-N checkpoint + T kalibracija + sigma sweep)
  Faza B: vseh 8 train crop-ov -> BLIND trening (brez peekanja) na
          median(best_epochs) iz Faze A -> ENA napoved na TEST datoteki

Arhitektura = DualStreamCNN (kot v10/v13), utezi = softened (soften=0.5, kot v13).
Privzeti n-ensemble=4 za hiter sanity-check (poveca na 12 za koncni tek).
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
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES  = 6
PATCH_SIZE   = 17
N_PCA        = 16
NEIGH_SCALES = [3, 5, 7]
RESULTS_FILE = "rezultati_report.txt"

# Margina okoli bounding-boxa anotiranih pikslov. Mora biti >= max potrebnega
# radija (patch_size//2=8, max(NEIGH_SCALES)//2=3) da je rezultat numericno
# enakovreden branju celotnega platna (brez aproksimacije).
BBOX_MARGIN = 12

# Dolocena dinamicno v main() iz dejanskih podatkov (813 kanalov pri novih podatkih)
SPEC_DIM_RAW = None
SPEC_DIM     = None

SIGMA_CHOICES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_SOFTEN = 0.5

METODOLOGIJA_OPOMBA = (
    "modelC_crossSlide_v1: PRAVI train/test TMA split iz clanka "
    "(train=br1003-br2085b, 8 crop-ov; test=brc961-br1001, locena datoteka). "
    "Preprocessing (rubber-band+Amide I) ze narejen v podatkih. Notranji split "
    "= 1 naraven crop kot inner-val (najmanjsi z vsemi 6 razredi), 7 kot "
    "inner-train (Faza A). Vseh 8 crop-ov = outer-train (Faza B), blind trening, "
    "en dotik s pravim TEST filom na koncu. Softened class weights (0.5), "
    "sigma sweep na inner-val, mediana za final_epochs (kot v13). RAM-varcna "
    f"verzija: bere samo bounding box (margina={BBOX_MARGIN}) namesto celega "
    "platna — numericno enakovredno, ker margina presega max potreben radij."
)


# ---------------------------------------------------------------------------
# Nalaganje — SAMO bounding box okoli anotiranih pikslov (RAM-varcno)
# ---------------------------------------------------------------------------
def peek_classes_and_wns(path):
    """Hitro branje samo classes+wns (majhno), brez nalaganja ogromnega 'data'."""
    with h5py.File(path, 'r') as f:
        classes = np.array(f['classes'])
        wns     = np.array(f['wns'])
    return classes, wns


def get_annotated_bbox(classes, margin=BBOX_MARGIN):
    """Vrne (r0,r1,c0,c1) — bounding box anotiranih pikslov + margina, obrezan
    na meje slike."""
    H, W = classes.shape
    coords = np.argwhere(classes != -1)
    r0 = max(0, int(coords[:, 0].min()) - margin)
    r1 = min(H, int(coords[:, 0].max()) + margin + 1)
    c0 = max(0, int(coords[:, 1].min()) - margin)
    c1 = min(W, int(coords[:, 1].max()) + margin + 1)
    return r0, r1, c0, c1


def read_crop_bbox(path, margin=BBOX_MARGIN):
    """
    Prebere SAMO bounding box okoli anotiranih pikslov iz HDF5 — izogne se
    nalaganju celotnega (~2.9GB) platna. h5py fancy-slicing (f['data'][r0:r1,
    c0:c1,:]) bere neposredno z diska, celoten dataset se nikoli ne
    materializira v pomnilniku.

    Numericno enakovredno branju celega platna, ker je margin >= najvecjemu
    potrebnemu radiju (patch_size//2=8, max(NEIGH_SCALES)//2=3) — noben
    anotiran piksel se nikoli ne "dotakne" umetnega reflect-roba bbox-a.
    """
    with h5py.File(path, 'r') as f:
        classes_full = np.array(f['classes'])
        r0, r1, c0, c1 = get_annotated_bbox(classes_full, margin)
        data_bbox        = np.array(f['data'][r0:r1, c0:c1, :], dtype=np.float32)
        tissue_mask_bbox = np.array(f['tissue_mask'][r0:r1, c0:c1])
    classes_bbox = classes_full[r0:r1, c0:c1]
    return data_bbox, tissue_mask_bbox, classes_bbox, r0, c0


def choose_inner_val_crop(crop_paths, seed=42, verbose=True):
    """
    Prebere samo 'classes' (poceni) iz vseh crop-ov, izbere NAJMANJSI crop
    z vsemi 6 razredi kot inner-val (ohrani cim vec train podatkov za
    inner-train). Ce noben nima vseh razredov, vzame najmanjsega.
    """
    info = []
    if verbose:
        print("  Pregled train crop-ov (samo 'classes', poceni branje):")
    for p in crop_paths:
        classes, _ = peek_classes_and_wns(p)
        ann = (classes != -1)
        n = int(ann.sum())
        cls_present = np.unique(classes[ann]) if n > 0 else np.array([])
        has_all = len(cls_present) == NUM_CLASSES
        info.append({"path": p, "n": n, "has_all": has_all, "classes": cls_present})
        marker = "*" if has_all else " "
        if verbose:
            print(f"    {marker} {os.path.basename(p)}: {n:,} anotiranih, razredi={list(cls_present)}")

    with_all = [x for x in info if x["has_all"]]
    if with_all:
        with_all.sort(key=lambda x: x["n"])  # najmanjsi prvi -> ohrani vec za train
        chosen = with_all[0]
    else:
        info_sorted = sorted(info, key=lambda x: x["n"])
        chosen = info_sorted[0]
        if verbose:
            print("  OPOZORILO: noben crop nima vseh 6 razredov. Vzet najmanjsi.")

    if verbose:
        print(f"\n  Izbran inner-val crop: {os.path.basename(chosen['path'])} "
              f"({chosen['n']:,} anotiranih pikslov)")

    inner_val_path   = chosen["path"]
    inner_train_paths = [x["path"] for x in info if x["path"] != inner_val_path]
    return inner_train_paths, inner_val_path


# ---------------------------------------------------------------------------
# Cache za PCA-NEODVISEN del (spec_dense = surov spekter + sosedska povprecja).
# To je najdrazji korak (uniform_filter), a je popolnoma neodvisen od tega,
# kateri PCA (Faza A ali Faza B) se kasneje uporabi -- zato se lahko varno
# predpomni na disk in ponovno uporabi:
#   - znotraj ENEGA teka: vseh 7 inner-train + 1 inner-val crop se v Fazi B
#     spet pojavi (kot del vseh 8 outer-train) -> cache HIT, brez uniform_filter
#   - med VEC teki skripte (npr. spreminjanje --n-ensemble): cel spec_dense
#     ostane na disku, naslednji tek ga samo prebere
# Cache NE hrani surovega bbox platna (lahko je vec GB na velik crop) -- to
# se za PCA transform vedno znova prebere iz HDF5 (poceni, par sekund).
# ---------------------------------------------------------------------------
def _crop_cache_key(path, margin):
    import hashlib
    key_str = f"{os.path.abspath(path)}|margin={margin}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def get_spec_dense_cached(path, scales=NEIGH_SCALES, margin=BBOX_MARGIN,
                          cache_dir=None, label=""):
    """Vrne (coords, y, spec_dense, tissue_mask, H, W, r0, c0). Cache-ira samo
    te (majhne/srednje) izpeljane podatke, ne surovega bbox platna."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _crop_cache_key(path, margin)
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


# ---------------------------------------------------------------------------
# PCA fit na zbirki anotiranih pikslov iz vec crop-ov (uporabi cache, kjer obstaja)
# ---------------------------------------------------------------------------
def fit_pca_pooled(crop_paths, n_pca=N_PCA, seed=42, label="", cache_dir=None):
    print(f"  PCA fit ({label}): zbiranje anotiranih spektrov iz {len(crop_paths)} crop-ov...")
    pooled = []
    for p in crop_paths:
        coords, y, spec_dense, tissue_mask, H, W, r0, c0 = get_spec_dense_cached(
            p, NEIGH_SCALES, BBOX_MARGIN, cache_dir, label=label)
        # prvih SPEC_DIM_RAW stolpcev spec_dense = surov (ne-povprecen) spekter
        X_ann = spec_dense[:, :SPEC_DIM_RAW].copy()
        pooled.append(X_ann)
        print(f"    {os.path.basename(p)}: {len(X_ann):,} spektrov")
    X_pool = np.concatenate(pooled, axis=0)
    del pooled
    print(f"  Skupaj za PCA fit: {len(X_pool):,} spektrov, dim={X_pool.shape[1]}")
    t0 = time.time()
    pca = PCA(n_components=n_pca, random_state=seed)
    pca.fit(X_pool)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%  "
          f"({time.time()-t0:.1f}s)")
    del X_pool
    return pca


# ---------------------------------------------------------------------------
# Procesiranje enega crop-a: cache-iran spec_dense + sveza PCA transformacija
# (PCA transform je poceni/linearen, zato se vedno racuna sproti -- razlikuje
# se med Fazo A in Fazo B).
# ---------------------------------------------------------------------------
def process_crop(path, pca, scales=NEIGH_SCALES, label="", cache_dir=None):
    coords, y, spec_dense, tissue_mask, H, W, r0, c0 = get_spec_dense_cached(
        path, scales, BBOX_MARGIN, cache_dir, label=label)

    t0 = time.time()
    print(f"  [{label}] PCA transform ({H*W:,} pikslov v bbox)...")
    data, _, _, _, _ = read_crop_bbox(path, BBOX_MARGIN)
    flat = data.reshape(-1, data.shape[-1])
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
    """Zdruzi vec procesiranih crop-ov v (samples[N,3], spec_dense[N,D], y[N])."""
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
# Dataset — vec crop-ov, vsak s svojim (padded) platnom
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
        self.padded_pca_by_crop = padded_pca_by_crop  # dict: crop_id -> padded (H+2p,W+2p,N_PCA)
        self.spec_dense = spec_dense
        self.samples = samples  # (N,3): crop_id, r, c (r,c v bbox-lokalnih, ne-padded koordinatah)
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
# Arhitektura: Dual-Stream CNN — identicno v10/v13
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
# Helpers
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
    print(f"  Temperature: T={T:.4f}  {neg_ll(1.0):.5f} -> {neg_ll(T):.5f}")
    return T


def apply_temperature(logits, T):
    s = logits / T
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


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
    """H,W so zdaj bbox-dimenzije (ne 800x1200) — crop je ze celo pravokotno
    platno znotraj bbox-a, ni potreben dodaten bbox trik."""
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
    print(f"  Najboljsa sigma: {best_sigma:.1f} (inner-val ll={best_ll:.5f})")
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
                          output_path, t_opt, sigma, max_epochs, best_epoch,
                          final_epochs, n_ensemble, extra_note=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"TEST_OA={test_oa*100:6.2f}%  TEST_ll={test_ll:.5f}  "
        f"T={t_opt:.4f}  sigma={sigma:.1f}  n_ensemble={n_ensemble}  "
        f"max_ep={max_epochs}(best={best_epoch})  final_ep={final_epochs}"
        f"  -> {output_path}\n"
    ]
    if extra_note:
        lines.append(f"{'':>19}  Opomba: {extra_note}\n")
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("# Rezultati modelov — FTIR klasifikacija tkiva\n")
            f.write(f"# {'-'*90}\n")
        print(f"  -> {RESULTS_FILE} (ustvarjena nova)")
    else:
        print(f"  -> {RESULTS_FILE} (dodana vrstica)")
    with open(RESULTS_FILE, "a") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Trening — Faza A (best-of-N na inner-val) in Faza B (blind)
# ---------------------------------------------------------------------------
def train_single(padded_pca_train, spec_train, samples_train, y_train,
                 val_ds, y_val, device, weights,
                 max_epochs, batch_size, lr, seed, spec_noise):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = MultiCropDataset(padded_pca_train, spec_train, samples_train, y_train,
                                augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = DualStreamCNN(spec_dim=SPEC_DIM).to(device)
    criterion = nn.NLLLoss(weight=weights.to(device))
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


def train_blind(padded_pca_train, spec_train, samples_train, y_train, device, weights,
                final_epochs, batch_size, lr, seed, spec_noise):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = MultiCropDataset(padded_pca_train, spec_train, samples_train, y_train,
                                augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = DualStreamCNN(spec_dim=SPEC_DIM).to(device)
    criterion = nn.NLLLoss(weight=weights.to(device))
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
    global SPEC_DIM_RAW, SPEC_DIM, BBOX_MARGIN

    parser = argparse.ArgumentParser(
        description="Model C Cross-Slide v1: pravi train/test TMA split iz clanka (RAM-varcen, bbox branje)"
    )
    parser.add_argument("--train-dir",  default="FTIR-data/train_preprocessed")
    parser.add_argument("--test-file",  default="FTIR-data/test_preprocessed/test_expanded_crop_preprocessed.hdf5")
    parser.add_argument("--output",     default="modelC_crossSlide_v1_test.npy")
    parser.add_argument("--max-epochs", type=int,   default=15,
                        help="Zgornja meja epoh za Fazo A (best-of-N na inner-val).")
    parser.add_argument("--final-epochs", type=int, default=None,
                        help="Rocna preglasitev epoh za Fazo B (privzeto: mediana "
                             "best_epochs iz Faze A). Uporabno, ce mediana prezre "
                             "osamelce, ki bi radi trenirali dlje (npr. best_epochs "
                             "zelo razprseni).")
    parser.add_argument("--calib-ensemble", type=int, default=4,
                        help="Stevilo modelov za rekalibracijski ensemble (T na "
                             "final_epochs-usklajenem modelu). Manjse od n-ensemble "
                             "za hitrost, T ostane robusten ze pri nekaj modelih.")
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--n-ensemble", type=int,   default=4,
                        help="Privzeto 4 za hiter sanity-check; koncni tek naj bo 12.")
    parser.add_argument("--spec-noise", type=float, default=0.005)
    parser.add_argument("--weight-soften", type=float, default=WEIGHT_SOFTEN)
    parser.add_argument("--bbox-margin", type=int, default=BBOX_MARGIN,
                        help="Margina okoli anotiranih pikslov pri bbox branju (RAM varcevanje).")
    parser.add_argument("--cache-dir",  default="FTIR-data/_cache",
                        help="Predpomnilnik za spec_dense (PCA-neodvisen, najdrazji korak). "
                             "Prazen niz ('') izklopi cache.")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    BBOX_MARGIN = args.bbox_margin
    cache_dir = args.cache_dir if args.cache_dir else None

    # ------------------------------------------------------------------
    print("\n=== 1. Odkrivanje train crop-ov in izbira inner-val ===")
    train_crop_paths = sorted(glob.glob(os.path.join(args.train_dir, "train_crop_*.hdf5")))
    print(f"  Najdenih train crop-ov: {len(train_crop_paths)}")
    for p in train_crop_paths:
        print(f"    {p}")

    _, wns_ref = peek_classes_and_wns(train_crop_paths[0])
    SPEC_DIM_RAW = len(wns_ref)
    SPEC_DIM     = SPEC_DIM_RAW * (1 + len(NEIGH_SCALES))
    print(f"\n  Spektralnih kanalov (SPEC_DIM_RAW): {SPEC_DIM_RAW}  ->  SPEC_DIM={SPEC_DIM}")
    print(f"  Bbox margina (RAM varcevanje): {BBOX_MARGIN} pikslov")

    inner_train_paths, inner_val_path = choose_inner_val_crop(train_crop_paths, seed=args.seed)

    device = get_device()
    n_param = sum(p.numel() for p in DualStreamCNN(spec_dim=SPEC_DIM).parameters()
                 if p.requires_grad)
    print(f"\n  Konfiguracija: Patch {PATCH_SIZE}x{PATCH_SIZE} | PCA({N_PCA}) | TTA 8 | "
          f"Ensemble {args.n_ensemble} | DualStreamCNN ({n_param:,} param)")
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    seeds = list(range(args.seed, args.seed + args.n_ensemble))

    # ==================================================================
    # FAZA A — 7 inner-train crop-ov, 1 inner-val crop (TEST se ne dotakne)
    # ==================================================================
    print(f"\n=== 2. Faza A — PCA fit na {len(inner_train_paths)} inner-train crop-ih ===")
    pca_A = fit_pca_pooled(inner_train_paths, N_PCA, args.seed, label="Faza A", cache_dir=cache_dir)

    print(f"\n=== 3. Faza A — procesiranje crop-ov (en naenkrat, samo bbox) ===")
    inner_train_results, inner_train_ids = [], []
    for i, p in enumerate(inner_train_paths):
        res = process_crop(p, pca_A, label=f"inner-train {i}", cache_dir=cache_dir)
        inner_train_results.append(res)
        inner_train_ids.append(i)
    inner_val_res = process_crop(inner_val_path, pca_A, label="inner-val", cache_dir=cache_dir)

    samples_it, spec_it, y_it = stack_crops(inner_train_results, inner_train_ids)
    padded_it = {cid: pad_pca(res["data_pca"]) for cid, res in zip(inner_train_ids, inner_train_results)}
    print(f"\n  Skupaj inner-train: {len(y_it):,} pikslov iz {len(inner_train_paths)} crop-ov")
    print(f"  Inner-val: {len(inner_val_res['y']):,} pikslov")
    for c in range(NUM_CLASSES):
        nt = (y_it == c).sum(); nv = (inner_val_res["y"] == c).sum()
        print(f"    R{c}: inner-train={nt:6d} ({100*nt/len(y_it):.1f}%)  "
              f"inner-val={nv:6d} ({100*nv/max(len(inner_val_res['y']),1):.1f}%)")

    padded_iv = {0: pad_pca(inner_val_res["data_pca"])}
    samples_iv = np.concatenate([
        np.zeros((len(inner_val_res["y"]), 1), dtype=np.int64), inner_val_res["coords"]
    ], axis=1)
    inner_val_ds = MultiCropDataset(padded_iv, inner_val_res["spec_dense"], samples_iv,
                                    inner_val_res["y"], augment=False)

    weights_A = compute_class_weights(y_it, soften=args.weight_soften)

    print(f"\n=== 4. Faza A — Ensemble ({args.n_ensemble} modelov x do {args.max_epochs} epoh) ===")
    innerval_logits_list, best_epochs = [], []
    for i, seed in enumerate(seeds):
        print(f"\n  -- Faza A model {i+1}/{args.n_ensemble} (seed={seed}) --")
        model, best_epoch = train_single(
            padded_it, spec_it, samples_it, y_it,
            inner_val_ds, inner_val_res["y"], device, weights_A,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, spec_noise=args.spec_noise,
        )
        best_epochs.append(best_epoch)
        print(f"  TTA na inner-val...")
        innerval_logits_list.append(
            get_logits_tta(model, padded_iv, inner_val_res["spec_dense"], samples_iv,
                           inner_val_res["y"], device)
        )

    innerval_logits_avg = np.mean(innerval_logits_list, axis=0)
    median_epochs = int(round(float(np.median(best_epochs))))
    print(f"\n  Best-of-{args.max_epochs} epohe: {best_epochs}")
    print(f"  Mediana: {median_epochs}  (povprecje bi bilo: {round(np.mean(best_epochs))})")

    if args.final_epochs is not None:
        final_epochs = args.final_epochs
        print(f"  final_epochs ROCNO PREGLASEN: {final_epochs} "
              f"(namesto mediane {median_epochs}) — --final-epochs={args.final_epochs}")
    else:
        final_epochs = median_epochs

    print(f"\n=== 4b. Rekalibracijski ensemble ({min(args.calib_ensemble, args.n_ensemble)} "
          f"modelov x {final_epochs} epoh, BREZ peeka — usklajen s final_epochs) ===")
    print(f"  Namen: T mora biti kalibriran na modelu, treniranem ENAKO dolgo kot "
          f"Faza B ({final_epochs} epoh), ne na best-of-N Fazi A ({best_epochs}), "
          f"sicer T ne ustreza dejanski samozavesti Faze B modelov.")
    calib_seeds = seeds[:max(1, min(args.calib_ensemble, args.n_ensemble))]
    calib_logits_list = []
    for i, seed in enumerate(calib_seeds):
        print(f"\n  -- Rekalibracijski model {i+1}/{len(calib_seeds)} (seed={seed}) --")
        model_c = train_blind(
            padded_it, spec_it, samples_it, y_it, device, weights_A,
            final_epochs=final_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, spec_noise=args.spec_noise,
        )
        calib_logits_list.append(
            get_logits_tta(model_c, padded_iv, inner_val_res["spec_dense"], samples_iv,
                           inner_val_res["y"], device)
        )
    calib_logits_avg = np.mean(calib_logits_list, axis=0)
    del calib_logits_list

    print("\n=== 5. Temperature scaling in evaluacija (na inner-val, usklajen model) ===")
    T_opt = find_temperature(calib_logits_avg, inner_val_res["y"])
    innerval_probs = apply_temperature(calib_logits_avg, T_opt)
    innerval_pred  = np.argmax(innerval_probs, axis=1)
    innerval_oa    = accuracy_score(inner_val_res["y"], innerval_pred)
    innerval_ll    = log_loss(inner_val_res["y"], innerval_probs, labels=np.arange(NUM_CLASSES))
    print(f"  INNERVAL OA: {innerval_oa*100:.2f}%  |  INNERVAL ll: {innerval_ll:.5f}")
    print_per_class_table(inner_val_res["y"], innerval_pred, innerval_probs,
                          "Per-class OA in log-loss (rekalibracijski model, inner-val):")

    print("\n=== 6. Faza A — sigma sweep na inner-val ===")
    prior_A = np.bincount(y_it, minlength=NUM_CLASSES).astype(np.float32)
    prior_A /= prior_A.sum()
    sigma_opt = find_best_sigma(
        inner_val_res["H"], inner_val_res["W"], inner_val_res["coords"],
        innerval_probs, inner_val_res["y"], inner_val_res["tissue_mask"], prior_A,
        sigma_choices=SIGMA_CHOICES)

    # sprosti Fazo A pomnilnik pred Fazo B
    del inner_train_results, padded_it, padded_iv, samples_it, spec_it
    del innerval_logits_list

    # ==================================================================
    # FAZA B — vseh 8 crop-ov, blind trening, en dotik s pravim TEST filom
    # ==================================================================
    print(f"\n=== 7. Faza B — PCA fit na vseh {len(train_crop_paths)} train crop-ih ===")
    pca_B = fit_pca_pooled(train_crop_paths, N_PCA, args.seed, label="Faza B", cache_dir=cache_dir)

    print(f"\n=== 8. Faza B — procesiranje vseh train crop-ov (samo bbox) ===")
    outer_train_results, outer_train_ids = [], []
    for i, p in enumerate(train_crop_paths):
        res = process_crop(p, pca_B, label=f"outer-train {i}", cache_dir=cache_dir)
        outer_train_results.append(res)
        outer_train_ids.append(i)

    samples_ot, spec_ot, y_ot = stack_crops(outer_train_results, outer_train_ids)
    padded_ot = {cid: pad_pca(res["data_pca"]) for cid, res in zip(outer_train_ids, outer_train_results)}
    print(f"\n  Skupaj outer-train (Faza B): {len(y_ot):,} pikslov iz {len(train_crop_paths)} crop-ov")

    weights_B = compute_class_weights(y_ot, soften=args.weight_soften)
    del outer_train_results

    print(f"\n=== 9. Faza B — procesiranje TEST datoteke (locen fizicni slajd, samo bbox) ===")
    test_res = process_crop(args.test_file, pca_B, label="TEST", cache_dir=cache_dir)
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
            padded_ot, spec_ot, samples_ot, y_ot, device, weights_B,
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
    test_probs = apply_temperature(
        (test_logits_sum / args.n_ensemble).astype(np.float32), T_opt
    )
    test_pred = np.argmax(test_probs, axis=1)
    test_oa   = accuracy_score(test_res["y"], test_pred)
    test_ll   = log_loss(test_res["y"], test_probs, labels=np.arange(NUM_CLASSES))
    print(f"  TEST OA (pred smoothing): {test_oa*100:.2f}%")
    print(f"  TEST ll (pred smoothing): {test_ll:.5f}")
    print(f"  Primerjava Faza A (inner-val): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Ref clanek CNN (SD, isti split): OA=79.45% +/- 1.25")
    print(f"  Ref clanek SVM (SD, isti split): OA=56.41%")
    print_per_class_table(test_res["y"], test_pred, test_probs,
                          "Per-class OA in log-loss (Faza B, TEST — koncni test):")

    prior_B = np.bincount(y_ot, minlength=NUM_CLASSES).astype(np.float32)
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
    print("\n=== POVZETEK (crossSlide v1) ===")
    print(f"  Train: {len(train_crop_paths)} crop-ov (br1003-br2085b), {len(y_ot):,} pikslov")
    print(f"  Test:  1 datoteka (brc961-br1001), {len(test_res['y']):,} pikslov")
    print(f"  Faza A: max_epochs={args.max_epochs}, best_epochs={best_epochs}, mediana={median_epochs}")
    if args.final_epochs is not None:
        print(f"  Faza B: final_epochs={final_epochs} (ROCNO PREGLASEN), ensemble={args.n_ensemble}, brez peeka")
    else:
        print(f"  Faza B: final_epochs={final_epochs} (=mediana), ensemble={args.n_ensemble}, brez peeka")
    print(f"  T={T_opt:.4f} | sigma={sigma_opt:.1f} | weight_soften={args.weight_soften} | "
          f"bbox_margin={BBOX_MARGIN}")
    print(f"\n  Faza A (inner-val):        OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (TEST, KONCNI):     OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")
    print(f"\n  Primerjava:")
    print(f"    Clanek SVM (isti split):  OA=56.41%")
    print(f"    Clanek CNN (isti split):  OA=79.45% +/- 1.25")
    print(f"    crossSlide v1 (ta tek, n_ensemble={args.n_ensemble}): OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")

    # ------------------------------------------------------------------
    print(f"\n=== 12. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelC_crossSlide_v1",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa_sm, test_ll=test_ll_sm,
        output_path=args.output,
        t_opt=T_opt, sigma=sigma_opt,
        max_epochs=args.max_epochs, best_epoch=final_epochs,
        final_epochs=final_epochs, n_ensemble=args.n_ensemble,
        extra_note=METODOLOGIJA_OPOMBA
    )


if __name__ == "__main__":
    main()
