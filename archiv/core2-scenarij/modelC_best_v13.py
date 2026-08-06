"""
Model C — Best v13  (v10 + 3 majhna popravka, blizu clanku)
================================================================
Osnova: v10 (gnezdena validacija, Core 2 nikoli "pogledan" pred koncem).
Namen: majhne, dobro utemeljene izboljsave, BREZ oddaljevanja od metodologije
clanka (glavni diplomski cilj je kasneje nov cross-slide dataset + PCA/
transformerji — v13 je vmesni korak, ne velik nov eksperiment).

TRI SPREMEMBE glede na v10:

1. Omiljene class weights (namesto ostrih inverse-frequency utezi).
   v10: weight = N / (n_classes * count)          — ekstremno za R3 (redek razred)
   v13: weight = (N / (n_classes * count)) ** 0.5  — sqrt, manj agresivno
   Utemeljitev: ekstremne utezi za R3 lahko destabilizirajo trening podobno
   kot je oversampling (v11) povzrocil overfitting — sqrt je standardna,
   blazja alternativa, ki ostane pri konceptu utezene loss (ne oversamplinga,
   torej se ne oddaljujemo dodatno od ze ugotovljenih tezav).

2. Sigma (Gaussian smoothing) izbrana na inner-val, ne fiksna 1.5.
   Sweep cez vec vrednosti na Fazi A (inner-val), izbere tisto z najnizjim
   log-lossom, uporabi v Fazi B. Post-processing korak ni del clanka
   (clanek ga ne uporablja), zato je to nevtralno glede na zvestobo clanku
   — le naša lastna tehnika bolje nastavljena.

3. Mediana namesto povprecja za dolocitev final_epochs.
   Best-of-N epohe med 12 modeli Faze A so pogosto zelo raztresene
   (npr. [1,4,2,5,5,5,1,11,13,9,4,1]) — povprecje je obcutljivo na osamelce.
   Mediana je robustnejsa ocena "tipicnega" najboljsega epoha.

Vse ostalo identicno v10: DualStreamCNN (302k), SPEC_DIM=748, gnezdena
Core2 validacija (Faza A/B), StandardScaler+PCA, TTA 8x, ensemble 12.
"""

import argparse
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES  = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660
PATCH_SIZE   = 17
N_PCA        = 16
SPEC_DIM_RAW = 187
NEIGH_SCALES = [3, 5, 7]
SPEC_DIM     = SPEC_DIM_RAW * (1 + len(NEIGH_SCALES))
RESULTS_FILE = "rezultati_report.txt"

SIGMA_CHOICES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_SOFTEN = 0.5     #<- spremenjeno iz 0.5

METODOLOGIJA_OPOMBA = (
    "v13 = v10 + 3 majhna popravka: (1) omiljene class weights "
    f"(** {WEIGHT_SOFTEN} namesto ** 1.0 — manj ekstremne za redke razrede), "
    "(2) sigma izbrana na inner-val (sweep), ne fiksna 1.5, "
    "(3) mediana namesto povprecja za final_epochs (robustnejsa na osamelce "
    "v best_epochs). Ostalo identicno v10 (gnezdena Core2 validacija)."
)


# ---------------------------------------------------------------------------
# Nalaganje
# ---------------------------------------------------------------------------
def load_data(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        data        = np.array(f["data"],        dtype=np.float32)
        wns         = np.array(f["wns"])
        tissue_mask = np.array(f["tissue_mask"])
        classes     = np.array(f["classes"])
    return data, wns, tissue_mask, classes


def make_prediction_crop_mask(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1] = True
    return mask


# ---------------------------------------------------------------------------
# Splosen prostorski split (KMeans) — identicno v10
# ---------------------------------------------------------------------------
def make_spatial_split(tissue_mask, classes, exclude_mask, n_cores, seed=42,
                        verbose=True, label="core"):
    usable = (classes != -1) & (~exclude_mask)
    total  = int(usable.sum())
    coords = np.argwhere(usable)
    flat_cls = classes[coords[:, 0], coords[:, 1]]

    print(f"  KMeans(k={n_cores}, seed={seed}) na {len(coords):,} pikslih ({label})...")
    km = KMeans(n_clusters=n_cores, random_state=seed, n_init=10)
    core_ids = km.fit_predict(coords)

    print(f"  Pregled {label}-ov:")
    candidates = []
    for c_id in range(n_cores):
        idx = (core_ids == c_id)
        n   = int(idx.sum())
        cls = np.unique(flat_cls[idx])
        has_all = len(cls) == NUM_CLASSES
        candidates.append((c_id, n, has_all))
        marker = "*" if has_all else " "
        print(f"    {marker} {label} {c_id}: {n:5,} pikslov, razredi={list(cls)}")

    with_all = [(c_id, n) for c_id, n, has_all in candidates if has_all]
    with_all.sort(key=lambda x: x[1], reverse=True)
    if not with_all:
        candidates.sort(key=lambda x: x[1], reverse=True)
        held_id = candidates[0][0]
        print(f"  OPOZORILO: noben {label} nima vseh razredov. Vzet najvecji.")
    else:
        held_id = with_all[0][0]

    held_idx    = (core_ids == held_id)
    held_coords = coords[held_idx]
    held_mask   = np.zeros(classes.shape, dtype=bool)
    held_mask[held_coords[:, 0], held_coords[:, 1]] = True
    rest_mask   = usable & ~held_mask

    if verbose:
        held_n = int(held_mask.sum())
        print(f"\n  Held-out ({label} {held_id}): {held_n:,} pikslov ({100*held_n/total:.1f}%)")
        print(f"  Ostalo:                {rest_mask.sum():,} pikslov ({100*rest_mask.sum()/total:.1f}%)")

    return rest_mask, held_mask


# ---------------------------------------------------------------------------
# Neighbourhood MEAN features — identicno v10
# ---------------------------------------------------------------------------
def extract_spec_features(flat_sc, H, W, coords, scales=NEIGH_SCALES):
    D        = flat_sc.shape[1]
    spec_map = flat_sc.reshape(H, W, D)
    r, c     = coords[:, 0], coords[:, 1]
    parts    = [flat_sc[r * W + c].copy()]
    for scale in scales:
        t0 = time.time()
        print(f"    mean filter scale={scale}...", end=" ", flush=True)
        mean_map = uniform_filter(spec_map, size=[scale, scale, 1], mode='reflect')
        parts.append(mean_map[r, c].copy())
        del mean_map
        print(f"{time.time()-t0:.1f}s")
    out = np.concatenate(parts, axis=1).astype(np.float32)
    print(f"    -> shape={out.shape}  ({out.nbytes/1e6:.0f} MB)")
    return out


# ---------------------------------------------------------------------------
# Preprocessing — identicno v10
# ---------------------------------------------------------------------------
def build_preprocessed(data, fit_mask, n_pca=N_PCA, seed=42):
    H, W, D = data.shape
    flat    = data.reshape(-1, D)
    X_fit_raw = flat[fit_mask.ravel()]
    print(f"  StandardScaler fit na {len(X_fit_raw):,} pikslih...")
    t0 = time.time()
    scaler   = StandardScaler()
    X_fit_sc = scaler.fit_transform(X_fit_raw)
    print(f"    -> {time.time()-t0:.1f}s")
    print(f"  PCA({n_pca}) fit...")
    pca = PCA(n_components=n_pca, random_state=seed)
    pca.fit(X_fit_sc)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"  Transform vseh {H*W:,} pikslov...")
    t0 = time.time()
    flat_sc  = scaler.transform(flat).astype(np.float32)
    flat_pca = pca.transform(flat_sc).astype(np.float32)
    data_pca = flat_pca.reshape(H, W, n_pca)
    print(f"    -> {time.time()-t0:.1f}s")
    return data_pca, flat_sc, scaler, pca


# ---------------------------------------------------------------------------
# Dataset — identicno v10
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


class DualStreamDataset(Dataset):
    def __init__(self, data_pca, spec_dense, coords, labels,
                 patch_size=PATCH_SIZE, augment=False, tta_idx=-1, spec_noise=0.0):
        self.pad        = patch_size // 2
        self.coords     = coords
        self.labels     = labels
        self.augment    = augment
        self.tta_idx    = tta_idx
        self.spec_noise = spec_noise
        self.spec_dense = spec_dense
        self.padded = np.pad(
            data_pca,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='reflect'
        ).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r, c   = self.coords[idx]
        rp, cp = r + self.pad, c + self.pad
        patch  = self.padded[rp-self.pad:rp+self.pad+1,
                              cp-self.pad:cp+self.pad+1]
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
# Arhitektura: Dual-Stream CNN — identicno v10
# ---------------------------------------------------------------------------
class DualStreamCNN(nn.Module):
    def __init__(self, n_channels=N_PCA, spec_dim=SPEC_DIM,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
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
    """
    soften=1.0 -> standardna inverse-frequency utez (v10)
    soften=0.5 -> sqrt(inverse frequency) — blazja, manj ekstremna za redke razrede (v13)
    """
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    raw     = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    weights = raw ** soften
    print(f"  Class weights (soften={soften}): {[f'{w:.2f}' for w in weights]}  "
          f"(nesoftened bi bilo: {[f'{w:.2f}' for w in raw]})")
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
def get_logits_tta(model, data_pca, spec_dense, coords, labels, device, batch_size=512):
    logits_sum = None
    for aug_idx in range(8):
        ds = DualStreamDataset(data_pca, spec_dense, coords, labels,
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


def build_bbox_prob_map(coords, probs, tissue_mask, fallback_prior):
    """Postavi probs (N, C) na (r,c) koordinate v bounding-box mapo za smoothing."""
    r_min = int(coords[:, 0].min()); r_max = int(coords[:, 0].max())
    c_min = int(coords[:, 1].min()); c_max = int(coords[:, 1].max())
    bbox_h = r_max - r_min + 1; bbox_w = c_max - c_min + 1
    prob_map = np.tile(fallback_prior, (bbox_h * bbox_w, 1)).reshape(bbox_h, bbox_w, NUM_CLASSES)
    for (r, c), p in zip(coords, probs):
        prob_map[r - r_min, c - c_min] = p
    tissue_2d = tissue_mask[r_min:r_max+1, c_min:c_max+1]
    return prob_map, tissue_2d, (r_min, c_min)


def find_best_sigma(coords, probs, y_true, tissue_mask, fallback_prior,
                    sigma_choices=SIGMA_CHOICES):
    """
    Sweep sigma vrednosti na (inner-val) probs, izbere tisto z najnizjim log-lossom.
    Uporabljeno SAMO v Fazi A (na inner-val), rezultat se prenese v Fazo B.
    """
    prob_map, tissue_2d, (r_min, c_min) = build_bbox_prob_map(
        coords, probs, tissue_mask, fallback_prior)

    print(f"  {'sigma':>6}  {'log-loss':>10}")
    best_sigma, best_ll = 0.0, None
    for sigma in sigma_choices:
        if sigma > 0:
            smoothed = gaussian_smooth_probs(prob_map, tissue_2d, sigma)
            m = prob_map.copy()
            m[tissue_2d] = smoothed[tissue_2d]
        else:
            m = prob_map

        probs_at_coords = np.array([m[r - r_min, c - c_min] for r, c in coords])
        probs_at_coords = np.clip(probs_at_coords, 1e-7, 1.0)
        probs_at_coords /= probs_at_coords.sum(axis=1, keepdims=True)
        ll = log_loss(y_true, probs_at_coords, labels=np.arange(NUM_CLASSES))
        marker = ""
        if best_ll is None or ll < best_ll:
            best_ll = ll; best_sigma = sigma
            marker = " *"
        print(f"  {sigma:>6.1f}  {ll:>10.5f}{marker}")

    print(f"  Najboljsa sigma: {best_sigma:.1f} (inner-val ll={best_ll:.5f})")
    return best_sigma


def print_per_class_table(y_true, y_pred, probs, title="Per-class rezultati"):
    print(f"\n  {title}")
    print(f"  {'Razred':>8}  {'N':>6}  {'OA':>8}  {'Log-loss':>10}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}")
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"  {c:>8}  {'─':>6}  {'─':>8}  {'─':>10}")
            continue
        oa_c = accuracy_score(y_true[mask], y_pred[mask])
        ll_c = log_loss(y_true[mask], probs[mask], labels=np.arange(NUM_CLASSES))
        print(f"  {c:>8}  {mask.sum():>6}  {oa_c*100:>7.2f}%  {ll_c:>10.5f}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}")
    oa_tot = accuracy_score(y_true, y_pred)
    ll_tot = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))
    print(f"  {'SKUPAJ':>8}  {len(y_true):>6}  {oa_tot*100:>7.2f}%  {ll_tot:>10.5f}")


def write_results_report(model_name, innerval_oa, innerval_ll, test_oa, test_ll,
                          output_path, t_opt, sigma, max_epochs, best_epoch,
                          final_epochs, extra_note=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"CORE2_OA={test_oa*100:6.2f}%  CORE2_ll={test_ll:.5f}  "
        f"T={t_opt:.4f}  sigma={sigma:.1f}  "
        f"max_ep={max_epochs}(best={best_epoch})  final_ep={final_epochs}"
        f"  -> {output_path}\n"
    ]
    if extra_note:
        lines.append(f"{'':>19}  Opomba: {extra_note}\n")

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("# Rezultati modelov — FTIR klasifikacija tkiva\n")
            f.write(f"# {'─'*90}\n")
        print(f"  -> {RESULTS_FILE} (ustvarjena nova)")
    else:
        print(f"  -> {RESULTS_FILE} (dodana vrstica)")

    with open(RESULTS_FILE, "a") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Trening enega modela — FIKSNE EPOHE, brez early stopping, best-of-N na val_ds
# ---------------------------------------------------------------------------
def train_single(data_pca, train_spec, val_ds, y_train, y_val,
                 train_coords, device, weights,
                 max_epochs, batch_size, lr, seed, spec_noise):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = DualStreamDataset(data_pca, train_spec, train_coords, y_train,
                                  augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = DualStreamCNN().to(device)
    criterion = nn.NLLLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_loss, best_state, best_epoch = float('inf'), None, 1
    print(f"  {'Ep':>4}  {'Train ll':>10}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}")
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
        lr_now = optimizer.param_groups[0]['lr']

        val_logits = get_logits_array(model, val_ds, device)
        s = val_logits - val_logits.max(axis=1, keepdims=True)
        e = np.exp(s); val_probs = e / e.sum(axis=1, keepdims=True)
        val_oa = accuracy_score(y_val, np.argmax(val_probs, axis=1))
        val_ll = log_loss(y_val, val_probs, labels=np.arange(NUM_CLASSES))
        marker = " *" if val_ll < best_val_loss else "  "
        print(f"  {epoch:>4}  {train_loss:>10.5f}  {val_oa*100:>8.2f}%  "
              f"{val_ll:>9.5f}{marker}  {lr_now:>9.2e}")

        if val_ll < best_val_loss:
            best_val_loss = val_ll
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch

    model.load_state_dict(best_state)
    print(f"  Treniran v {time.time()-t0:.1f}s | best ep={best_epoch}, val_ll={best_val_loss:.5f}")
    return model, best_epoch


def train_blind(data_pca, train_spec, train_coords, y_train, device, weights,
                final_epochs, batch_size, lr, seed, spec_noise):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = DualStreamDataset(data_pca, train_spec, train_coords, y_train,
                                  augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = DualStreamCNN().to(device)
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
    parser = argparse.ArgumentParser(
        description="Model C Best v13: v10 + omiljene utezi + sigma sweep + mediana"
    )
    parser.add_argument("--input",           default="image1-competition.hdf5")
    parser.add_argument("--output",          default="modelC_best_v13_core2.npy")
    parser.add_argument("--max-epochs",      type=int,   default=15,
                        help="Zgornja meja epoh za Fazo A (best-of-N na inner-val).")
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--n-ensemble",      type=int,   default=12)
    parser.add_argument("--spec-noise",      type=float, default=0.005)
    parser.add_argument("--weight-soften",   type=float, default=WEIGHT_SOFTEN,
                        help="Eksponent za omilitev class weights (1.0=v10, 0.5=v13).")
    parser.add_argument("--n-cores",         type=int,   default=6)
    parser.add_argument("--n-inner-cores",   type=int,   default=5)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, wns, tissue_mask, classes = load_data(args.input)
    H, W, _ = data.shape
    print(f"  data: {data.shape} | Anotiranih: {(classes != -1).sum()}")
    prediction_crop_mask = make_prediction_crop_mask(H, W)
    device = get_device()
    n_param = sum(p.numel() for p in DualStreamCNN().parameters() if p.requires_grad)
    print(f"\n  Konfiguracija (v13 — v10 + 3 popravki):")
    print(f"    SPEC_DIM={SPEC_DIM}: raw(187) + mean@{NEIGH_SCALES}")
    print(f"    Patch {PATCH_SIZE}x{PATCH_SIZE} | PCA({N_PCA}) | TTA 8 | Ensemble {args.n_ensemble}")
    print(f"    max_epochs={args.max_epochs} (Faza A, best-of-N) | DualStreamCNN ({n_param:,} param)")
    print(f"    weight_soften={args.weight_soften} | sigma: sweep {SIGMA_CHOICES} na inner-val")
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    # ------------------------------------------------------------------
    print("\n=== 2. Zunanji split: 5 krogcev (outer-train) | Core 2 (TEST) ===")
    outer_train_mask, test_mask = make_spatial_split(
        tissue_mask, classes, prediction_crop_mask,
        n_cores=args.n_cores, seed=args.seed, label="core")
    test_coords = np.argwhere(test_mask)
    y_test      = classes[test_mask].astype(np.int64)

    print("\n=== 3. Notranji split: 4 (inner-train) | 1 (inner-val), znotraj outer-train ===")
    exclude_for_inner = prediction_crop_mask | test_mask
    inner_train_mask, inner_val_mask = make_spatial_split(
        tissue_mask, classes, exclude_for_inner,
        n_cores=args.n_inner_cores, seed=args.seed, label="subcore")

    inner_train_coords = np.argwhere(inner_train_mask)
    inner_val_coords   = np.argwhere(inner_val_mask)
    y_inner_train = classes[inner_train_mask].astype(np.int64)
    y_inner_val   = classes[inner_val_mask].astype(np.int64)

    print(f"\n  Porazdelitev razredov (Faza A: inner-train / inner-val):")
    for c in range(NUM_CLASSES):
        nt = (y_inner_train == c).sum(); nv = (y_inner_val == c).sum()
        print(f"    R{c}: inner-train={nt:5d} ({100*nt/len(y_inner_train):.1f}%)  "
              f"inner-val={nv:5d} ({100*nv/max(len(y_inner_val),1):.1f}%)")

    outer_train_coords = np.argwhere(outer_train_mask)
    y_outer_train = classes[outer_train_mask].astype(np.int64)

    # ==================================================================
    # FAZA A — iskanje hiperparametrov (Core 2 se NE dotakne)
    # ==================================================================
    print("\n=== 4. Faza A — Preprocessing (fit na 4 inner-train krogcih) ===")
    data_pca_A, flat_sc_A, scaler_A, pca_A = build_preprocessed(
        data, inner_train_mask, N_PCA, args.seed)
    weights_A = compute_class_weights(y_inner_train, soften=args.weight_soften)

    print(f"\n=== 5. Faza A — Neighbourhood features ({SPEC_DIM} dim) ===")
    print(f"  Inner-train ({len(inner_train_coords):,} pikslov):")
    inner_train_spec = extract_spec_features(flat_sc_A, H, W, inner_train_coords)
    print(f"  Inner-val ({len(inner_val_coords):,} pikslov):")
    inner_val_spec   = extract_spec_features(flat_sc_A, H, W, inner_val_coords)
    inner_val_ds = DualStreamDataset(data_pca_A, inner_val_spec, inner_val_coords,
                                      y_inner_val, augment=False)

    print(f"\n=== 6. Faza A — Ensemble ({args.n_ensemble} modelov x do {args.max_epochs} epoh) ===")
    print(f"  Best-of-N na INNER-VAL (Core 2 ni vpleten).")
    seeds        = list(range(args.seed, args.seed + args.n_ensemble))
    innerval_logits_list = []
    best_epochs  = []
    for i, seed in enumerate(seeds):
        print(f"\n  ── Faza A model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        model, best_epoch = train_single(
            data_pca=data_pca_A, train_spec=inner_train_spec,
            val_ds=inner_val_ds, y_train=y_inner_train, y_val=y_inner_val,
            train_coords=inner_train_coords, device=device, weights=weights_A,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, spec_noise=args.spec_noise,
        )
        best_epochs.append(best_epoch)
        print(f"  TTA na inner-val...")
        innerval_logits_list.append(
            get_logits_tta(model, data_pca_A, inner_val_spec, inner_val_coords,
                           y_inner_val, device)
        )

    innerval_logits_avg = np.mean(innerval_logits_list, axis=0)
    # v13: mediana namesto povprecja — robustnejsa na osamelce v best_epochs
    final_epochs = int(round(float(np.median(best_epochs))))
    print(f"\n  Best-of-{args.max_epochs} epohe: {best_epochs}")
    print(f"  Mediana: {final_epochs}  (povprecje bi bilo: {round(np.mean(best_epochs))})")

    print("\n=== 7. Faza A — Temperature scaling (na inner-val) ===")
    T_opt = find_temperature(innerval_logits_avg, y_inner_val)

    print("\n=== 8. Faza A — evaluacija na inner-val ===")
    innerval_probs = apply_temperature(innerval_logits_avg, T_opt)
    innerval_pred  = np.argmax(innerval_probs, axis=1)
    innerval_oa    = accuracy_score(y_inner_val, innerval_pred)
    innerval_ll    = log_loss(y_inner_val, innerval_probs, labels=np.arange(NUM_CLASSES))
    print(f"  INNERVAL OA: {innerval_oa*100:.2f}%")
    print(f"  INNERVAL ll: {innerval_ll:.5f}")
    print_per_class_table(y_inner_val, innerval_pred, innerval_probs,
                          "Per-class OA in log-loss (Faza A, inner-val):")

    print("\n=== 9. Faza A — sigma sweep (Gaussian smoothing) na inner-val ===")
    prior_A = np.bincount(y_inner_train, minlength=NUM_CLASSES).astype(np.float32)
    prior_A /= prior_A.sum()
    sigma_opt = find_best_sigma(
        inner_val_coords, innerval_probs, y_inner_val, tissue_mask, prior_A,
        sigma_choices=SIGMA_CHOICES)

    # ==================================================================
    # FAZA B — finalni model, edini dotik s Core 2
    # ==================================================================
    print(f"\n=== 10. Faza B — Preprocessing (fit na vseh 5 outer-train krogcih) ===")
    data_pca_B, flat_sc_B, scaler_B, pca_B = build_preprocessed(
        data, outer_train_mask, N_PCA, args.seed)
    weights_B = compute_class_weights(y_outer_train, soften=args.weight_soften)

    print(f"\n=== 11. Faza B — Neighbourhood features ({SPEC_DIM} dim) ===")
    print(f"  Outer-train ({len(outer_train_coords):,} pikslov):")
    outer_train_spec = extract_spec_features(flat_sc_B, H, W, outer_train_coords)
    print(f"  Core 2 / test ({len(test_coords):,} pikslov):")
    test_spec = extract_spec_features(flat_sc_B, H, W, test_coords)

    print(f"\n=== 12. Faza B — Ensemble ({args.n_ensemble} modelov x {final_epochs} epoh, brez peeka) ===")
    print(f"  Core 2 se prvic dotakne SELE po koncanem treningu (TTA napoved).")
    test_logits_sum = np.zeros((len(test_coords), NUM_CLASSES), dtype=np.float64)
    for i, seed in enumerate(seeds):
        print(f"\n  ── Faza B model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        model_b = train_blind(
            data_pca=data_pca_B, train_spec=outer_train_spec,
            train_coords=outer_train_coords, y_train=y_outer_train,
            device=device, weights=weights_B,
            final_epochs=final_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, spec_noise=args.spec_noise,
        )
        print(f"  TTA na Core 2 (prvic in edinkrat)...")
        test_logits_sum += get_logits_tta(
            model_b, data_pca_B, test_spec, test_coords, y_test, device
        ).astype(np.float64)

    # ------------------------------------------------------------------
    print("\n=== 13. KONCNA evaluacija na Core 2 (edini dotik) ===")
    test_probs = apply_temperature(
        (test_logits_sum / args.n_ensemble).astype(np.float32), T_opt
    )
    test_pred = np.argmax(test_probs, axis=1)
    test_oa   = accuracy_score(y_test, test_pred)
    test_ll   = log_loss(y_test, test_probs, labels=np.arange(NUM_CLASSES))
    print(f"  CORE2 OA (pred smoothing): {test_oa*100:.2f}%")
    print(f"  CORE2 ll (pred smoothing): {test_ll:.5f}")
    print(f"  Primerjava Faza A (inner-val): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Ref clanek SD (cross-slide):   OA=56.41%")
    print_per_class_table(y_test, test_pred, test_probs,
                          "Per-class OA in log-loss (Faza B, Core 2 — koncni test):")

    # Bounding box Core 2 za shranjevanje + Gaussian smoothing (sigma_opt iz Faze A)
    prior_B = np.bincount(y_outer_train, minlength=NUM_CLASSES).astype(np.float32)
    prior_B /= prior_B.sum()
    prob_map, test_tissue_2d, (r_min, c_min) = build_bbox_prob_map(
        test_coords, test_probs, tissue_mask, prior_B)

    if sigma_opt > 0:
        smoothed  = gaussian_smooth_probs(prob_map, test_tissue_2d, sigma_opt)
        final_map = prob_map.copy()
        final_map[test_tissue_2d] = smoothed[test_tissue_2d]
    else:
        final_map = prob_map

    final_map = np.clip(final_map, 1e-7, 1.0)
    final_map /= final_map.sum(axis=-1, keepdims=True)

    # Po-smoothing OA/ll na Core 2 (dejanska koncna stevilka z izbrano sigma)
    smoothed_at_coords = np.array([
        final_map[r - r_min, c - c_min] for r, c in test_coords
    ])
    smoothed_at_coords = np.clip(smoothed_at_coords, 1e-7, 1.0)
    smoothed_at_coords /= smoothed_at_coords.sum(axis=1, keepdims=True)
    test_pred_sm = np.argmax(smoothed_at_coords, axis=1)
    test_oa_sm   = accuracy_score(y_test, test_pred_sm)
    test_ll_sm   = log_loss(y_test, smoothed_at_coords, labels=np.arange(NUM_CLASSES))
    print(f"\n  CORE2 OA (po smoothing, sigma={sigma_opt:.1f}): {test_oa_sm*100:.2f}%")
    print(f"  CORE2 ll (po smoothing, sigma={sigma_opt:.1f}): {test_ll_sm:.5f}")

    np.save(args.output, final_map.astype(np.float32))
    print(f"\n  Shranjeno: {args.output}  shape={final_map.shape}")
    print(f"  (bbox Core 2: vrstice {r_min}-{r_min+final_map.shape[0]-1}, "
          f"stolpci {c_min}-{c_min+final_map.shape[1]-1})")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (v13) ===")
    print(f"  Zunanji split: 5 krogcev (outer-train) | Core 2 (TEST)")
    print(f"  Notranji split: 4 (inner-train) | 1 (inner-val)")
    print(f"  Faza A: max_epochs={args.max_epochs}, best_epochs={best_epochs}, "
          f"mediana={final_epochs}")
    print(f"  Faza B: final_epochs={final_epochs}, brez peeka, en dotik s Core 2")
    print(f"  T={T_opt:.4f} | sigma={sigma_opt:.1f} (izbrana na inner-val) | "
          f"weight_soften={args.weight_soften}")
    print(f"\n  Faza A (inner-val, hiperparametri): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (Core 2, pred smoothing):     OA={test_oa*100:.2f}%  ll={test_ll:.5f}")
    print(f"  Faza B (Core 2, KONCNI, po smoothing): OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")
    print(f"\n  Primerjava:")
    print(f"    v10 (osnova):        OA=73.82%  ll=0.84206")
    print(f"    v13 (ta tek):        OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")
    print(f"    Clanek SD (cross-slide, tezja naloga): OA=56.41% (SVM) / 79.45% (CNN)")

    # ------------------------------------------------------------------
    print(f"\n=== 14. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelC_best_v13",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa_sm, test_ll=test_ll_sm,
        output_path=args.output,
        t_opt=T_opt, sigma=sigma_opt,
        max_epochs=args.max_epochs,
        best_epoch=final_epochs,
        final_epochs=final_epochs,
        extra_note=METODOLOGIJA_OPOMBA
    )


if __name__ == "__main__":
    main()
