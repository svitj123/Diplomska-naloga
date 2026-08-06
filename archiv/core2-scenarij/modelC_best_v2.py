"""
Model C — Best v2
=================
Spremembe glede na modelC_best.py:

  1. Neighbourhood spectral features:
     Spectral MLP zdaj dobiva [original(187) + 3×3 mean(187) + 5×5 mean(187)] = 561 dim.
     Neighbourhood mean zajame lokalni kemijski kontekst — Vid je dokazal da deluje
     (Final=0.452 samo z neighbourhood MLP brez CNN-ja).

  2. Ensemble: 7 modelov (bilo 5).

  3. Final epochs min = 15 (bilo ~7 — premalo za konvergenco na 40k pikslih).

  4. DualStreamDataset: sprejema pre-ekstrahirane dense spektralne matrike
     namesto flat_spec[flat_idx] — omogoča neighbourhood features brez huge flat arrays.

Vse ostalo je enako kot v modelC_best.py:
  - Patch 17×17 z reflect paddingom
  - StandardScaler + PCA(16)
  - DualStreamCNN arhitektura (spatial CNN 64 feat + spectral MLP 64 feat)
  - TTA 8 D4 augmentacij
  - Temperature scaling
  - Gaussian smoothing sigma=1.5
"""

import argparse
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from scipy import ndimage as ndi

NUM_CLASSES = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660
PATCH_SIZE  = 17
N_PCA       = 16
SPEC_DIM_RAW = 187

# Neighbourhood scales za spectral stream
NEIGH_SCALES = [3, 5]
# SPEC_DIM = raw(187) + mean_3(187) + mean_5(187) = 561
SPEC_DIM = SPEC_DIM_RAW * (1 + len(NEIGH_SCALES))


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
# Prostorski 2-way split (80/20) — greedy po komponentah
# ---------------------------------------------------------------------------
def make_spatial_two_way_split(tissue_mask, classes, prediction_crop_mask,
                                val_fraction=0.20, min_size=20, verbose=True):
    usable = (classes != -1) & (~prediction_crop_mask)
    total  = int(usable.sum())

    labeled, _ = ndi.label(tissue_mask)
    cids = np.unique(labeled[usable])
    cids = cids[cids != 0]
    comps = []
    for cid in cids:
        mask = (labeled == cid) & usable
        n = int(mask.sum())
        if n >= min_size:
            comps.append((cid, n))

    comps = sorted(comps, key=lambda x: x[1], reverse=True)
    target_val = int(val_fraction * total)
    val_cids, train_cids = [], []
    running = 0
    toggle  = True

    for cid, cnt in comps:
        if running < target_val and toggle:
            val_cids.append(cid)
            running += cnt
        else:
            train_cids.append(cid)
        toggle = not toggle

    train_mask = np.isin(labeled, train_cids) & usable
    val_mask   = np.isin(labeled, val_cids)   & usable

    if verbose:
        print(f"  Train: {train_mask.sum():,} ({100*train_mask.sum()/total:.1f}%)")
        print(f"  Val:   {val_mask.sum():,}   ({100*val_mask.sum()/total:.1f}%)")

    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Neighbourhood spectral features
# ---------------------------------------------------------------------------
def extract_spec_features(flat_sc, H, W, coords, scales=NEIGH_SCALES):
    """
    Za vsak piksel vrne [original_spectrum(187), mean_scale1(187), mean_scale2(187), ...]
    flat_sc:  (H*W, 187) float32 — StandardScaler normalized spectra (ves H×W)
    coords:   (N, 2) piksli za katere računamo
    Returns:  (N, 187*(1+len(scales))) float32

    Spomin: za vsak scale začasno ustvari (H, W, 187) float32 = 908 MB.
    Peak: ~1.8 GB (flat_sc + mean_map). Po ekstrakciji sprosti.
    """
    D       = flat_sc.shape[1]  # 187
    spec_map = flat_sc.reshape(H, W, D)  # view
    r, c    = coords[:, 0], coords[:, 1]

    parts = [flat_sc[r * W + c].copy()]  # (N, 187) — original

    for scale in scales:
        t0 = time.time()
        print(f"    uniform_filter scale={scale}...", end=" ", flush=True)
        # uniform_filter z reflect modom — primerno za robove
        mean_map = uniform_filter(spec_map, size=[scale, scale, 1], mode='reflect')
        parts.append(mean_map[r, c].copy())  # (N, 187)
        del mean_map
        print(f"{time.time()-t0:.1f}s")

    return np.concatenate(parts, axis=1).astype(np.float32)  # (N, 187*(1+len(scales)))


# ---------------------------------------------------------------------------
# Preprocessing: StandardScaler + PCA
# ---------------------------------------------------------------------------
def build_preprocessed(data, train_mask, n_pca=N_PCA, seed=42):
    """
    Fit StandardScaler + PCA na train pikslih.
    Returns:
      data_pca:  (H, W, n_pca) — za patch CNN
      flat_sc:   (H*W, 187)    — za neighbourhood extraction
      scaler, pca
    """
    H, W, D = data.shape
    flat    = data.reshape(-1, D)

    X_train_raw = flat[train_mask.ravel()]
    print(f"  StandardScaler fit na {len(X_train_raw):,} train pikslih...")
    t0 = time.time()
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train_raw)
    print(f"    → {time.time()-t0:.1f}s")

    print(f"  PCA({n_pca}) fit...")
    pca         = PCA(n_components=n_pca, random_state=seed)
    pca.fit(X_train_sc)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    print(f"  Transform vseh {H*W:,} pikslov...")
    t0 = time.time()
    flat_sc   = scaler.transform(flat).astype(np.float32)  # (H*W, 187)
    flat_pca  = pca.transform(flat_sc).astype(np.float32)
    data_pca  = flat_pca.reshape(H, W, n_pca)
    print(f"    → {time.time()-t0:.1f}s")

    return data_pca, flat_sc, scaler, pca


# ---------------------------------------------------------------------------
# Dataset
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
    """
    Vrne (patch, spectrum_extended, label).
    spec_dense: (N, SPEC_DIM) — pre-ekstrahi neighbourhood features, indeksirani po
                poziciji v datasetu (ne po flat pikslu).
    """

    def __init__(self, data_pca, spec_dense, coords, labels,
                 patch_size=PATCH_SIZE, augment=False, tta_idx=-1, spec_noise=0.0):
        self.pad        = patch_size // 2
        self.coords     = coords
        self.labels     = labels
        self.augment    = augment
        self.tta_idx    = tta_idx
        self.spec_noise = spec_noise
        self.spec_dense = spec_dense  # (N, SPEC_DIM)

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
            # Perturbira samo original del spektra (prvih 187 dimenzij)
            noise = torch.zeros_like(spectrum_t)
            noise[:SPEC_DIM_RAW] = torch.randn(SPEC_DIM_RAW) * self.spec_noise
            spectrum_t = spectrum_t + noise

        return patch_t, spectrum_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Arhitektura: Dual-Stream CNN (enaka kot v1, samo SPEC_DIM = 561)
# ---------------------------------------------------------------------------
class DualStreamCNN(nn.Module):
    """
    Spatial: 17×17×16 PCA patch → CNN → 64 feat
    Spectral: 561-dim [raw+3×3mean+5×5mean] → MLP → 64 feat
    Fusion: 128 → 6 razredov
    """

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

        # Spectral MLP: zdaj 561 dim vhod (was 187)
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


def compute_class_weights(y):
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
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
    """TTA: povprečenje logitov čez 8 D4 simetrij."""
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
    print(f"  Temperature: T={T:.4f}  {neg_ll(1.0):.5f} → {neg_ll(T):.5f}")
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


# ---------------------------------------------------------------------------
# Trening enega modela
# ---------------------------------------------------------------------------
def train_single(data_pca, train_spec, val_ds, y_train, y_val,
                 train_coords, device, weights,
                 epochs, batch_size, lr, patience, seed, spec_noise):
    torch.manual_seed(seed); random.seed(seed)

    train_ds = DualStreamDataset(data_pca, train_spec, train_coords, y_train,
                                  augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model     = DualStreamCNN().to(device)
    criterion = nn.NLLLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss, best_state, best_epoch = float('inf'), None, epochs
    patience_counter = 0

    print(f"  {'Ep':>4}  {'Train ll':>10}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
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
        e = np.exp(s)
        val_probs = e / e.sum(axis=1, keepdims=True)
        val_oa = accuracy_score(y_val, np.argmax(val_probs, axis=1))
        val_ll = log_loss(y_val, val_probs, labels=np.arange(NUM_CLASSES))

        print(f"  {epoch:>4}  {train_loss:>10.5f}  {val_oa*100:>8.2f}%  "
              f"{val_ll:>9.5f}  {lr_now:>9.2e}")

        if val_ll < best_val_loss - 1e-6:
            best_val_loss    = val_ll
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch       = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping: epoha {epoch} (best={best_val_loss:.5f}, ep={best_epoch})")
                break

    model.load_state_dict(best_state)
    print(f"  Treniran v {time.time()-t0:.1f}s | ep={best_epoch}, val_ll={best_val_loss:.5f}")
    return model, best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model C Best v2: dual-stream + neighbourhood spectral features"
    )
    parser.add_argument("--input",      default="image1-competition.hdf5")
    parser.add_argument("--output",     default="modelC_best_v2.npy")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--n-ensemble", type=int,   default=7)
    parser.add_argument("--spec-noise", type=float, default=0.005)
    parser.add_argument("--val-frac",   type=float, default=0.20)
    parser.add_argument("--sigma",      type=float, default=1.5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, wns, tissue_mask, classes = load_data(args.input)
    H, W, _ = data.shape
    print(f"  data: {data.shape} | Anotiranih: {(classes != -1).sum()}")
    prediction_crop_mask = make_prediction_crop_mask(H, W)
    device = get_device()

    n_param = sum(p.numel() for p in DualStreamCNN().parameters() if p.requires_grad)
    print(f"\n  Konfiguracija:")
    print(f"    Patch: {PATCH_SIZE}×{PATCH_SIZE} (reflect padding)")
    print(f"    Preprocessing: StandardScaler + PCA({N_PCA})")
    print(f"    Spektralni tok: {SPEC_DIM_RAW} raw + "
          f"{SPEC_DIM_RAW*len(NEIGH_SCALES)} neighbourhood mean "
          f"(scales={NEIGH_SCALES}) = {SPEC_DIM} dim")
    print(f"    Arhitektura: DualStreamCNN ({n_param:,} param)")
    print(f"    TTA: 8 augmentacij | Ensemble: {args.n_ensemble} modelov")

    # ------------------------------------------------------------------
    print("\n=== 2. Prostorski 80/20 split ===")
    train_mask, val_mask = make_spatial_two_way_split(
        tissue_mask=tissue_mask, classes=classes,
        prediction_crop_mask=prediction_crop_mask,
        val_fraction=args.val_frac, verbose=True,
    )
    train_coords = np.argwhere(train_mask)
    val_coords   = np.argwhere(val_mask)
    y_train = classes[train_mask].astype(np.int64)
    y_val   = classes[val_mask].astype(np.int64)

    print(f"\n  Porazdelitev razredov:")
    for c in range(NUM_CLASSES):
        nt = (y_train == c).sum(); nv = (y_val == c).sum()
        print(f"    Razred {c}: train={nt:5d} ({100*nt/len(y_train):.1f}%)  "
              f"val={nv:5d} ({100*nv/len(y_val):.1f}%)")

    # ------------------------------------------------------------------
    print("\n=== 3. Preprocessing: StandardScaler + PCA ===")
    data_pca, flat_sc, scaler, pca = build_preprocessed(data, train_mask, N_PCA, args.seed)
    weights = compute_class_weights(y_train)

    # ------------------------------------------------------------------
    print("\n=== 4. Neighbourhood spectral features (eval split) ===")
    print(f"  Scales: {NEIGH_SCALES} | SPEC_DIM = {SPEC_DIM}")
    print(f"  Extrakcija za train ({len(train_coords):,} pikslov)...")
    train_spec = extract_spec_features(flat_sc, H, W, train_coords, NEIGH_SCALES)
    print(f"  Extrakcija za val ({len(val_coords):,} pikslov)...")
    val_spec   = extract_spec_features(flat_sc, H, W, val_coords, NEIGH_SCALES)
    print(f"  train_spec: {train_spec.shape}, val_spec: {val_spec.shape}")

    val_ds = DualStreamDataset(data_pca, val_spec, val_coords, y_val, augment=False)

    # ------------------------------------------------------------------
    print(f"\n=== 5. Ensemble trening ({args.n_ensemble} modelov) ===")
    seeds           = list(range(args.seed, args.seed + args.n_ensemble))
    val_logits_list = []
    best_epochs     = []

    for i, seed in enumerate(seeds):
        print(f"\n  ── Model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        model, best_epoch = train_single(
            data_pca=data_pca, train_spec=train_spec,
            val_ds=val_ds, y_train=y_train, y_val=y_val,
            train_coords=train_coords,
            device=device, weights=weights,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience,
            seed=seed, spec_noise=args.spec_noise,
        )
        best_epochs.append(best_epoch)

        print(f"  TTA na val...")
        val_logits_list.append(
            get_logits_tta(model, data_pca, val_spec, val_coords, y_val, device)
        )

    val_logits_avg = np.mean(val_logits_list, axis=0)
    avg_best_epoch = round(np.mean(best_epochs))
    print(f"\n  Best epochs: {best_epochs} (avg={avg_best_epoch})")

    # ------------------------------------------------------------------
    print("\n=== 6. Temperature scaling ===")
    T_opt = find_temperature(val_logits_avg, y_val)

    # ------------------------------------------------------------------
    print("\n=== 7. Val evaluacija ===")
    val_probs = apply_temperature(val_logits_avg, T_opt)
    val_oa    = accuracy_score(y_val, np.argmax(val_probs, axis=1))
    val_ll    = log_loss(y_val, val_probs, labels=np.arange(NUM_CLASSES))
    print(f"  VAL OA:       {val_oa*100:.2f}%")
    print(f"  VAL log loss: {val_ll:.5f}")
    print(f"  Ref v1 best:  VAL ll=0.25168 | Final=0.459")

    # ------------------------------------------------------------------
    n_all = int(((classes != -1) & (~prediction_crop_mask)).sum())
    proportional = round(avg_best_epoch * n_all / max(len(y_train), 1))
    final_epochs = max(proportional, 15)  # min 15 epoh (v1 je imel samo 7!)
    print(f"\n=== 8. Finalni ensemble (100%, {final_epochs} epoh × {args.n_ensemble} modelov) ===")
    print(f"  (proporcionalno={proportional}, min=15 → final_epochs={final_epochs})")

    usable_mask = (classes != -1) & (~prediction_crop_mask)
    all_coords  = np.argwhere(usable_mask)
    y_all       = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov: {len(y_all)}")

    # Refit na vseh podatkih
    print("\n  StandardScaler + PCA na vseh anotiranih pikslih...")
    flat         = data.reshape(-1, data.shape[-1])
    X_all_raw    = flat[usable_mask.ravel()]
    scaler_final = StandardScaler()
    X_all_sc     = scaler_final.fit_transform(X_all_raw)
    pca_final    = PCA(n_components=N_PCA, random_state=args.seed)
    pca_final.fit(X_all_sc)
    print(f"  PCA varianca: {pca_final.explained_variance_ratio_.sum()*100:.2f}%")

    flat_sc_final  = scaler_final.transform(flat).astype(np.float32)
    data_pca_final = pca_final.transform(flat_sc_final).reshape(H, W, N_PCA).astype(np.float32)

    # Neighbourhood features za final
    print("\n  Neighbourhood features za final (all_coords + crop_coords)...")
    print(f"  all_coords ({len(all_coords):,} pikslov):")
    all_spec_final = extract_spec_features(flat_sc_final, H, W, all_coords, NEIGH_SCALES)

    rs = np.repeat(np.arange(PRED_R0, PRED_R1), PRED_C1 - PRED_C0)
    cs = np.tile(  np.arange(PRED_C0, PRED_C1), PRED_R1 - PRED_R0)
    crop_coords  = np.stack([rs, cs], axis=1)
    n_crop       = len(crop_coords)
    crop_tissue  = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)

    print(f"  crop_coords ({n_crop:,} pikslov):")
    crop_spec_final = extract_spec_features(flat_sc_final, H, W, crop_coords, NEIGH_SCALES)

    weights_final    = compute_class_weights(y_all)
    crop_logits_sum  = np.zeros((n_crop, NUM_CLASSES), dtype=np.float64)

    for i, seed in enumerate(seeds):
        print(f"\n  ── Finalni model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        torch.manual_seed(seed); random.seed(seed)

        final_ds  = DualStreamDataset(data_pca_final, all_spec_final, all_coords, y_all,
                                       augment=True, spec_noise=args.spec_noise)
        final_ldr = DataLoader(final_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
        model_f   = DualStreamCNN().to(device)
        opt_f     = optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=1e-4)
        sched_f   = optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=final_epochs, eta_min=1e-6)
        crit_f    = nn.NLLLoss(weight=weights_final.to(device))

        print(f"  {'Ep':>4}  {'Train ll':>10}  {'LR':>9}")
        t0 = time.time()
        for epoch in range(1, final_epochs + 1):
            model_f.train()
            total = 0.0
            for patches, spectra, labels in final_ldr:
                patches = patches.to(device); spectra = spectra.to(device)
                labels  = labels.to(device)
                opt_f.zero_grad()
                loss = crit_f(model_f(patches, spectra), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model_f.parameters(), 1.0)
                opt_f.step()
                total += loss.item() * len(labels)
            sched_f.step()
            if epoch % 5 == 0 or epoch in (1, final_epochs):
                print(f"  {epoch:>4}  {total/len(y_all):>10.5f}  "
                      f"{opt_f.param_groups[0]['lr']:>9.2e}")
        print(f"  Treniran v {time.time()-t0:.1f}s")

        print(f"  TTA za crop...")
        crop_logits_tta = get_logits_tta(
            model_f, data_pca_final, crop_spec_final,
            crop_coords, np.zeros(n_crop, dtype=np.int64), device
        )
        crop_logits_sum += crop_logits_tta.astype(np.float64)

    # ------------------------------------------------------------------
    print("\n=== 9. Submission ===")
    crop_probs = apply_temperature(
        (crop_logits_sum / args.n_ensemble).astype(np.float32), T_opt
    )

    prior = np.bincount(y_all, minlength=NUM_CLASSES).astype(np.float32)
    prior /= prior.sum()
    crop_probs[~crop_tissue] = prior

    crop_h, crop_w = PRED_R1 - PRED_R0, PRED_C1 - PRED_C0
    prob_map       = crop_probs.reshape(crop_h, crop_w, NUM_CLASSES)
    crop_tissue_2d = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1]

    print(f"  Gaussian smoothing (sigma={args.sigma:.1f})...")
    if args.sigma > 0:
        smoothed   = gaussian_smooth_probs(prob_map, crop_tissue_2d, args.sigma)
        final_map  = prob_map.copy()
        final_map[crop_tissue_2d] = smoothed[crop_tissue_2d]
    else:
        final_map = prob_map

    final_map = np.clip(final_map, 1e-7, 1.0)
    final_map /= final_map.sum(axis=-1, keepdims=True)
    np.save(args.output, final_map.astype(np.float32))

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  SPEC_DIM: {SPEC_DIM} (raw {SPEC_DIM_RAW} + neighbourhood {SPEC_DIM-SPEC_DIM_RAW})")
    print(f"  Ensemble: {args.n_ensemble} modelov | final_epochs={final_epochs}")
    print(f"  Best epochs (eval): {best_epochs}")
    print(f"  T={T_opt:.4f} | σ={args.sigma}")
    print(f"  VAL OA={val_oa*100:.2f}%  VAL ll={val_ll:.5f}")
    print(f"\n  Primerjava (Final — nižje je boljše):")
    print(f"    Zmagovalec:   0.401")
    print(f"    modelC_best:  0.459  ← naš prejšnji best")
    print(f"    patchcnn:     0.519")
    print(f"\n  Submission: {args.output}")


if __name__ == "__main__":
    main()
