"""
Model C — Dual-Stream CNN (spektralni + prostorni tok)
=======================================================
Mentor hint: "The best models will likely use both spatial AND spectral information"

Problem z v2/v3:
  - PCA(16) komprimira 187 dimenzij → 16 → izgubimo spektralno informacijo
  - CNN vidi samo prostorne vzorce, spektralna diskriminacija je šibka
  - Razredi ki so podobni prostorsko a spektralno različni → napačne napovedi

Rešitev — dual-stream arhitektura:
  ┌─────────────────────────────────┐
  │  PROSTORNI TOK (2D CNN)         │
  │  patch 33×33×16 (PCA)           │
  │  → prostorni vzorci, morfologija│
  │  → feature vektor (64 dim)      │
  └──────────────┬──────────────────┘
                 │ FUZIJA
  ┌──────────────┴──────────────────┐    ┌─────────────────┐
  │  SPEKTRALNI TOK (MLP)           │    │  concatenate    │
  │  centralni piksel: 187 dim      │    │  64 + 64 = 128  │
  │  → absorption peaks             │────→  FC → 6 razredov│
  │  → feature vektor (64 dim)      │    └─────────────────┘
  └─────────────────────────────────┘

Zakaj deluje:
  - Prostorni tok: "ali je ta piksel obkrožen z epitelijem ali kolagenem?"
  - Spektralni tok: "kakšno kemično sestavo ima ta piksel?"
  - Skupaj: oba aspekta hkrati → boljša diskriminacija mejnih primerov

Preprocessing:
  - Prostorni tok: rubberband + Amide I → PCA(16) → datacube (H,W,16)
  - Spektralni tok: rubberband + Amide I → 187 dim (za anotirane)
                    Amide I only → 187 dim (za neanotirane / competition crop)
"""

import argparse
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from torch.utils.data import DataLoader, Dataset

from spatial_split import make_spatial_three_way_split, print_split_summary

# ---------------------------------------------------------------------------
# Konstante
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660
AMIDE_I_TARGET_WN = 1650.0
PATCH_SIZE = 33   # nazaj na v2 (49 je bilo slabše)
N_PCA = 16
SPEC_DIM = 187


# ---------------------------------------------------------------------------
# Nalaganje
# ---------------------------------------------------------------------------
def load_data(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        data        = np.array(f["data"],        dtype=np.float32)
        wns         = np.array(f["wns"])
        tissue_mask = np.array(f["tissue_mask"])
        classes     = np.array(f["classes"])
    return data, wns, tissue_mask, classes


def find_amide_i_index(wns):
    idx = int(np.argmin(np.abs(wns - AMIDE_I_TARGET_WN)))
    print(f"  Amide I: actual={wns[idx]:.2f} cm-1 | idx={idx}")
    return idx


def make_prediction_crop_mask(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1] = True
    return mask


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _rubberband_single(spectrum):
    n = len(spectrum)
    x = np.arange(n, dtype=np.float64)
    y = spectrum.astype(np.float64)
    lower = []
    for i in range(n):
        p = (x[i], y[i])
        while len(lower) >= 2:
            o, a = lower[-2], lower[-1]
            if (a[0]-o[0])*(p[1]-o[1]) - (a[1]-o[1])*(p[0]-o[0]) <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    lx = np.array([p[0] for p in lower])
    ly = np.array([p[1] for p in lower])
    return (y - np.interp(x, lx, ly)).astype(np.float32)


def rubberband_correction(spectra):
    out = np.empty_like(spectra, dtype=np.float32)
    for i in range(len(spectra)):
        out[i] = _rubberband_single(spectra[i])
    return out


def amide_i_normalize(spectra, amide_idx, eps=1e-6):
    vals = spectra[:, amide_idx].astype(np.float64)
    safe = np.where(vals > eps, vals, eps)
    return (spectra / safe[:, np.newaxis]).astype(np.float32)


def build_preprocessed_datacube(data, wns, amide_idx, train_mask,
                                 n_components=N_PCA, seed=42):
    """
    Zgradi:
      1. data_pca: (H, W, 16)  — za prostorni tok (patchi)
      2. flat_pp:  (H*W, 187)  — za spektralni tok (centralni piksli)
                  rubberband+Amide I za train anotirane piksle,
                  samo Amide I za vse ostale.

    Vrne: (data_pca, flat_pp, pca_model)
    """
    H, W, D = data.shape
    print(f"  Preprocessing datacube: {H}×{W}×{D}")

    # 1. Amide I na VSEH pikslih (hitro)
    print(f"  Amide I na vseh {H*W:,} pikslih...")
    t0 = time.time()
    flat = data.reshape(-1, D)
    flat_pp = amide_i_normalize(flat, amide_idx)
    print(f"    → {time.time()-t0:.1f}s")

    # 2. Rubberband SAMO za train anotirane piksle
    train_coords = np.argwhere(train_mask)
    train_idx = train_coords[:, 0] * W + train_coords[:, 1]
    print(f"  Rubberband za {len(train_idx):,} train pikslov...")
    t0 = time.time()
    train_spectra = flat[train_idx]
    train_pp = rubberband_correction(amide_i_normalize(train_spectra, amide_idx))
    flat_pp[train_idx] = train_pp   # vstavi nazaj v flat_pp
    print(f"    → {time.time()-t0:.1f}s")

    # 3. PCA fit na train pikslih → za prostorni tok
    print(f"  PCA({n_components}) fit na {len(train_idx)} pikslih...")
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_pp)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # 4. PCA transform VSEH pikslov → data_pca
    print(f"  PCA transform {H*W:,} pikslov...")
    t0 = time.time()
    data_pca = pca.transform(flat_pp).reshape(H, W, n_components).astype(np.float32)
    print(f"    → {time.time()-t0:.1f}s | PCA shape={data_pca.shape}")

    return data_pca, flat_pp, pca


# ---------------------------------------------------------------------------
# Dual-Stream Dataset
# ---------------------------------------------------------------------------
class DualStreamDataset(Dataset):
    """
    Za vsak piksel vrne:
      - patch_t:   (N_PCA, PATCH_SIZE, PATCH_SIZE) — za prostorni tok
      - spectrum_t: (SPEC_DIM,)                     — za spektralni tok
      - label_t:   int

    Augmentacija (samo za trening):
      - Flip H, flip V (p=0.5 vsak)
      - Rotacija 90°/180°/270° (naključno)
    Centralni spekter se ne augmentira (piksel ostane isti).
    """

    def __init__(self, data_pca: np.ndarray, flat_pp: np.ndarray,
                 coords: np.ndarray, labels: np.ndarray,
                 image_width: int,
                 patch_size: int = PATCH_SIZE,
                 augment: bool = False):
        self.pad       = patch_size // 2
        self.labels    = labels
        self.coords    = coords
        self.augment   = augment
        self.W         = image_width

        # Precompute flat indekse za hiter dostop do spektrov
        self.flat_idx  = coords[:, 0] * image_width + coords[:, 1]
        self.flat_pp   = flat_pp   # (H*W, SPEC_DIM)

        # Zero-padded PCA datacube za patch ekstrakcijo
        self.padded = np.pad(
            data_pca,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='constant', constant_values=0.0
        ).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        rp, cp = r + self.pad, c + self.pad

        # Prostorni patch: (PATCH_SIZE, PATCH_SIZE, N_PCA) → (N_PCA, PS, PS)
        patch = self.padded[rp-self.pad:rp+self.pad+1,
                            cp-self.pad:cp+self.pad+1]
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1).copy())

        # Augmentacija patcha (spekter ostane enak!)
        if self.augment:
            if random.random() > 0.5:
                patch_t = torch.flip(patch_t, dims=[1])
            if random.random() > 0.5:
                patch_t = torch.flip(patch_t, dims=[2])
            k = random.randint(0, 3)
            if k > 0:
                patch_t = torch.rot90(patch_t, k=k, dims=[1, 2])

        # Centralni spekter: (SPEC_DIM,)
        spectrum_t = torch.from_numpy(self.flat_pp[self.flat_idx[idx]].copy())

        label_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return patch_t, spectrum_t, label_t


# ---------------------------------------------------------------------------
# Tehtana loss
# ---------------------------------------------------------------------------
def compute_class_weights(y):
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Arhitektura: DualStreamCNN
# ---------------------------------------------------------------------------
class DualStreamCNN(nn.Module):
    """
    Dual-stream arhitektura:

    Prostorni tok (2D CNN na PCA patchu 33×33×16):
      Conv(16→32) + BN + ReLU + Drop + MaxPool → (32, 16, 16)
      Conv(32→64) + BN + ReLU + Drop           → (64, 16, 16)
      Conv(64→64) + BN + ReLU + Drop + MaxPool → (64,  8,  8)
      AdaptiveAvgPool → (64,)

    Spektralni tok (MLP na centralnem spektru 187 dim):
      Linear(187→256) + BN + ReLU + Drop(0.4)
      Linear(256→128) + BN + ReLU + Drop(0.3)
      Linear(128→64)  + BN + ReLU
      → (64,)

    Fuzija:
      Concat(64 + 64) = 128
      Linear(128→64) + ReLU + Drop(0.3)
      Linear(64→6)   → LogSoftmax
    """

    def __init__(self, n_channels=N_PCA, spec_dim=SPEC_DIM,
                 num_classes=NUM_CLASSES, patch_size=PATCH_SIZE,
                 spatial_feat=64, spectral_feat=64, dropout=0.3):
        super().__init__()

        # --- Prostorni tok ---
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),                              # 33→16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),                              # 16→8
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)      # → (64, 1, 1) → 64

        # --- Spektralni tok ---
        self.spectral_mlp = nn.Sequential(
            nn.Linear(spec_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout + 0.1),

            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout),

            nn.Linear(128, spectral_feat), nn.BatchNorm1d(spectral_feat),
            nn.ReLU(),
        )

        # --- Fuzija ---
        fusion_in = spatial_feat + spectral_feat  # 64 + 64 = 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(),
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
        x = self.spatial_cnn(patch)
        return self.spatial_pool(x).squeeze(-1).squeeze(-1)   # (B, 64)

    def get_spectral_features(self, spectrum):
        return self.spectral_mlp(spectrum)                     # (B, 64)

    def get_logits(self, patch, spectrum):
        f_spatial   = self.get_spatial_features(patch)
        f_spectral  = self.get_spectral_features(spectrum)
        f_combined  = torch.cat([f_spatial, f_spectral], dim=1)
        return self.fusion(f_combined)

    def forward(self, patch, spectrum):
        return self.log_softmax(self.get_logits(patch, spectrum))


# ---------------------------------------------------------------------------
# Naprava
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu"); print("  Naprava: CPU")
    return d


# ---------------------------------------------------------------------------
# Napoved
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_p = []
    for patches, spectra, _ in loader:
        p = torch.exp(model(patches.to(device), spectra.to(device)))
        all_p.append(p.cpu().numpy())
    return np.concatenate(all_p)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def find_temperature(model, val_ds, y_val, device):
    model.eval()
    logits_list = []
    loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for patches, spectra, _ in loader:
            logits_list.append(
                model.get_logits(patches.to(device), spectra.to(device)).cpu().numpy()
            )
    logits = np.concatenate(logits_list)

    def neg_ll(T):
        s = logits / T
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-9, 1.0)
        return -np.mean(np.log(p[np.arange(len(y_val)), y_val]))

    res = minimize_scalar(neg_ll, bounds=(0.1, 10.0), method='bounded')
    T = res.x
    print(f"  Temperature: T={T:.4f}  "
          f"log_loss: {neg_ll(1.0):.5f} → {neg_ll(T):.5f}")
    return T


# ---------------------------------------------------------------------------
# Gaussian smoothing
# ---------------------------------------------------------------------------
def gaussian_smooth_probs(probs, tissue_mask, sigma):
    if sigma <= 0:
        return probs
    smoothed = np.zeros_like(probs)
    mask_f = tissue_mask.astype(np.float32)
    for c in range(probs.shape[-1]):
        num = gaussian_filter(probs[:, :, c] * mask_f, sigma=sigma)
        den = gaussian_filter(mask_f, sigma=sigma)
        smoothed[:, :, c] = num / np.where(den < 1e-8, 1e-8, den)
    smoothed = np.clip(smoothed, 1e-7, 1.0)
    smoothed /= smoothed.sum(axis=-1, keepdims=True)
    return smoothed.astype(np.float32)


# ---------------------------------------------------------------------------
# Fitanje
# ---------------------------------------------------------------------------
def fit_model(data_pca, flat_pp, W,
              train_coords, y_train,
              val_coords, y_val,
              device,
              epochs=150, batch_size=128, lr=1e-4,
              patience=15, seed=42):
    torch.manual_seed(seed); random.seed(seed)

    weights  = compute_class_weights(y_train)
    train_ds = DualStreamDataset(data_pca, flat_pp, train_coords, y_train,
                                  W, augment=True)
    val_ds   = DualStreamDataset(data_pca, flat_pp, val_coords,   y_val,
                                  W, augment=False)
    loader   = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=0)

    model   = DualStreamCNN().to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  DualStreamCNN: {n_param:,} param")
    print(f"  Prostorni tok: patch {PATCH_SIZE}×{PATCH_SIZE}×{N_PCA}")
    print(f"  Spektralni tok: {SPEC_DIM} dim → 64")
    print(f"  Fuzija: 128 → 64 → 6")

    criterion = nn.NLLLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss, best_state, best_epoch = float('inf'), None, epochs
    patience_counter = 0

    print(f"\n  {'Epoha':>6}  {'Train ll':>10}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for patches, spectra, labels in loader:
            patches  = patches.to(device)
            spectra  = spectra.to(device)
            labels   = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(patches, spectra), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        train_loss = total / len(train_ds)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        probs  = predict_proba(model, val_ds, device, 256)
        val_oa = accuracy_score(y_val, np.argmax(probs, axis=1))
        val_ll = log_loss(y_val, probs, labels=np.arange(NUM_CLASSES))
        print(f"  {epoch:>6}  {train_loss:>10.5f}  {val_oa*100:>8.2f}%  {val_ll:>9.5f}  {lr_now:>9.2e}")

        if val_ll < best_val_loss - 1e-6:
            best_val_loss = val_ll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping: epoha {epoch} "
                      f"(best={best_val_loss:.5f}, epoha={best_epoch})")
                break

    print(f"\n  Trening: {time.time()-t0:.1f}s")
    model.load_state_dict(best_state)
    print(f"  Naložene najboljše uteži: epoha={best_epoch}, val_ll={best_val_loss:.5f}")
    return model, best_epoch, val_ds


# ---------------------------------------------------------------------------
# Evaluacija
# ---------------------------------------------------------------------------
def evaluate(model, data_pca, flat_pp, W,
             coords, y_true, device,
             split_name="", temperature=1.0):
    ds    = DualStreamDataset(data_pca, flat_pp, coords, y_true, W, augment=False)

    if temperature == 1.0:
        probs = predict_proba(model, ds, device, 256)
    else:
        model.eval()
        logits_list = []
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        with torch.no_grad():
            for patches, spectra, _ in loader:
                logits_list.append(
                    model.get_logits(patches.to(device), spectra.to(device)).cpu().numpy()
                )
        logits = np.concatenate(logits_list)
        s = logits / temperature
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    preds = np.argmax(probs, axis=1)
    oa    = accuracy_score(y_true, preds)
    ll    = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))

    t_str = f" [T={temperature:.3f}]" if temperature != 1.0 else ""
    print(f"\n  ── {split_name}{t_str} ──")
    print(f"  OA:       {oa*100:.2f}%")
    print(f"  Log loss: {ll:.5f}")
    print(f"  Ref Model A v3: TEST OA=92.69%, ll=0.347")
    print(f"  Ref Model C v2: TEST OA=88.40%, ll=0.337")

    per_class = []
    print(f"\n  Natancnost po razredih:")
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"    Razred {c}: N/A"); continue
        acc_c = (preds[mask] == y_true[mask]).mean()
        per_class.append(acc_c)
        print(f"    Razred {c}: {acc_c*100:.2f}%  (n={mask.sum()})")
    print(f"\n  Macro OA: {np.mean(per_class)*100:.2f}%")
    cm = confusion_matrix(y_true, preds, labels=np.arange(NUM_CLASSES))
    print(f"\n  Matrika zmede:\n{cm}")
    return oa, ll


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def make_submission(model, data_pca, flat_pp, W,
                    tissue_mask, device, output_path,
                    train_class_counts, temperature=1.0, sigma=1.5):
    crop_h, crop_w = PRED_R1 - PRED_R0, PRED_C1 - PRED_C0
    n_crop = crop_h * crop_w
    rs = np.repeat(np.arange(PRED_R0, PRED_R1), crop_w)
    cs = np.tile(  np.arange(PRED_C0, PRED_C1), crop_h)
    coords = np.stack([rs, cs], axis=1)

    prior = train_class_counts.astype(np.float32)
    prior /= prior.sum()
    crop_tissue = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)
    print(f"  Crop: {n_crop} | tkivo: {int(crop_tissue.sum())} | "
          f"ozadje: {n_crop - int(crop_tissue.sum())}")

    dummy = np.zeros(n_crop, dtype=np.int64)
    ds    = DualStreamDataset(data_pca, flat_pp, coords, dummy, W, augment=False)

    model.eval()
    logits_list = []
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for patches, spectra, _ in loader:
            logits_list.append(
                model.get_logits(patches.to(device), spectra.to(device)).cpu().numpy()
            )
    logits = np.concatenate(logits_list)
    s = logits / temperature
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    probs[~crop_tissue] = prior
    prob_map = probs.reshape(crop_h, crop_w, NUM_CLASSES)

    crop_tissue_2d = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1]
    if sigma > 0:
        print(f"  Gaussian smoothing (sigma={sigma:.1f})...")
        smoothed = gaussian_smooth_probs(prob_map, crop_tissue_2d, sigma)
        final = prob_map.copy()
        final[crop_tissue_2d] = smoothed[crop_tissue_2d]
    else:
        final = prob_map

    final = np.clip(final, 1e-7, 1.0)
    final /= final.sum(axis=-1, keepdims=True)
    np.save(output_path, final.astype(np.float32))
    print(f"  Submission: {output_path}  T={temperature:.3f}  σ={sigma:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model C Dual-Stream — 2D CNN patch + 1D spektralni MLP"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output",             default="modelC_dual.npy")
    parser.add_argument("--epochs",             type=int,   default=150)
    parser.add_argument("--batch-size",         type=int,   default=128)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--patience",           type=int,   default=15)
    parser.add_argument("--sigma",              type=float, default=1.5)
    parser.add_argument("--val-fraction",       type=float, default=0.20)
    parser.add_argument("--test-fraction",      type=float, default=0.20)
    parser.add_argument("--min-component-size", type=int,   default=20)
    parser.add_argument("--seed",               type=int,   default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, wns, tissue_mask, classes = load_data(args.input)
    H, W, _ = data.shape
    print(f"  data: {data.shape} | Anotiranih: {(classes != -1).sum()}")
    amide_idx            = find_amide_i_index(wns)
    prediction_crop_mask = make_prediction_crop_mask(H, W)
    device               = get_device()

    # ------------------------------------------------------------------
    print("\n=== 2. Prostorski 3-way split ===")
    train_mask, val_mask, test_mask = make_spatial_three_way_split(
        tissue_mask=tissue_mask, classes=classes,
        prediction_crop_mask=prediction_crop_mask,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        min_component_size=args.min_component_size, verbose=True,
    )
    print_split_summary(train_mask, val_mask, test_mask, classes)

    train_coords = np.argwhere(train_mask)
    val_coords   = np.argwhere(val_mask)
    test_coords  = np.argwhere(test_mask)
    y_train = classes[train_mask].astype(np.int64)
    y_val   = classes[val_mask].astype(np.int64)
    y_test  = classes[test_mask].astype(np.int64)

    # ------------------------------------------------------------------
    print("\n=== 3. Preprocessing (prostorni + spektralni tok) ===")
    data_pca_eval, flat_pp_eval, _ = build_preprocessed_datacube(
        data, wns, amide_idx, train_mask, N_PCA, args.seed
    )

    # ------------------------------------------------------------------
    print("\n=== 4. Ucenje Dual-Stream CNN ===")
    model, best_epoch, val_ds = fit_model(
        data_pca=data_pca_eval, flat_pp=flat_pp_eval, W=W,
        train_coords=train_coords, y_train=y_train,
        val_coords=val_coords,     y_val=y_val,
        device=device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, seed=args.seed,
    )

    # ------------------------------------------------------------------
    print("\n=== 5. Temperature scaling ===")
    T_opt = find_temperature(model, val_ds, y_val, device)

    # ------------------------------------------------------------------
    print("\n=== 6. Evaluacija ===")
    oa_val, ll_val = evaluate(
        model, data_pca_eval, flat_pp_eval, W,
        val_coords, y_val, device, "VAL", T_opt
    )
    print()
    oa_test, ll_test = evaluate(
        model, data_pca_eval, flat_pp_eval, W,
        test_coords, y_test, device, "TEST (zaklenjen)", T_opt
    )

    # ------------------------------------------------------------------
    print(f"\n=== 7. Finalni model (100% podatkov) ===")
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    all_coords  = np.argwhere(usable_mask)
    y_all       = classes[usable_mask].astype(np.int64)
    n_all, n_tr = len(y_all), len(y_train)
    final_epochs = max(best_epoch, round(best_epoch * n_all / n_tr))
    print(f"  Skupaj pikslov: {n_all} | Finalne epohe: {final_epochs}")

    print("\n  Preprocessing za finalni model...")
    data_pca_final, flat_pp_final, _ = build_preprocessed_datacube(
        data, wns, amide_idx, usable_mask, N_PCA, args.seed
    )

    weights_f = compute_class_weights(y_all)
    final_ds  = DualStreamDataset(data_pca_final, flat_pp_final,
                                   all_coords, y_all, W, augment=True)
    final_ldr = DataLoader(final_ds, batch_size=args.batch_size,
                           shuffle=True, num_workers=0)
    model_f   = DualStreamCNN().to(device)
    opt_f     = optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_f   = optim.lr_scheduler.CosineAnnealingLR(
        opt_f, T_max=final_epochs, eta_min=1e-6)
    crit_f    = nn.NLLLoss(weight=weights_f.to(device))

    print(f"\n  {'Epoha':>6}  {'Train ll':>10}  {'LR':>9}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*9}")
    torch.manual_seed(args.seed); random.seed(args.seed)
    t0 = time.time()
    for epoch in range(1, final_epochs + 1):
        model_f.train()
        total = 0.0
        for patches, spectra, labels in final_ldr:
            patches  = patches.to(device)
            spectra  = spectra.to(device)
            labels   = labels.to(device)
            opt_f.zero_grad()
            loss = crit_f(model_f(patches, spectra), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model_f.parameters(), 1.0)
            opt_f.step()
            total += loss.item() * len(labels)
        sched_f.step()
        if epoch % 5 == 0 or epoch in (1, final_epochs):
            lr_now = opt_f.param_groups[0]['lr']
            print(f"  {epoch:>6}  {total/n_all:>10.5f}  {lr_now:>9.2e}")
    print(f"  Treniran v {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print("\n=== 8. Submission ===")
    make_submission(
        model=model_f, data_pca=data_pca_final, flat_pp=flat_pp_final,
        W=W, tissue_mask=tissue_mask, device=device,
        output_path=args.output,
        train_class_counts=np.bincount(y_all, minlength=NUM_CLASSES),
        temperature=T_opt, sigma=args.sigma,
    )

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  DualStreamCNN: patch {PATCH_SIZE}×{PATCH_SIZE}×{N_PCA} + spekter {SPEC_DIM} dim")
    print(f"  Best epoch: {best_epoch} | T={T_opt:.4f} | σ={args.sigma}")
    print(f"\n  {'':25} {'OA':>8}  {'Log loss':>10}")
    print(f"  {'VAL':25} {oa_val*100:>7.2f}%  {ll_val:>10.5f}")
    print(f"  {'TEST (diploma)':25} {oa_test*100:>7.2f}%  {ll_test:>10.5f}")
    print(f"\n  Primerjava (TEST):")
    print(f"  {'Članek CNN prostorni':25} {'92.85%':>8}")
    print(f"  {'Model A SVM+PCA':25} {'92.69%':>8}  {'0.34700':>10}")
    print(f"  {'Model C v2 (patch CNN)':25} {'88.40%':>8}  {'0.33703':>10}")
    print(f"\n  Submission: {args.output}")


if __name__ == "__main__":
    main()
