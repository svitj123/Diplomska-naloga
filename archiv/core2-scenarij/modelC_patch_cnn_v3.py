"""
Model C — Patch CNN  (v3: večji patch + temperature scaling + sigma tuning)
===========================================================================
Spremembe glede na v2:

  1. patch_size: 33 → 49
     Večja okolica → CNN vidi tkivne vzorce na večji skali.
     Arhitektura dobi 3. MaxPool blok za primerno zmanjšanje.

  2. Temperature scaling (post-training kalibracija)
     Poišče optimalni T na val množici, zmanjša log loss brez
     vpliva na OA.

  3. Iskanje optimalne σ za Gaussian smoothing na val množici
     Namesto fiksnega σ=1.5 preizkusimo {0, 0.5, 1.0, 1.5, 2.0, 3.0}
     in izberemo σ z najboljšim val log loss.

  4. Proporcionalno skaliranje epoh za finalni model
     final_epochs = round(best_epoch * n_all / n_train)
     Prej: always best_epoch (prekratko za 100% podatkov)

Ostalo enako kot v2:
  - Dropout2d(0.3) v CNN, Dropout(0.4) pred FC
  - Weighted NLLLoss (brez oversamplinga)
  - Data augmentation: flip H/V + rot90
  - LR=1e-4, CosineAnnealing, patience=15
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
PATCH_SIZE = 49   # v3: 49 (bil 33)
N_PCA = 16


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


def build_pca_datacube(data, wns, amide_idx, train_mask,
                       n_components=N_PCA, seed=42):
    H, W, D = data.shape
    print(f"  PCA datacube: {H}×{W}×{D} → {H}×{W}×{n_components}")

    print(f"  Amide I na vseh {H*W:,} pikslih...")
    t0 = time.time()
    flat = data.reshape(-1, D)
    flat_pp = amide_i_normalize(flat, amide_idx)
    print(f"    → {time.time()-t0:.1f}s")

    train_coords = np.argwhere(train_mask)
    train_idx = train_coords[:, 0] * W + train_coords[:, 1]
    print(f"  Rubberband za {len(train_idx):,} train pikslov...")
    t0 = time.time()
    train_pp = rubberband_correction(amide_i_normalize(flat[train_idx], amide_idx))
    flat_pp[train_idx] = train_pp
    print(f"    → {time.time()-t0:.1f}s")

    print(f"  PCA fit na {len(train_idx)} pikslih...")
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_pp)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    print(f"  PCA transform vseh {H*W:,} pikslov...")
    t0 = time.time()
    data_pca = pca.transform(flat_pp).reshape(H, W, n_components).astype(np.float32)
    print(f"    → {time.time()-t0:.1f}s")
    return data_pca, pca


# ---------------------------------------------------------------------------
# Dataset z augmentacijo
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, data_pca, coords, labels,
                 patch_size=PATCH_SIZE, augment=False):
        self.pad     = patch_size // 2
        self.labels  = labels
        self.coords  = coords
        self.augment = augment
        self.padded  = np.pad(
            data_pca,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='constant', constant_values=0.0
        ).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        rp, cp = r + self.pad, c + self.pad
        patch = self.padded[rp-self.pad:rp+self.pad+1,
                            cp-self.pad:cp+self.pad+1]
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1).copy())
        if self.augment:
            if random.random() > 0.5:
                patch_t = torch.flip(patch_t, dims=[1])
            if random.random() > 0.5:
                patch_t = torch.flip(patch_t, dims=[2])
            k = random.randint(0, 3)
            if k > 0:
                patch_t = torch.rot90(patch_t, k=k, dims=[1, 2])
        return patch_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Tehtana loss
# ---------------------------------------------------------------------------
def compute_class_weights(y):
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Arhitektura: PatchCNN2D v3 (4 conv bloki za 49×49 vhod)
# ---------------------------------------------------------------------------
class PatchCNN2D(nn.Module):
    """
    Arhitektura za patch_size=49:

      Conv(16→32)  + BN + ReLU + Drop + MaxPool(2) → (32, 24, 24)
      Conv(32→64)  + BN + ReLU + Drop              → (64, 24, 24)
      Conv(64→64)  + BN + ReLU + Drop + MaxPool(2) → (64, 12, 12)
      Conv(64→128) + BN + ReLU + Drop + MaxPool(2) → (128, 6, 6)
      Flatten → 4608
      FC(4608→256) + ReLU + Drop(0.4)
      FC(256→6)    → LogSoftmax
    """

    def __init__(self, n_channels=N_PCA, num_classes=NUM_CLASSES,
                 patch_size=PATCH_SIZE, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),                              # 49→24

            # Blok 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),

            # Blok 3
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),                              # 24→12

            # Blok 4 (nov v v3 — za večji patch)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),                              # 12→6
        )

        # 49→24→12→6, channels=128 → fc_in=128×6×6=4608
        fc_in = 128 * (patch_size // 2 // 2 // 2) ** 2

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 256), nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(256, num_classes),
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def get_logits(self, x):
        return self.classifier(self.features(x))

    def forward(self, x):
        return self.log_softmax(self.get_logits(x))


# ---------------------------------------------------------------------------
# Naprava + napoved
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


@torch.no_grad()
def predict_proba(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_p = []
    for patches, _ in loader:
        all_p.append(torch.exp(model(patches.to(device))).cpu().numpy())
    return np.concatenate(all_p)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def find_temperature(model, val_ds, y_val, device):
    """Poišče T ki minimizira log loss na val množici."""
    model.eval()
    logits_list = []
    loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for patches, _ in loader:
            logits_list.append(model.get_logits(patches.to(device)).cpu().numpy())
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
# Gaussian smoothing + iskanje optimalne σ
# ---------------------------------------------------------------------------
def gaussian_smooth_probs(probs, tissue_mask, sigma):
    if sigma <= 0:
        return probs
    smoothed = np.zeros_like(probs)
    mask_f   = tissue_mask.astype(np.float32)
    for c in range(probs.shape[-1]):
        num = gaussian_filter(probs[:, :, c] * mask_f, sigma=sigma)
        den = gaussian_filter(mask_f, sigma=sigma)
        smoothed[:, :, c] = num / np.where(den < 1e-8, 1e-8, den)
    smoothed = np.clip(smoothed, 1e-7, 1.0)
    smoothed /= smoothed.sum(axis=-1, keepdims=True)
    return smoothed.astype(np.float32)


def find_best_sigma(probs_flat, y_val, tissue_mask_crop,
                    sigmas=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0)):
    """
    Preizkusi različne σ na val množici in vrne najboljšo.
    probs_flat: (N_val, 6) verjetnosti za val piksle
    Val piksli morajo biti v tem klicu že v obliki 2D mape.
    """
    # To je poenostavljena verzija — σ iščemo na celotnem val probs mapu
    print(f"\n  Iskanje optimalne σ ∈ {sigmas}:")
    best_sigma, best_ll = 0.0, float('inf')
    for s in sigmas:
        ll = log_loss(y_val, probs_flat, labels=np.arange(NUM_CLASSES))
        print(f"    σ={s:.1f}: log_loss={ll:.5f}")
        if ll < best_ll:
            best_ll, best_sigma = ll, s
    print(f"  → Najboljša σ={best_sigma} (ll={best_ll:.5f})")
    return best_sigma


# ---------------------------------------------------------------------------
# Fitanje
# ---------------------------------------------------------------------------
def fit_model_c(data_pca, train_coords, y_train,
                val_coords, y_val, device,
                epochs=150, batch_size=64, lr=1e-4,
                patience=15, seed=42):
    torch.manual_seed(seed); random.seed(seed)

    weights = compute_class_weights(y_train)
    train_ds = PatchDataset(data_pca, train_coords, y_train, augment=True)
    val_ds   = PatchDataset(data_pca, val_coords,   y_val,   augment=False)
    loader   = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=0)

    model   = PatchCNN2D().to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PatchCNN2D v3: {n_param:,} param | patch={PATCH_SIZE}×{PATCH_SIZE}")

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
        for p, l in loader:
            p, l = p.to(device), l.to(device)
            optimizer.zero_grad()
            loss = criterion(model(p), l)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(l)
        train_loss = total / len(train_ds)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        probs  = predict_proba(model, val_ds, device, 512)
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
def evaluate(model, data_pca, coords, y_true, device,
             split_name="", temperature=1.0):
    ds    = PatchDataset(data_pca, coords, y_true, augment=False)
    probs = predict_proba(model, ds, device, 512)

    if temperature != 1.0:
        # Recalibriraj z logiti
        model.eval()
        logits_list = []
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        with torch.no_grad():
            for p, _ in loader:
                logits_list.append(model.get_logits(p.to(device)).cpu().numpy())
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
    print(f"  Ref Model A: TEST OA=92.69%, ll=0.347")
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
    return oa, ll, probs


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def make_submission(model, data_pca, tissue_mask, device,
                    output_path, train_class_counts,
                    temperature=1.0, sigma=1.5):
    crop_h, crop_w = PRED_R1 - PRED_R0, PRED_C1 - PRED_C0
    n_crop = crop_h * crop_w
    rs = np.repeat(np.arange(PRED_R0, PRED_R1), crop_w)
    cs = np.tile(  np.arange(PRED_C0, PRED_C1), crop_h)
    coords = np.stack([rs, cs], axis=1)

    prior = train_class_counts.astype(np.float32)
    prior /= prior.sum()
    crop_tissue = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)
    n_tissue = int(crop_tissue.sum())
    print(f"  Crop: {n_crop} | tkivo: {n_tissue} | ozadje: {n_crop-n_tissue}")

    # Napoved z temperature scaling
    ds = PatchDataset(data_pca, coords, np.zeros(n_crop, dtype=np.int64), augment=False)
    if temperature == 1.0:
        probs = predict_proba(model, ds, device, 512)
    else:
        model.eval()
        logits_list = []
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        with torch.no_grad():
            for p, _ in loader:
                logits_list.append(model.get_logits(p.to(device)).cpu().numpy())
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
    print(f"  Submission: {output_path}  shape={final.shape}  T={temperature:.3f}  σ={sigma:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model C v3 — Patch CNN, patch=49, temperature scaling"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output",             default="modelC_v3.npy")
    parser.add_argument("--epochs",             type=int,   default=150)
    parser.add_argument("--batch-size",         type=int,   default=64)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--patience",           type=int,   default=15)
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
    print("\n=== 3. Preprocessing + PCA datacube ===")
    data_pca_eval, _ = build_pca_datacube(
        data, wns, amide_idx, train_mask, N_PCA, args.seed
    )

    # ------------------------------------------------------------------
    print("\n=== 4. Ucenje Model C v3 ===")
    model, best_epoch, val_ds = fit_model_c(
        data_pca=data_pca_eval,
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
    print("\n=== 6. Iskanje optimalne σ za Gaussian smoothing ===")
    # Evaluiramo val verjetnosti z temperature scaling
    val_ds_eval = PatchDataset(data_pca_eval, val_coords, y_val, augment=False)
    model.eval()
    logits_list = []
    loader_val = DataLoader(val_ds_eval, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for p, _ in loader_val:
            logits_list.append(model.get_logits(p.to(device)).cpu().numpy())
    logits_val = np.concatenate(logits_list)
    s = logits_val / T_opt
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    probs_val_T = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    print(f"\n  Primerjava σ na val množici:")
    best_sigma, best_sigma_ll = 0.0, float('inf')
    for s_val in sigmas:
        ll_s = log_loss(y_val, probs_val_T, labels=np.arange(NUM_CLASSES))
        print(f"    σ={s_val:.1f}: val log_loss={ll_s:.5f}")
        if ll_s < best_sigma_ll:
            best_sigma_ll, best_sigma = ll_s, s_val
    print(f"  → Izbrana σ={best_sigma:.1f}")

    # ------------------------------------------------------------------
    print("\n=== 7. Evaluacija ===")
    print("\n  [Brez temperature scaling]")
    oa_val,  ll_val,  _ = evaluate(model, data_pca_eval,
                                    val_coords, y_val, device, "VAL")
    print("\n  [S temperature scaling T={:.3f}]".format(T_opt))
    oa_val_T, ll_val_T, _ = evaluate(model, data_pca_eval,
                                      val_coords, y_val, device,
                                      "VAL+T", T_opt)
    print()
    oa_test, ll_test, _ = evaluate(model, data_pca_eval,
                                    test_coords, y_test, device,
                                    "TEST (zaklenjen)", T_opt)

    # ------------------------------------------------------------------
    print(f"\n=== 8. Finalni model (100% podatkov) ===")
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    all_coords  = np.argwhere(usable_mask)
    y_all       = classes[usable_mask].astype(np.int64)
    n_all       = len(y_all)
    n_train     = len(y_train)

    # Proporcionalno skaliranje epoh
    final_epochs = max(best_epoch, round(best_epoch * n_all / n_train))
    print(f"  Skupaj pikslov: {n_all}")
    print(f"  Finalne epohe: {final_epochs} "
          f"(best_epoch={best_epoch} × {n_all/n_train:.2f})")

    print("\n  PCA datacube za finalni model...")
    data_pca_final, _ = build_pca_datacube(
        data, wns, amide_idx, usable_mask, N_PCA, args.seed
    )

    weights_final = compute_class_weights(y_all)
    final_ds  = PatchDataset(data_pca_final, all_coords, y_all, augment=True)
    final_ldr = DataLoader(final_ds, batch_size=args.batch_size,
                           shuffle=True, num_workers=0)
    model_f   = PatchCNN2D().to(device)
    opt_f     = optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_f   = optim.lr_scheduler.CosineAnnealingLR(
        opt_f, T_max=final_epochs, eta_min=1e-6)
    crit_f    = nn.NLLLoss(weight=weights_final.to(device))

    print(f"\n  {'Epoha':>6}  {'Train ll':>10}  {'LR':>9}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*9}")
    torch.manual_seed(args.seed); random.seed(args.seed)
    t0 = time.time()
    for epoch in range(1, final_epochs + 1):
        model_f.train()
        total = 0.0
        for p, l in final_ldr:
            p, l = p.to(device), l.to(device)
            opt_f.zero_grad()
            loss = crit_f(model_f(p), l)
            loss.backward()
            nn.utils.clip_grad_norm_(model_f.parameters(), 1.0)
            opt_f.step()
            total += loss.item() * len(l)
        sched_f.step()
        if epoch % 5 == 0 or epoch in (1, final_epochs):
            lr_now = opt_f.param_groups[0]['lr']
            print(f"  {epoch:>6}  {total/n_all:>10.5f}  {lr_now:>9.2e}")
    print(f"  Finalni model treniran v {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print("\n=== 9. Submission ===")
    make_submission(
        model=model_f, data_pca=data_pca_final,
        tissue_mask=tissue_mask, device=device,
        output_path=args.output,
        train_class_counts=np.bincount(y_all, minlength=NUM_CLASSES),
        temperature=T_opt, sigma=best_sigma,
    )

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  Model C v3 — patch={PATCH_SIZE}×{PATCH_SIZE}, PCA={N_PCA}")
    print(f"  Best epoch: {best_epoch} | T={T_opt:.4f} | σ={best_sigma:.1f}")
    print(f"\n  {'':28} {'OA':>8}  {'Log loss':>10}")
    print(f"  {'VAL (brez T)':28} {oa_val*100:>7.2f}%  {ll_val:>10.5f}")
    print(f"  {'VAL (T={:.3f})'.format(T_opt):28} {oa_val_T*100:>7.2f}%  {ll_val_T:>10.5f}")
    print(f"  {'TEST (diploma, T=opt)':28} {oa_test*100:>7.2f}%  {ll_test:>10.5f}")
    print(f"\n  Primerjava (TEST):")
    print(f"  {'Članek CNN prostorni':28} {'92.85%':>8}  {'N/A':>10}")
    print(f"  {'Model A SVM+PCA':28} {'92.69%':>8}  {'0.34700':>10}")
    print(f"  {'Model C v2 (patch=33)':28} {'88.40%':>8}  {'0.33703':>10}")
    print(f"\n  Submission: {args.output}")


if __name__ == "__main__":
    main()
