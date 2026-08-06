"""
Model C — Patch CNN s prostornim kontekstom  (v2: pravilna zasnova)
=====================================================================
Popravki glede na v1:

  Problem 1 — Brez Dropoutа:
    V1: model memoriral patche v 1 epohi (train_loss: 0.22 → 0.03 v 1 epohi)
    V2: Dropout2d(0.3) po vsakem conv bloku + Dropout(0.4) pred FC

  Problem 2 — LR=1e-3 previsok:
    V1: ista napaka kot Model B v1 CNN (nestabilna val metrika)
    V2: LR=1e-4, ista rešitev kot Model B v2

  Problem 3 — Oversampling pokvari kalibrацijo:
    V1: uniform distribution po oversamplu ≠ realna val porazdelitev
    V2: weighted NLLLoss namesto oversamplinga (ista lekcija kot Model B)

  Problem 4 — Pomanjkanje raznovrstnosti patchev:
    Članek: 504 TMA coresh od različnih pacientov → pravi raznovrstni patchi
    Mi: 1 slide → patchi iste komponente so skoraj identični → memorization
    V2: DATA AUGMENTATION — random flip (H, V) + rotacija (90°, 180°, 270°)
        Za tkivne slike smiselno: tkivo nima "pravilne" orientacije.
        Efekt: 8× več efektivnih patchev, model se ne more naučiti na pamet.

Ostalo ostaja enako kot v1:
  - Arhitektura: zvesta članku (Conv2d × 3, MaxPool × 2, FC(128), Softmax(6))
  - Preprocessing: rubberband + Amide I na anotiranih, Amide I na vseh → PCA(16)
  - Post-processing: Gaussian glajenje verjetnosti (σ=1.5)
  - Split: prostorski 3-way za eval, 100% za submission
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
PATCH_SIZE = 33
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


def find_amide_i_index(wns: np.ndarray) -> int:
    idx = int(np.argmin(np.abs(wns - AMIDE_I_TARGET_WN)))
    print(f"  Amide I: target={AMIDE_I_TARGET_WN:.1f} | actual={wns[idx]:.2f} | idx={idx}")
    return idx


def make_prediction_crop_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1] = True
    return mask


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _rubberband_single(spectrum: np.ndarray) -> np.ndarray:
    n = len(spectrum)
    x = np.arange(n, dtype=np.float64)
    y = spectrum.astype(np.float64)
    lower = []
    for i in range(n):
        p = (x[i], y[i])
        while len(lower) >= 2:
            o, a = lower[-2], lower[-1]
            cross = (a[0]-o[0])*(p[1]-o[1]) - (a[1]-o[1])*(p[0]-o[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    lx = np.array([p[0] for p in lower])
    ly = np.array([p[1] for p in lower])
    return (y - np.interp(x, lx, ly)).astype(np.float32)


def rubberband_correction(spectra: np.ndarray) -> np.ndarray:
    out = np.empty_like(spectra, dtype=np.float32)
    for i in range(len(spectra)):
        out[i] = _rubberband_single(spectra[i])
    return out


def amide_i_normalize(spectra: np.ndarray, amide_idx: int,
                      eps: float = 1e-6) -> np.ndarray:
    vals = spectra[:, amide_idx].astype(np.float64)
    safe = np.where(vals > eps, vals, eps)
    return (spectra / safe[:, np.newaxis]).astype(np.float32)


def build_pca_datacube(data, wns, amide_idx, train_mask,
                       n_components=N_PCA, seed=42):
    H, W, D = data.shape
    print(f"  PCA datacube: {H}×{W}×{D} → {H}×{W}×{n_components}")

    # Amide I na vseh pikslih (hitro, vektorizirano)
    print(f"  Amide I normalizacija vseh {H*W:,} pikslov...")
    t0 = time.time()
    flat = data.reshape(-1, D)
    flat_amide = amide_i_normalize(flat, amide_idx)
    print(f"    → {time.time()-t0:.1f}s")

    # Rubberband samo za train anotirane piksle
    train_coords = np.argwhere(train_mask)
    N_train = len(train_coords)
    print(f"  Rubberband za {N_train:,} train pikslov...")
    t0 = time.time()
    train_flat_idx = train_coords[:, 0] * W + train_coords[:, 1]
    train_spectra = flat[train_flat_idx]
    train_pp = rubberband_correction(amide_i_normalize(train_spectra, amide_idx))
    print(f"    → {time.time()-t0:.1f}s")

    # Vstavi train pp nazaj
    flat_pp = flat_amide.copy()
    flat_pp[train_flat_idx] = train_pp

    # PCA fit samo na train pikslih
    print(f"  PCA fit na {N_train} train pikslih...")
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_pp)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA pojasnjena varianca: {var*100:.2f}%")

    # PCA transform vseh pikslov
    print(f"  PCA transform vseh {H*W:,} pikslov...")
    t0 = time.time()
    data_pca = pca.transform(flat_pp).reshape(H, W, n_components).astype(np.float32)
    print(f"    → {time.time()-t0:.1f}s | shape={data_pca.shape}")

    return data_pca, pca


# ---------------------------------------------------------------------------
# Dataset z data augmentation
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """
    Patch ekstrakcija on-the-fly iz PCA datacuba.

    augment=True (samo za trening!):
      - Random horizontal flip (p=0.5)
      - Random vertical flip   (p=0.5)
      - Random 90° rotacija    (k ∈ {0,1,2,3})
    Skupaj 8 možnih transformacij → 8× več efektivnih patchev.

    Biološka utemeljitev: tkivo nima naravne orientacije, zato
    flip/rotacija ne ustvari nerealistične vzorce.
    """

    def __init__(self, data_pca: np.ndarray, coords: np.ndarray,
                 labels: np.ndarray, patch_size: int = PATCH_SIZE,
                 augment: bool = False):
        self.patch_size = patch_size
        self.pad        = patch_size // 2
        self.labels     = labels
        self.coords     = coords
        self.augment    = augment

        # Zero padding za robne piksle
        self.padded = np.pad(
            data_pca,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='constant', constant_values=0.0
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        r, c = self.coords[idx]
        rp, cp = r + self.pad, c + self.pad
        patch = self.padded[rp-self.pad:rp+self.pad+1,
                            cp-self.pad:cp+self.pad+1]
        # (H, W, C) → (C, H, W)
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1).copy())

        # Augmentacija (samo za train)
        if self.augment:
            if random.random() > 0.5:
                patch_t = torch.flip(patch_t, dims=[1])   # horizontal
            if random.random() > 0.5:
                patch_t = torch.flip(patch_t, dims=[2])   # vertical
            k = random.randint(0, 3)
            if k > 0:
                patch_t = torch.rot90(patch_t, k=k, dims=[1, 2])

        label_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return patch_t, label_t


# ---------------------------------------------------------------------------
# Tehtana loss (namesto oversamplinga)
# ---------------------------------------------------------------------------
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Arhitektura: PatchCNN2D v2 (+ Dropout)
# ---------------------------------------------------------------------------
class PatchCNN2D(nn.Module):
    """
    2D CNN zvest članku + Dropout za regularizacijo.

    Vhod:  (batch, 16, 33, 33)
    Izhod: (batch, 6) log-verjetnosti

    Arhitektura (Fig. 3 + Dropout):
      Conv2d(16→32, 3×3) + BN + ReLU + Dropout2d(0.3) + MaxPool(2)
      Conv2d(32→64, 3×3) + BN + ReLU + Dropout2d(0.3)
      Conv2d(64→64, 3×3) + BN + ReLU + Dropout2d(0.3) + MaxPool(2)
      Flatten
      Linear(4096→128) + ReLU + Dropout(0.4)
      Linear(128→6) → LogSoftmax
    """

    def __init__(self, n_channels: int = N_PCA,
                 num_classes: int = NUM_CLASSES,
                 patch_size: int = PATCH_SIZE,
                 dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),                           # 33 → 16

            # Blok 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout),

            # Blok 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),                           # 16 → 8
        )

        spatial_out = patch_size // 2 // 2            # 33→16→8
        fc_in = 64 * spatial_out * spatial_out        # 4096

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 128), nn.ReLU(),
            nn.Dropout(dropout + 0.1),                 # 0.4
            nn.Linear(128, num_classes),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_logits(self, x):
        return self.classifier(self.features(x))

    def forward(self, x):
        return self.log_softmax(self.get_logits(x))


# ---------------------------------------------------------------------------
# Naprava
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS (Apple Silicon)")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


# ---------------------------------------------------------------------------
# Napoved verjetnosti
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model: nn.Module, dataset: PatchDataset,
                  device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_probs = []
    for patches, _ in loader:
        probs = torch.exp(model(patches.to(device)))
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Post-processing: Gaussian glajenje
# ---------------------------------------------------------------------------
def gaussian_smooth_probs(probs: np.ndarray, tissue_mask: np.ndarray,
                          sigma: float = 1.5) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Fitanje
# ---------------------------------------------------------------------------
def fit_model_c(
    data_pca: np.ndarray,
    train_coords: np.ndarray, y_train: np.ndarray,
    val_coords:   np.ndarray, y_val:   np.ndarray,
    device: torch.device,
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 1e-4,           # v2: 10× nižji LR
    patience: int = 15,
    seed: int = 42,
) -> tuple:
    torch.manual_seed(seed)
    random.seed(seed)

    # Tehtana loss (namesto oversamplinga)
    weights = compute_class_weights(y_train)

    # Dataseti — train Z augmentacijo, val BREZ
    train_ds = PatchDataset(data_pca, train_coords, y_train, augment=True)
    val_ds   = PatchDataset(data_pca, val_coords,   y_val,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    model   = PatchCNN2D().to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PatchCNN2D v2: {n_param:,} param | LR={lr} | batch={batch_size}")
    print(f"  Augmentacija: flip H/V + rot90 (8× raznovrstnost)")

    criterion = nn.NLLLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_loss    = float('inf')
    best_state       = None
    best_epoch       = epochs
    patience_counter = 0

    print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*9}  {'─'*9}  {'─'*9}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # Trening
        model.train()
        total_loss = 0.0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(patches), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        train_loss = total_loss / len(train_ds)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        # Validacija
        probs  = predict_proba(model, val_ds, device, batch_size=512)
        preds  = np.argmax(probs, axis=1)
        val_oa = accuracy_score(y_val, preds)
        val_ll = log_loss(y_val, probs, labels=np.arange(NUM_CLASSES))

        print(f"  {epoch:>6}  {train_loss:>11.5f}  {val_oa*100:>8.2f}%  {val_ll:>9.5f}  {lr_now:>9.2e}")

        if val_ll < best_val_loss - 1e-6:
            best_val_loss    = val_ll
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch       = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping pri epohi {epoch} "
                      f"(best={best_val_loss:.5f}, epoha={best_epoch})")
                break

    print(f"\n  Trening: {time.time()-t0:.1f}s")
    if best_state:
        model.load_state_dict(best_state)
        print(f"  Naložene najboljše uteži: epoha={best_epoch}, val_ll={best_val_loss:.5f}")

    return model, best_epoch


# ---------------------------------------------------------------------------
# Evaluacija
# ---------------------------------------------------------------------------
def evaluate(model, data_pca, coords, y_true, device,
             split_name="") -> tuple:
    ds    = PatchDataset(data_pca, coords, y_true, augment=False)
    probs = predict_proba(model, ds, device, batch_size=512)
    preds = np.argmax(probs, axis=1)
    oa    = accuracy_score(y_true, preds)
    ll    = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))

    print(f"\n  ── {split_name} ──")
    print(f"  OA:       {oa*100:.2f}%   (članek HD CNN: 92.85%)")
    print(f"  Log loss: {ll:.5f}")
    print(f"  Ref — Model A: TEST OA=92.69%, ll=0.347")
    print(f"  Ref — Model B: TEST OA=90.79%, ll=0.400")

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
def make_submission(model, data_pca, tissue_mask, device,
                    output_path, train_class_counts, sigma=1.5):
    crop_h, crop_w = PRED_R1 - PRED_R0, PRED_C1 - PRED_C0
    n_crop = crop_h * crop_w

    rs = np.repeat(np.arange(PRED_R0, PRED_R1), crop_w)
    cs = np.tile(  np.arange(PRED_C0, PRED_C1), crop_h)
    crop_coords = np.stack([rs, cs], axis=1)

    prior = train_class_counts.astype(np.float32)
    prior /= prior.sum()
    crop_tissue = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)
    n_tissue = int(crop_tissue.sum())
    print(f"  Crop: {n_crop} | tkivo: {n_tissue} | ozadje: {n_crop-n_tissue}")

    dummy_labels = np.zeros(n_crop, dtype=np.int64)
    ds    = PatchDataset(data_pca, crop_coords, dummy_labels, augment=False)
    probs = predict_proba(model, ds, device, batch_size=512)

    # Ozadje dobi class prior
    probs[~crop_tissue] = prior

    prob_map = probs.reshape(crop_h, crop_w, NUM_CLASSES)

    # Gaussian smoothing
    crop_tissue_2d = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1]
    if sigma > 0:
        print(f"  Gaussian smoothing (sigma={sigma})...")
        smoothed = gaussian_smooth_probs(prob_map, crop_tissue_2d, sigma)
        final = prob_map.copy()
        final[crop_tissue_2d] = smoothed[crop_tissue_2d]
    else:
        final = prob_map

    final = np.clip(final, 1e-7, 1.0)
    final /= final.sum(axis=-1, keepdims=True)
    np.save(output_path, final.astype(np.float32))
    print(f"  Submission: {output_path}  shape={final.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model C v2 — Patch CNN + Dropout + Augmentacija + Weighted loss"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output",             default="modelC_v2.npy")
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
    print("\n=== 3. Preprocessing + PCA datacube ===")
    data_pca_eval, _ = build_pca_datacube(
        data, wns, amide_idx, train_mask, N_PCA, args.seed
    )

    # ------------------------------------------------------------------
    print("\n=== 4. Ucenje Model C v2 ===")
    model, best_epoch = fit_model_c(
        data_pca=data_pca_eval,
        train_coords=train_coords, y_train=y_train,
        val_coords=val_coords,     y_val=y_val,
        device=device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, seed=args.seed,
    )

    # ------------------------------------------------------------------
    print("\n=== 5. Evaluacija ===")
    oa_val,  ll_val  = evaluate(model, data_pca_eval,
                                 val_coords,  y_val,  device, "VAL")
    print()
    # TEST — odpre se enkrat, rezultat za diplomo
    oa_test, ll_test = evaluate(model, data_pca_eval,
                                 test_coords, y_test, device, "TEST (zaklenjen)")

    # ------------------------------------------------------------------
    print(f"\n=== 6. Finalni model (100% anotiranih, {best_epoch} epoh) ===")
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    all_coords  = np.argwhere(usable_mask)
    y_all       = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov: {len(y_all)}")

    print("\n  PCA datacube za finalni model...")
    data_pca_final, _ = build_pca_datacube(
        data, wns, amide_idx, usable_mask, N_PCA, args.seed
    )

    weights_final = compute_class_weights(y_all)
    final_ds      = PatchDataset(data_pca_final, all_coords, y_all, augment=True)
    final_loader  = DataLoader(final_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)

    model_final  = PatchCNN2D().to(device)
    optimizer_f  = optim.Adam(model_final.parameters(),
                               lr=args.lr, weight_decay=1e-4)
    scheduler_f  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_f, T_max=best_epoch, eta_min=1e-6
    )
    criterion_f  = nn.NLLLoss(weight=weights_final.to(device))

    print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'LR':>9}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*9}")
    torch.manual_seed(args.seed); random.seed(args.seed)
    t0 = time.time()
    for epoch in range(1, best_epoch + 1):
        model_final.train()
        total = 0.0
        for patches, labels in final_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer_f.zero_grad()
            loss = criterion_f(model_final(patches), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model_final.parameters(), 1.0)
            optimizer_f.step()
            total += loss.item() * len(labels)
        scheduler_f.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == best_epoch:
            lr_now = optimizer_f.param_groups[0]['lr']
            print(f"  {epoch:>6}  {total/len(final_ds):>11.5f}  {lr_now:>9.2e}")
    print(f"  Finalni model treniran v {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print("\n=== 7. Submission ===")
    make_submission(
        model=model_final, data_pca=data_pca_final,
        tissue_mask=tissue_mask, device=device,
        output_path=args.output,
        train_class_counts=np.bincount(y_all, minlength=NUM_CLASSES),
        sigma=args.sigma,
    )

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  Model C v2 — Patch CNN + Dropout + Augmentacija + Weighted loss")
    print(f"  Best epoch: {best_epoch} | Sigma: {args.sigma}")
    print(f"\n  {'':25} {'OA':>8}  {'Log loss':>10}")
    print(f"  {'VAL':25} {oa_val*100:>7.2f}%  {ll_val:>10.5f}")
    print(f"  {'TEST (diploma)':25} {oa_test*100:>7.2f}%  {ll_test:>10.5f}")
    print(f"\n  Primerjava (TEST):")
    print(f"  {'Članek CNN prostorni':25} {'92.85%':>8}  {'N/A':>10}")
    print(f"  {'Model A SVM+PCA':25} {'92.69%':>8}  {'0.34700':>10}")
    print(f"  {'Model B MLP':25} {'90.79%':>8}  {'0.40008':>10}")
    print(f"\n  Submission: {args.output}")


if __name__ == "__main__":
    main()
