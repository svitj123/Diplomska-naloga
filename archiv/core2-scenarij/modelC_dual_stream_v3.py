"""
Model C — Dual-Stream CNN v3 (ensemble v1 arhitekture)
=======================================================
Lekcija iz v2: večja arhitektura + N_PCA=32 je SLABŠA od v1.
  - v1: 159k param, N_PCA=16, best_epoch=7, TEST ll=0.222
  - v2: 373k param, N_PCA=32, best_epoch=3-5, TEST ll=0.531  ← slabše!

Razlog: premalo podatkov (23k spektrov) → večji model se hitreje
prepeče. N_PCA=32 dodaja samo šum (16 komponent → 97.47% variance,
17-32 → le ~1% variance = šum ki moti CNN).

V3 strategija: OHRANIMO v1 arhitekturo (ki je dokazano dobra) +
dodamo ENSEMBLE (3 modeli, različni seedi).

Zakaj ensemble pomaga:
  - Vsak model konvergira k malo drugačnemu lokalnemu minimumu
  - Povprečenje logitov reducira varianco napovedi
  - Bolje kalibrirane verjetnosti → manjši log loss
  - Brez spremembe arhitekture = brez tveganja regresije

Pričakovano: TEST ll 0.222 → ~0.18-0.20 (-10% do -15%)
"""

import argparse
import random
import time

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
PATCH_SIZE = 33
N_PCA      = 16   # v3: nazaj na 16 (v2's 32 je bilo slabše!)
SPEC_DIM   = 187


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
    H, W, D = data.shape
    print(f"  Preprocessing: {H}×{W}×{D} → PCA({n_components})")

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

    print(f"  PCA({n_components}) fit na {len(train_idx)} pikslih...")
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_pp)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    print(f"  PCA transform {H*W:,} pikslov...")
    t0 = time.time()
    data_pca = pca.transform(flat_pp).reshape(H, W, n_components).astype(np.float32)
    print(f"    → {time.time()-t0:.1f}s")

    return data_pca, flat_pp, pca


# ---------------------------------------------------------------------------
# Dataset (z spektralnim šumom)
# ---------------------------------------------------------------------------
class DualStreamDataset(Dataset):
    def __init__(self, data_pca, flat_pp, coords, labels,
                 image_width, patch_size=PATCH_SIZE,
                 augment=False, spec_noise=0.0):
        self.pad        = patch_size // 2
        self.labels     = labels
        self.coords     = coords
        self.augment    = augment
        self.spec_noise = spec_noise
        self.flat_pp    = flat_pp
        self.flat_idx   = coords[:, 0] * image_width + coords[:, 1]
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

        spectrum_t = torch.from_numpy(self.flat_pp[self.flat_idx[idx]].copy())

        if self.augment and self.spec_noise > 0:
            spectrum_t = spectrum_t + torch.randn_like(spectrum_t) * self.spec_noise

        return patch_t, spectrum_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Tehtana loss
# ---------------------------------------------------------------------------
def compute_class_weights(y):
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Arhitektura: IDENTIČNA v1 (dokazano deluje!)
# ---------------------------------------------------------------------------
class DualStreamCNN(nn.Module):
    """
    ISTA arhitektura kot v1 (159k param).
    v1 je dosegel TEST OA=94.41%, ll=0.222 — ne dotikamo se!

    Prostorni tok:
      Conv(16→32) + BN + ReLU + Drop(0.3) + MaxPool → (32, 16, 16)
      Conv(32→64) + BN + ReLU + Drop(0.3)           → (64, 16, 16)
      Conv(64→64) + BN + ReLU + Drop(0.3) + MaxPool → (64,  8,  8)
      AdaptiveAvgPool(1)                             → 64

    Spektralni tok:
      Linear(187→256) + BN + ReLU + Drop(0.4)
      Linear(256→128) + BN + ReLU + Drop(0.3)
      Linear(128→64)  + BN + ReLU                   → 64

    Fuzija: Concat(128) → Linear(128→64) + ReLU + Drop(0.3) → Linear(64→6)
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
# Naprava
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


# ---------------------------------------------------------------------------
# Napoved + logiti
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_logits_array(model, dataset, device, batch_size=512):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_l = []
    for patches, spectra, _ in loader:
        all_l.append(
            model.get_logits(patches.to(device), spectra.to(device)).cpu().numpy()
        )
    return np.concatenate(all_l)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def find_temperature(logits, y):
    def neg_ll(T):
        s = logits / T
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-9, 1.0)
        return -np.mean(np.log(p[np.arange(len(y)), y]))
    res = minimize_scalar(neg_ll, bounds=(0.1, 10.0), method='bounded')
    T = res.x
    print(f"  Temperature: T={T:.4f}  "
          f"log_loss: {neg_ll(1.0):.5f} → {neg_ll(T):.5f}")
    return T


def apply_temperature(logits, T):
    s = logits / T
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Gaussian smoothing
# ---------------------------------------------------------------------------
def gaussian_smooth_probs(probs, tissue_mask, sigma):
    if sigma <= 0: return probs
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
# Trening enega modela
# ---------------------------------------------------------------------------
def train_single(data_pca, flat_pp, W,
                 train_coords, y_train,
                 val_ds, y_val,
                 device, weights,
                 epochs, batch_size, lr, patience, seed, spec_noise):
    torch.manual_seed(seed); random.seed(seed)

    train_ds = DualStreamDataset(data_pca, flat_pp, train_coords, y_train,
                                  W, augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=0)
    model     = DualStreamCNN().to(device)
    criterion = nn.NLLLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

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

        # Val evaluacija
        val_logits = get_logits_array(model, val_ds, device)
        s = val_logits - val_logits.max(axis=1, keepdims=True)
        e = np.exp(s)
        val_probs = e / e.sum(axis=1, keepdims=True)
        val_oa = accuracy_score(y_val, np.argmax(val_probs, axis=1))
        val_ll = log_loss(y_val, val_probs, labels=np.arange(NUM_CLASSES))

        print(f"  {epoch:>4}  {train_loss:>10.5f}  {val_oa*100:>8.2f}%  "
              f"{val_ll:>9.5f}  {lr_now:>9.2e}")

        if val_ll < best_val_loss - 1e-6:
            best_val_loss = val_ll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping: epoha {epoch} "
                      f"(best={best_val_loss:.5f}, epoha={best_epoch})")
                break

    model.load_state_dict(best_state)
    print(f"  Treniran v {time.time()-t0:.1f}s | best epoha={best_epoch}, "
          f"val_ll={best_val_loss:.5f}")
    return model, best_epoch


# ---------------------------------------------------------------------------
# Evaluacija
# ---------------------------------------------------------------------------
def evaluate(probs, y_true, split_name="", T=1.0):
    preds = np.argmax(probs, axis=1)
    oa    = accuracy_score(y_true, preds)
    ll    = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))
    t_str = f" [T={T:.3f}]" if T != 1.0 else ""
    print(f"\n  ── {split_name}{t_str} ──")
    print(f"  OA:       {oa*100:.2f}%")
    print(f"  Log loss: {ll:.5f}")
    print(f"  Ref — Dual v1: TEST OA=94.41%, ll=0.222")
    per_class = []
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"    Razred {c}: N/A"); continue
        acc_c = (preds[mask] == y_true[mask]).mean()
        per_class.append(acc_c)
        print(f"    Razred {c}: {acc_c*100:.2f}%  (n={mask.sum()})")
    print(f"  Macro OA: {np.mean(per_class)*100:.2f}%")
    cm = confusion_matrix(y_true, preds, labels=np.arange(NUM_CLASSES))
    print(f"  Matrika zmede:\n{cm}")
    return oa, ll


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Dual-Stream v3: v1 arhitektura + ensemble (pravilna kombinacija)"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output",             default="modelC_dual_v3.npy")
    parser.add_argument("--epochs",             type=int,   default=150)
    parser.add_argument("--batch-size",         type=int,   default=128)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--patience",           type=int,   default=20)
    parser.add_argument("--sigma",              type=float, default=1.5)
    parser.add_argument("--n-ensemble",         type=int,   default=5,
                        help="Število modelov v ensemblu (5 = optimalno)")
    parser.add_argument("--spec-noise",         type=float, default=0.01)
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
    print(f"\n=== 3. Preprocessing (N_PCA={N_PCA} — enako kot v1) ===")
    data_pca_eval, flat_pp_eval, _ = build_preprocessed_datacube(
        data, wns, amide_idx, train_mask, N_PCA, args.seed
    )
    weights = compute_class_weights(y_train)

    n_param = sum(p.numel() for p in DualStreamCNN().parameters() if p.requires_grad)
    seeds   = list(range(args.seed, args.seed + args.n_ensemble))
    print(f"\n  DualStreamCNN v3: {n_param:,} param (identična v1 arhitekturi)")
    print(f"  Ensemble: {args.n_ensemble} modelov | seedi: {seeds}")
    print(f"  Spektralni šum: σ={args.spec_noise}")

    # Skupni val/test dataseti (brez augmentacije)
    val_ds  = DualStreamDataset(data_pca_eval, flat_pp_eval, val_coords,
                                 y_val,  W, augment=False)
    test_ds = DualStreamDataset(data_pca_eval, flat_pp_eval, test_coords,
                                 y_test, W, augment=False)

    # ------------------------------------------------------------------
    print(f"\n=== 4. Ensemble trening ({args.n_ensemble} modelov) ===")

    val_logits_list  = []
    test_logits_list = []
    best_epochs      = []

    for i, seed in enumerate(seeds):
        print(f"\n  ── Model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        model, best_epoch = train_single(
            data_pca=data_pca_eval, flat_pp=flat_pp_eval, W=W,
            train_coords=train_coords, y_train=y_train,
            val_ds=val_ds, y_val=y_val,
            device=device, weights=weights,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience,
            seed=seed, spec_noise=args.spec_noise,
        )
        best_epochs.append(best_epoch)
        val_logits_list.append(get_logits_array(model, val_ds, device))
        test_logits_list.append(get_logits_array(model, test_ds, device))

    # Ensemble: povprečenje logitov
    val_logits_avg  = np.mean(val_logits_list,  axis=0)
    test_logits_avg = np.mean(test_logits_list, axis=0)

    print(f"\n  Best epochs: {best_epochs} (avg={np.mean(best_epochs):.1f})")

    # ------------------------------------------------------------------
    print("\n=== 5. Temperature scaling (na ensemble logitih) ===")
    T_opt = find_temperature(val_logits_avg, y_val)

    # ------------------------------------------------------------------
    print("\n=== 6. Evaluacija ===")
    val_probs  = apply_temperature(val_logits_avg,  T_opt)
    test_probs = apply_temperature(test_logits_avg, T_opt)

    oa_val,  ll_val  = evaluate(val_probs,  y_val,  "VAL",              T_opt)
    print()
    oa_test, ll_test = evaluate(test_probs, y_test, "TEST (zaklenjen)", T_opt)

    # ------------------------------------------------------------------
    n_all       = int(((classes != -1) & (~prediction_crop_mask)).sum())
    avg_best    = round(np.mean(best_epochs))
    final_epochs = max(avg_best, round(avg_best * n_all / len(y_train)))

    print(f"\n=== 7. Finalni ensemble (100% podatkov, {final_epochs} epoh × "
          f"{args.n_ensemble} modelov) ===")

    usable_mask = (classes != -1) & (~prediction_crop_mask)
    all_coords  = np.argwhere(usable_mask)
    y_all       = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov: {len(y_all)}")

    print("\n  Preprocessing za finalni model...")
    data_pca_final, flat_pp_final, _ = build_preprocessed_datacube(
        data, wns, amide_idx, usable_mask, N_PCA, args.seed
    )
    weights_final = compute_class_weights(y_all)

    # Crop koordinate
    rs = np.repeat(np.arange(PRED_R0, PRED_R1), PRED_C1 - PRED_C0)
    cs = np.tile(  np.arange(PRED_C0, PRED_C1), PRED_R1 - PRED_R0)
    crop_coords = np.stack([rs, cs], axis=1)
    n_crop = len(crop_coords)
    crop_tissue = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)

    crop_logits_sum = np.zeros((n_crop, NUM_CLASSES), dtype=np.float64)

    for i, seed in enumerate(seeds):
        print(f"\n  ── Finalni model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        torch.manual_seed(seed); random.seed(seed)

        final_ds  = DualStreamDataset(data_pca_final, flat_pp_final,
                                       all_coords, y_all, W,
                                       augment=True, spec_noise=args.spec_noise)
        final_ldr = DataLoader(final_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
        model_f   = DualStreamCNN().to(device)
        opt_f     = optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=1e-4)
        sched_f   = optim.lr_scheduler.CosineAnnealingLR(
            opt_f, T_max=final_epochs, eta_min=1e-6)
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

        crop_ds = DualStreamDataset(data_pca_final, flat_pp_final,
                                     crop_coords,
                                     np.zeros(n_crop, dtype=np.int64),
                                     W, augment=False)
        crop_logits_sum += get_logits_array(model_f, crop_ds, device).astype(np.float64)

    # Ensemble + temperature
    crop_probs = apply_temperature(
        (crop_logits_sum / args.n_ensemble).astype(np.float32), T_opt
    )

    # ------------------------------------------------------------------
    print("\n=== 8. Submission ===")
    prior = np.bincount(y_all, minlength=NUM_CLASSES).astype(np.float32)
    prior /= prior.sum()
    crop_probs[~crop_tissue] = prior

    crop_h, crop_w = PRED_R1 - PRED_R0, PRED_C1 - PRED_C0
    prob_map = crop_probs.reshape(crop_h, crop_w, NUM_CLASSES)
    crop_tissue_2d = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1]

    if args.sigma > 0:
        print(f"  Gaussian smoothing (sigma={args.sigma:.1f})...")
        smoothed = gaussian_smooth_probs(prob_map, crop_tissue_2d, args.sigma)
        final_map = prob_map.copy()
        final_map[crop_tissue_2d] = smoothed[crop_tissue_2d]
    else:
        final_map = prob_map

    final_map = np.clip(final_map, 1e-7, 1.0)
    final_map /= final_map.sum(axis=-1, keepdims=True)
    np.save(args.output, final_map.astype(np.float32))
    print(f"  Submission: {args.output}  T={T_opt:.3f}  σ={args.sigma}")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  DualStreamCNN v3: {args.n_ensemble} modelov, N_PCA={N_PCA}")
    print(f"  Best epochs: {best_epochs}")
    print(f"  T={T_opt:.4f} | σ={args.sigma}")
    print(f"\n  {'':25} {'OA':>8}  {'Log loss':>10}")
    print(f"  {'VAL':25} {oa_val*100:>7.2f}%  {ll_val:>10.5f}")
    print(f"  {'TEST (diploma)':25} {oa_test*100:>7.2f}%  {ll_test:>10.5f}")
    print(f"\n  Primerjava (TEST):")
    print(f"  {'Dual v1 (1 model)':25} {'94.41%':>8}  {'0.22209':>10}")
    print(f"  {'Model A SVM+PCA':25} {'92.69%':>8}  {'0.34700':>10}")
    print(f"\n  Submission: {args.output}")


if __name__ == "__main__":
    main()
