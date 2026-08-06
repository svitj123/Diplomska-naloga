"""
Model C — Best v9  (Trifazni model: eval → tekmovanje → diploma)
=================================================================
Osnova: v8 (cross-core, fiksne epohe, brez early stopping).

TRI FAZE:
  Faza 1 (eval):        5 krogcev × 6 epoh  | Core 2 → val
                        Najde avg_best_epoch, kalibira T — brez early stopping.

  Faza 2 (tekmovanje):  6 krogcev × 15 epoh | competition crop → oddaja
                        Scaler/PCA refittan na vseh 6 krogcih (max podatkov).

  Faza 3 (diploma):     5 krogcev × 15 epoh | Core 2 → evalvacija
                        Scaler/PCA iz Faze 1 (fit samo na 5 krogcih, brez leakage).
                        Core 2 NI nikoli viden med treningom → poštena cross-core ocena.

RAZLIKA od v8:
  v8: Faza 1 (eval) + Faza 2 (6 krogcev → competition crop)
  v9: Faza 1 (eval) + Faza 2 (6 krogcev → competition crop)
                    + Faza 3 (5 krogcev → Core 2 evalvacija za diplomo)

Arhitektura = identično v5/v8 (DualStreamCNN 302k, SPEC_DIM=748, 12 ensemble, sigma=1.5).
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
from scipy import ndimage as ndi

NUM_CLASSES  = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660
PATCH_SIZE   = 17
N_PCA        = 16
SPEC_DIM_RAW = 187
NEIGH_SCALES = [3, 5, 7]
SPEC_DIM     = SPEC_DIM_RAW * (1 + len(NEIGH_SCALES))
RESULTS_FILE = "rezultati_report.txt"

METODOLOGIJA_OPOMBA = (
    "Faza 1: eval ensemble (5 krogcev, 6 epoh) — najde avg_best_epoch + T, brez early stopping. "
    "Faza 2: tekmovalni ensemble (6 krogcev, 15 epoh) → competition crop. "
    "Faza 3: diplomska eval (5 krogcev, 15 epoh) → Core 2 napoved; Core 2 nikoli viden v treningu."
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
# Leave-One-Core-Out split (KMeans) — identično v6/v8
# ---------------------------------------------------------------------------
def make_spatial_two_way_split(tissue_mask, classes, prediction_crop_mask,
                                n_cores=6, seed=42, verbose=True):
    """
    KMeans(k=n_cores) na prostorskih koordinatah -> TMA core-i.
    Val = največji core z vsemi 6 razredi -> SAMO za T + metrike.
    Train = vsi ostali core-i.
    """
    usable = (classes != -1) & (~prediction_crop_mask)
    total  = int(usable.sum())
    coords = np.argwhere(usable)
    flat_cls = classes[coords[:, 0], coords[:, 1]]

    print(f"  KMeans(k={n_cores}, seed={seed}) na {len(coords):,} pikslih...")
    km = KMeans(n_clusters=n_cores, random_state=seed, n_init=10)
    core_ids = km.fit_predict(coords)

    print("  Pregled core-ov:")
    candidates = []
    for c_id in range(n_cores):
        idx = (core_ids == c_id)
        n   = int(idx.sum())
        cls = np.unique(flat_cls[idx])
        has_all = len(cls) == NUM_CLASSES
        candidates.append((c_id, n, has_all))
        marker = "*" if has_all else " "
        print(f"    {marker} Core {c_id}: {n:5,} pikslov, razredi={list(cls)}")

    with_all = [(c_id, n) for c_id, n, has_all in candidates if has_all]
    with_all.sort(key=lambda x: x[1], reverse=True)
    if not with_all:
        candidates.sort(key=lambda x: x[1], reverse=True)
        val_core_id = candidates[0][0]
        print("  OPOZORILO: noben core nima vseh razredov. Vzet največji.")
    else:
        val_core_id = with_all[0][0]

    val_idx    = (core_ids == val_core_id)
    val_coords = coords[val_idx]
    val_mask   = np.zeros(classes.shape, dtype=bool)
    val_mask[val_coords[:, 0], val_coords[:, 1]] = True
    train_mask = usable & ~val_mask

    if verbose:
        val_n = int(val_mask.sum())
        print(f"\n  Val core (cluster {val_core_id}): {val_n:,} pikslov ({100*val_n/total:.1f}%) <- T + metrike")
        print(f"  Train:                {train_mask.sum():,} pikslov ({100*train_mask.sum()/total:.1f}%)")

    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Neighbourhood MEAN features
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
# Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessed(data, train_mask, n_pca=N_PCA, seed=42):
    H, W, D = data.shape
    flat    = data.reshape(-1, D)
    X_train_raw = flat[train_mask.ravel()]
    print(f"  StandardScaler fit na {len(X_train_raw):,} train pikslih...")
    t0 = time.time()
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    print(f"    -> {time.time()-t0:.1f}s")
    print(f"  PCA({n_pca}) fit...")
    pca = PCA(n_components=n_pca, random_state=seed)
    pca.fit(X_train_sc)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"  Transform vseh {H*W:,} pikslov...")
    t0 = time.time()
    flat_sc  = scaler.transform(flat).astype(np.float32)
    flat_pca = pca.transform(flat_sc).astype(np.float32)
    data_pca = flat_pca.reshape(H, W, n_pca)
    print(f"    -> {time.time()-t0:.1f}s")
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
# Arhitektura: Dual-Stream CNN (identično v5/v8)
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


def write_results_report(model_name, eval_oa, eval_ll, diploma_oa, diploma_ll,
                          competition_output, diploma_output,
                          t_opt, sigma, eval_epochs, best_epoch, final_epochs,
                          extra_note=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"EVAL_OA={eval_oa*100:6.2f}%  EVAL_ll={eval_ll:.5f}  "
        f"DIPLOMA_OA={diploma_oa*100:6.2f}%  DIPLOMA_ll={diploma_ll:.5f}  "
        f"T={t_opt:.4f}  sigma={sigma:.1f}  "
        f"eval_ep={eval_epochs}(best={best_epoch})  final_ep={final_epochs}\n"
        f"{'':>19}  Tekmovanje: {competition_output}  |  Diploma: {diploma_output}\n"
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
# Trening enega modela — FIKSNE EPOHE, brez early stopping
# ---------------------------------------------------------------------------
def train_single(data_pca, train_spec, val_ds, y_train, y_val,
                 train_coords, device, weights,
                 eval_epochs, batch_size, lr, seed, spec_noise):
    """
    Trenira točno eval_epochs epoh. Brez early stopping.
    Best-of-N selekcija: shrani stanje z najnižjim val loss med vsemi epohi.
    Val set se NE uporablja za odločitev kdaj ustaviti — le za izbiro najboljšega stanja.
    """
    torch.manual_seed(seed); random.seed(seed)
    train_ds = DualStreamDataset(data_pca, train_spec, train_coords, y_train,
                                  augment=True, spec_noise=spec_noise)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = DualStreamCNN().to(device)
    criterion = nn.NLLLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eval_epochs, eta_min=1e-6)

    best_val_loss, best_state, best_epoch = float('inf'), None, 1
    print(f"  {'Ep':>4}  {'Train ll':>10}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}")
    t0 = time.time()

    for epoch in range(1, eval_epochs + 1):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model C Best v9: trifazni model (eval + tekmovanje + diploma)"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output-competition", default="modelC_best_v9.npy",
                        help="Izhod Faze 2: competition crop za oddajo.")
    parser.add_argument("--output-diploma",     default="modelC_best_v9_core2.npy",
                        help="Izhod Faze 3: napoved na Core 2 za diplomo.")
    parser.add_argument("--eval-epochs",        type=int,   default=6,
                        help="Epohe za eval fazo (Faza 1). Iz v5 (avg best=4) + buffer.")
    parser.add_argument("--batch-size",         type=int,   default=256)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--n-ensemble",         type=int,   default=12)
    parser.add_argument("--spec-noise",         type=float, default=0.005)
    parser.add_argument("--sigma",              type=float, default=1.5)
    parser.add_argument("--min-final-epochs",   type=int,   default=15,
                        help="Epohe za Fazo 2 in Fazo 3.")
    parser.add_argument("--n-cores",            type=int,   default=6)
    parser.add_argument("--seed",               type=int,   default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, wns, tissue_mask, classes = load_data(args.input)
    H, W, _ = data.shape
    print(f"  data: {data.shape} | Anotiranih: {(classes != -1).sum()}")
    prediction_crop_mask = make_prediction_crop_mask(H, W)
    device = get_device()
    n_param = sum(p.numel() for p in DualStreamCNN().parameters() if p.requires_grad)
    print(f"\n  Konfiguracija (v9 — trifazni: eval + tekmovanje + diploma):")
    print(f"    SPEC_DIM={SPEC_DIM}: raw(187) + mean@{NEIGH_SCALES}")
    print(f"    Patch {PATCH_SIZE}x{PATCH_SIZE} | PCA({N_PCA}) | TTA 8 | Ensemble {args.n_ensemble}")
    print(f"    eval_epochs={args.eval_epochs} (Faza 1) | final_epochs={args.min_final_epochs} (Fazi 2+3)")
    print(f"    sigma={args.sigma} FIKSNA | DualStreamCNN ({n_param:,} param)")
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    # ------------------------------------------------------------------
    print("\n=== 2. Prostorski split: leave-one-core-out (KMeans) ===")
    train_mask, val_mask = make_spatial_two_way_split(
        tissue_mask, classes, prediction_crop_mask,
        n_cores=args.n_cores, seed=args.seed)
    train_coords = np.argwhere(train_mask)
    val_coords   = np.argwhere(val_mask)
    y_train = classes[train_mask].astype(np.int64)
    y_val   = classes[val_mask].astype(np.int64)
    print(f"\n  Porazdelitev razredov:")
    for c in range(NUM_CLASSES):
        nt = (y_train == c).sum(); nv = (y_val == c).sum()
        print(f"    R{c}: train={nt:5d} ({100*nt/len(y_train):.1f}%)  "
              f"val={nv:5d} ({100*nv/len(y_val):.1f}%)")

    # ------------------------------------------------------------------
    print("\n=== 3. Preprocessing: StandardScaler + PCA (fit na 5 krogcih) ===")
    data_pca, flat_sc, scaler, pca = build_preprocessed(data, train_mask, N_PCA, args.seed)
    weights = compute_class_weights(y_train)

    # ------------------------------------------------------------------
    print(f"\n=== 4. Neighbourhood mean features ({SPEC_DIM} dim) ===")
    print(f"  Train ({len(train_coords):,} pikslov):")
    train_spec = extract_spec_features(flat_sc, H, W, train_coords)
    print(f"  Val/Core2 ({len(val_coords):,} pikslov):")
    val_spec   = extract_spec_features(flat_sc, H, W, val_coords)
    val_ds     = DualStreamDataset(data_pca, val_spec, val_coords, y_val, augment=False)

    # ------------------------------------------------------------------
    print(f"\n=== 5. Faza 1 — Eval ensemble ({args.n_ensemble} modelov x {args.eval_epochs} epoh) ===")
    print(f"  Brez early stopping — val krogec samo za T, best_epoch in metrike.")
    seeds           = list(range(args.seed, args.seed + args.n_ensemble))
    val_logits_list = []
    best_epochs     = []
    for i, seed in enumerate(seeds):
        print(f"\n  ── Model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        model, best_epoch = train_single(
            data_pca=data_pca, train_spec=train_spec,
            val_ds=val_ds, y_train=y_train, y_val=y_val,
            train_coords=train_coords, device=device, weights=weights,
            eval_epochs=args.eval_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, spec_noise=args.spec_noise,
        )
        best_epochs.append(best_epoch)
        print(f"  TTA na Core 2...")
        val_logits_list.append(
            get_logits_tta(model, data_pca, val_spec, val_coords, y_val, device)
        )

    val_logits_avg = np.mean(val_logits_list, axis=0)
    avg_best_epoch = round(np.mean(best_epochs))
    print(f"\n  Best-of-{args.eval_epochs} epohe: {best_epochs} (avg={avg_best_epoch})")

    # ------------------------------------------------------------------
    print("\n=== 6. Temperature scaling (na Core 2) ===")
    T_opt = find_temperature(val_logits_avg, y_val)

    # ------------------------------------------------------------------
    print("\n=== 7. Faza 1 — Eval evaluacija na Core 2 ===")
    val_probs_eval = apply_temperature(val_logits_avg, T_opt)
    val_pred_eval  = np.argmax(val_probs_eval, axis=1)
    val_oa_eval    = accuracy_score(y_val, val_pred_eval)
    val_ll_eval    = log_loss(y_val, val_probs_eval, labels=np.arange(NUM_CLASSES))
    print(f"  EVAL OA  (Faza 1, best-of-{args.eval_epochs}): {val_oa_eval*100:.2f}%")
    print(f"  EVAL ll  (Faza 1, best-of-{args.eval_epochs}): {val_ll_eval:.5f}")
    print(f"  Ref v8:  EVAL OA=74.17%  EVAL ll=0.77597")
    print_per_class_table(y_val, val_pred_eval, val_probs_eval,
                          "Per-class OA in log-loss (Faza 1 eval):")

    sigma_opt    = args.sigma
    final_epochs = max(avg_best_epoch, args.min_final_epochs)

    # ------------------------------------------------------------------
    # Faza 2: Tekmovalni ensemble — VSI 6 krogci, 15 epoh → competition crop
    # Scaler/PCA refittan na vseh 6 krogcih (max podatkov za tekmovanje).
    # ------------------------------------------------------------------
    print(f"\n=== 8. Faza 2 — Tekmovalni ensemble (6 krogcev, {final_epochs} epoh x {args.n_ensemble} modelov) ===")
    print(f"  Scaler/PCA refittan na vseh 6 krogcih. Napoved: competition crop.")

    usable_mask = (classes != -1) & (~prediction_crop_mask)
    all_coords  = np.argwhere(usable_mask)
    y_all       = classes[usable_mask].astype(np.int64)

    flat          = data.reshape(-1, data.shape[-1])
    scaler_comp   = StandardScaler()
    X_all_sc      = scaler_comp.fit_transform(flat[usable_mask.ravel()])
    pca_comp      = PCA(n_components=N_PCA, random_state=args.seed)
    pca_comp.fit(X_all_sc)
    print(f"  PCA varianca (6 krogci): {pca_comp.explained_variance_ratio_.sum()*100:.2f}%")
    flat_sc_comp   = scaler_comp.transform(flat).astype(np.float32)
    data_pca_comp  = pca_comp.transform(flat_sc_comp).reshape(H, W, N_PCA).astype(np.float32)

    print(f"\n  Neighbourhood features za vse krogce ({len(all_coords):,}):")
    all_spec_comp  = extract_spec_features(flat_sc_comp, H, W, all_coords)

    rs = np.repeat(np.arange(PRED_R0, PRED_R1), PRED_C1 - PRED_C0)
    cs = np.tile(  np.arange(PRED_C0, PRED_C1), PRED_R1 - PRED_R0)
    crop_coords    = np.stack([rs, cs], axis=1)
    n_crop         = len(crop_coords)
    crop_tissue    = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)
    crop_spec_comp = extract_spec_features(flat_sc_comp, H, W, crop_coords)

    weights_comp      = compute_class_weights(y_all)
    crop_logits_sum   = np.zeros((n_crop, NUM_CLASSES), dtype=np.float64)

    for i, seed in enumerate(seeds):
        print(f"\n  ── Tekmovalni model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        torch.manual_seed(seed); random.seed(seed)
        comp_ds  = DualStreamDataset(data_pca_comp, all_spec_comp, all_coords, y_all,
                                      augment=True, spec_noise=args.spec_noise)
        comp_ldr = DataLoader(comp_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
        model_c  = DualStreamCNN().to(device)
        opt_c    = optim.Adam(model_c.parameters(), lr=args.lr, weight_decay=1e-4)
        sched_c  = optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=final_epochs, eta_min=1e-6)
        crit_c   = nn.NLLLoss(weight=weights_comp.to(device))
        print(f"  {'Ep':>4}  {'Train ll':>10}  {'LR':>9}")
        t0 = time.time()
        for epoch in range(1, final_epochs + 1):
            model_c.train()
            total = 0.0
            for patches, spectra, labels in comp_ldr:
                patches = patches.to(device); spectra = spectra.to(device)
                labels  = labels.to(device)
                opt_c.zero_grad()
                loss = crit_c(model_c(patches, spectra), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
                opt_c.step()
                total += loss.item() * len(labels)
            sched_c.step()
            if epoch % 3 == 0 or epoch in (1, final_epochs):
                print(f"  {epoch:>4}  {total/len(y_all):>10.5f}  "
                      f"{opt_c.param_groups[0]['lr']:>9.2e}")
        print(f"  Treniran v {time.time()-t0:.1f}s")
        crop_logits_sum += get_logits_tta(
            model_c, data_pca_comp, crop_spec_comp,
            crop_coords, np.zeros(n_crop, dtype=np.int64), device
        ).astype(np.float64)

    crop_probs = apply_temperature(
        (crop_logits_sum / args.n_ensemble).astype(np.float32), T_opt
    )
    prior_comp = np.bincount(y_all, minlength=NUM_CLASSES).astype(np.float32)
    prior_comp /= prior_comp.sum()
    crop_probs[~crop_tissue] = prior_comp

    crop_h, crop_w = PRED_R1 - PRED_R0, PRED_C1 - PRED_C0
    prob_map_comp  = crop_probs.reshape(crop_h, crop_w, NUM_CLASSES)
    crop_tissue_2d = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1]

    if sigma_opt > 0:
        smoothed      = gaussian_smooth_probs(prob_map_comp, crop_tissue_2d, sigma_opt)
        final_map_comp = prob_map_comp.copy()
        final_map_comp[crop_tissue_2d] = smoothed[crop_tissue_2d]
    else:
        final_map_comp = prob_map_comp

    final_map_comp = np.clip(final_map_comp, 1e-7, 1.0)
    final_map_comp /= final_map_comp.sum(axis=-1, keepdims=True)
    np.save(args.output_competition, final_map_comp.astype(np.float32))
    print(f"\n  Shranjeno (tekmovanje): {args.output_competition}  shape={final_map_comp.shape}")

    # ------------------------------------------------------------------
    # Faza 3: Diplomska evaluacija — ISTI 5 krogci, 15 epoh, napoved na Core 2
    # Scaler/PCA iz Faze 1 — Core 2 nikoli viden v treningu ali preprocessingu.
    # ------------------------------------------------------------------
    print(f"\n=== 9. Faza 3 — Diplomska eval (5 krogcev, {final_epochs} epoh x {args.n_ensemble} modelov) ===")
    print(f"  Core 2 NI vključen v trening. Scaler/PCA iz Faze 1 (fit na 5 krogcih).")

    weights_dip      = compute_class_weights(y_train)
    val_logits_final = np.zeros((len(val_coords), NUM_CLASSES), dtype=np.float64)

    for i, seed in enumerate(seeds):
        print(f"\n  ── Diplomski model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        torch.manual_seed(seed); random.seed(seed)
        dip_ds  = DualStreamDataset(data_pca, train_spec, train_coords, y_train,
                                     augment=True, spec_noise=args.spec_noise)
        dip_ldr = DataLoader(dip_ds, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
        model_d = DualStreamCNN().to(device)
        opt_d   = optim.Adam(model_d.parameters(), lr=args.lr, weight_decay=1e-4)
        sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=final_epochs, eta_min=1e-6)
        crit_d  = nn.NLLLoss(weight=weights_dip.to(device))
        print(f"  {'Ep':>4}  {'Train ll':>10}  {'LR':>9}")
        t0 = time.time()
        for epoch in range(1, final_epochs + 1):
            model_d.train()
            total = 0.0
            for patches, spectra, labels in dip_ldr:
                patches = patches.to(device); spectra = spectra.to(device)
                labels  = labels.to(device)
                opt_d.zero_grad()
                loss = crit_d(model_d(patches, spectra), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model_d.parameters(), 1.0)
                opt_d.step()
                total += loss.item() * len(labels)
            sched_d.step()
            if epoch % 3 == 0 or epoch in (1, final_epochs):
                print(f"  {epoch:>4}  {total/len(y_train):>10.5f}  "
                      f"{opt_d.param_groups[0]['lr']:>9.2e}")
        print(f"  Treniran v {time.time()-t0:.1f}s")
        val_logits_final += get_logits_tta(
            model_d, data_pca, val_spec, val_coords, y_val, device
        ).astype(np.float64)

    # ------------------------------------------------------------------
    print("\n=== 9b. Finalna evaluacija na Core 2 ===")
    val_probs_final = apply_temperature(
        (val_logits_final / args.n_ensemble).astype(np.float32), T_opt
    )
    val_pred_final = np.argmax(val_probs_final, axis=1)
    val_oa_final   = accuracy_score(y_val, val_pred_final)
    val_ll_final   = log_loss(y_val, val_probs_final, labels=np.arange(NUM_CLASSES))
    print(f"  DIPLOMA OA  (Core 2, {final_epochs} epoh): {val_oa_final*100:.2f}%")
    print(f"  DIPLOMA ll  (Core 2, {final_epochs} epoh): {val_ll_final:.5f}")
    print(f"  Primerjava Faza 1 eval: OA={val_oa_eval*100:.2f}%  ll={val_ll_eval:.5f}")
    print(f"  Ref članek SD:          OA=56.41%  (cross-slide)")
    print_per_class_table(y_val, val_pred_final, val_probs_final,
                          "Per-class OA in log-loss (Faza 3, Core 2):")

    # Bounding box Core 2 za Gaussian smoothing in shranjevanje
    r_min = int(val_coords[:, 0].min())
    r_max = int(val_coords[:, 0].max())
    c_min = int(val_coords[:, 1].min())
    c_max = int(val_coords[:, 1].max())
    bbox_h = r_max - r_min + 1
    bbox_w = c_max - c_min + 1

    prior_dip = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    prior_dip /= prior_dip.sum()
    prob_map_dip  = np.tile(prior_dip, (bbox_h * bbox_w, 1)).reshape(bbox_h, bbox_w, NUM_CLASSES)
    val_tissue_2d = tissue_mask[r_min:r_max+1, c_min:c_max+1]
    for (r, c), prob in zip(val_coords, val_probs_final):
        prob_map_dip[r - r_min, c - c_min] = prob

    if sigma_opt > 0:
        smoothed     = gaussian_smooth_probs(prob_map_dip, val_tissue_2d, sigma_opt)
        final_map_dip = prob_map_dip.copy()
        final_map_dip[val_tissue_2d] = smoothed[val_tissue_2d]
    else:
        final_map_dip = prob_map_dip

    final_map_dip = np.clip(final_map_dip, 1e-7, 1.0)
    final_map_dip /= final_map_dip.sum(axis=-1, keepdims=True)
    np.save(args.output_diploma, final_map_dip.astype(np.float32))
    print(f"\n  Shranjeno (diploma): {args.output_diploma}  shape={final_map_dip.shape}")
    print(f"  (bbox Core 2: vrstice {r_min}-{r_max}, stolpci {c_min}-{c_max})")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (v9) ===")
    print(f"  Split: 5 krogcev train | Core 2 val (nikoli viden v treningu)")
    print(f"  T={T_opt:.4f} | sigma={sigma_opt:.1f}")
    print(f"\n  Faza 1 (eval,       {args.eval_epochs:2d} epoh): OA={val_oa_eval*100:.2f}%  ll={val_ll_eval:.5f}  [best_epochs={best_epochs}]")
    print(f"  Faza 2 (tekmovanje, {final_epochs:2d} epoh): → {args.output_competition}")
    print(f"  Faza 3 (diploma,    {final_epochs:2d} epoh): OA={val_oa_final*100:.2f}%  ll={val_ll_final:.5f}  [Core 2]")
    print(f"\n  Primerjava z člankom:")
    print(f"    SD (cross-slide): OA=56.41%")
    print(f"    v9 (cross-core):  OA={val_oa_final*100:.2f}%")

    # ------------------------------------------------------------------
    print(f"\n=== 10. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelC_best_v9",
        eval_oa=val_oa_eval, eval_ll=val_ll_eval,
        diploma_oa=val_oa_final, diploma_ll=val_ll_final,
        competition_output=args.output_competition,
        diploma_output=args.output_diploma,
        t_opt=T_opt, sigma=sigma_opt,
        eval_epochs=args.eval_epochs,
        best_epoch=avg_best_epoch,
        final_epochs=final_epochs,
        extra_note=METODOLOGIJA_OPOMBA
    )


if __name__ == "__main__":
    main()
