"""
Model C — Best v7  (Three-way split + spektralna skalirna augmentacija)
=======================================================================
Spremembe glede na v6:

PROBLEM v6: cross-core val je tako drugacen od train corov, da early stopping
  ustavi trening pri epohi 1 -> podučeni modeli -> OA 74%, ll 0.776.

RESITEV v7: TRIZNAK split
  - val krogec (en cel TMA core, KMeans) -> SAMO za porocanje mentorju
  - inner val (10% komponent iz train corov, razprseni) -> SAMO za early stopping
  - train (90% od 5 train corov) -> dejansko treniranje

Zakaj to ohranja legitimnost cross-core validacije:
  Model med treningom nikoli ne vidi niti enega piksla iz val krogca.
  Metriki (OA, log-loss) se stereta na val krogcu -> neodvisna ocena.
  Inner val je samo interna ura za ustavljanje -> ne vpliva na porocanje.

NOVA AUGMENTACIJA: spektralna skalirna perturbacija
  Vsak TMA core ima rahlo drugacno intenziteto absorbance (razlicna debelina,
  barvanje). Multiplikativni sum U(0.9, 1.1) na raw spekter prisili model, da
  se nauci oblike spektra, ne absolutnih vrednosti -> boljsa cross-core generalizacija.

Arhitektura = identicno v5/v6 (DualStreamCNN 302k, SPEC_DIM=748, ensemble 12, sigma=1.5).
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
from sklearn.model_selection import train_test_split
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
SPEC_DIM     = SPEC_DIM_RAW * (1 + len(NEIGH_SCALES))   # 748
RESULTS_FILE = "rezultati_report.txt"


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
# Triznak split: train / inner_val / val_core
# ---------------------------------------------------------------------------
def make_three_way_split(tissue_mask, classes, prediction_crop_mask,
                          n_cores=6, inner_val_frac=0.10, min_size=20,
                          seed=42, verbose=True):
    """
    1. KMeans(k=n_cores) na prostorskih koord. anotiranih pikslov -> TMA core-i
    2. Val core = najvecji core z vsemi 6 razredi -> samo za porocanje
    3. Inner val = ~inner_val_frac komponent iz train corov -> samo za early stopping
    4. Train = ostalo

    Model nikoli ne vidi val core med treningom -> cross-core eval ostane legitimen.
    """
    usable = (classes != -1) & (~prediction_crop_mask)
    total  = int(usable.sum())
    coords = np.argwhere(usable)
    flat_cls = classes[coords[:, 0], coords[:, 1]]

    # --- Korak 1: KMeans -> TMA core-i ---
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

    # --- Korak 2: Val core ---
    with_all = [(c_id, n) for c_id, n, has_all in candidates if has_all]
    with_all.sort(key=lambda x: x[1], reverse=True)
    if not with_all:
        candidates.sort(key=lambda x: x[1], reverse=True)
        val_core_id = candidates[0][0]
        print("  OPOZORILO: noben core nima vseh razredov. Vzet najvecji.")
    else:
        val_core_id = with_all[0][0]

    val_idx    = (core_ids == val_core_id)
    val_coords = coords[val_idx]
    val_mask   = np.zeros(classes.shape, dtype=bool)
    val_mask[val_coords[:, 0], val_coords[:, 1]] = True

    # --- Korak 3: Inner val iz train corov (samo za early stopping) ---
    # Stratificiran naključni split po razredih (seed=seed -> reproducibilno).
    # Ne component-based: komponente so mono-razredne -> inner val bi dobil samo en razred.
    # Inner val je SAMO za early stopping, ne za porocanje -> pixelwise split je OK.
    train_cores_mask = usable & ~val_mask
    tc_coords   = np.argwhere(train_cores_mask)
    tc_classes  = classes[tc_coords[:, 0], tc_coords[:, 1]]

    tr_idx, iv_idx = train_test_split(
        np.arange(len(tc_coords)),
        test_size=inner_val_frac,
        stratify=tc_classes,
        random_state=seed
    )

    inner_val_mask = np.zeros(classes.shape, dtype=bool)
    iv_c = tc_coords[iv_idx]
    inner_val_mask[iv_c[:, 0], iv_c[:, 1]] = True

    train_mask = np.zeros(classes.shape, dtype=bool)
    tr_c = tc_coords[tr_idx]
    train_mask[tr_c[:, 0], tr_c[:, 1]] = True

    if verbose:
        val_n = int(val_mask.sum())
        iv_n  = int(inner_val_mask.sum())
        tr_n  = int(train_mask.sum())
        print(f"\n  Val core  (cluster {val_core_id}): {val_n:,} pikslov ({100*val_n/total:.1f}%) <- porocanje")
        print(f"  Inner val (razprseni v train):     {iv_n:,} pikslov ({100*iv_n/total:.1f}%) <- early stop")
        print(f"  Train:                             {tr_n:,} pikslov ({100*tr_n/total:.1f}%)")

    return train_mask, inner_val_mask, val_mask


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
# Dataset (reflect padding + TTA + spektralna skalirna augmentacija)
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
                 patch_size=PATCH_SIZE, augment=False, tta_idx=-1,
                 spec_noise=0.0, spec_scale=0.0):
        self.pad        = patch_size // 2
        self.coords     = coords
        self.labels     = labels
        self.augment    = augment
        self.tta_idx    = tta_idx
        self.spec_noise = spec_noise
        self.spec_scale = spec_scale   # nov: multiplikativni sum za cross-core robustnost
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
        if self.augment:
            if self.spec_noise > 0:
                # Aditivni sum na raw del spektra
                noise = torch.zeros_like(spectrum_t)
                noise[:SPEC_DIM_RAW] = torch.randn(SPEC_DIM_RAW) * self.spec_noise
                spectrum_t = spectrum_t + noise
            if self.spec_scale > 0:
                # Multiplikativni sum: simulira razlike v debelini/barvanju med core-i
                scale = 1.0 + (random.random() - 0.5) * 2 * self.spec_scale
                spectrum_t = spectrum_t.clone()
                spectrum_t[:SPEC_DIM_RAW] = spectrum_t[:SPEC_DIM_RAW] * scale

        return patch_t, spectrum_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Arhitektura: Dual-Stream CNN (identicno v5/v6)
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
        oa_c  = accuracy_score(y_true[mask], y_pred[mask])
        ll_c  = log_loss(y_true[mask], probs[mask], labels=np.arange(NUM_CLASSES))
        print(f"  {c:>8}  {mask.sum():>6}  {oa_c*100:>7.2f}%  {ll_c:>10.5f}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}")
    oa_tot = accuracy_score(y_true, y_pred)
    ll_tot = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))
    print(f"  {'SKUPAJ':>8}  {len(y_true):>6}  {oa_tot*100:>7.2f}%  {ll_tot:>10.5f}")


def write_results_report(model_name, val_oa, val_ll, output_path, t_opt, sigma, extra=""):
    """Zapise rezultate v rezultati_report.txt (ustvari ce ne obstaja, sicer doda vrstico)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = (f"{timestamp}  {model_name:<25}  "
            f"VAL_OA={val_oa*100:6.2f}%  VAL_ll={val_ll:.5f}  "
            f"T={t_opt:.4f}  sigma={sigma:.1f}"
            + (f"  {extra}" if extra else "")
            + f"  -> {output_path}\n")

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("# Rezultati modelov — FTIR klasifikacija tkiva\n")
            f.write(f"# {'─'*90}\n")
            f.write(f"# {'Cas':>16}  {'Model':<25}  {'VAL_OA':>12}  {'VAL_ll':>12}  "
                    f"{'T':>8}  {'sigma':>7}  {'Output'}\n")
            f.write(f"# {'─'*90}\n")
        print(f"  -> {RESULTS_FILE} (ustvarjena nova)")
    else:
        print(f"  -> {RESULTS_FILE} (dodana vrstica)")

    with open(RESULTS_FILE, "a") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Trening enega modela (z inner val za early stopping)
# ---------------------------------------------------------------------------
def train_single(data_pca, train_spec, inner_val_ds, y_train, y_inner_val,
                 train_coords, device, weights,
                 epochs, batch_size, lr, patience, seed, spec_noise, spec_scale):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = DualStreamDataset(data_pca, train_spec, train_coords, y_train,
                                  augment=True, spec_noise=spec_noise,
                                  spec_scale=spec_scale)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = DualStreamCNN().to(device)
    criterion = nn.NLLLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss, best_state, best_epoch = float('inf'), None, epochs
    patience_counter = 0
    print(f"  {'Ep':>4}  {'Train ll':>10}  {'iVal OA':>9}  {'iVal ll':>9}  {'LR':>9}")
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
        # Early stopping glede na inner val (ne val krogec!)
        iv_logits = get_logits_array(model, inner_val_ds, device)
        s = iv_logits - iv_logits.max(axis=1, keepdims=True)
        e = np.exp(s); iv_probs = e / e.sum(axis=1, keepdims=True)
        iv_oa = accuracy_score(y_inner_val, np.argmax(iv_probs, axis=1))
        iv_ll = log_loss(y_inner_val, iv_probs, labels=np.arange(NUM_CLASSES))
        print(f"  {epoch:>4}  {train_loss:>10.5f}  {iv_oa*100:>8.2f}%  "
              f"{iv_ll:>9.5f}  {lr_now:>9.2e}")
        if iv_ll < best_val_loss - 1e-6:
            best_val_loss = iv_ll
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch; patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping: ep {epoch} (best iVal ll={best_val_loss:.5f}, ep={best_epoch})")
                break
    model.load_state_dict(best_state)
    print(f"  Treniran v {time.time()-t0:.1f}s | ep={best_epoch}, iVal_ll={best_val_loss:.5f}")
    return model, best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model C Best v7: triznak split + spektralna skalirna aug"
    )
    parser.add_argument("--input",           default="image1-competition.hdf5")
    parser.add_argument("--output",          default="modelC_best_v7.npy")
    parser.add_argument("--epochs",          type=int,   default=150)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--patience",        type=int,   default=10)
    parser.add_argument("--n-ensemble",      type=int,   default=12)
    parser.add_argument("--spec-noise",      type=float, default=0.005)
    parser.add_argument("--spec-scale",      type=float, default=0.10,
                        help="Spektralna skalirna aug: U(1-scale, 1+scale) na raw spekter.")
    parser.add_argument("--sigma",           type=float, default=1.5)
    parser.add_argument("--min-final-epochs",type=int,   default=15)
    parser.add_argument("--n-cores",         type=int,   default=6)
    parser.add_argument("--inner-val-frac",  type=float, default=0.10)
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
    print(f"\n  Konfiguracija (v7 — triznak split + scale aug):")
    print(f"    SPEC_DIM={SPEC_DIM}: raw(187) + mean@{NEIGH_SCALES}")
    print(f"    Patch {PATCH_SIZE}x{PATCH_SIZE} | PCA({N_PCA}) | TTA 8 | Ensemble {args.n_ensemble}")
    print(f"    spec_noise={args.spec_noise}  spec_scale={args.spec_scale} (nov)")
    print(f"    sigma={args.sigma} FIKSNA | DualStreamCNN ({n_param:,} param)")

    # ------------------------------------------------------------------
    print("\n=== 2. Triznak split (train / inner_val / val_core) ===")
    train_mask, inner_val_mask, val_mask = make_three_way_split(
        tissue_mask, classes, prediction_crop_mask,
        n_cores=args.n_cores, inner_val_frac=args.inner_val_frac, seed=args.seed)

    train_coords    = np.argwhere(train_mask)
    inner_val_coords= np.argwhere(inner_val_mask)
    val_coords      = np.argwhere(val_mask)
    y_train     = classes[train_mask].astype(np.int64)
    y_inner_val = classes[inner_val_mask].astype(np.int64)
    y_val       = classes[val_mask].astype(np.int64)

    print(f"\n  Porazdelitev razredov (train / inner_val / val_core):")
    for c in range(NUM_CLASSES):
        nt = (y_train == c).sum()
        ni = (y_inner_val == c).sum()
        nv = (y_val == c).sum()
        print(f"    R{c}: train={nt:5d} ({100*nt/len(y_train):.1f}%)  "
              f"iVal={ni:4d} ({100*ni/len(y_inner_val):.1f}%)  "
              f"val={nv:5d} ({100*nv/len(y_val):.1f}%)")

    # ------------------------------------------------------------------
    print("\n=== 3. Preprocessing: StandardScaler + PCA ===")
    data_pca, flat_sc, scaler, pca = build_preprocessed(data, train_mask, N_PCA, args.seed)
    weights = compute_class_weights(y_train)

    # ------------------------------------------------------------------
    print(f"\n=== 4. Neighbourhood mean features ({SPEC_DIM} dim) ===")
    print(f"  Train ({len(train_coords):,} pikslov):")
    train_spec     = extract_spec_features(flat_sc, H, W, train_coords)
    print(f"  Inner val ({len(inner_val_coords):,} pikslov):")
    inner_val_spec = extract_spec_features(flat_sc, H, W, inner_val_coords)
    print(f"  Val core ({len(val_coords):,} pikslov):")
    val_spec       = extract_spec_features(flat_sc, H, W, val_coords)

    inner_val_ds = DualStreamDataset(data_pca, inner_val_spec, inner_val_coords,
                                      y_inner_val, augment=False)
    val_ds       = DualStreamDataset(data_pca, val_spec, val_coords, y_val, augment=False)

    # ------------------------------------------------------------------
    print(f"\n=== 5. Ensemble trening ({args.n_ensemble} modelov) ===")
    print(f"  Early stopping glede na inner val (ne val core!)")
    seeds           = list(range(args.seed, args.seed + args.n_ensemble))
    val_logits_list = []
    best_epochs     = []
    for i, seed in enumerate(seeds):
        print(f"\n  ── Model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        model, best_epoch = train_single(
            data_pca=data_pca, train_spec=train_spec,
            inner_val_ds=inner_val_ds, y_train=y_train, y_inner_val=y_inner_val,
            train_coords=train_coords, device=device, weights=weights,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience,
            seed=seed, spec_noise=args.spec_noise, spec_scale=args.spec_scale,
        )
        best_epochs.append(best_epoch)
        print(f"  TTA na val core...")
        val_logits_list.append(
            get_logits_tta(model, data_pca, val_spec, val_coords, y_val, device)
        )

    val_logits_avg = np.mean(val_logits_list, axis=0)
    avg_best_epoch = round(np.mean(best_epochs))
    print(f"\n  Best epochs (iVal): {best_epochs} (avg={avg_best_epoch})")

    # ------------------------------------------------------------------
    print("\n=== 6. Temperature scaling (na val core) ===")
    T_opt = find_temperature(val_logits_avg, y_val)

    # ------------------------------------------------------------------
    print("\n=== 7. Val core evaluacija (cross-core, za porocanje) ===")
    val_probs = apply_temperature(val_logits_avg, T_opt)
    val_pred  = np.argmax(val_probs, axis=1)
    val_oa    = accuracy_score(y_val, val_pred)
    val_ll    = log_loss(y_val, val_probs, labels=np.arange(NUM_CLASSES))
    print(f"  VAL OA:       {val_oa*100:.2f}%")
    print(f"  VAL log loss: {val_ll:.5f}")
    print(f"  Ref v6:       VAL OA=74.17%  VAL ll=0.77597")
    print(f"  Ref v5:       VAL OA=92.35%  VAL ll=0.23094  (within-core, za primerjavo)")

    print_per_class_table(y_val, val_pred, val_probs, "Per-class OA in log-loss (val core):")

    sigma_opt = args.sigma

    # ------------------------------------------------------------------
    n_all        = int(((classes != -1) & (~prediction_crop_mask)).sum())
    proportional = round(avg_best_epoch * n_all / max(len(y_train), 1))
    final_epochs = max(proportional, args.min_final_epochs)
    print(f"\n=== 8. Finalni ensemble (100%, {final_epochs} epoh x {args.n_ensemble} modelov) ===")

    usable_mask = (classes != -1) & (~prediction_crop_mask)
    all_coords  = np.argwhere(usable_mask)
    y_all       = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov: {len(y_all)}")

    print("\n  Refit StandardScaler + PCA na vseh anotiranih pikslih...")
    flat         = data.reshape(-1, data.shape[-1])
    scaler_final = StandardScaler()
    X_all_sc     = scaler_final.fit_transform(flat[usable_mask.ravel()])
    pca_final    = PCA(n_components=N_PCA, random_state=args.seed)
    pca_final.fit(X_all_sc)
    print(f"  PCA varianca: {pca_final.explained_variance_ratio_.sum()*100:.2f}%")
    flat_sc_final  = scaler_final.transform(flat).astype(np.float32)
    data_pca_final = pca_final.transform(flat_sc_final).reshape(H, W, N_PCA).astype(np.float32)

    print(f"\n  Neighbourhood features za all_coords ({len(all_coords):,}):")
    all_spec_final = extract_spec_features(flat_sc_final, H, W, all_coords)

    rs = np.repeat(np.arange(PRED_R0, PRED_R1), PRED_C1 - PRED_C0)
    cs = np.tile(  np.arange(PRED_C0, PRED_C1), PRED_R1 - PRED_R0)
    crop_coords  = np.stack([rs, cs], axis=1)
    n_crop       = len(crop_coords)
    crop_tissue  = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)

    print(f"  Neighbourhood features za crop ({n_crop:,}):")
    crop_spec_final = extract_spec_features(flat_sc_final, H, W, crop_coords)

    weights_final   = compute_class_weights(y_all)
    crop_logits_sum = np.zeros((n_crop, NUM_CLASSES), dtype=np.float64)

    for i, seed in enumerate(seeds):
        print(f"\n  ── Finalni model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        torch.manual_seed(seed); random.seed(seed)
        final_ds  = DualStreamDataset(data_pca_final, all_spec_final, all_coords, y_all,
                                       augment=True, spec_noise=args.spec_noise,
                                       spec_scale=args.spec_scale)
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

    print(f"  Gaussian smoothing (sigma={sigma_opt:.1f}, FIKSNA)...")
    if sigma_opt > 0:
        smoothed  = gaussian_smooth_probs(prob_map, crop_tissue_2d, sigma_opt)
        final_map = prob_map.copy()
        final_map[crop_tissue_2d] = smoothed[crop_tissue_2d]
    else:
        final_map = prob_map

    final_map = np.clip(final_map, 1e-7, 1.0)
    final_map /= final_map.sum(axis=-1, keepdims=True)
    np.save(args.output, final_map.astype(np.float32))

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (v7) ===")
    print(f"  SPEC_DIM={SPEC_DIM} | Ensemble={args.n_ensemble} | final_epochs={final_epochs}")
    print(f"  spec_scale={args.spec_scale} | spec_noise={args.spec_noise}")
    print(f"  Best epochs (iVal): {best_epochs} (avg={avg_best_epoch})")
    print(f"  T={T_opt:.4f} | sigma={sigma_opt:.1f}")
    print(f"  VAL OA={val_oa*100:.2f}%  VAL ll={val_ll:.5f}  (cross-core, val krogec)")
    print(f"\n  Primerjava (nizje = boljse):")
    print(f"    v5 (within-core):  VAL OA=92.35%  VAL ll=0.23094  Final=0.39533")
    print(f"    v6 (cross-core):   VAL OA=74.17%  VAL ll=0.77597  Final=0.40396")
    print(f"    v7 (cross-core+):  VAL OA={val_oa*100:.2f}%  VAL ll={val_ll:.5f}  Final=???")
    print(f"\n  Submission: {args.output}")

    # ------------------------------------------------------------------
    print(f"\n=== 10. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelC_best_v7",
        val_oa=val_oa, val_ll=val_ll,
        output_path=args.output,
        t_opt=T_opt, sigma=sigma_opt,
        extra=f"cross-core iVal best_ep={best_epochs}"
    )


if __name__ == "__main__":
    main()
