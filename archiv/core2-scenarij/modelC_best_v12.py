"""
Model C — Best v12  (Zvesta replika arhitekture/treninga iz clanka)
=====================================================================
Izhodiscna tocka za diplomo: "kako blizu smo clanku, ce posnemamo NJIHOVO
arhitekturo/trening recept, znotraj NASE poštene (gnezdene) validacijske
metodologije?" v13 bo nato eksperimentiral z izboljsavami iz te tocke naprej.

KAJ JE ZVESTO POSNEMANO IZ CLANKA:
  - Single-stream arhitektura (BREZ locene spektralne MLP veje — samo
    PCA(16) patch -> conv/pool/FC/softmax, kot v Fig. 3 clanka)
  - BREZ BatchNorm (clanek: "same architecture without BN... for SD data")
  - Softplus aktivacija namesto ReLU (clanek: "provided better convergence
    than ReLU")
  - Dropout keep_prob=0.5 (torej rate=0.5) namesto nasih 0.3
  - Weight init: normal(0, 0.02) namesto kaiming_normal
  - Batch size 128 namesto 256
  - BREZ StandardScaler — PCA fit direktno na (ze baseline+Amide
    normaliziranih) podatkih, kot v11
  - Oversampling manjsinskih razredov namesto class-weighted loss
    (clanek: "stack copies of underrepresented classes")

KAJ NAMERNO NI SPREMENJENO (nasa infrastruktura, ne del "zvestobe" arhitekturi):
  - Gnezdena leave-one-core-out validacija (Core 2 = pravi, enkratni test) —
    clanek nima direktnega analoga te metode (oni imajo fiksen cross-slide
    train/test split), a je nujna za nasa diplomska primerjavo
  - Ensemble 12 modelov, TTA 8x D4, temperature scaling, Gaussian smoothing
  - Adam namesto Adadelta — Adadelta lr=0.1 se ne prevede smiselno na Adam;
    zamenjava optimizerja brez pravega razloga tvega nestabilnost brez
    jasne koristi, zato ostane Adam (kot v5-v11)
  - Trening-casovna D4 augmentacija (flip/rotate) patch-ev — clanek tega
    eksplicitno ne omenja, a ga tudi ne izkljucuje; sodobna dobra praksa

Ta locitev omogoca, da razlika v10 vs v12 pripisemo SAMO arhitekturi in
treningu, ne razlikam v evalvacijski infrastrukturi.

Metodologija (nested split) identicna v10/v11:
  Zunanji split (KMeans k=6):  5 krogcev (outer-train)  |  Core 2 (TEST)
  Notranji split (KMeans k=5): 4 (inner-train) | 1 (inner-val)
  Faza A: iskanje hiperparametrov (best-of-N na inner-val, Core 2 ni vpleten)
  Faza B: finalni model (5 krogcev, brez peekanja) — en sam dotik s Core 2
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
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES  = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660
PATCH_SIZE   = 17
N_PCA        = 16
RESULTS_FILE = "rezultati_report.txt"

REF_CNN_SD_OA   = 79.45
REF_CNN_SD_STD  = 1.25

METODOLOGIJA_OPOMBA = (
    "v12 = zvesta replika clanka: single-stream arhitektura (brez spektralne "
    "MLP veje), brez BatchNorm (SD varianta), softplus namesto ReLU, "
    "dropout=0.5, weight init normal(0,0.02), batch=128, brez StandardScaler, "
    "oversampling namesto class weights. Nasa gnezdena Core2-validacija, "
    "ensemble/TTA/T-scaling infrastruktura ostane enaka v10/v11, da je "
    "primerjava izolirana na arhitekturo/trening recept. Adam obdrzan namesto "
    "Adadelta (lr=0.1 se ne prevede smiselno). Ref. clanek CNN (SD, "
    "cross-slide, tezja naloga od nase): OA=79.45%+/-1.25%."
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
# Splosen prostorski split (KMeans) — zunanji IN notranji split (identicno v10/v11)
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
# Oversampling manjsinskih razredov (namesto class-weighted loss)
# ---------------------------------------------------------------------------
def oversample_coords(coords, y, seed=42, verbose=True, label="train"):
    rng = np.random.RandomState(seed)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    target = int(counts.max())
    if verbose:
        print(f"  Oversampling ({label}) na target={target:,} (najvecji razred):")
    idx_parts = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(y == c)[0]
        if len(cls_idx) == 0:
            if verbose: print(f"    R{c}: 0 (izpuscen — ni prisoten)")
            continue
        if len(cls_idx) < target:
            extra = rng.choice(cls_idx, size=target - len(cls_idx), replace=True)
            full = np.concatenate([cls_idx, extra])
        else:
            full = cls_idx
        if verbose:
            print(f"    R{c}: {len(cls_idx):5,} -> {len(full):6,}")
        idx_parts.append(full)
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    if verbose:
        print(f"  Skupaj po oversamplingu: {len(idx):,} (prej {len(y):,})")
    return coords[idx], y[idx]


# ---------------------------------------------------------------------------
# Preprocessing — BREZ StandardScaler, samo PCA (kot v11)
# ---------------------------------------------------------------------------
def build_preprocessed(data, fit_mask, n_pca=N_PCA, seed=42):
    H, W, D = data.shape
    flat = data.reshape(-1, D)
    print(f"  PCA({n_pca}) fit na {int(fit_mask.sum()):,} pikslih (BREZ StandardScaler)...")
    t0 = time.time()
    pca = PCA(n_components=n_pca, random_state=seed)
    pca.fit(flat[fit_mask.ravel()])
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%  "
          f"(clanek SD: 90.03%)")
    flat_pca = pca.transform(flat).astype(np.float32)
    data_pca = flat_pca.reshape(H, W, n_pca)
    print(f"    -> {time.time()-t0:.1f}s")
    return data_pca, pca


# ---------------------------------------------------------------------------
# Dataset — SAMO patch (brez spektralne veje)
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
    def __init__(self, data_pca, coords, labels,
                 patch_size=PATCH_SIZE, augment=False, tta_idx=-1):
        self.pad     = patch_size // 2
        self.coords  = coords
        self.labels  = labels
        self.augment = augment
        self.tta_idx = tta_idx
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
        return patch_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Arhitektura: Single-Stream CNN (posnema Fig. 3 clanka, SD varianta brez BN)
# ---------------------------------------------------------------------------
class SingleStreamCNN(nn.Module):
    def __init__(self, n_channels=N_PCA, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.Softplus(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Softplus(), nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Softplus(), nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, PATCH_SIZE, PATCH_SIZE)
            flatten_dim = self.features(dummy).numel()
        self.fc1        = nn.Linear(flatten_dim, 128)
        self.fc1_act     = nn.Softplus()
        self.dropout_fc  = nn.Dropout(dropout)
        self.fc2         = nn.Linear(128, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def get_logits(self, patch):
        f = self.features(patch)
        f = f.flatten(1)
        f = self.dropout_fc(self.fc1_act(self.fc1(f)))
        return self.fc2(f)

    def forward(self, patch):
        return self.log_softmax(self.get_logits(patch))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


def print_class_distribution(y, label="razredi"):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    print(f"  Porazdelitev ({label}): " +
          ", ".join(f"R{c}={counts[c]:,}" for c in range(NUM_CLASSES)))


@torch.no_grad()
def get_logits_array(model, dataset, device, batch_size=512):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_l  = []
    for patches, _ in loader:
        all_l.append(model.get_logits(patches.to(device)).cpu().numpy())
    return np.concatenate(all_l)


@torch.no_grad()
def get_logits_tta(model, data_pca, coords, labels, device, batch_size=512):
    logits_sum = None
    for aug_idx in range(8):
        ds = PatchDataset(data_pca, coords, labels, augment=False, tta_idx=aug_idx)
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


def write_results_report(model_name, innerval_oa, innerval_ll, test_oa, test_ll,
                          output_path, t_opt, sigma, max_epochs, best_epoch,
                          final_epochs, extra_note=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"CORE2_OA={test_oa*100:6.2f}%  CORE2_ll={test_ll:.5f}  "
        f"T={t_opt:.4f}  sigma={sigma:.1f}  "
        f"max_ep={max_epochs}(best={best_epoch})  final_ep={final_epochs}\n"
        f"{'':>19}  Primerjava: CNN(clanek,SD,cross-slide)=79.45%+/-1.25%"
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
def train_single(data_pca, val_ds, y_train, y_val,
                 train_coords, device, max_epochs, batch_size, lr, seed, dropout):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = PatchDataset(data_pca, train_coords, y_train, augment=True)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = SingleStreamCNN(dropout=dropout).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_loss, best_state, best_epoch = float('inf'), None, 1
    print(f"  {'Ep':>4}  {'Train ll':>10}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}")
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
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


def train_blind(data_pca, train_coords, y_train, device,
                final_epochs, batch_size, lr, seed, dropout):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = PatchDataset(data_pca, train_coords, y_train, augment=True)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model     = SingleStreamCNN(dropout=dropout).to(device)
    criterion = nn.NLLLoss()
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
        description="Model C Best v12: zvesta replika arhitekture/treninga iz clanka"
    )
    parser.add_argument("--input",           default="image1-competition.hdf5")
    parser.add_argument("--output",          default="modelC_best_v12_core2.npy")
    parser.add_argument("--max-epochs",      type=int,   default=15,
                        help="Zgornja meja epoh za Fazo A (best-of-N na inner-val).")
    parser.add_argument("--batch-size",      type=int,   default=128,
                        help="Clanek: batch size 128.")
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--dropout",         type=float, default=0.5,
                        help="Clanek: keep_prob=0.5 -> dropout rate=0.5.")
    parser.add_argument("--n-ensemble",      type=int,   default=12)
    parser.add_argument("--sigma",           type=float, default=1.5)
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
    n_param = sum(p.numel() for p in SingleStreamCNN().parameters() if p.requires_grad)
    print(f"\n  Konfiguracija (v12 — zvesta replika clanka):")
    print(f"    Single-stream CNN, brez BN, softplus, dropout={args.dropout}")
    print(f"    Patch {PATCH_SIZE}x{PATCH_SIZE} | PCA({N_PCA}) | TTA 8 | Ensemble {args.n_ensemble}")
    print(f"    Batch={args.batch_size} | max_epochs={args.max_epochs} (Faza A) | ({n_param:,} param)")
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
    print_class_distribution(y_inner_train, "inner-train")
    print_class_distribution(y_inner_val,   "inner-val")

    outer_train_coords = np.argwhere(outer_train_mask)
    y_outer_train = classes[outer_train_mask].astype(np.int64)

    # ==================================================================
    # FAZA A — iskanje hiperparametrov (Core 2 se NE dotakne)
    # ==================================================================
    print("\n=== 4. Faza A — Preprocessing (PCA fit na 4 inner-train krogcih, brez StandardScaler) ===")
    data_pca_A, pca_A = build_preprocessed(data, inner_train_mask, N_PCA, args.seed)

    inner_val_ds = PatchDataset(data_pca_A, inner_val_coords, y_inner_val, augment=False)

    print(f"\n=== 5. Faza A — Oversampling inner-train ===")
    it_coords_os, y_it_os = oversample_coords(
        inner_train_coords, y_inner_train, seed=args.seed, label="Faza A / inner-train")

    print(f"\n=== 6. Faza A — Ensemble ({args.n_ensemble} modelov x do {args.max_epochs} epoh) ===")
    print(f"  Best-of-N na INNER-VAL (Core 2 ni vpleten). Loss neutezen (oversampling).")
    seeds        = list(range(args.seed, args.seed + args.n_ensemble))
    innerval_logits_list = []
    best_epochs  = []
    for i, seed in enumerate(seeds):
        print(f"\n  ── Faza A model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        model, best_epoch = train_single(
            data_pca=data_pca_A, val_ds=inner_val_ds,
            y_train=y_it_os, y_val=y_inner_val,
            train_coords=it_coords_os, device=device,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, dropout=args.dropout,
        )
        best_epochs.append(best_epoch)
        print(f"  TTA na inner-val...")
        innerval_logits_list.append(
            get_logits_tta(model, data_pca_A, inner_val_coords, y_inner_val, device)
        )

    innerval_logits_avg = np.mean(innerval_logits_list, axis=0)
    avg_best_epoch = round(np.mean(best_epochs))
    print(f"\n  Best-of-{args.max_epochs} epohe: {best_epochs} (avg={avg_best_epoch})")

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

    sigma_opt    = args.sigma
    final_epochs = avg_best_epoch

    # ==================================================================
    # FAZA B — finalni model, edini dotik s Core 2
    # ==================================================================
    print(f"\n=== 9. Faza B — Preprocessing (PCA fit na vseh 5 outer-train krogcih) ===")
    data_pca_B, pca_B = build_preprocessed(data, outer_train_mask, N_PCA, args.seed)

    print(f"\n=== 10. Faza B — Oversampling outer-train ===")
    ot_coords_os, y_ot_os = oversample_coords(
        outer_train_coords, y_outer_train, seed=args.seed, label="Faza B / outer-train")

    print(f"\n=== 11. Faza B — Ensemble ({args.n_ensemble} modelov x {final_epochs} epoh, brez peeka) ===")
    print(f"  Core 2 se prvic dotakne SELE po koncanem treningu (TTA napoved).")
    test_logits_sum = np.zeros((len(test_coords), NUM_CLASSES), dtype=np.float64)
    for i, seed in enumerate(seeds):
        print(f"\n  ── Faza B model {i+1}/{args.n_ensemble} (seed={seed}) ──")
        model_b = train_blind(
            data_pca=data_pca_B, train_coords=ot_coords_os, y_train=y_ot_os,
            device=device, final_epochs=final_epochs, batch_size=args.batch_size,
            lr=args.lr, seed=seed, dropout=args.dropout,
        )
        print(f"  TTA na Core 2 (prvic in edinkrat)...")
        test_logits_sum += get_logits_tta(
            model_b, data_pca_B, test_coords, y_test, device
        ).astype(np.float64)

    # ------------------------------------------------------------------
    print("\n=== 12. KONCNA evaluacija na Core 2 (edini dotik) ===")
    test_probs = apply_temperature(
        (test_logits_sum / args.n_ensemble).astype(np.float32), T_opt
    )
    test_pred = np.argmax(test_probs, axis=1)
    test_oa   = accuracy_score(y_test, test_pred)
    test_ll   = log_loss(y_test, test_probs, labels=np.arange(NUM_CLASSES))
    print(f"  CORE2 OA: {test_oa*100:.2f}%")
    print(f"  CORE2 ll: {test_ll:.5f}")
    print(f"  Primerjava Faza A (inner-val): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Ref clanek CNN (SD, cross-slide): OA={REF_CNN_SD_OA:.2f}% +/- {REF_CNN_SD_STD:.2f}")
    print_per_class_table(y_test, test_pred, test_probs,
                          "Per-class OA in log-loss (Faza B, Core 2 — koncni test):")

    r_min = int(test_coords[:, 0].min())
    r_max = int(test_coords[:, 0].max())
    c_min = int(test_coords[:, 1].min())
    c_max = int(test_coords[:, 1].max())
    bbox_h = r_max - r_min + 1
    bbox_w = c_max - c_min + 1

    prior = np.bincount(y_outer_train, minlength=NUM_CLASSES).astype(np.float32)
    prior /= prior.sum()
    prob_map = np.tile(prior, (bbox_h * bbox_w, 1)).reshape(bbox_h, bbox_w, NUM_CLASSES)
    test_tissue_2d = tissue_mask[r_min:r_max+1, c_min:c_max+1]
    for (r, c), prob in zip(test_coords, test_probs):
        prob_map[r - r_min, c - c_min] = prob

    if sigma_opt > 0:
        smoothed  = gaussian_smooth_probs(prob_map, test_tissue_2d, sigma_opt)
        final_map = prob_map.copy()
        final_map[test_tissue_2d] = smoothed[test_tissue_2d]
    else:
        final_map = prob_map

    final_map = np.clip(final_map, 1e-7, 1.0)
    final_map /= final_map.sum(axis=-1, keepdims=True)
    np.save(args.output, final_map.astype(np.float32))
    print(f"\n  Shranjeno: {args.output}  shape={final_map.shape}")
    print(f"  (bbox Core 2: vrstice {r_min}-{r_max}, stolpci {c_min}-{c_max})")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (v12) ===")
    print(f"  Zunanji split: 5 krogcev (outer-train) | Core 2 (TEST)")
    print(f"  Notranji split: 4 (inner-train) | 1 (inner-val)")
    print(f"  Faza A: max_epochs={args.max_epochs}, best_epochs={best_epochs} (avg={avg_best_epoch})")
    print(f"  Faza B: final_epochs={final_epochs}, brez peeka, en dotik s Core 2")
    print(f"  T={T_opt:.4f} | sigma={sigma_opt:.1f} | dropout={args.dropout} | batch={args.batch_size}")
    print(f"  Arhitektura: single-stream, brez BN, softplus, weight init N(0,0.02)")
    print(f"\n  Faza A (inner-val, hiperparametri): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (Core 2, KONCNI test):        OA={test_oa*100:.2f}%  ll={test_ll:.5f}")
    print(f"\n  Primerjava:")
    print(f"    CNN (clanek, SD, cross-slide, tezja naloga): OA={REF_CNN_SD_OA:.2f}% +/- {REF_CNN_SD_STD:.2f}")
    print(f"    v10 (nasa gnezdena, dual-stream, moderne izbire): OA=73.82%  ll=0.84206")
    print(f"    v12 (nasa gnezdena, single-stream, zvesta replika): OA={test_oa*100:.2f}%  ll={test_ll:.5f}")

    # ------------------------------------------------------------------
    print(f"\n=== 13. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelC_best_v12",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa, test_ll=test_ll,
        output_path=args.output,
        t_opt=T_opt, sigma=sigma_opt,
        max_epochs=args.max_epochs,
        best_epoch=avg_best_epoch,
        final_epochs=final_epochs,
        extra_note=METODOLOGIJA_OPOMBA
    )


if __name__ == "__main__":
    main()
