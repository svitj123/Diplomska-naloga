"""
Model B — Spektralni 1D CNN  (v3: gnezdena Core2 validacija)
================================================================
Razlika od modelB_spectral_cnn_v2: zamenjan STAR 3-way komponentni split
(spatial_split.py, 60/20/20) z ISTO gnezdeno KMeans leave-one-core-out
validacijo kot modelC v10-v12 in modelA v4 — poštena primerjava na istem
Core2 test setu.

  Zunanji split (KMeans k=6):  5 krogcev (outer-train)  |  Core 2 (TEST)
  Notranji split (KMeans k=5): 4 (inner-train) | 1 (inner-val)

Faza A (Core 2 se ne dotakne): trenira na 4 inner-train krogcih, best-of-N
selekcija checkpointa na inner-val (brez early stopping) -> avg_best_epoch.
Faza B (edini dotik s Core 2): trenira na vseh 5 outer-train krogcev,
fiksno avg_best_epoch, BREZ peekanja -> ena TTA-manj napoved na Core 2.

Arhitektura (SpectralCNN1D) = identicna v1/v2:
  Conv1d(1->32->64->128->256) + BN + ReLU + MaxPool, AdaptiveAvgPool,
  FC(256->128->64->6).

Preprocessing = identicen modelA v4: rubberband + Amide I normalizacija
(brez PCA — 1D CNN dela direktno na 187-dim spektru).
Oversampling manjsinskih razredov namesto class-weighted loss.

Primerjava: CNN (spektralno, brez prostorske informacije, clanek, SD):
OA=62.52% (Tabela 8).
"""

import argparse
import os
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES  = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660
AMIDE_I_TARGET_WN = 1650.0
RESULTS_FILE = "rezultati_report.txt"

REF_CNN_SPEC_SD_OA = 62.52

METODOLOGIJA_OPOMBA = (
    "modelB_v3: Spektralni 1D CNN (clanek-zvest preprocessing: rubberband+"
    "Amide I, brez PCA) z ISTO gnezdeno Core2 validacijo kot modelC v10-v12 "
    "in modelA v4 (prej: star 3-way komponentni split). Oversampling "
    "namesto class weights. Ref. clanek CNN spektralno (SD): OA=62.52%."
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


def find_amide_i_index(wns):
    idx = int(np.argmin(np.abs(wns - AMIDE_I_TARGET_WN)))
    print(f"  Amide I: target={AMIDE_I_TARGET_WN:.1f} cm-1 | "
          f"actual={wns[idx]:.2f} cm-1 | index={idx}")
    return idx


def make_prediction_crop_mask(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1] = True
    return mask


# ---------------------------------------------------------------------------
# Gnezdeni prostorski split (identicno modelA v4 / modelC v10-v12)
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
# Preprocessing (zvest clanku — rubberband + Amide I)
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
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    lower_x = np.array([p[0] for p in lower])
    lower_y = np.array([p[1] for p in lower])
    return (y - np.interp(x, lower_x, lower_y)).astype(np.float32)


def rubberband_baseline_correction(spectra):
    out = np.empty_like(spectra, dtype=np.float32)
    for i in range(len(spectra)):
        out[i] = _rubberband_single(spectra[i])
    return out


def amide_i_normalize(spectra, amide_i_idx, eps=1e-6):
    amide_vals = spectra[:, amide_i_idx].astype(np.float64)
    n_bad = int(np.sum(amide_vals <= eps))
    if n_bad > 0:
        print(f"  Opozorilo: {n_bad} spektrov ima Amide I <= {eps}.")
    amide_safe = np.where(amide_vals > eps, amide_vals, eps)
    return (spectra / amide_safe[:, np.newaxis]).astype(np.float32)


def preprocess(spectra, amide_i_idx, label=""):
    prefix = f"  [{label}] " if label else "  "
    t0 = time.time()
    print(f"{prefix}Rubberband korekcija ({len(spectra)} spektrov)...")
    bc = rubberband_baseline_correction(spectra)
    print(f"{prefix}  -> {time.time()-t0:.1f}s")
    print(f"{prefix}Amide I normalizacija (idx={amide_i_idx})...")
    t1 = time.time()
    normed = amide_i_normalize(bc, amide_i_idx)
    print(f"{prefix}  -> {time.time()-t1:.1f}s")
    return normed


# ---------------------------------------------------------------------------
# Oversampling manjsinskih razredov (namesto class weights)
# ---------------------------------------------------------------------------
def oversample_to_max_class(X, y, seed=42, verbose=True, label="train"):
    rng = np.random.default_rng(seed)
    counts = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    target = max(len(idx) for idx in counts if len(idx) > 0)
    if verbose:
        print(f"  Oversampling ({label}): {[len(i) for i in counts]} -> {target}/razred")
    sampled = [rng.choice(idx, size=target, replace=True)
               for idx in counts if len(idx) > 0]
    idx_all = np.concatenate(sampled)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


# ---------------------------------------------------------------------------
# Arhitektura (identicna v1/v2)
# ---------------------------------------------------------------------------
class SpectralCNN1D(nn.Module):
    def __init__(self, input_len=187, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=7, padding=3),
            nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,  64,  kernel_size=5, padding=2),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def get_logits(self, x):
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

    def forward(self, x):
        return self.log_softmax(self.get_logits(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


def make_dataloader(X, y, batch_size=512, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict_proba(model, X, device, batch_size=1024):
    model.eval()
    all_probs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i+batch_size].to(device)
        all_probs.append(torch.exp(model(batch)).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


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
                          output_path, max_epochs, best_epoch, final_epochs,
                          extra_note=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"CORE2_OA={test_oa*100:6.2f}%  CORE2_ll={test_ll:.5f}  "
        f"max_ep={max_epochs}(best={best_epoch})  final_ep={final_epochs}\n"
        f"{'':>19}  Primerjava: CNN-spektralno(clanek,SD)=62.52%"
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
# Trening — Faza A (best-of-N, brez early stopping)
# ---------------------------------------------------------------------------
def train_single(X_train, y_train, X_val, y_val, device,
                 max_epochs, batch_size, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    model     = SpectralCNN1D().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_loss, best_state, best_epoch = float('inf'), None, 1
    print(f"  {'Ep':>4}  {'Train ll':>10}  {'Val OA':>9}  {'Val ll':>9}")
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(yb)
        train_loss = total / len(loader.dataset)
        scheduler.step()

        probs  = predict_proba(model, X_val, device)
        val_oa = accuracy_score(y_val, np.argmax(probs, axis=1))
        val_ll = log_loss(y_val, probs, labels=np.arange(NUM_CLASSES))
        marker = " *" if val_ll < best_val_loss else "  "
        print(f"  {epoch:>4}  {train_loss:>10.5f}  {val_oa*100:>8.2f}%  {val_ll:>9.5f}{marker}")

        if val_ll < best_val_loss:
            best_val_loss = val_ll
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch

    model.load_state_dict(best_state)
    print(f"  Treniran v {time.time()-t0:.1f}s | best ep={best_epoch}, val_ll={best_val_loss:.5f}")
    return model, best_epoch


def train_blind(X_train, y_train, device, final_epochs, batch_size, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    model     = SpectralCNN1D().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs, eta_min=1e-6)

    print(f"  {'Ep':>4}  {'Train ll':>10}")
    t0 = time.time()
    for epoch in range(1, final_epochs + 1):
        model.train()
        total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(yb)
        scheduler.step()
        if epoch % 5 == 0 or epoch in (1, final_epochs):
            print(f"  {epoch:>4}  {total/len(loader.dataset):>10.5f}")
    print(f"  Treniran v {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model B v3 — spektralni 1D CNN, gnezdena Core2 validacija"
    )
    parser.add_argument("--input",         default="image1-competition.hdf5")
    parser.add_argument("--output",        default="modelB_v3_core2.npy")
    parser.add_argument("--max-epochs",    type=int,   default=40,
                        help="Zgornja meja epoh za Fazo A (best-of-N na inner-val).")
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--n-cores",       type=int,   default=6)
    parser.add_argument("--n-inner-cores", type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, wns, tissue_mask, classes = load_data(args.input)
    H, W, _ = data.shape
    print(f"  data: {data.shape} | Anotiranih: {(classes != -1).sum()}")
    amide_i_idx = find_amide_i_index(wns)
    prediction_crop_mask = make_prediction_crop_mask(H, W)
    device = get_device()
    n_param = sum(p.numel() for p in SpectralCNN1D().parameters() if p.requires_grad)
    print(f"  SpectralCNN1D: {n_param:,} parametrov")
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    # ------------------------------------------------------------------
    print("\n=== 2. Zunanji split: 5 krogcev (outer-train) | Core 2 (TEST) ===")
    outer_train_mask, test_mask = make_spatial_split(
        tissue_mask, classes, prediction_crop_mask,
        n_cores=args.n_cores, seed=args.seed, label="core")
    test_coords = np.argwhere(test_mask)
    y_test = classes[test_mask].astype(np.int64)

    print("\n=== 3. Notranji split: 4 (inner-train) | 1 (inner-val), znotraj outer-train ===")
    exclude_for_inner = prediction_crop_mask | test_mask
    inner_train_mask, inner_val_mask = make_spatial_split(
        tissue_mask, classes, exclude_for_inner,
        n_cores=args.n_inner_cores, seed=args.seed, label="subcore")

    X_it_raw = data[inner_train_mask]; y_it = classes[inner_train_mask].astype(np.int64)
    X_iv_raw = data[inner_val_mask];   y_iv = classes[inner_val_mask].astype(np.int64)
    X_ot_raw = data[outer_train_mask]; y_ot = classes[outer_train_mask].astype(np.int64)
    X_test_raw = data[test_mask]

    # ==================================================================
    # FAZA A — iskanje hiperparametrov (Core 2 se NE dotakne)
    # ==================================================================
    print("\n=== 4. Faza A — Preprocessing (rubberband + Amide I) ===")
    X_it = preprocess(X_it_raw, amide_i_idx, label="Faza A / inner-train")
    X_iv = preprocess(X_iv_raw, amide_i_idx, label="Faza A / inner-val")

    print(f"\n=== 5. Faza A — Oversampling inner-train ===")
    X_it_os, y_it_os = oversample_to_max_class(X_it, y_it, seed=args.seed,
                                               label="Faza A / inner-train")

    print(f"\n=== 6. Faza A — trening (do {args.max_epochs} epoh, best-of-N na inner-val) ===")
    model_A, best_epoch = train_single(
        X_it_os, y_it_os, X_iv, y_iv, device,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed,
    )

    print("\n=== 7. Faza A — evaluacija na inner-val ===")
    innerval_probs = predict_proba(model_A, X_iv, device)
    innerval_pred  = np.argmax(innerval_probs, axis=1)
    innerval_oa    = accuracy_score(y_iv, innerval_pred)
    innerval_ll    = log_loss(y_iv, innerval_probs, labels=np.arange(NUM_CLASSES))
    print(f"  INNERVAL OA: {innerval_oa*100:.2f}%")
    print(f"  INNERVAL ll: {innerval_ll:.5f}")
    print_per_class_table(y_iv, innerval_pred, innerval_probs,
                          "Per-class OA in log-loss (Faza A, inner-val):")

    final_epochs = best_epoch

    # ==================================================================
    # FAZA B — finalni model, edini dotik s Core 2
    # ==================================================================
    print(f"\n=== 8. Faza B — Preprocessing (rubberband + Amide I) ===")
    X_ot   = preprocess(X_ot_raw,   amide_i_idx, label="Faza B / outer-train")
    X_test = preprocess(X_test_raw, amide_i_idx, label="Faza B / Core 2")

    print(f"\n=== 9. Faza B — Oversampling outer-train ===")
    X_ot_os, y_ot_os = oversample_to_max_class(X_ot, y_ot, seed=args.seed,
                                               label="Faza B / outer-train")

    print(f"\n=== 10. Faza B — trening ({final_epochs} epoh, brez peeka) ===")
    model_B = train_blind(X_ot_os, y_ot_os, device,
                          final_epochs=final_epochs, batch_size=args.batch_size,
                          lr=args.lr, seed=args.seed)

    print("\n=== 11. KONCNA evaluacija na Core 2 (edini dotik) ===")
    test_probs = predict_proba(model_B, X_test, device)
    test_pred  = np.argmax(test_probs, axis=1)
    test_oa    = accuracy_score(y_test, test_pred)
    test_ll    = log_loss(y_test, test_probs, labels=np.arange(NUM_CLASSES))
    print(f"  CORE2 OA: {test_oa*100:.2f}%")
    print(f"  CORE2 ll: {test_ll:.5f}")
    print(f"  Primerjava Faza A (inner-val): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Ref clanek CNN-spektralno (SD): OA={REF_CNN_SPEC_SD_OA:.2f}%")
    print_per_class_table(y_test, test_pred, test_probs,
                          "Per-class OA in log-loss (Faza B, Core 2 — koncni test):")

    r_min = int(test_coords[:, 0].min()); r_max = int(test_coords[:, 0].max())
    c_min = int(test_coords[:, 1].min()); c_max = int(test_coords[:, 1].max())
    bbox_h = r_max - r_min + 1; bbox_w = c_max - c_min + 1
    prior = np.bincount(y_ot, minlength=NUM_CLASSES).astype(np.float32)
    prior /= prior.sum()
    prob_map = np.tile(prior, (bbox_h * bbox_w, 1)).reshape(bbox_h, bbox_w, NUM_CLASSES)
    for (r, c), prob in zip(test_coords, test_probs):
        prob_map[r - r_min, c - c_min] = prob
    np.save(args.output, prob_map.astype(np.float32))
    print(f"\n  Shranjeno: {args.output}  shape={prob_map.shape}")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (modelB_v3) ===")
    print(f"  Faza A: max_epochs={args.max_epochs}, best_epoch={best_epoch}")
    print(f"  Faza B: final_epochs={final_epochs}, brez peeka, en dotik s Core 2")
    print(f"  Faza A (inner-val, hiperparametri): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (Core 2, KONCNI test):        OA={test_oa*100:.2f}%  ll={test_ll:.5f}")
    print(f"  Ref clanek CNN-spektralno (SD): OA={REF_CNN_SPEC_SD_OA:.2f}%")

    print(f"\n=== 12. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelB_v3",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa, test_ll=test_ll,
        output_path=args.output,
        max_epochs=args.max_epochs, best_epoch=best_epoch, final_epochs=final_epochs,
        extra_note=METODOLOGIJA_OPOMBA,
    )


if __name__ == "__main__":
    main()
