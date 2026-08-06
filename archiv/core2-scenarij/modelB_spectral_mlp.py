"""
Model B — Spektralni MLP  (v3: pravilna zasnova)
=================================================
Zakaj CNN verziji (v1, v2) nista premagali LR:

  Problem 1 — Napačna arhitektura:
    1D konvolucija + MaxPool predpostavlja translacijsko invariantnost.
    Pri FTIR spektrih je vsaka točka fiksna (Amide I je VEDNO pri idx 162).
    MaxPool uniči to pozicijsko informacijo. LR tega ne počne.

  Problem 2 — Oversampling pokvari kalibrацijo:
    Po oversamplu CNN vidi enako porazdelitev razredov (1/6 vsak).
    Na val/test je porazdelitev neenaka → napovedi so slabo kalibrirane
    → visok log loss kljub OK točnosti.

  Problem 3 — Preveč parametrov:
    176k param / 23k vzorcev = 7:1 → garantiran overfitting.

Rešitve v tej verziji:
  ✓ MLP (brez konvolucij) — vsaka vhodna dimenzija dobi svojo utež
  ✓ Tehtana cross-entropy namesto oversamplinga — ohranja realno porazdelitev
  ✓ Temperature scaling — post-hoc kalibracija verjetnosti na val množici
  ✓ Manjša arhitektura (90k param) + agresivni Dropout (0.5/0.4/0.3)
  ✓ Finalni model: train za best_epoch epoh (ne slepo 150)

Preprocessing: enak (rubberband + Amide I, brez PCA).
Split: isti prostorski 3-way (60/20/20).
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from torch.utils.data import DataLoader, TensorDataset

from spatial_split import make_spatial_three_way_split, print_split_summary


# ---------------------------------------------------------------------------
# Konstante
# ---------------------------------------------------------------------------
NUM_CLASSES  = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660
AMIDE_I_TARGET_WN = 1650.0


# ---------------------------------------------------------------------------
# Nalaganje podatkov
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
    print(f"  Amide I: target={AMIDE_I_TARGET_WN:.1f} cm-1 | "
          f"actual={wns[idx]:.2f} cm-1 | index={idx}")
    return idx


def make_prediction_crop_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1] = True
    return mask


# ---------------------------------------------------------------------------
# Preprocessing (identičen Model A/B)
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


def rubberband_baseline_correction(spectra: np.ndarray) -> np.ndarray:
    out = np.empty_like(spectra, dtype=np.float32)
    for i in range(len(spectra)):
        out[i] = _rubberband_single(spectra[i])
    return out


def amide_i_normalize(spectra: np.ndarray, amide_i_idx: int,
                      eps: float = 1e-6) -> np.ndarray:
    amide_vals = spectra[:, amide_i_idx].astype(np.float64)
    n_bad = int(np.sum(amide_vals <= eps))
    if n_bad > 0:
        print(f"  Opozorilo: {n_bad} spektrov ima Amide I <= {eps}.")
    amide_safe = np.where(amide_vals > eps, amide_vals, eps)
    return (spectra / amide_safe[:, np.newaxis]).astype(np.float32)


def preprocess(spectra: np.ndarray, amide_i_idx: int,
               label: str = "") -> np.ndarray:
    prefix = f"  [{label}] " if label else "  "
    t0 = time.time()
    print(f"{prefix}Rubberband korekcija ({len(spectra)} spektrov)...")
    bc = rubberband_baseline_correction(spectra)
    print(f"{prefix}  → {time.time()-t0:.1f}s")
    print(f"{prefix}Amide I normalizacija (idx={amide_i_idx})...")
    t1 = time.time()
    normed = amide_i_normalize(bc, amide_i_idx)
    print(f"{prefix}  → {time.time()-t1:.1f}s")
    return normed


# ---------------------------------------------------------------------------
# Tehtana loss — nadomešča oversampling
# ---------------------------------------------------------------------------
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Standardne inverzne frekvenčne uteži:
        w_c = N / (K * n_c)
    kjer je N skupno število vzorcev, K število razredov, n_c velikost razreda c.

    Z utežmi je vsak razred v loss funkciji zastopan enakovredno,
    brez da bi spremenili porazdelitev vzorcev → verjetnosti ostanejo kalibrirane.
    """
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    print(f"  Class weights: {[f'{w:.3f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model: spektralni MLP
# ---------------------------------------------------------------------------
class SpectralMLP(nn.Module):
    """
    Multi-layer perceptron za klasifikacijo FTIR spektrov.

    Vsaka vhodna dimenzija (wavenumber) dobi svojo utež — brez predpostavke
    o translacijski invariantnosti (ki bi jo naredil CNN).

    Vhod:  (batch, 187)
    Izhod: (batch, NUM_CLASSES)  — log-verjetnosti

    Arhitektura:
      Linear(187 → 256) + BN + ReLU + Dropout(0.50)
      Linear(256 → 128) + BN + ReLU + Dropout(0.40)
      Linear(128 →  64) + BN + ReLU + Dropout(0.30)
      Linear( 64 →   6)              [logiti]
      LogSoftmax
    """

    def __init__(self, input_dim: int = 187, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.50),

            nn.Linear(256, 128),       nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.40),

            nn.Linear(128, 64),        nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.30),
        )
        self.head = nn.Linear(64, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Vrne logite (pred LogSoftmax) — potrebno za temperature scaling."""
        return self.head(self.body(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_softmax(self.get_logits(x))


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def find_temperature(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray,
                     device: torch.device) -> float:
    """
    Poišči optimalno temperaturo T, ki minimizira log loss na val množici.

    T > 1: zmehčaj verjetnosti (model je preveč samozavesten)
    T < 1: zaostrí verjetnosti (model je premalo samozavesten)
    T = 1: brez spremembe

    Metoda: scipy.optimize.minimize_scalar (bounded, brez gradient).
    """
    model.eval()
    logits_list = []
    X_t = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), 1024):
            batch = X_t[i:i+1024].to(device)
            logits_list.append(model.get_logits(batch).cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)  # (N, K)

    def neg_ll(T: float) -> float:
        scaled = logits / T
        # Numerično stabilen softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-9, 1.0)
        return -np.mean(np.log(probs[np.arange(len(y_val)), y_val]))

    result = minimize_scalar(neg_ll, bounds=(0.05, 10.0), method='bounded')
    T_opt = result.x
    ll_before = neg_ll(1.0)
    ll_after  = neg_ll(T_opt)
    print(f"  Temperature scaling: T={T_opt:.4f}  "
          f"log_loss: {ll_before:.5f} → {ll_after:.5f}  "
          f"(izboljšava: {ll_before - ll_after:.5f})")
    return T_opt


@torch.no_grad()
def predict_proba_mlp(model: nn.Module, X: np.ndarray,
                      device: torch.device,
                      temperature: float = 1.0,
                      batch_size: int = 2048) -> np.ndarray:
    """Vrne verjetnosti. Če temperature != 1.0, skalira logite."""
    model.eval()
    all_probs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i+batch_size].to(device)
        if temperature == 1.0:
            log_probs = model(batch)
            probs = torch.exp(log_probs)
        else:
            logits = model.get_logits(batch) / temperature
            probs  = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Naprava
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Naprava: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("  Naprava: CPU")
    return device


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def make_dataloader(X: np.ndarray, y: np.ndarray,
                    batch_size: int = 512, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size, shuffle=shuffle, num_workers=0,
    )


# ---------------------------------------------------------------------------
# Ena epoha treninga
# ---------------------------------------------------------------------------
def _train_epoch(model, loader, optimizer, criterion, device,
                 clip_grad: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Fitanje
# ---------------------------------------------------------------------------
def fit_model_b(
    X_pp: np.ndarray,           # že preprocesirani spektri
    y: np.ndarray,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 20,
    seed: int = 42,
    label: str = "",
    X_val_pp: np.ndarray = None,
    y_val: np.ndarray = None,
) -> tuple:
    """
    Trenira SpectralMLP. Vrne (model, best_epoch).

    Brez oversamplinga — class_weights v loss funkciji poskrbijo za balans.
    """
    torch.manual_seed(seed)

    loader = make_dataloader(X_pp, y, batch_size=batch_size, shuffle=True)

    model     = SpectralMLP(input_dim=X_pp.shape[1]).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SpectralMLP: {n_params:,} parametrov | input={X_pp.shape[1]}")

    criterion = nn.NLLLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    has_val          = (X_val_pp is not None)
    best_val_loss    = float('inf')
    best_state       = None
    best_epoch       = epochs
    patience_counter = 0

    if has_val:
        print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
        print(f"  {'─'*6}  {'─'*11}  {'─'*9}  {'─'*9}  {'─'*9}")
    else:
        print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'LR':>9}")
        print(f"  {'─'*6}  {'─'*11}  {'─'*9}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        if has_val:
            probs  = predict_proba_mlp(model, X_val_pp, device)
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
                    print(f"\n  Early stopping pri epohi {epoch}  "
                          f"(best val loss={best_val_loss:.5f} pri epohi {best_epoch})")
                    break
        else:
            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>6}  {train_loss:>11.5f}  {lr_now:>9.2e}")

    print(f"\n  Trening zaključen v {time.time()-t0:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Naložene najboljše uteži (val loss={best_val_loss:.5f}, epoha={best_epoch})")

    return model, best_epoch


# ---------------------------------------------------------------------------
# Evaluacija
# ---------------------------------------------------------------------------
def evaluate_mlp(model: nn.Module, device: torch.device,
                 X_pp: np.ndarray, y_true: np.ndarray,
                 split_name: str = "",
                 temperature: float = 1.0) -> tuple:
    probs = predict_proba_mlp(model, X_pp, device, temperature=temperature)
    preds = np.argmax(probs, axis=1)

    oa = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))

    temp_str = f" [T={temperature:.3f}]" if temperature != 1.0 else ""
    print(f"\n  ── {split_name}{temp_str} ──")
    print(f"  OA:       {oa*100:.2f}%")
    print(f"  Log loss: {ll:.5f}")
    print(f"  Ref — Model A SVM:    VAL 77.14%/0.607  | TEST 92.69%/0.347")
    print(f"  Ref — Model B CNN v2: VAL 75.21%/0.745  | TEST 90.79%/0.400")

    print(f"\n  Natancnost po razredih:")
    per_class = []
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"    Razred {c}: N/A")
            continue
        acc_c = (preds[mask] == y_true[mask]).mean()
        per_class.append(acc_c)
        print(f"    Razred {c}: {acc_c*100:.2f}%  (n={mask.sum()})")

    print(f"\n  Macro OA: {np.mean(per_class)*100:.2f}%")

    cm = confusion_matrix(y_true, preds, labels=np.arange(NUM_CLASSES))
    print(f"\n  Matrika zmede:")
    print(cm)

    return oa, ll


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def make_submission_mlp(
    model: nn.Module,
    device: torch.device,
    data: np.ndarray,
    tissue_mask: np.ndarray,
    amide_i_idx: int,
    output_path: str,
    temperature: float = 1.0,
    train_class_counts: np.ndarray = None,
) -> None:
    crop_h, crop_w = PRED_R1 - PRED_R0, PRED_C1 - PRED_C0
    n_crop         = crop_h * crop_w
    crop_data      = data[PRED_R0:PRED_R1, PRED_C0:PRED_C1]
    crop_tissue    = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)
    n_tissue       = int(crop_tissue.sum())

    print(f"  Crop: {n_crop} pikslov | tkivo: {n_tissue} | ozadje: {n_crop-n_tissue}")

    prior = (train_class_counts.astype(np.float32) / train_class_counts.sum()
             if train_class_counts is not None
             else np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES)

    submission_flat = np.tile(prior, (n_crop, 1)).astype(np.float32)

    if n_tissue > 0:
        X_tissue = crop_data.reshape(-1, crop_data.shape[-1])[crop_tissue]
        X_pp     = preprocess(X_tissue, amide_i_idx, label="submission")
        probs    = predict_proba_mlp(model, X_pp, device, temperature=temperature)
        submission_flat[crop_tissue] = probs.astype(np.float32)

    submission = submission_flat.reshape(crop_h, crop_w, NUM_CLASSES)
    np.save(output_path, submission)
    print(f"  Submission shranjen: {output_path}  shape={submission.shape}  T={temperature:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model B v3 — Spektralni MLP (tehtana loss + temperature scaling)"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output",             default="modelB_mlp.npy")
    parser.add_argument("--epochs",             type=int,   default=200)
    parser.add_argument("--batch-size",         type=int,   default=512)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--patience",           type=int,   default=20)
    parser.add_argument("--val-fraction",       type=float, default=0.20)
    parser.add_argument("--test-fraction",      type=float, default=0.20)
    parser.add_argument("--min-component-size", type=int,   default=20)
    parser.add_argument("--seed",               type=int,   default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, wns, tissue_mask, classes = load_data(args.input)
    h, w, _ = data.shape
    print(f"  data: {data.shape} | Anotiranih: {(classes != -1).sum()}")
    amide_i_idx          = find_amide_i_index(wns)
    prediction_crop_mask = make_prediction_crop_mask(h, w)
    device               = get_device()

    # ------------------------------------------------------------------
    print("\n=== 2. Prostorski 3-way split ===")
    train_mask, val_mask, test_mask = make_spatial_three_way_split(
        tissue_mask          = tissue_mask,
        classes              = classes,
        prediction_crop_mask = prediction_crop_mask,
        val_fraction         = args.val_fraction,
        test_fraction        = args.test_fraction,
        min_component_size   = args.min_component_size,
        verbose              = True,
    )
    print_split_summary(train_mask, val_mask, test_mask, classes)

    X_train_raw = data[train_mask];  y_train = classes[train_mask].astype(np.int64)
    X_val_raw   = data[val_mask];    y_val   = classes[val_mask].astype(np.int64)
    X_test_raw  = data[test_mask];   y_test  = classes[test_mask].astype(np.int64)

    # ------------------------------------------------------------------
    print("\n=== 3. Preprocessing ===")
    X_train_pp = preprocess(X_train_raw, amide_i_idx, label="train")
    X_val_pp   = preprocess(X_val_raw,   amide_i_idx, label="val")
    X_test_pp  = preprocess(X_test_raw,  amide_i_idx, label="test")

    # Tehtana loss iz TRAIN porazdelitve (ne oversamplinga)
    weights = compute_class_weights(y_train)

    # ------------------------------------------------------------------
    print("\n=== 4. Ucenje MLP na TRAIN splitu ===")
    model, best_epoch = fit_model_b(
        X_pp          = X_train_pp,
        y             = y_train,
        class_weights = weights,
        device        = device,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        patience      = args.patience,
        seed          = args.seed,
        label         = "train",
        X_val_pp      = X_val_pp,
        y_val         = y_val,
    )

    # ------------------------------------------------------------------
    print("\n=== 5. Temperature scaling (kalibracija na VAL) ===")
    T_opt = find_temperature(model, X_val_pp, y_val, device)

    # ------------------------------------------------------------------
    print("\n=== 6. Evaluacija ===")
    print("\n--- Brez temperature scaling ---")
    oa_val,  ll_val  = evaluate_mlp(model, device, X_val_pp,  y_val,  "VAL")
    print()
    oa_test, ll_test = evaluate_mlp(model, device, X_test_pp, y_test, "TEST (zaklenjen)")

    print("\n--- S temperature scaling (T={:.4f}) ---".format(T_opt))
    oa_val_t,  ll_val_t  = evaluate_mlp(model, device, X_val_pp,  y_val,  "VAL",  T_opt)
    print()
    oa_test_t, ll_test_t = evaluate_mlp(model, device, X_test_pp, y_test, "TEST", T_opt)

    # ------------------------------------------------------------------
    print("\n=== 7. Finalni model (vse anotacije, best_epoch={}) ===".format(best_epoch))
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    X_all_raw   = data[usable_mask]
    y_all       = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov: {len(y_all)}")

    X_all_pp      = preprocess(X_all_raw, amide_i_idx, label="final")
    weights_final = compute_class_weights(y_all)

    # Finalni model: iste epohe kot train split (best_epoch), brez val
    model_final, _ = fit_model_b(
        X_pp          = X_all_pp,
        y             = y_all,
        class_weights = weights_final,
        device        = device,
        epochs        = best_epoch,      # ← točno toliko epoh kot pri train splitu
        batch_size    = args.batch_size,
        lr            = args.lr,
        patience      = args.patience,   # ni v uporabi brez val
        seed          = args.seed,
        label         = "final",
        X_val_pp      = None,
        y_val         = None,
    )

    # Temperature scaling za finalni model: uporabi T iz val eksperimenta
    print(f"  Temperature scaling: uporabimo T={T_opt:.4f} (iz val eksperimenta)")

    # ------------------------------------------------------------------
    print("\n=== 8. Submission ===")
    make_submission_mlp(
        model              = model_final,
        device             = device,
        data               = data,
        tissue_mask        = tissue_mask,
        amide_i_idx        = amide_i_idx,
        output_path        = args.output,
        temperature        = T_opt,
        train_class_counts = np.bincount(y_all, minlength=NUM_CLASSES),
    )

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  Model B MLP (spektralni, tehtana loss + temperature scaling)")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Temperature: T={T_opt:.4f}")
    print(f"\n  {'':20} {'VAL OA':>10} {'VAL ll':>10} {'TEST OA':>10} {'TEST ll':>10}")
    print(f"  {'Brez T-scaling':20} {oa_val*100:>9.2f}% {ll_val:>10.5f} {oa_test*100:>9.2f}% {ll_test:>10.5f}")
    print(f"  {'S T-scalingom':20} {oa_val_t*100:>9.2f}% {ll_val_t:>10.5f} {oa_test_t*100:>9.2f}% {ll_test_t:>10.5f}")
    print(f"  {'Model A (SVM)':20} {'77.14%':>10} {'0.60700':>10} {'92.69%':>10} {'0.34700':>10}")
    print(f"  {'Model B CNN v2':20} {'75.21%':>10} {'0.74488':>10} {'90.79%':>10} {'0.40008':>10}")
    print(f"\n  Submission: {args.output}")


if __name__ == "__main__":
    main()
