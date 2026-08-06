"""
Model B — Spektralni 1D CNN  (v2: stabiliziran trening)
=========================================================
Razlike glede na v1:
  - lr: 1e-3 → 1e-4  (nižji začetni LR, manj osciliranja)
  - Scheduler: ReduceLROnPlateau → CosineAnnealingLR
    (gladko zniževanje LR, brez nenadnih skokov ki destabilizirajo trening)
  - Gradient clipping: max_norm=1.0 (preprečuje eksplodirajoče gradiente)
  - Patience: 15 → 20 (modelu damo več časa da najde minimum)

V1 problem: val loss je osciliral med 2.23 in 9.57 — model je "odskakoval"
čez minimume. Najboljša epoha je bila 3 (val_loss=2.23), kar pomeni da se
model sploh ni naučil nič koristnega. Vzrok: previsok LR.

Arhitektura (SpectralCNN1D): enaka kot v1.
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
# Preprocessing (identičen Model A in Model B v1)
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
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    lower_x = np.array([p[0] for p in lower])
    lower_y = np.array([p[1] for p in lower])
    return (y - np.interp(x, lower_x, lower_y)).astype(np.float32)


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
# Balansiranje razredov
# ---------------------------------------------------------------------------
def oversample_to_max_class(X: np.ndarray, y: np.ndarray,
                             seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    counts = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    target = max(len(idx) for idx in counts)
    print(f"  Oversampling: {[len(i) for i in counts]} → {target}/razred")
    sampled = [rng.choice(idx, size=target, replace=True)
               for idx in counts if len(idx) > 0]
    idx_all = np.concatenate(sampled)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


# ---------------------------------------------------------------------------
# Model (identičen v1)
# ---------------------------------------------------------------------------
class SpectralCNN1D(nn.Module):
    def __init__(self, input_len: int = 187, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return self.log_softmax(x)


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
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=0)


# ---------------------------------------------------------------------------
# Trening — ena epoha
# ---------------------------------------------------------------------------
def _train_epoch(model: nn.Module, loader: DataLoader,
                 optimizer: optim.Optimizer, criterion: nn.Module,
                 device: torch.device,
                 clip_grad: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        log_probs = model(X_batch)
        loss = criterion(log_probs, y_batch)
        loss.backward()
        # Gradient clipping — preprečuje eksplodirajoče gradiente
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Napoved verjetnosti
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba_cnn(model: nn.Module, X: np.ndarray,
                      device: torch.device,
                      batch_size: int = 1024) -> np.ndarray:
    model.eval()
    all_probs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i + batch_size].to(device)
        log_probs = model(batch)
        all_probs.append(torch.exp(log_probs).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Fitanje modela B v2
# ---------------------------------------------------------------------------
def fit_model_b(
    X_spec: np.ndarray,
    y: np.ndarray,
    amide_i_idx: int,
    epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-4,           # v2: 1e-4 (bil 1e-3)
    patience: int = 20,          # v2: 20 (bil 15)
    clip_grad: float = 1.0,      # v2: gradient clipping
    seed: int = 42,
    label: str = "",
    val_spec: np.ndarray = None,
    val_y: np.ndarray = None,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()

    # 1. Preprocessing
    X_pp = preprocess(X_spec, amide_i_idx, label=label)

    # 2. Oversampling
    X_bal, y_bal = oversample_to_max_class(X_pp, y, seed=seed)

    # 3. DataLoader
    train_loader = make_dataloader(X_bal, y_bal, batch_size=batch_size, shuffle=True)

    # 4. Val preprocessing
    X_val_pp = None
    if val_spec is not None:
        X_val_pp = preprocess(val_spec, amide_i_idx, label=f"{label}-val")

    # 5. Model
    input_len = X_pp.shape[1]
    model = SpectralCNN1D(input_len=input_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SpectralCNN1D: {n_params:,} parametrov | input={input_len}")
    print(f"  Hiperparametri v2: lr={lr}, patience={patience}, clip_grad={clip_grad}")

    # 6. Optimizer + scheduler
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # CosineAnnealingLR: gladko znižuje LR od lr do eta_min čez T_max epoh
    # (brez nenadnih skokov ReduceLROnPlateau, ki so destabilizirali v1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    has_val = (val_spec is not None)
    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0

    if has_val:
        print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
        print(f"  {'─'*6}  {'─'*11}  {'─'*9}  {'─'*9}  {'─'*9}")
    else:
        print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'LR':>9}")
        print(f"  {'─'*6}  {'─'*11}  {'─'*9}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion,
                                  device, clip_grad=clip_grad)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if has_val:
            probs  = predict_proba_cnn(model, X_val_pp, device)
            preds  = np.argmax(probs, axis=1)
            val_oa = accuracy_score(val_y, preds)
            val_ll = log_loss(val_y, probs, labels=np.arange(NUM_CLASSES))

            print(f"  {epoch:>6}  {train_loss:>11.5f}  {val_oa*100:>8.2f}%  {val_ll:>9.5f}  {current_lr:>9.2e}")

            if val_ll < best_val_loss - 1e-6:
                best_val_loss    = val_ll
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping pri epohi {epoch}  "
                          f"(najboljši val loss={best_val_loss:.5f}, patience={patience})")
                    break
        else:
            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>6}  {train_loss:>11.5f}  {current_lr:>9.2e}")

    elapsed = time.time() - t_start
    print(f"\n  Trening zaključen v {elapsed:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Naložene najboljše uteži (val loss={best_val_loss:.5f})")

    return model, device


# ---------------------------------------------------------------------------
# Evaluacija
# ---------------------------------------------------------------------------
def evaluate_cnn(model: nn.Module, device: torch.device,
                 X_spec: np.ndarray, y_true: np.ndarray,
                 amide_i_idx: int, split_name: str = "") -> tuple:
    X_pp  = preprocess(X_spec, amide_i_idx, label=split_name)
    probs = predict_proba_cnn(model, X_pp, device)
    preds = np.argmax(probs, axis=1)

    oa = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))

    print(f"\n  ── {split_name} ──")
    print(f"  OA:       {oa*100:.2f}%")
    print(f"  Log loss: {ll:.5f}")
    print(f"  (Model A ref: VAL OA=77.14%/ll=0.607 | TEST OA=92.69%/ll=0.347)")
    print(f"  (Model B v1: VAL OA=43.40%/ll=2.231  | TEST OA=64.22%/ll=0.998)")

    print(f"\n  Natancnost po razredih:")
    per_class = []
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"    Razred {c}: N/A")
        else:
            acc_c = (preds[mask] == y_true[mask]).mean()
            per_class.append(acc_c)
            print(f"    Razred {c}: {acc_c*100:.2f}%  (n={mask.sum()})")

    macro_acc = np.mean(per_class)
    print(f"\n  Macro OA: {macro_acc*100:.2f}%")

    cm = confusion_matrix(y_true, preds, labels=np.arange(NUM_CLASSES))
    print(f"\n  Matrika zmede:")
    print(cm)

    return oa, ll


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def make_submission_cnn(
    model: nn.Module,
    device: torch.device,
    data: np.ndarray,
    tissue_mask: np.ndarray,
    amide_i_idx: int,
    output_path: str,
    train_class_counts: np.ndarray = None,
) -> None:
    crop_h, crop_w = PRED_R1 - PRED_R0, PRED_C1 - PRED_C0
    n_crop = crop_h * crop_w

    crop_data   = data[PRED_R0:PRED_R1, PRED_C0:PRED_C1]
    crop_tissue = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)

    n_tissue = int(crop_tissue.sum())
    print(f"  Crop: {n_crop} pikslov | tkivo: {n_tissue} | "
          f"ozadje: {n_crop - n_tissue}")

    if train_class_counts is not None:
        prior = train_class_counts.astype(np.float32)
        prior /= prior.sum()
    else:
        prior = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES

    submission_flat = np.tile(prior, (n_crop, 1)).astype(np.float32)

    if n_tissue > 0:
        X_tissue = crop_data.reshape(-1, crop_data.shape[-1])[crop_tissue]
        X_pp     = preprocess(X_tissue, amide_i_idx, label="submission")
        probs    = predict_proba_cnn(model, X_pp, device)
        submission_flat[crop_tissue] = probs.astype(np.float32)

    submission = submission_flat.reshape(crop_h, crop_w, NUM_CLASSES)
    np.save(output_path, submission)
    print(f"  Submission shranjen: {output_path}  shape={submission.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model B v2 — Spektralni 1D CNN (stabiliziran trening)"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output",             default="modelB_v2.npy")
    parser.add_argument("--epochs",             type=int,   default=150)
    parser.add_argument("--batch-size",         type=int,   default=512)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--patience",           type=int,   default=20)
    parser.add_argument("--clip-grad",          type=float, default=1.0)
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

    X_train = data[train_mask];  y_train = classes[train_mask].astype(np.int64)
    X_val   = data[val_mask];    y_val   = classes[val_mask].astype(np.int64)
    X_test  = data[test_mask];   y_test  = classes[test_mask].astype(np.int64)

    # ------------------------------------------------------------------
    print("\n=== 3. Ucenje Model B v2 na TRAIN splitu ===")
    model, device = fit_model_b(
        X_spec      = X_train,
        y           = y_train,
        amide_i_idx = amide_i_idx,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        patience    = args.patience,
        clip_grad   = args.clip_grad,
        seed        = args.seed,
        label       = "train-pp",
        val_spec    = X_val,
        val_y       = y_val,
    )

    # ------------------------------------------------------------------
    print("\n=== 4. Evaluacija ===")
    oa_val,  ll_val  = evaluate_cnn(model, device, X_val,  y_val,  amide_i_idx, "VAL")
    print()
    oa_test, ll_test = evaluate_cnn(model, device, X_test, y_test, amide_i_idx, "TEST (zaklenjen)")

    # ------------------------------------------------------------------
    print("\n=== 5. Finalni model (vse anotacije) ===")
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    X_all = data[usable_mask]
    y_all = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov za finalni model: {len(y_all)}")

    model_final, _ = fit_model_b(
        X_spec      = X_all,
        y           = y_all,
        amide_i_idx = amide_i_idx,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        patience    = args.patience,
        clip_grad   = args.clip_grad,
        seed        = args.seed,
        label       = "final-pp",
        val_spec    = None,
        val_y       = None,
    )

    # ------------------------------------------------------------------
    print("\n=== 6. Submission ===")
    make_submission_cnn(
        model              = model_final,
        device             = device,
        data               = data,
        tissue_mask        = tissue_mask,
        amide_i_idx        = amide_i_idx,
        output_path        = args.output,
        train_class_counts = np.bincount(y_all, minlength=NUM_CLASSES),
    )

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  Model B v2 (1D CNN, stabiliziran trening)")
    print(f"  VAL  — OA: {oa_val*100:.2f}%  |  log loss: {ll_val:.5f}")
    print(f"  TEST — OA: {oa_test*100:.2f}%  |  log loss: {ll_test:.5f}  <- koncna stevilka")
    print(f"  Model A ref:  VAL OA=77.14%/ll=0.607 | TEST OA=92.69%/ll=0.347")
    print(f"  Model B v1:   VAL OA=43.40%/ll=2.231 | TEST OA=64.22%/ll=0.998")
    print(f"  Submission: {args.output}")


if __name__ == "__main__":
    main()
