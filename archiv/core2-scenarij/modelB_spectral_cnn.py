"""
Model B — Spektralni 1D CNN  (v1)
==================================
Razlika glede na Model A:
  - Klasifikator: 1D konvolucijska nevronska mreža (namesto PCA + RBF SVM)
  - Preprocessing: ENAK (rubberband baseline + Amide I normalizacija)
  - Brez PCA: CNN sam se nauči relevantnih spektralnih značilk iz vhodnih 187 točk
  - Loss: direktna cross-entropy (log loss) → bolje kalibrirano za tekmovanje
  - Hardware: MPS (Apple Silicon GPU) → samodejni CPU fallback

Namenska primerjava:
  Model A:  rubberband + Amide I  → PCA(16) → RBF SVM
  Model B:  rubberband + Amide I            → 1D CNN      ← samo klasifikator se razlikuje

Arhitektura (SpectralCNN1D):
  Input(187) → reshape(1×187)
  Conv1d(1→32,  k=7) + BN + ReLU + MaxPool(2)   →  32 ×  93
  Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)   →  64 ×  46
  Conv1d(64→128,k=3) + BN + ReLU + MaxPool(2)   → 128 ×  23
  Conv1d(128→256,k=3)+ BN + ReLU                → 256 ×  23
  AdaptiveAvgPool1d(1)                           → 256
  Dropout → Linear(256→128) → ReLU
  Dropout → Linear(128→64)  → ReLU
  Dropout → Linear(64→6)    → LogSoftmax

Isti prostorski split kot Model A: 60 / 20 / 20 iz spatial_split.py.
Early stopping na val log loss (patience=15 epoh).
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
# Nalaganje podatkov (identično Model A)
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
# Preprocessing — identičen Model A (rubberband + Amide I, BEZ PCA)
# ---------------------------------------------------------------------------
def _rubberband_single(spectrum: np.ndarray) -> np.ndarray:
    """Piece-wise linear (rubber band) baseline korekcija za en spekter."""
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
    """Rubberband + Amide I normalizacija (brez PCA — CNN sam se nauči značilk)."""
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
# Balansiranje razredov (identično Model A)
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
# Model: 1D CNN za spektre
# ---------------------------------------------------------------------------
class SpectralCNN1D(nn.Module):
    """
    1D konvolucijska nevronska mreža za klasifikacijo FTIR spektrov.

    Vhod:  (batch, input_len)      — en preprocesiran spekter na piksel
    Izhod: (batch, NUM_CLASSES)    — log-softmax log-verjetnosti

    Konvolucijski stolp ekstrahira lokalne spektralne vzorce na različnih
    skalah (absorpcijski vrhovi, kombinacije trakov). GlobalAvgPool agregira
    prostorsko invariantno reprezentacijo pred klasifikatorskim FC stolpom.
    """

    def __init__(self, input_len: int = 187, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Blok 1: grobe spektralne značilke (k=7 → ~±3 valovne dolžine)
            nn.Conv1d(1,    32,  kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            # Blok 2
            nn.Conv1d(32,   64,  kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            # Blok 3
            nn.Conv1d(64,   128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            # Blok 4: fine spektralne značilke (brez pooling — ohrani resolucijo)
            nn.Conv1d(128,  256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)  # → (batch, 256)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.log_softmax = nn.LogSoftmax(dim=1)

        # Inicializacija uteži (Kaiming za ReLU)
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
        x = x.unsqueeze(1)             # (batch, input_len) → (batch, 1, input_len)
        x = self.conv_blocks(x)        # (batch, 256, L)
        x = self.pool(x).squeeze(-1)   # (batch, 256)
        x = self.classifier(x)         # (batch, num_classes)
        return self.log_softmax(x)     # log-verjetnosti


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
                 device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        log_probs = model(X_batch)
        loss = criterion(log_probs, y_batch)
        loss.backward()
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
    """Vrne verjetnosti oblike (N, NUM_CLASSES) kot numpy array."""
    model.eval()
    all_probs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i + batch_size].to(device)
        log_probs = model(batch)
        all_probs.append(torch.exp(log_probs).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Fitanje modela B
# ---------------------------------------------------------------------------
def fit_model_b(
    X_spec: np.ndarray,
    y: np.ndarray,
    amide_i_idx: int,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 15,
    seed: int = 42,
    label: str = "",
    val_spec: np.ndarray = None,
    val_y: np.ndarray = None,
) -> tuple:
    """
    Trenira SpectralCNN1D.

    Parametri
    ---------
    val_spec / val_y : če sta podana, se izvaja early stopping na val log loss.
                       Za finalni model (brez val splita) ju pusti None.

    Vrne
    ----
    (model, device)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()

    # 1. Preprocessing
    X_pp = preprocess(X_spec, amide_i_idx, label=label)

    # 2. Oversampling za balans razredov
    X_bal, y_bal = oversample_to_max_class(X_pp, y, seed=seed)

    # 3. DataLoader
    train_loader = make_dataloader(X_bal, y_bal, batch_size=batch_size, shuffle=True)

    # 4. Val preprocessing (brez oversamplinga — realne porazdelitve)
    X_val_pp = None
    if val_spec is not None:
        X_val_pp = preprocess(val_spec, amide_i_idx, label=f"{label}-val")

    # 5. Model
    input_len = X_pp.shape[1]
    model = SpectralCNN1D(input_len=input_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SpectralCNN1D: {n_params:,} parametrov | input={input_len}")

    # 6. Optimizer + scheduler
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if val_spec is not None:
        # Scheduler na val loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
    else:
        # Scheduler na train loss (finalni model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
        )

    # 7. Trening
    best_val_loss     = float('inf')
    best_state        = None
    patience_counter  = 0

    has_val = (val_spec is not None)

    if has_val:
        print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'Val OA':>9}  {'Val ll':>9}  {'LR':>9}")
        print(f"  {'─'*6}  {'─'*11}  {'─'*9}  {'─'*9}  {'─'*9}")
    else:
        print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'LR':>9}")
        print(f"  {'─'*6}  {'─'*11}  {'─'*9}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        if has_val:
            probs  = predict_proba_cnn(model, X_val_pp, device)
            preds  = np.argmax(probs, axis=1)
            val_oa = accuracy_score(val_y, preds)
            val_ll = log_loss(val_y, probs, labels=np.arange(NUM_CLASSES))

            scheduler.step(val_ll)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"  {epoch:>6}  {train_loss:>11.5f}  {val_oa*100:>8.2f}%  {val_ll:>9.5f}  {current_lr:>9.2e}")

            # Early stopping
            if val_ll < best_val_loss - 1e-6:
                best_val_loss   = val_ll
                best_state      = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping pri epohi {epoch}  "
                          f"(najboljši val loss={best_val_loss:.5f}, patience={patience})")
                    break
        else:
            scheduler.step(train_loss)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>6}  {train_loss:>11.5f}  {current_lr:>9.2e}")

    elapsed = time.time() - t_start
    print(f"\n  Trening zaključen v {elapsed:.1f}s")

    # Naloži najboljše uteži (samo če je val na voljo)
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
    """Evaluacija na poljubnem splitu. Vrne (OA, log_loss)."""
    X_pp  = preprocess(X_spec, amide_i_idx, label=split_name)
    probs = predict_proba_cnn(model, X_pp, device)
    preds = np.argmax(probs, axis=1)

    oa = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))

    print(f"\n  ── {split_name} ──")
    print(f"  OA:       {oa*100:.2f}%   (Model A ref: VAL 77.14%, TEST 92.69%)")
    print(f"  Log loss: {ll:.5f}       (Model A ref: VAL 0.607,  TEST 0.347)")

    # Per-class accuracy
    print(f"\n  Natancnost po razredih:")
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"    Razred {c}: N/A")
        else:
            acc_c = (preds[mask] == y_true[mask]).mean()
            print(f"    Razred {c}: {acc_c*100:.2f}%  (n={mask.sum()})")

    # Macro accuracy
    per_class = []
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() > 0:
            per_class.append((preds[mask] == y_true[mask]).mean())
    macro_acc = np.mean(per_class)
    print(f"\n  Macro OA: {macro_acc*100:.2f}%  (neodvisno od razrednih velikosti)")

    # Matrika zmede
    cm = confusion_matrix(y_true, preds, labels=np.arange(NUM_CLASSES))
    print(f"\n  Matrika zmede:")
    print(cm)

    return oa, ll


# ---------------------------------------------------------------------------
# Submission (identična logika kot Model A v3)
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
    """
    Submission z ločevanjem tkivo / ozadje.
    - Tkivni piksli: napoved z modelom.
    - Ozadje: class prior iz train seta.
    """
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
        description="Model B v1 — Spektralni 1D CNN + prostorski 3-way split"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output",             default="modelB_v1.npy")
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--batch-size",         type=int,   default=512)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--patience",           type=int,   default=15)
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
    print("\n=== 3. Ucenje Model B na TRAIN splitu (val za early stopping) ===")
    model, device = fit_model_b(
        X_spec      = X_train,
        y           = y_train,
        amide_i_idx = amide_i_idx,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        patience    = args.patience,
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
    print("\n=== 5. Finalni model (train + val + test = vse anotacije) ===")
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    X_all = data[usable_mask]
    y_all = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov za finalni model: {len(y_all)}")
    print(f"  (Brez val early stopping — scheduler na train loss)")

    model_final, _ = fit_model_b(
        X_spec      = X_all,
        y           = y_all,
        amide_i_idx = amide_i_idx,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        patience    = args.patience,  # ni v uporabi brez val
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
    print(f"  Model B (1D CNN) — spektralni, brez prostornega konteksta")
    print(f"  VAL  — OA: {oa_val*100:.2f}%  |  log loss: {ll_val:.5f}")
    print(f"  TEST — OA: {oa_test*100:.2f}%  |  log loss: {ll_test:.5f}  <- koncna stevilka za diplomo")
    print(f"  Primerjava Model A: VAL OA=77.14%/ll=0.607 | TEST OA=92.69%/ll=0.347")
    print(f"  Submission: {args.output}")


if __name__ == "__main__":
    main()
