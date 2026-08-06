"""
Model B — Spektralni MLP  (v4: poštena primerjava z LR)
=========================================================
Problem z v3: MLP je tekmoval pod neenakimi pogoji kot LR.

LR (predict_lr.ipynb):
  - Podatki: VSI anotirani piksli (classes != -1), brez prostornega splita
  - Preprocessing: StandardScaler na surovih spektrih
  - Regularizacija: C=0.001 (enakovredno weight_decay ~ 500)
  - Distribucija: naravna (brez oversamplinga ali tehtane loss)
  - Submission: vsi piksli v cropu (brez tkivo/ozadje ločevanja)

Naš MLP v3:
  - Podatki: samo 60% anotiranih (train split)         ← slabše
  - Preprocessing: rubberband + Amide I                 ← drugačno
  - Regularizacija: weight_decay=1e-4                   ← šibkejša
  - Distribucija: tehtana loss (različna od LR)         ← drugačno

Ta verzija izenači pogoje:
  ✓ Isti podatki: vsi anotirani piksli za treniranje
  ✓ Isti preprocessing: StandardScaler na surovih spektrih
  ✓ Enakovredna regularizacija: weight_decay=0.1 (≈ L2 norm C=0.01)
  ✓ Naravna porazdelitev razredov (brez oversamplinga)
  ✓ Isti submission pristop: vsi piksli v cropu

MLP ima prednost pred LR: nelinearne aktivacije → kompleksnejše meje odločitev.
Z enakimi pogoji mora MLP preseči LR.

Za evaluacijo: 80/20 stratificiran random split (ne prostorski).
"""

import argparse
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES  = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660


# ---------------------------------------------------------------------------
# Nalaganje podatkov
# ---------------------------------------------------------------------------
def load_data(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        data        = np.array(f["data"],        dtype=np.float32)
        tissue_mask = np.array(f["tissue_mask"])
        classes     = np.array(f["classes"])
    return data, tissue_mask, classes


# ---------------------------------------------------------------------------
# Preprocessing — enak kot LR: StandardScaler na surovih spektrih
# ---------------------------------------------------------------------------
def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"  StandardScaler fit: mean=[{X_train.mean():.4f}], std=[{X_train.std():.4f}]")
    return scaler


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SpectralMLP(nn.Module):
    """
    MLP z enako strukturo kot v3, prilagojen za StandardScaler vhod.

    Arhitektura:
      Linear(187 → 512) + BN + ReLU + Dropout(0.4)
      Linear(512 → 256) + BN + ReLU + Dropout(0.3)
      Linear(256 → 128) + BN + ReLU + Dropout(0.2)
      Linear(128 → 6)   [logiti]
      LogSoftmax

    Večja arhitektura kot v3 ker imamo več podatkov (100% vs 60%).
    Močna regularizacija (weight_decay=0.1) kompenzira večje število parametrov.
    """

    def __init__(self, input_dim: int = 187, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.4),

            nn.Linear(512, 256),       nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(256, 128),       nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(128, num_classes)
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
        return self.head(self.body(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_softmax(self.get_logits(x))


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
# Temperature scaling
# ---------------------------------------------------------------------------
def find_temperature(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray,
                     device: torch.device) -> float:
    model.eval()
    logits_list = []
    X_t = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), 2048):
            logits_list.append(model.get_logits(X_t[i:i+2048].to(device)).cpu().numpy())
    logits = np.concatenate(logits_list)

    def neg_ll(T: float) -> float:
        s = logits / T
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-9, 1.0)
        return -np.mean(np.log(p[np.arange(len(y_val)), y_val]))

    res = minimize_scalar(neg_ll, bounds=(0.05, 10.0), method='bounded')
    T = res.x
    print(f"  Temperature scaling: T={T:.4f}  "
          f"log_loss: {neg_ll(1.0):.5f} → {neg_ll(T):.5f}")
    return T


# ---------------------------------------------------------------------------
# Napoved
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device,
                  temperature: float = 1.0, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    all_probs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X_t), batch_size):
        b = X_t[i:i+batch_size].to(device)
        if temperature == 1.0:
            p = torch.exp(model(b))
        else:
            p = torch.softmax(model.get_logits(b) / temperature, dim=1)
        all_probs.append(p.cpu().numpy())
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Trening
# ---------------------------------------------------------------------------
def train_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    device: torch.device,
    epochs: int = 300,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.1,   # močna regularizacija (≈ LR C=0.01)
    patience: int = 25,
    seed: int = 42,
) -> tuple:
    torch.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )

    model    = SpectralMLP(input_dim=X_train.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SpectralMLP: {n_params:,} parametrov")
    print(f"  weight_decay={weight_decay}  lr={lr}  batch={batch_size}")

    criterion = nn.NLLLoss()   # brez uteži — naravna porazdelitev kot LR
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        # --- trening ---
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
        train_loss = total_loss / len(train_loader.dataset)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        # --- validacija ---
        probs  = predict_proba(model, X_val, device)
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
                      f"(best={best_val_loss:.5f} pri epohi {best_epoch})")
                break

    print(f"\n  Trening zaključen v {time.time()-t0:.1f}s")
    model.load_state_dict(best_state)
    print(f"  Najboljše uteži: epoha {best_epoch}, val loss={best_val_loss:.5f}")
    return model, best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model B v4 — MLP, poštena primerjava z LR"
    )
    parser.add_argument("--input",        default="image1-competition.hdf5")
    parser.add_argument("--output",       default="modelB_mlp_v2.npy")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--batch-size",   type=int,   default=1024)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--patience",     type=int,   default=25)
    parser.add_argument("--val-frac",     type=float, default=0.20)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, tissue_mask, classes = load_data(args.input)
    print(f"  data: {data.shape} | Anotiranih: {(classes != -1).sum()}")

    # Enako kot LR: vsi anotirani piksli (brez prostornega splita)
    ann_mask = (classes != -1)
    X_all = data[ann_mask].astype(np.float32)
    y_all = classes[ann_mask].astype(np.int64)
    print(f"  Skupaj anotiranih: {len(y_all)}")
    for c in range(NUM_CLASSES):
        print(f"    Razred {c}: {(y_all == c).sum()} ({100*(y_all==c).mean():.1f}%)")

    # ------------------------------------------------------------------
    print("\n=== 2. Preprocessing: StandardScaler (enako kot LR) ===")
    # Stratificiran train/val split (ne prostorski — za pošteno primerjavo z LR)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_all, y_all,
        test_size=args.val_frac, random_state=args.seed, stratify=y_all
    )
    print(f"  Train: {len(y_tr)} | Val: {len(y_v)}")

    scaler = fit_scaler(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_v_s  = scaler.transform(X_v).astype(np.float32)

    # ------------------------------------------------------------------
    device = get_device()

    print("\n=== 3. Ucenje MLP ===")
    model, best_epoch = train_model(
        X_train      = X_tr_s,
        y_train      = y_tr,
        X_val        = X_v_s,
        y_val        = y_v,
        device       = device,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        patience     = args.patience,
        seed         = args.seed,
    )

    # ------------------------------------------------------------------
    print("\n=== 4. Temperature scaling ===")
    T_opt = find_temperature(model, X_v_s, y_v, device)

    # ------------------------------------------------------------------
    print("\n=== 5. Evaluacija na VAL ===")
    probs_v  = predict_proba(model, X_v_s, device, temperature=T_opt)
    preds_v  = np.argmax(probs_v, axis=1)
    oa_v     = accuracy_score(y_v, preds_v)
    ll_v     = log_loss(y_v, probs_v, labels=np.arange(NUM_CLASSES))

    print(f"  OA:       {oa_v*100:.2f}%")
    print(f"  Log loss: {ll_v:.5f}  [T={T_opt:.4f}]")
    print(f"  Ref LR:   OA=?, log_loss=? (LR ne prikazuje val metrik)")
    print(f"  Ref MLP v3 (VAL): OA=75.21%, log_loss=0.745")

    per_class = []
    for c in range(NUM_CLASSES):
        mask = (y_v == c)
        if mask.sum() > 0:
            acc_c = (preds_v[mask] == y_v[mask]).mean()
            per_class.append(acc_c)
            print(f"    Razred {c}: {acc_c*100:.2f}%  (n={mask.sum()})")
    print(f"  Macro OA: {np.mean(per_class)*100:.2f}%")

    # ------------------------------------------------------------------
    print(f"\n=== 6. Finalni model (vsi podatki, {best_epoch} epoh) ===")
    # Scaler fitted na celotnem datasetu (enako kot LR)
    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X_all).astype(np.float32)

    # Retrain za best_epoch epoh brez val
    torch.manual_seed(args.seed)
    model_final = SpectralMLP(input_dim=X_all_s.shape[1]).to(device)
    criterion   = nn.NLLLoss()
    optimizer_f = optim.Adam(model_final.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)
    scheduler_f = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_f, T_max=best_epoch, eta_min=1e-6
    )
    loader_f = DataLoader(
        TensorDataset(torch.tensor(X_all_s, dtype=torch.float32),
                      torch.tensor(y_all,   dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    print(f"\n  {'Epoha':>6}  {'Train loss':>11}  {'LR':>9}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*9}")

    t0 = time.time()
    for epoch in range(1, best_epoch + 1):
        model_final.train()
        total_loss = 0.0
        for Xb, yb in loader_f:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer_f.zero_grad()
            loss = criterion(model_final(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model_final.parameters(), 1.0)
            optimizer_f.step()
            total_loss += loss.item() * len(yb)
        scheduler_f.step()
        if epoch % 10 == 0 or epoch == 1 or epoch == best_epoch:
            print(f"  {epoch:>6}  {total_loss/len(y_all):>11.5f}  "
                  f"{optimizer_f.param_groups[0]['lr']:>9.2e}")

    print(f"  Finalni model treniran v {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print("\n=== 7. Submission (enako kot LR: vsi piksli v cropu) ===")
    crop_data    = data[PRED_R0:PRED_R1, PRED_C0:PRED_C1]
    crop_flat    = crop_data.reshape(-1, crop_data.shape[-1]).astype(np.float32)
    crop_flat_s  = scaler_final.transform(crop_flat).astype(np.float32)

    probs_crop   = predict_proba(model_final, crop_flat_s, device, temperature=T_opt)
    submission   = probs_crop.reshape(PRED_R1-PRED_R0, PRED_C1-PRED_C0, NUM_CLASSES)
    np.save(args.output, submission.astype(np.float32))
    print(f"  Submission shranjen: {args.output}  shape={submission.shape}")
    print(f"  Temperature: T={T_opt:.4f}")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  Model: MLP v4 (StandardScaler + weight_decay={args.weight_decay})")
    print(f"  Best epoch: {best_epoch}")
    print(f"  VAL — OA: {oa_v*100:.2f}%  |  log loss: {ll_v:.5f}")
    print(f"\n  Primerjava (LB score, nižje = boljše):")
    print(f"    LR (multilogreg):  0.637")
    print(f"    MLP v3:            0.812")
    print(f"    SVM+PCA (Model A): 0.905")
    print(f"\n  Submission: {args.output}")


if __name__ == "__main__":
    main()
