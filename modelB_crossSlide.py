"""
Model B — Cross-Slide Spektralni 1D CNN  (clanek-zvest CNN brez PCA/prostorske info)
=====================================================================================
Namen: tretji clen trojice iz clanka Table 8 (SVM / CNN-spektralno / CNN-prostorsko)
na PRAVEM cross-slide splitu -- doslej je modelB (modelB_v3.py) tekel samo na
STAREM Core2 leave-one-out scenariju, ne na pravem cross-slide splitu.

Table 8 (clanek, SD stolpec): SVM=56.41%, CNN(spectral)=62.52%, CNN(spatial)=79.45%.

POMEMBNO: FTIR-data/*.hdf5 so ZE predprocesirani (rubber-band baseline +
Amide I normalizacija). Ta skripta NE ponavlja preprocesiranja -- surovi
(ze normalizirani) spektri gredo direktno v 1D CNN, BREZ PCA (za razliko od
modelA_crossSlide.py in modelC_crossSlide_faithful.py).

Arhitektura (SpectralCNN1D, identicna modelB_v3.py):
  Conv1d(1->32->64->128->256, kernel 7/5/3/3) + BatchNorm1d + ReLU + MaxPool,
  AdaptiveAvgPool1d, FC(256->128->64->6). Deluje na POLNEM spektru (813
  kanalov po SPECTRAL_STEP=2 downsamplingu), ne na PCA komponentah.

Gnezdena Faza A/B struktura (isto ogrodje kot modelA_crossSlide.py /
modelC_crossSlide_faithful.py):
  Faza A: trening na inner-train crop-ih, best-of-N izbira checkpointa na
          inner-val (po val log-lossu, brez early stopping) -> best_epoch,
          T/sigma kalibracija -- TEST se ne dotakne.
  Faza B: trening na VSEH train crop-ih (fiksen best_epoch, brez peeka),
          EN dotik s celotnim TEST setom (--test-dir, vsi crop-i).

Oversampling do velikosti najvecjega razreda (isto kot modelB_v3.py) --
razlika od modelA_crossSlide.py, ki uporablja fiksen target=10,000 (clanek
SVM specifikacija); za spektralni CNN clanek ne navaja natancnega stevila,
zato je obdrzana ze uveljavljena modelB_v3 konvencija.
"""

import argparse
import glob
import os
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES  = 6
RESULTS_FILE = "rezultati_report.txt"
BBOX_MARGIN  = 12
SIGMA_CHOICES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_SOFTEN = 0.5

REF_SVM_SD_OA      = 56.41
REF_CNN_SPEC_SD_OA = 62.52
REF_CNN_SD_OA      = 79.45

METODOLOGIJA_OPOMBA = (
    "modelB_crossSlide: clanek-zvest spektralni 1D CNN (Conv1d 1->32->64->128->256, "
    "BREZ PCA, BREZ prostorske informacije -- deluje samo na spektru vsakega "
    "piksla) na PRAVEM cross-slide train/test TMA splitu (isti kot modelA/"
    "modelC_crossSlide_faithful). Podatki so ZE predprocesirani (rubber-band+"
    "Amide I) -- brez ponovnega preprocesiranja. Privzeto balance-strategy=weights "
    "(softened inverse-frequency, isto kot modelC -- boljse od trdega oversamplinga), "
    "per-class temperature (sirsa meja 0.1-50, isto popravilo kot modelC), opcijski "
    "label smoothing. Ref. clanek Table 8 (SD): SVM=56.41%, "
    "CNN-spektralno=62.52%, CNN-prostorsko=79.45%."
)


# ---------------------------------------------------------------------------
# Nalaganje — SAMO bounding box okoli anotiranih pikslov (RAM-varcno)
# ---------------------------------------------------------------------------
def get_annotated_bbox(classes, margin=BBOX_MARGIN):
    H, W = classes.shape
    coords = np.argwhere(classes != -1)
    r0 = max(0, int(coords[:, 0].min()) - margin)
    r1 = min(H, int(coords[:, 0].max()) + margin + 1)
    c0 = max(0, int(coords[:, 1].min()) - margin)
    c1 = min(W, int(coords[:, 1].max()) + margin + 1)
    return r0, r1, c0, c1


def peek_classes(path):
    with h5py.File(path, 'r') as f:
        return np.array(f['classes'])


def load_annotated_spectra(path, margin=BBOX_MARGIN, label=""):
    t0 = time.time()
    with h5py.File(path, 'r') as f:
        classes_full = np.array(f['classes'])
        r0, r1, c0, c1 = get_annotated_bbox(classes_full, margin)
        classes_bbox = classes_full[r0:r1, c0:c1]
        ann = classes_bbox != -1
        data_bbox = np.array(f['data'][r0:r1, c0:c1, :], dtype=np.float32)
    X = data_bbox[ann]
    y = classes_bbox[ann].astype(np.int64)
    coords = np.argwhere(ann)
    n_bad = int((~np.isfinite(X)).any(axis=1).sum())
    if n_bad:
        print(f"  [{label}] OPOZORILO: {n_bad} anotiranih pikslov z NaN/Inf -- "
              f"sanitiziram na 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [{label}] {os.path.basename(path)}: {len(y):,} anotiranih  "
          f"({time.time()-t0:.1f}s)")
    return X, y, coords, classes_bbox.shape, (r0, c0)


def select_inner_val_candidates(crop_paths, verbose=True):
    info = []
    if verbose:
        print("  Pregled train crop-ov (samo 'classes', poceni branje):")
    for p in crop_paths:
        classes = peek_classes(p)
        ann = (classes != -1)
        n = int(ann.sum())
        if n == 0:
            continue
        vals, counts = np.unique(classes[ann], return_counts=True)
        has_all = len(vals) == NUM_CLASSES
        min_count = int(counts.min()) if has_all else 0
        info.append({"path": p, "n": n, "has_all": has_all, "min_count": min_count})
        marker = "*" if has_all else " "
        if verbose:
            mc_str = str(min_count) if has_all else "-"
            print(f"    {marker} {os.path.basename(p)}: {n:,} anotiranih, "
                  f"min_razred={mc_str}")

    with_all = [x for x in info if x["has_all"]]
    if with_all:
        with_all.sort(key=lambda x: x["min_count"], reverse=True)
        candidates = [x["path"] for x in with_all]
    else:
        info_sorted = sorted(info, key=lambda x: x["n"], reverse=True)
        candidates = [info_sorted[0]["path"]]
        if verbose:
            print("  OPOZORILO: noben crop nima vseh 6 razredov. Vzet najvecji.")
    if verbose:
        print(f"\n  Kandidati za inner-val (najboljsi prvi):")
        for i, p in enumerate(candidates):
            print(f"    {i+1}. {os.path.basename(p)}")
    return candidates


# ---------------------------------------------------------------------------
# Oversampling do velikosti najvecjega razreda (isto kot modelB_v3.py)
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


def compute_class_weights(y, soften=WEIGHT_SOFTEN):
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    raw     = len(y) / (NUM_CLASSES * np.where(counts > 0, counts, 1))
    weights = raw ** soften
    print(f"  Class weights (soften={soften}): {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


class SmoothedNLLLoss(nn.Module):
    """Label smoothing za NLLLoss (vhod = log-probs iz LogSoftmax) -- cilja
    pretirano samozavestne napovedi (visok log-loss), isto kot pri modelC."""
    def __init__(self, weight=None, smoothing=0.1, n_classes=NUM_CLASSES):
        super().__init__()
        self.weight = weight
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, log_probs, target):
        nll = F.nll_loss(log_probs, target, weight=self.weight, reduction='none')
        smooth = -log_probs.mean(dim=1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()


def build_criterion(strategy, weights, device, label_smoothing=0.0):
    w = None if strategy == "oversample" else weights.to(device)
    if label_smoothing > 0:
        return SmoothedNLLLoss(weight=w, smoothing=label_smoothing)
    if strategy == "oversample":
        return nn.NLLLoss()
    else:
        return nn.NLLLoss(weight=w)


# ---------------------------------------------------------------------------
# Arhitektura (identicna modelB_v3.py)
# ---------------------------------------------------------------------------
class SpectralCNN1D(nn.Module):
    def __init__(self, input_len, num_classes=NUM_CLASSES, dropout=0.3):
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
    if torch.cuda.is_available():
        d = torch.device("cuda"); print(f"  Naprava: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


def make_dataloader(X, y, batch_size=512, shuffle=True, num_workers=0):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


@torch.no_grad()
def predict_proba(model, X, device, batch_size=1024):
    model.eval()
    all_probs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i+batch_size].to(device)
        all_probs.append(torch.exp(model(batch)).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def train_single(X_train, y_train, X_val, y_val, device, input_len,
                 max_epochs, batch_size, lr, seed, criterion, num_workers=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = make_dataloader(X_train, y_train, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    model     = SpectralCNN1D(input_len=input_len).to(device)
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


def train_blind(X_train, y_train, device, input_len, final_epochs, batch_size,
                lr, seed, criterion, num_workers=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = make_dataloader(X_train, y_train, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    model     = SpectralCNN1D(input_len=input_len).to(device)
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
# Temperature scaling + sigma smoothing (isto kot modelA_crossSlide.py)
# ---------------------------------------------------------------------------
def find_temperature_per_class(probs, y, n_classes=NUM_CLASSES):
    """Per-class temperature s SIRSO mejo (0.1-50, prej 10) -- modelC je
    pokazal, da R2/R3 pristanejo tocno na stari meji (10.0), kar pomeni da
    model 'hoce' se vec mehcanja, kot mu je bilo dovoljeno."""
    eps = 1e-9
    log_probs = np.log(np.clip(probs, eps, 1.0))

    def neg_ll(T_vec):
        T_vec = np.clip(T_vec, 0.05, 60.0)
        s = log_probs / T_vec[None, :]
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), eps, 1.0)
        return -np.mean(np.log(p[np.arange(len(y)), y]))

    x0 = np.ones(n_classes)
    bounds = [(0.1, 50.0)] * n_classes
    res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
    T_vec = res.x
    print(f"  Temperature (per-class): " +
          ", ".join(f"R{c}={t:.2f}" for c, t in enumerate(T_vec)))
    print(f"  ll: {neg_ll(np.ones(n_classes)):.5f} -> {neg_ll(T_vec):.5f}")
    return T_vec


def apply_temperature_per_class(probs, T_vec):
    eps = 1e-9
    log_probs = np.log(np.clip(probs, eps, 1.0))
    s = log_probs / T_vec[None, :]
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def gaussian_smooth_probs(probs, sigma):
    if sigma <= 0: return probs
    smoothed = np.zeros_like(probs)
    for c in range(probs.shape[-1]):
        smoothed[:, :, c] = gaussian_filter(probs[:, :, c], sigma=sigma)
    smoothed = np.clip(smoothed, 1e-7, 1.0)
    smoothed /= smoothed.sum(axis=-1, keepdims=True)
    return smoothed.astype(np.float32)


def build_full_canvas_prob_map(shape, coords, probs, fallback_prior):
    H, W = shape
    prob_map = np.tile(fallback_prior, (H * W, 1)).reshape(H, W, NUM_CLASSES)
    prob_map[coords[:, 0], coords[:, 1]] = probs
    return prob_map


def find_best_sigma(shape, coords, probs, y_true, fallback_prior,
                    sigma_choices=SIGMA_CHOICES):
    prob_map = build_full_canvas_prob_map(shape, coords, probs, fallback_prior)
    print(f"  {'sigma':>6}  {'log-loss':>10}")
    best_sigma, best_ll = 0.0, None
    for sigma in sigma_choices:
        m = gaussian_smooth_probs(prob_map, sigma) if sigma > 0 else prob_map
        probs_at_coords = m[coords[:, 0], coords[:, 1]]
        probs_at_coords = np.clip(probs_at_coords, 1e-7, 1.0)
        probs_at_coords /= probs_at_coords.sum(axis=1, keepdims=True)
        ll = log_loss(y_true, probs_at_coords, labels=np.arange(NUM_CLASSES))
        marker = ""
        if best_ll is None or ll < best_ll:
            best_ll = ll; best_sigma = sigma; marker = " *"
        print(f"  {sigma:>6.1f}  {ll:>10.5f}{marker}")
    print(f"  Najboljsa sigma: {best_sigma:.1f} (ll={best_ll:.5f})")
    return best_sigma


def print_per_class_table(y_true, y_pred, probs, title=""):
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


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h {m}min {s}s"
    if m: return f"{m}min {s}s"
    return f"{s}s"


def format_T(t_opt):
    if np.isscalar(t_opt):
        return f"{float(t_opt):.4f}"
    return "[" + ",".join(f"{t:.3f}" for t in t_opt) + "]"


def write_results_report(model_name, innerval_oa, innerval_ll, test_oa, test_ll,
                          output_path, t_opt, sigma, extra_note="", total_duration_str=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dur_part = f"  cas={total_duration_str}" if total_duration_str else ""
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"TEST_OA={test_oa*100:6.2f}%  TEST_ll={test_ll:.5f}  "
        f"T={format_T(t_opt)}  sigma={sigma:.1f}{dur_part}  -> {output_path}\n"
    ]
    if extra_note:
        lines.append(f"{'':>19}  Opomba: {extra_note}\n")
    for attempt in range(3):
        try:
            if not os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, "w") as f:
                    f.write("# Rezultati modelov — FTIR klasifikacija tkiva\n")
                    f.write(f"# {'-'*90}\n")
            with open(RESULTS_FILE, "a") as f:
                f.writelines(lines)
            print(f"  -> {RESULTS_FILE} (dodana vrstica)")
            return
        except OSError as e:
            print(f"  OPOZORILO: pisanje ni uspelo (poskus {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2)
    print("  OPOZORILO: pisanje v log ni uspelo po 3 poskusih. Vsebina:")
    print("  " + "".join(lines).replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    run_start = time.time()
    parser = argparse.ArgumentParser(
        description="Model B Cross-Slide: clanek-zvest spektralni 1D CNN "
                    "(brez PCA) na pravem cross-slide train/test TMA splitu"
    )
    parser.add_argument("--train-dir", default="FTIR-data/train_preprocessed_no24")
    parser.add_argument("--test-dir",  default="FTIR-data/test_preprocessed_full")
    parser.add_argument("--output",    default="modelB_crossSlide_test.npy")
    parser.add_argument("--max-epochs", type=int, default=40,
                        help="Zgornja meja epoh za Fazo A (best-of-N na inner-val).")
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--bbox-margin", type=int, default=BBOX_MARGIN)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-strategy", choices=["weights", "oversample"],
                        default="weights",
                        help="weights = softened inverse-frequency class weights "
                             "(privzeto, boljse od trdega oversamplinga glede na "
                             "modelC ablacije). oversample = stara modelB_v3 "
                             "konvencija (do velikosti najvecjega razreda).")
    parser.add_argument("--weight-soften", type=float, default=WEIGHT_SOFTEN)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing (0=izklopljeno) -- regularizacija "
                             "med treningom proti pretirano samozavestnim "
                             "napovedim (visok log-loss).")
    args = parser.parse_args()

    print("\n=== 1. Odkrivanje train crop-ov in izbira inner-val kandidata ===")
    train_crop_paths = sorted(glob.glob(os.path.join(args.train_dir, "train_crop_*.hdf5")))
    print(f"  Najdenih train crop-ov: {len(train_crop_paths)}")
    for p in train_crop_paths:
        print(f"    {p}")
    candidates = select_inner_val_candidates(train_crop_paths)
    inner_val_path = candidates[0]
    inner_train_paths = [p for p in train_crop_paths if p != inner_val_path]
    print(f"\n  Inner-val: {os.path.basename(inner_val_path)}  "
          f"({len(inner_train_paths)} inner-train crop-ov)")
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    device = get_device()

    # ==================================================================
    # FAZA A — trening na inner-train, best-of-N na inner-val (TEST se ne dotakne)
    # ==================================================================
    print(f"\n=== 2. Faza A — nalaganje inner-train ({len(inner_train_paths)} crop-ov) ===")
    X_it_parts, y_it_parts = [], []
    for p in inner_train_paths:
        X, y, _, _, _ = load_annotated_spectra(p, args.bbox_margin, label="it")
        X_it_parts.append(X); y_it_parts.append(y)
    X_it = np.concatenate(X_it_parts, axis=0)
    y_it = np.concatenate(y_it_parts, axis=0)
    del X_it_parts, y_it_parts
    input_len = X_it.shape[1]
    print(f"  Skupaj inner-train: {len(y_it):,} pikslov  |  spektralnih kanalov: {input_len}")

    n_param = sum(p.numel() for p in SpectralCNN1D(input_len=input_len).parameters()
                 if p.requires_grad)
    print(f"  SpectralCNN1D: {n_param:,} parametrov")

    print(f"\n=== 3. Faza A — nalaganje inner-val ({os.path.basename(inner_val_path)}) ===")
    X_iv, y_iv, coords_iv, shape_iv, _ = load_annotated_spectra(
        inner_val_path, args.bbox_margin, label="iv")

    print(f"\n=== 4. Faza A — uravnotezenje razredov (strategy={args.balance_strategy}) ===")
    weights_A = compute_class_weights(y_it, soften=args.weight_soften)
    criterion_A = build_criterion(args.balance_strategy, weights_A, device,
                                  label_smoothing=args.label_smoothing)
    if args.balance_strategy == "oversample":
        X_it_os, y_it_os = oversample_to_max_class(X_it, y_it, seed=args.seed, label="Faza A")
    else:
        X_it_os, y_it_os = X_it, y_it
    del X_it, y_it

    print(f"\n=== 5. Faza A — trening (do {args.max_epochs} epoh, best-of-N na inner-val) ===")
    model_A, best_epoch = train_single(
        X_it_os, y_it_os, X_iv, y_iv, device, input_len,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, criterion=criterion_A, num_workers=args.num_workers,
    )
    del X_it_os, y_it_os

    print(f"\n=== 6. Faza A — napoved na inner-val, kalibracija ===")
    probs_iv = predict_proba(model_A, X_iv, device)
    T_opt = find_temperature_per_class(probs_iv, y_iv)
    probs_iv_cal = apply_temperature_per_class(probs_iv, T_opt)
    pred_iv = np.argmax(probs_iv_cal, axis=1)
    innerval_oa = accuracy_score(y_iv, pred_iv)
    innerval_ll = log_loss(y_iv, probs_iv_cal, labels=np.arange(NUM_CLASSES))
    print(f"  INNERVAL OA: {innerval_oa*100:.2f}%  |  ll: {innerval_ll:.5f}")
    print_per_class_table(y_iv, pred_iv, probs_iv_cal, "Per-class (inner-val):")

    print(f"\n=== 7. Sigma sweep (inner-val) ===")
    prior_it = np.bincount(y_iv, minlength=NUM_CLASSES).astype(np.float32)
    prior_it /= prior_it.sum()
    sigma_opt = find_best_sigma(shape_iv, coords_iv, probs_iv_cal, y_iv, prior_it)
    del model_A, X_iv, probs_iv, probs_iv_cal

    final_epochs = best_epoch

    # ==================================================================
    # FAZA B — trening na VSEH train crop-ih, EN dotik s TEST-om
    # ==================================================================
    print(f"\n=== 8. Faza B — nalaganje vseh {len(train_crop_paths)} train crop-ov ===")
    X_ot_parts, y_ot_parts = [], []
    for p in train_crop_paths:
        X, y, _, _, _ = load_annotated_spectra(p, args.bbox_margin, label="ot")
        X_ot_parts.append(X); y_ot_parts.append(y)
    X_ot = np.concatenate(X_ot_parts, axis=0)
    y_ot = np.concatenate(y_ot_parts, axis=0)
    del X_ot_parts, y_ot_parts
    print(f"  Skupaj train (Faza B): {len(y_ot):,} pikslov")

    print(f"\n=== 9. Faza B — uravnotezenje razredov (strategy={args.balance_strategy}) ===")
    weights_B = compute_class_weights(y_ot, soften=args.weight_soften)
    criterion_B = build_criterion(args.balance_strategy, weights_B, device,
                                  label_smoothing=args.label_smoothing)
    prior_B = np.bincount(y_ot, minlength=NUM_CLASSES).astype(np.float32)
    prior_B /= prior_B.sum()
    if args.balance_strategy == "oversample":
        X_ot_os, y_ot_os = oversample_to_max_class(X_ot, y_ot, seed=args.seed, label="Faza B")
    else:
        X_ot_os, y_ot_os = X_ot, y_ot
    del X_ot

    print(f"\n=== 10. Faza B — trening ({final_epochs} epoh, brez peeka) ===")
    model_B = train_blind(X_ot_os, y_ot_os, device, input_len,
                          final_epochs=final_epochs, batch_size=args.batch_size,
                          lr=args.lr, seed=args.seed, criterion=criterion_B,
                          num_workers=args.num_workers)
    del X_ot_os, y_ot_os

    print(f"\n=== 11. Faza B — nalaganje TEST podatkov ===")
    test_paths = sorted(glob.glob(os.path.join(args.test_dir, "test_crop_*.hdf5")))
    print(f"  Najdenih test crop-ov: {len(test_paths)}")
    test_results = []
    for i, p in enumerate(test_paths):
        X, y, coords, shape, offset = load_annotated_spectra(p, args.bbox_margin, label=f"TEST-{i}")
        test_results.append((X, y, coords, shape, offset))
    y_test_all = np.concatenate([r[1] for r in test_results])
    print(f"  Skupaj TEST: {len(y_test_all):,} pikslov iz {len(test_paths)} crop-ov")

    print(f"\n=== 12. KONCNA evaluacija na TEST (edini dotik, pravi cross-slide) ===")
    X_test_all = np.concatenate([r[0] for r in test_results])
    test_probs_all = predict_proba(model_B, X_test_all, device)
    test_probs_all_cal = apply_temperature_per_class(test_probs_all, T_opt)
    test_pred_all = np.argmax(test_probs_all_cal, axis=1)
    test_oa = accuracy_score(y_test_all, test_pred_all)
    test_ll = log_loss(y_test_all, test_probs_all_cal, labels=np.arange(NUM_CLASSES))
    print(f"  TEST OA (pred smoothing): {test_oa*100:.2f}%")
    print(f"  TEST ll (pred smoothing): {test_ll:.5f}")
    print(f"  Ref clanek SVM (SD): OA={REF_SVM_SD_OA:.2f}%")
    print(f"  Ref clanek CNN-spektralno (SD): OA={REF_CNN_SPEC_SD_OA:.2f}%")
    print(f"  Ref clanek CNN-prostorsko (SD): OA={REF_CNN_SD_OA:.2f}% +/- 1.25")
    print_per_class_table(y_test_all, test_pred_all, test_probs_all_cal,
                          "Per-class (TEST — koncni test):")

    # Per-crop smoothing + pooled evaluation
    offset_probs = 0
    smoothed_parts, y_parts = [], []
    for X, y, coords, shape, off in test_results:
        n = len(y)
        probs_c = test_probs_all_cal[offset_probs:offset_probs + n]
        prob_map = build_full_canvas_prob_map(shape, coords, probs_c, prior_B)
        final_map = gaussian_smooth_probs(prob_map, sigma_opt) if sigma_opt > 0 else prob_map
        final_map = np.clip(final_map, 1e-7, 1.0)
        final_map /= final_map.sum(axis=-1, keepdims=True)
        at_coords = final_map[coords[:, 0], coords[:, 1]]
        at_coords = np.clip(at_coords, 1e-7, 1.0)
        at_coords /= at_coords.sum(axis=1, keepdims=True)
        smoothed_parts.append(at_coords)
        y_parts.append(y)
        offset_probs += n

    smoothed_at_coords = np.concatenate(smoothed_parts, axis=0)
    y_test_sm = np.concatenate(y_parts, axis=0)
    test_pred_sm = np.argmax(smoothed_at_coords, axis=1)
    test_oa_sm = accuracy_score(y_test_sm, test_pred_sm)
    test_ll_sm = log_loss(y_test_sm, smoothed_at_coords, labels=np.arange(NUM_CLASSES))
    print(f"\n  TEST OA (po smoothing, sigma={sigma_opt:.1f}, zdruzeno cez "
          f"{len(test_paths)} crop-ov): {test_oa_sm*100:.2f}%")
    print(f"  TEST ll (po smoothing, sigma={sigma_opt:.1f}, zdruzeno): {test_ll_sm:.5f}")

    np.save(args.output, smoothed_at_coords.astype(np.float32))
    print(f"\n  Shranjeno: {args.output}")

    print("\n=== POVZETEK (modelB crossSlide spektralni CNN) ===")
    print(f"  SKUPAJ CAS TEKA: {format_duration(time.time() - run_start)}")
    print(f"  Train: {len(train_crop_paths)} crop-ov, {len(y_ot):,} pikslov")
    print(f"  Test:  {len(test_paths)} crop-ov, {len(y_test_all):,} pikslov")
    print(f"  Faza A: max_epochs={args.max_epochs}, best_epoch={best_epoch}")
    print(f"  Inner-val (diagnostika): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (TEST, KONCNI):   OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")
    print(f"\n  Primerjava (Table 8, SD):")
    print(f"    Clanek SVM:            OA={REF_SVM_SD_OA:.2f}%")
    print(f"    Clanek CNN-spektralno: OA={REF_CNN_SPEC_SD_OA:.2f}%")
    print(f"    Clanek CNN-prostorsko: OA={REF_CNN_SD_OA:.2f}% +/- 1.25")
    print(f"    modelB crossSlide (ta tek): OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")

    print(f"\n=== 13. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelB_crossSlide",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa_sm, test_ll=test_ll_sm,
        output_path=args.output, t_opt=T_opt, sigma=sigma_opt,
        total_duration_str=format_duration(time.time() - run_start),
        extra_note=(METODOLOGIJA_OPOMBA +
                   f" [strategy={args.balance_strategy}, "
                   f"weight_soften={args.weight_soften}, "
                   f"label_smoothing={args.label_smoothing}]"),
    )


if __name__ == "__main__":
    main()
