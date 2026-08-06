"""
Model A — Article-faithful spectral baseline  (v3: 3-way split)
===============================================================
Razlika glede na v2:
  - Uporablja prostorski TRAIN / VAL / TEST split (60 / 20 / 20)
    iz modula spatial_split.py.
  - VAL: za sledenje napredka med razvojem.
  - TEST: zaklenjen do konca — samo za končno porocanje v diplomi.
  - Finalni model za submission: ucen na vseh anotiranih pikslih
    (train + val + test), ker tekmovalni portal sluzi kot pravi test.

Preprocessing pipeline (Section 2.3, Berisha et al. 2018):
  1. Rubberband (piece-wise linear) baseline correction
  2. Amide I normalization: vsak spekter delimo z vrednostjo pri ~1650 cm-1
  3. PCA (16 komponent)
  4. RBF SVM  (C=1.0, gamma=1/16, probability=True)

Primarna metrika: OA (kot clanek). Log loss informativno.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.svm import SVC

from spatial_split import make_spatial_three_way_split, print_split_summary


# ---------------------------------------------------------------------------
# Konstante
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
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
# Preprocessing (zvest clanku — brez StandardScalerja)
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
    """Rubberband + Amide I normalizacija."""
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
# Fitanje modela
# ---------------------------------------------------------------------------
def fit_model_a(X_spec: np.ndarray, y: np.ndarray, amide_i_idx: int,
                n_components: int = 16, seed: int = 42,
                label: str = ""):
    X_pp  = preprocess(X_spec, amide_i_idx, label=label)
    pca   = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_pp)
    print(f"  PCA pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    X_bal, y_bal = oversample_to_max_class(X_pca, y, seed=seed)
    gamma = 1.0 / n_components
    print(f"  Ucenje RBF SVM (C=1.0, gamma={gamma:.4f}, verbose=True)...")
    svm = SVC(kernel="rbf", C=1.0, gamma=gamma,
              probability=True, random_state=seed, verbose=True)
    t0 = time.time()
    svm.fit(X_bal, y_bal)
    print(f"  SVM naučen v {time.time()-t0:.1f}s")
    return pca, svm


# ---------------------------------------------------------------------------
# Evaluacija
# ---------------------------------------------------------------------------
def evaluate(pca, svm, X_spec: np.ndarray, y_true: np.ndarray,
             amide_i_idx: int, split_name: str = ""):
    """Evaluacija na poljubnem splitu. Vrne (OA, log_loss)."""
    X_pp  = preprocess(X_spec, amide_i_idx, label=split_name)
    X_pca = pca.transform(X_pp)
    probs = svm.predict_proba(X_pca)
    preds = np.argmax(probs, axis=1)

    oa = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))

    print(f"\n  ── {split_name} ──")
    print(f"  OA:       {oa*100:.2f}%   (clanek: SD=56.41%, HD=76.28%)")
    print(f"  Log loss: {ll:.5f}")
    print(f"\n  Natancnost po razredih:")
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"    Razred {c}: N/A")
        else:
            acc_c = (preds[mask] == y_true[mask]).mean()
            print(f"    Razred {c}: {acc_c*100:.2f}%  (n={mask.sum()})")

    cm = confusion_matrix(y_true, preds, labels=np.arange(NUM_CLASSES))
    print(f"\n  Matrika zmede:")
    print(cm)

    return oa, ll


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def make_submission(pca, svm, data: np.ndarray, tissue_mask: np.ndarray,
                    amide_i_idx: int, output_path: str,
                    train_class_counts: np.ndarray = None):
    """
    Submission z locevanjem tkivo / ozadje.
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
        X_pca    = pca.transform(X_pp)
        submission_flat[crop_tissue] = svm.predict_proba(X_pca).astype(np.float32)

    submission = submission_flat.reshape(crop_h, crop_w, NUM_CLASSES)
    np.save(output_path, submission)
    print(f"  Submission shranjen: {output_path}  shape={submission.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model A v3 — article-faithful PP + 3-way prostorski split"
    )
    parser.add_argument("--input",              default="image1-competition.hdf5")
    parser.add_argument("--output",             default="modelA_v3.npy")
    parser.add_argument("--pca-components",     type=int,   default=16)
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
    amide_i_idx        = find_amide_i_index(wns)
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
    print("\n=== 3. Ucenje modela na TRAIN splitu ===")
    pca, svm = fit_model_a(
        X_spec       = X_train,
        y            = y_train,
        amide_i_idx  = amide_i_idx,
        n_components = args.pca_components,
        seed         = args.seed,
        label        = "train-pp",
    )

    # ------------------------------------------------------------------
    print("\n=== 4. Evaluacija ===")
    oa_val,  ll_val  = evaluate(pca, svm, X_val,  y_val,  amide_i_idx, "VAL")
    print()
    oa_test, ll_test = evaluate(pca, svm, X_test, y_test, amide_i_idx, "TEST (zaklenjen)")

    # ------------------------------------------------------------------
    print("\n=== 5. Finalni model (train + val + test = vse dovoljeno) ===")
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    X_all = data[usable_mask]
    y_all = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov za finalni model: {len(y_all)}")

    pca_final, svm_final = fit_model_a(
        X_spec       = X_all,
        y            = y_all,
        amide_i_idx  = amide_i_idx,
        n_components = args.pca_components,
        seed         = args.seed,
        label        = "final-pp",
    )

    # ------------------------------------------------------------------
    print("\n=== 6. Submission ===")
    make_submission(
        pca              = pca_final,
        svm              = svm_final,
        data             = data,
        tissue_mask      = tissue_mask,
        amide_i_idx      = amide_i_idx,
        output_path      = args.output,
        train_class_counts = np.bincount(y_all, minlength=NUM_CLASSES),
    )

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  VAL  — OA: {oa_val*100:.2f}%  |  log loss: {ll_val:.5f}")
    print(f"  TEST — OA: {oa_test*100:.2f}%  |  log loss: {ll_test:.5f}  ← koncna stevilka za diplomo")
    print(f"  Primerjava (clanek): SD=56.41%, HD=76.28% OA")
    print(f"  Submission: {args.output}")


if __name__ == "__main__":
    main()
