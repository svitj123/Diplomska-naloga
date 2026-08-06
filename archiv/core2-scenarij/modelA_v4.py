"""
Model A — Article-faithful spektralni SVM  (v4: gnezdena Core2 validacija)
=============================================================================
Razlika od v3: zamenjan STAR 3-way komponentni split (spatial_split.py,
train/val/test 60/20/20) z ISTO gnezdeno KMeans leave-one-core-out validacijo
kot modelC v10-v12 — poštena primerjava na istem Core2 test setu.

  Zunanji split (KMeans k=6):  5 krogcev (outer-train)  |  Core 2 (TEST)
  Notranji split (KMeans k=5): 4 (inner-train) | 1 (inner-val)

SVM nima "epoh" — ni potrebe po best-of-N iskanju. Inner-val je diagnosticen
prikaz (SVM fit na 4 inner-train krogcih). Koncni model se fitta na VSEH
5 outer-train krogcev in evaluira ENKRAT na Core 2 (edini pravi dotik).

Preprocessing (Section 2.3, Berisha et al. 2018) — nespremenjeno iz v3:
  1. Rubberband (piece-wise linear) baseline correction
  2. Amide I normalizacija (delitev z absorbanco pri ~1650 cm-1)
  3. PCA (16 komponent)
  4. RBF SVM (C=1.0, gamma=1/16, probability=True)
  5. Oversampling manjsinskih razredov (namesto class weights) — kot modelC v11/v12

Primerjava: SVM (clanek, SD, cross-slide, tezja naloga od nase): OA=56.41%.
"""

import argparse
import os
import time
from datetime import datetime

import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.svm import SVC

NUM_CLASSES = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660
AMIDE_I_TARGET_WN = 1650.0
RESULTS_FILE = "rezultati_report.txt"

REF_SVM_SD_OA = 56.41

METODOLOGIJA_OPOMBA = (
    "modelA_v4 (popravljen): SVM (clanek-zvest preprocessing: rubberband+"
    "Amide I+PCA16) z ISTO gnezdeno Core2 validacijo kot modelC v10-v12. "
    "Popravka: gamma='scale' (sklearn privzeto, clanek: 'automatically "
    "determined by the Scikit-learn implementation', ne rocno 1/16) in "
    "oversampling na fiksnih 10,000/razred (clanek: 'trained using 10,000 "
    "samples for each class', ne na velikost najvecjega razreda). "
    "Ref. clanek SVM (SD, cross-slide): OA=56.41%."
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
# Gnezdeni prostorski split (identicno modelC v10-v12)
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
def oversample_to_target(X, y, target=10_000, seed=42, verbose=True, label="train"):
    """Clanek (SVM): 'trained using 10,000 samples for each class'."""
    rng = np.random.default_rng(seed)
    counts = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    if verbose:
        print(f"  Oversampling ({label}): {[len(i) for i in counts]} -> {target}/razred (clanek: 10,000)")
    sampled = [rng.choice(idx, size=target, replace=True)
               for idx in counts if len(idx) > 0]
    idx_all = np.concatenate(sampled)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


# ---------------------------------------------------------------------------
# Fitanje SVM
# ---------------------------------------------------------------------------
def fit_svm(X_spec, y, amide_i_idx, n_components=16, seed=42, label="",
           oversample_target=10_000):
    X_pp  = preprocess(X_spec, amide_i_idx, label=label)
    pca   = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_pp)
    print(f"  PCA pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%  "
          f"(clanek SD: 90.03%)")
    X_bal, y_bal = oversample_to_target(X_pca, y, target=oversample_target,
                                        seed=seed, label=label)
    # clanek: gamma "automatically determined by the Scikit-learn implementation"
    # -> sklearn privzeto ('scale'), NE rocno fiksirano 1/n_components
    print(f"  Ucenje RBF SVM (C=1.0, gamma='scale', n={len(y_bal):,})...")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)
    t0 = time.time()
    svm.fit(X_bal, y_bal)
    print(f"  SVM naucen v {time.time()-t0:.1f}s")
    return pca, svm


def evaluate(pca, svm, X_spec, y_true, amide_i_idx, split_name=""):
    X_pp  = preprocess(X_spec, amide_i_idx, label=split_name)
    X_pca = pca.transform(X_pp)
    probs = svm.predict_proba(X_pca)
    preds = np.argmax(probs, axis=1)

    oa = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))

    print(f"\n  -- {split_name} --")
    print(f"  OA:       {oa*100:.2f}%   (clanek SVM SD: {REF_SVM_SD_OA:.2f}%)")
    print(f"  Log loss: {ll:.5f}")
    print(f"\n  Natancnost po razredih:")
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            print(f"    Razred {c}: N/A")
        else:
            ll_c = log_loss(y_true[mask], probs[mask], labels=np.arange(NUM_CLASSES))
            acc_c = (preds[mask] == y_true[mask]).mean()
            print(f"    Razred {c}: OA={acc_c*100:.2f}%  ll={ll_c:.5f}  (n={mask.sum()})")

    return oa, ll, probs


def write_results_report(model_name, innerval_oa, innerval_ll, test_oa, test_ll,
                          output_path, extra_note=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"CORE2_OA={test_oa*100:6.2f}%  CORE2_ll={test_ll:.5f}\n"
        f"{'':>19}  Primerjava: SVM(clanek,SD,cross-slide)=56.41%"
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model A v4 — article-faithful SVM, gnezdena Core2 validacija"
    )
    parser.add_argument("--input",           default="image1-competition.hdf5")
    parser.add_argument("--output",          default="modelA_v4_core2.npy")
    parser.add_argument("--pca-components",  type=int, default=16)
    parser.add_argument("--n-cores",         type=int, default=6)
    parser.add_argument("--n-inner-cores",   type=int, default=5)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, wns, tissue_mask, classes = load_data(args.input)
    H, W, _ = data.shape
    print(f"  data: {data.shape} | Anotiranih: {(classes != -1).sum()}")
    amide_i_idx = find_amide_i_index(wns)
    prediction_crop_mask = make_prediction_crop_mask(H, W)
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    # ------------------------------------------------------------------
    print("\n=== 2. Zunanji split: 5 krogcev (outer-train) | Core 2 (TEST) ===")
    outer_train_mask, test_mask = make_spatial_split(
        tissue_mask, classes, prediction_crop_mask,
        n_cores=args.n_cores, seed=args.seed, label="core")
    X_test = data[test_mask]
    y_test = classes[test_mask].astype(np.int64)

    print("\n=== 3. Notranji split: 4 (inner-train) | 1 (inner-val), znotraj outer-train ===")
    exclude_for_inner = prediction_crop_mask | test_mask
    inner_train_mask, inner_val_mask = make_spatial_split(
        tissue_mask, classes, exclude_for_inner,
        n_cores=args.n_inner_cores, seed=args.seed, label="subcore")

    X_it = data[inner_train_mask]; y_it = classes[inner_train_mask].astype(np.int64)
    X_iv = data[inner_val_mask];   y_iv = classes[inner_val_mask].astype(np.int64)
    X_ot = data[outer_train_mask]; y_ot = classes[outer_train_mask].astype(np.int64)

    # ==================================================================
    # FAZA A — diagnosticen SVM na inner-train, evalvacija na inner-val
    # (Core 2 se ne dotakne)
    # ==================================================================
    print("\n=== 4. Faza A — SVM na inner-train (diagnostika, Core 2 ni vpleten) ===")
    pca_A, svm_A = fit_svm(X_it, y_it, amide_i_idx,
                           n_components=args.pca_components, seed=args.seed,
                           label="Faza A / inner-train")
    innerval_oa, innerval_ll, _ = evaluate(
        pca_A, svm_A, X_iv, y_iv, amide_i_idx, "INNER-VAL (Faza A)")

    # ==================================================================
    # FAZA B — koncni SVM na vseh 5 outer-train krogcev, en dotik s Core 2
    # ==================================================================
    print("\n=== 5. Faza B — SVM na vseh 5 outer-train krogcev ===")
    pca_B, svm_B = fit_svm(X_ot, y_ot, amide_i_idx,
                           n_components=args.pca_components, seed=args.seed,
                           label="Faza B / outer-train")

    print("\n=== 6. KONCNA evaluacija na Core 2 (edini dotik) ===")
    test_oa, test_ll, test_probs = evaluate(
        pca_B, svm_B, X_test, y_test, amide_i_idx, "CORE 2 (koncni test)")

    # ------------------------------------------------------------------
    print("\n=== 7. Shranjevanje napovedi na Core 2 bbox ===")
    test_coords = np.argwhere(test_mask)
    r_min = int(test_coords[:, 0].min()); r_max = int(test_coords[:, 0].max())
    c_min = int(test_coords[:, 1].min()); c_max = int(test_coords[:, 1].max())
    bbox_h = r_max - r_min + 1; bbox_w = c_max - c_min + 1

    prior = np.bincount(y_ot, minlength=NUM_CLASSES).astype(np.float32)
    prior /= prior.sum()
    prob_map = np.tile(prior, (bbox_h * bbox_w, 1)).reshape(bbox_h, bbox_w, NUM_CLASSES)
    for (r, c), prob in zip(test_coords, test_probs):
        prob_map[r - r_min, c - c_min] = prob
    np.save(args.output, prob_map.astype(np.float32))
    print(f"  Shranjeno: {args.output}  shape={prob_map.shape}")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (modelA_v4) ===")
    print(f"  Faza A (inner-val, diagnostika): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (Core 2, KONCNI test):    OA={test_oa*100:.2f}%  ll={test_ll:.5f}")
    print(f"  Ref clanek SVM (SD, cross-slide): OA={REF_SVM_SD_OA:.2f}%")

    print(f"\n=== 8. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelA_v4",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa, test_ll=test_ll,
        output_path=args.output,
        extra_note=METODOLOGIJA_OPOMBA,
    )


if __name__ == "__main__":
    main()
