"""
Model A — Cross-Slide RBF SVM  (clanek-zvest spektralni SVM na PRAVEM cross-slide splitu)
=============================================================================================
Namen: isti train/test TMA split kot modelC_crossSlide_v2/faithful (FTIR-data/
train_preprocessed = 11 crop-ov iz br1003-br2085b, FTIR-data/test_preprocessed =
locen fizicni slajd brc961-br1001), da dobimo pravo, primerljivo SVM referenco
poleg CNN modelov -- doslej je edini SVM (modelA_v4.py) tekel na STAREM Core2
leave-one-out scenariju, ne na pravem cross-slide splitu.

POMEMBNO: FTIR-data/*.hdf5 so ZE predprocesirani (rubber-band baseline +
Amide I normalizacija -- preveri attrs vsake datoteke). Ta skripta NE
ponavlja preprocesiranja (za razliko od modelA_v4.py, ki dela na surovih
tekmovalnih podatkih) -- spektri se direktno uporabijo za PCA+SVM.

ZVESTO CLANKU (2.5, Table 3 SVM stolpec):
  1. PCA (16 komponent)
  2. RBF SVM, C=1.0
  3. gamma = "automatically determined by the Scikit-learn implementation"
     -> sklearn privzeto 'scale', NE rocno 1/16 (popravek iz modelA_v4)
  4. Oversampling: "trained using 10,000 samples for each class"
     -> --oversample-target privzeto 10000

Gnezdena Faza A/B struktura (isto ogrodje kot modelC_crossSlide_faithful.py):
  Faza A: PCA+SVM na inner-train (10 crop-ov), diagnostika + T/sigma kalibracija
          na inner-val (izloceni 11. crop) -- Core2/TEST se ne dotakne.
  Faza B: PCA+SVM na VSEH 11 train crop-ih, EN dotik s pravim TEST filom.

Brez epoh/ensembla/TTA (SVM nima tega) -- bistveno hitreje od CNN modelov.
"""

import argparse
import glob
import os
import time
from datetime import datetime

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC

NUM_CLASSES  = 6
RESULTS_FILE = "rezultati_report.txt"
BBOX_MARGIN  = 12
SIGMA_CHOICES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

REF_SVM_SD_OA = 56.41
REF_CNN_SD_OA = 79.45

METODOLOGIJA_OPOMBA = (
    "modelA_crossSlide: clanek-zvest RBF SVM (PCA16, C=1.0, gamma='scale', "
    "oversampling 10,000/razred) na PRAVEM cross-slide train/test TMA splitu "
    "(isti kot modelC_crossSlide_v2/faithful). Podatki so ZE predprocesirani "
    "(rubber-band+Amide I) -- brez ponovnega preprocesiranja, za razliko od "
    "modelA_v4.py (star Core2 scenarij, surovi vhod). "
    "Ref. clanek: SVM=56.41%, CNN=79.45%+/-1.25 (isti split)."
)


# ---------------------------------------------------------------------------
# Nalaganje — SAMO bounding box okoli anotiranih pikslov (RAM-varcno)
# ---------------------------------------------------------------------------
def peek_classes(path):
    with h5py.File(path, 'r') as f:
        return np.array(f['classes'])


def get_annotated_bbox(classes, margin=BBOX_MARGIN):
    H, W = classes.shape
    coords = np.argwhere(classes != -1)
    r0 = max(0, int(coords[:, 0].min()) - margin)
    r1 = min(H, int(coords[:, 0].max()) + margin + 1)
    c0 = max(0, int(coords[:, 1].min()) - margin)
    c1 = min(W, int(coords[:, 1].max()) + margin + 1)
    return r0, r1, c0, c1


def load_annotated_spectra(path, margin=BBOX_MARGIN, label=""):
    """Prebere SAMO bbox okoli anotiranih pikslov, nato SAMO anotirane spektre
    (ne cel bbox) -- brez patch/neighbourhood potrebe, SVM dela per-pixel."""
    t0 = time.time()
    with h5py.File(path, 'r') as f:
        classes_full = np.array(f['classes'])
        r0, r1, c0, c1 = get_annotated_bbox(classes_full, margin)
        classes_bbox = classes_full[r0:r1, c0:c1]
        ann = classes_bbox != -1
        data_bbox = np.array(f['data'][r0:r1, c0:c1, :], dtype=np.float32)
        tissue_mask_bbox = np.array(f['tissue_mask'][r0:r1, c0:c1])
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
    return X, y, coords, tissue_mask_bbox, classes_bbox.shape, (r0, c0)


def select_inner_val_candidates(crop_paths, verbose=True):
    """Enako kot modelC_crossSlide_*.py."""
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
# Oversampling — clanek: "10,000 samples for each class"
# ---------------------------------------------------------------------------
def oversample_to_target(X, y, target=10_000, seed=42, verbose=True, label="train"):
    rng = np.random.default_rng(seed)
    counts = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    if verbose:
        print(f"  Oversampling ({label}): {[len(i) for i in counts]} -> {target}/razred")
    sampled = [rng.choice(idx, size=target, replace=True)
               for idx in counts if len(idx) > 0]
    idx_all = np.concatenate(sampled)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


# ---------------------------------------------------------------------------
# PCA + SVM fit
# ---------------------------------------------------------------------------
def fit_pca_svm(X, y, n_components=16, oversample_target=10_000, seed=42, label=""):
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    print(f"  [{label}] PCA pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%  "
          f"(clanek SD: 90.03%)")
    X_bal, y_bal = oversample_to_target(X_pca, y, target=oversample_target,
                                        seed=seed, label=label)
    print(f"  [{label}] Ucenje RBF SVM (C=1.0, gamma='scale', n={len(y_bal):,})...")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)
    svm.fit(X_bal, y_bal)
    print(f"  [{label}] SVM naucen v {time.time()-t0:.1f}s")
    return pca, svm


# ---------------------------------------------------------------------------
# Temperature scaling + sigma smoothing (isto kot modelC_crossSlide_*.py)
# ---------------------------------------------------------------------------
def find_temperature(probs, y):
    """SVC.predict_proba je ze verjetnost, ne logit -- delamo na log(probs)
    kot na 'logits' (T=1 => nespremenjeno)."""
    eps = 1e-9
    log_probs = np.log(np.clip(probs, eps, 1.0))

    def neg_ll(T):
        s = log_probs / T
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), eps, 1.0)
        return -np.mean(np.log(p[np.arange(len(y)), y]))

    res = minimize_scalar(neg_ll, bounds=(0.1, 10.0), method='bounded')
    T = res.x
    print(f"  Temperature: T={T:.4f}  {neg_ll(1.0):.5f} -> {neg_ll(T):.5f}")
    return T


def apply_temperature(probs, T):
    eps = 1e-9
    log_probs = np.log(np.clip(probs, eps, 1.0))
    s = log_probs / T
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


def build_full_canvas_prob_map(shape, coords, probs, fallback_prior):
    H, W = shape
    prob_map = np.tile(fallback_prior, (H * W, 1)).reshape(H, W, NUM_CLASSES)
    prob_map[coords[:, 0], coords[:, 1]] = probs
    return prob_map


def find_best_sigma(shape, coords, probs, y_true, tissue_mask, fallback_prior,
                    sigma_choices=SIGMA_CHOICES):
    prob_map = build_full_canvas_prob_map(shape, coords, probs, fallback_prior)
    print(f"  {'sigma':>6}  {'log-loss':>10}")
    best_sigma, best_ll = 0.0, None
    for sigma in sigma_choices:
        if sigma > 0:
            smoothed = gaussian_smooth_probs(prob_map, tissue_mask, sigma)
            m = prob_map.copy()
            m[tissue_mask] = smoothed[tissue_mask]
        else:
            m = prob_map
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


def write_results_report(model_name, innerval_oa, innerval_ll, test_oa, test_ll,
                          output_path, t_opt, sigma, extra_note="", total_duration_str=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dur_part = f"  cas={total_duration_str}" if total_duration_str else ""
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"TEST_OA={test_oa*100:6.2f}%  TEST_ll={test_ll:.5f}  "
        f"T={t_opt:.4f}  sigma={sigma:.1f}{dur_part}  -> {output_path}\n"
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
        description="Model A Cross-Slide: clanek-zvest RBF SVM na pravem "
                    "cross-slide train/test TMA splitu"
    )
    parser.add_argument("--train-dir", default="FTIR-data/train_preprocessed")
    parser.add_argument("--test-file", default="FTIR-data/test_preprocessed/test_expanded_crop_preprocessed.hdf5")
    parser.add_argument("--output",    default="modelA_crossSlide_test.npy")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--oversample-target", type=int, default=10_000,
                        help="clanek: 10,000 pikslov/razred za SVM.")
    parser.add_argument("--bbox-margin", type=int, default=BBOX_MARGIN)
    parser.add_argument("--seed", type=int, default=42)
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

    # ==================================================================
    # FAZA A — PCA+SVM na inner-train, kalibracija na inner-val (Core2/TEST
    # se ne dotakne)
    # ==================================================================
    print(f"\n=== 2. Faza A — nalaganje inner-train ({len(inner_train_paths)} crop-ov) ===")
    X_it_parts, y_it_parts = [], []
    for p in inner_train_paths:
        X, y, _, _, _, _ = load_annotated_spectra(p, args.bbox_margin, label="it")
        X_it_parts.append(X); y_it_parts.append(y)
    X_it = np.concatenate(X_it_parts, axis=0)
    y_it = np.concatenate(y_it_parts, axis=0)
    del X_it_parts, y_it_parts
    print(f"  Skupaj inner-train: {len(y_it):,} pikslov")

    print(f"\n=== 3. Faza A — nalaganje inner-val ({os.path.basename(inner_val_path)}) ===")
    X_iv, y_iv, coords_iv, tissue_mask_iv, shape_iv, _ = load_annotated_spectra(
        inner_val_path, args.bbox_margin, label="iv")

    print(f"\n=== 4. Faza A — PCA + SVM fit (inner-train) ===")
    pca_A, svm_A = fit_pca_svm(X_it, y_it, args.pca_components,
                               args.oversample_target, args.seed, label="Faza A")
    del X_it, y_it

    print(f"\n=== 5. Faza A — napoved na inner-val, kalibracija ===")
    X_iv_pca = pca_A.transform(X_iv)
    probs_iv = svm_A.predict_proba(X_iv_pca)
    T_opt = find_temperature(probs_iv, y_iv)
    probs_iv_cal = apply_temperature(probs_iv, T_opt)
    pred_iv = np.argmax(probs_iv_cal, axis=1)
    innerval_oa = accuracy_score(y_iv, pred_iv)
    innerval_ll = log_loss(y_iv, probs_iv_cal, labels=np.arange(NUM_CLASSES))
    print(f"  INNERVAL OA: {innerval_oa*100:.2f}%  |  ll: {innerval_ll:.5f}")
    print_per_class_table(y_iv, pred_iv, probs_iv_cal, "Per-class (inner-val):")

    print(f"\n=== 6. Sigma sweep (inner-val) ===")
    prior_it = np.bincount(y_iv, minlength=NUM_CLASSES).astype(np.float32)  # priblizek
    prior_it /= prior_it.sum()
    sigma_opt = find_best_sigma(shape_iv, coords_iv, probs_iv_cal, y_iv,
                                tissue_mask_iv, prior_it)
    del svm_A, pca_A, X_iv, X_iv_pca, probs_iv, probs_iv_cal

    # ==================================================================
    # FAZA B — PCA+SVM na VSEH 11 train crop-ih, EN dotik s TEST-om
    # ==================================================================
    print(f"\n=== 7. Faza B — nalaganje vseh {len(train_crop_paths)} train crop-ov ===")
    X_ot_parts, y_ot_parts = [], []
    for p in train_crop_paths:
        X, y, _, _, _, _ = load_annotated_spectra(p, args.bbox_margin, label="ot")
        X_ot_parts.append(X); y_ot_parts.append(y)
    X_ot = np.concatenate(X_ot_parts, axis=0)
    y_ot = np.concatenate(y_ot_parts, axis=0)
    del X_ot_parts, y_ot_parts
    print(f"  Skupaj train (Faza B): {len(y_ot):,} pikslov")

    print(f"\n=== 8. Faza B — PCA + SVM fit (vsi train crop-i) ===")
    pca_B, svm_B = fit_pca_svm(X_ot, y_ot, args.pca_components,
                               args.oversample_target, args.seed, label="Faza B")
    prior_B = np.bincount(y_ot, minlength=NUM_CLASSES).astype(np.float32)
    prior_B /= prior_B.sum()
    del X_ot

    print(f"\n=== 9. Faza B — nalaganje TEST datoteke (locen fizicni slajd) ===")
    X_test, y_test, coords_test, tissue_mask_test, shape_test, bbox_offset = \
        load_annotated_spectra(args.test_file, args.bbox_margin, label="TEST")

    print(f"\n=== 10. KONCNA evaluacija na TEST (edini dotik, pravi cross-slide) ===")
    X_test_pca = pca_B.transform(X_test)
    probs_test = svm_B.predict_proba(X_test_pca)
    probs_test_cal = apply_temperature(probs_test, T_opt)
    pred_test = np.argmax(probs_test_cal, axis=1)
    test_oa = accuracy_score(y_test, pred_test)
    test_ll = log_loss(y_test, probs_test_cal, labels=np.arange(NUM_CLASSES))
    print(f"  TEST OA (pred smoothing): {test_oa*100:.2f}%")
    print(f"  TEST ll (pred smoothing): {test_ll:.5f}")
    print(f"  Ref clanek SVM (SD, isti split): OA={REF_SVM_SD_OA:.2f}%")
    print(f"  Ref clanek CNN (SD, isti split): OA={REF_CNN_SD_OA:.2f}% +/- 1.25")
    print_per_class_table(y_test, pred_test, probs_test_cal,
                          "Per-class (TEST — koncni test):")

    prob_map = build_full_canvas_prob_map(shape_test, coords_test, probs_test_cal, prior_B)
    if sigma_opt > 0:
        smoothed = gaussian_smooth_probs(prob_map, tissue_mask_test, sigma_opt)
        final_map = prob_map.copy()
        final_map[tissue_mask_test] = smoothed[tissue_mask_test]
    else:
        final_map = prob_map
    final_map = np.clip(final_map, 1e-7, 1.0)
    final_map /= final_map.sum(axis=-1, keepdims=True)

    smoothed_at_coords = final_map[coords_test[:, 0], coords_test[:, 1]]
    smoothed_at_coords = np.clip(smoothed_at_coords, 1e-7, 1.0)
    smoothed_at_coords /= smoothed_at_coords.sum(axis=1, keepdims=True)
    pred_test_sm = np.argmax(smoothed_at_coords, axis=1)
    test_oa_sm = accuracy_score(y_test, pred_test_sm)
    test_ll_sm = log_loss(y_test, smoothed_at_coords, labels=np.arange(NUM_CLASSES))
    print(f"\n  TEST OA (po smoothing, sigma={sigma_opt:.1f}): {test_oa_sm*100:.2f}%")
    print(f"  TEST ll (po smoothing, sigma={sigma_opt:.1f}): {test_ll_sm:.5f}")

    np.save(args.output, final_map.astype(np.float32))
    r0, c0 = bbox_offset
    print(f"\n  Shranjeno: {args.output}  shape={final_map.shape}")
    print(f"  (bbox znotraj izvornega TEST platna: offset r0={r0}, c0={c0})")

    print("\n=== POVZETEK (modelA crossSlide SVM) ===")
    print(f"  SKUPAJ CAS TEKA: {format_duration(time.time() - run_start)}")
    print(f"  Train: {len(train_crop_paths)} crop-ov (br1003-br2085b), {len(y_ot):,} pikslov")
    print(f"  Test:  1 datoteka (brc961-br1001), {len(y_test):,} pikslov")
    print(f"  Inner-val (diagnostika): OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (TEST, KONCNI):   OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")
    print(f"\n  Primerjava:")
    print(f"    Clanek SVM (isti split): OA={REF_SVM_SD_OA:.2f}%")
    print(f"    Clanek CNN (isti split): OA={REF_CNN_SD_OA:.2f}% +/- 1.25")
    print(f"    modelA crossSlide (ta tek): OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")

    print(f"\n=== 11. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelA_crossSlide",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa_sm, test_ll=test_ll_sm,
        output_path=args.output, t_opt=T_opt, sigma=sigma_opt,
        total_duration_str=format_duration(time.time() - run_start),
        extra_note=METODOLOGIJA_OPOMBA,
    )


if __name__ == "__main__":
    main()
