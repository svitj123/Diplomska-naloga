"""
Model A — Table 1 replikacija (vsi klasicni klasifikatorji iz clanka)
=======================================================================
Clanek (Table 1) je poleg RBF SVM primerjal se 8 drugih spektralnih
klasifikatorjev na SD splitu (isti kot midva -- br1003/2085b train,
br961/1001 test): KNN, Linear SVM, Decision Tree (DT), Random Forest (RF),
neprostorska nevronska mreza (NN, ne CNN), AdaBoost, Naive Bayes (NB),
Quadratic Discriminant Analysis (QDA).

Clanek za te klasifikatorje ne navaja natancnih hiperparametrov (samo za
RBF SVM: C=1.0, gamma avtomatsko) -- verjetno privzete/razumne scikit-learn
nastavitve, enako naredimo tukaj.

Referencne vrednosti clanka (Table 1, SD stolpec, %):
  KNN=52.43  LinearSVM=53.99  RBFSVM=56.83  DT=46.47  RF=46.76
  NN=45.36  AdaBoost=50.26  NB=47.90  QDA=47.09
  (CNN=79.45 +/- 1.25, locena primerjava)

Namen: hitra (CPU, brez GPU) primerjalna tabela na PRAVEM cross-slide
splitu, isti podatki (PCA16 + oversampling 10,000/razred) kot
modelA_crossSlide.py (RBF SVM), da dopolnimo replikacijo clanka preko
samo enega klasifikatorja.

Brez nested Faza A/B kalibracije (ti modeli so dopolnilna primerjava, ne
"uradni" rezultat) -- en sam fit na VSEH train crop-ih, ena evaluacija na
CELEM TEST setu (22 crop-ov, ce --test-dir).
"""

import argparse
import glob
import os
import time
from datetime import datetime

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

NUM_CLASSES  = 6
RESULTS_FILE = "rezultati_report.txt"
BBOX_MARGIN  = 12

# Table 1, SD stolpec (clanek)
REF_TABLE1_SD = {
    "KNN": 52.43, "LinearSVM": 53.99, "RBFSVM": 56.83, "DT": 46.47,
    "RF": 46.76, "NN": 45.36, "AdaBoost": 50.26, "NB": 47.90, "QDA": 47.09,
}

METODOLOGIJA_OPOMBA = (
    "modelA_classic_baselines: replikacija clanka Table1 (KNN, LinearSVM, DT, "
    "RF, NN(MLP), AdaBoost, NB, QDA) na PRAVEM cross-slide splitu, isti PCA16 "
    "+ oversampling(10000/razred) pipeline kot modelA_crossSlide.py (RBF SVM). "
    "Clanek ne navaja natancnih hiperparametrov za te klasifikatorje -- "
    "uporabljene razumne/privzete scikit-learn nastavitve. Brez nested "
    "kalibracije (dopolnilna primerjava, ne 'uradni' rezultat)."
)


def get_annotated_bbox(classes, margin=BBOX_MARGIN):
    H, W = classes.shape
    coords = np.argwhere(classes != -1)
    r0 = max(0, int(coords[:, 0].min()) - margin)
    r1 = min(H, int(coords[:, 0].max()) + margin + 1)
    c0 = max(0, int(coords[:, 1].min()) - margin)
    c1 = min(W, int(coords[:, 1].max()) + margin + 1)
    return r0, r1, c0, c1


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
    n_bad = int((~np.isfinite(X)).any(axis=1).sum())
    if n_bad:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [{label}] {os.path.basename(path)}: {len(y):,} anotiranih  "
          f"({time.time()-t0:.1f}s)")
    return X, y


def oversample_to_target(X, y, target=10_000, seed=42, label="train"):
    rng = np.random.default_rng(seed)
    counts = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    print(f"  Oversampling ({label}): {[len(i) for i in counts]} -> {target}/razred")
    sampled = [rng.choice(idx, size=target, replace=True)
               for idx in counts if len(idx) > 0]
    idx_all = np.concatenate(sampled)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


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


def write_results_report(model_name, test_oa, test_ll, extra_note="", total_duration_str=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dur_part = f"  cas={total_duration_str}" if total_duration_str else ""
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"TEST_OA={test_oa*100:6.2f}%  TEST_ll={test_ll:.5f}{dur_part}\n"
    ]
    if extra_note:
        lines.append(f"{'':>19}  Opomba: {extra_note}\n")
    for attempt in range(3):
        try:
            if not os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, "w") as f:
                    f.write("# Rezultati modelov — FTIR klasifikacija tkiva\n")
            with open(RESULTS_FILE, "a") as f:
                f.writelines(lines)
            print(f"  -> {RESULTS_FILE} (dodana vrstica)")
            return
        except OSError as e:
            print(f"  OPOZORILO: pisanje ni uspelo (poskus {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2)


def build_classifiers(seed):
    return {
        "KNN":       KNeighborsClassifier(n_neighbors=5),
        "LinearSVM": SVC(kernel="linear", probability=True, random_state=seed),
        "DT":        DecisionTreeClassifier(random_state=seed),
        "RF":        RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "NN":        MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=seed),
        "AdaBoost":  AdaBoostClassifier(random_state=seed),
        "NB":        GaussianNB(),
        "QDA":       QuadraticDiscriminantAnalysis(),
    }


def main():
    run_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="FTIR-data/train_preprocessed")
    parser.add_argument("--test-dir", default="FTIR-data/test_preprocessed_full")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--oversample-target", type=int, default=10_000)
    parser.add_argument("--bbox-margin", type=int, default=BBOX_MARGIN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None,
                        help="Vejica-loceni podniz klasifikatorjev (privzeto vsi 8).")
    args = parser.parse_args()

    print("=== 1. Nalaganje train podatkov ===")
    train_paths = sorted(glob.glob(os.path.join(args.train_dir, "train_crop_*.hdf5")))
    print(f"  Najdenih train crop-ov: {len(train_paths)}")
    X_parts, y_parts = [], []
    for p in train_paths:
        X, y = load_annotated_spectra(p, args.bbox_margin, label="train")
        X_parts.append(X); y_parts.append(y)
    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    del X_parts, y_parts
    print(f"  Skupaj train: {len(y_train):,} pikslov")

    print("\n=== 2. Nalaganje test podatkov ===")
    test_paths = sorted(glob.glob(os.path.join(args.test_dir, "test_crop_*.hdf5")))
    print(f"  Najdenih test crop-ov: {len(test_paths)}")
    X_parts, y_parts = [], []
    for p in test_paths:
        X, y = load_annotated_spectra(p, args.bbox_margin, label="test")
        X_parts.append(X); y_parts.append(y)
    X_test = np.concatenate(X_parts, axis=0)
    y_test = np.concatenate(y_parts, axis=0)
    del X_parts, y_parts
    print(f"  Skupaj test: {len(y_test):,} pikslov")

    print(f"\n=== 3. PCA({args.pca_components}) fit na train ===")
    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"  Pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    X_bal, y_bal = oversample_to_target(X_train_pca, y_train,
                                        target=args.oversample_target, seed=args.seed)

    classifiers = build_classifiers(args.seed)
    if args.only:
        wanted = set(args.only.split(","))
        classifiers = {k: v for k, v in classifiers.items() if k in wanted}

    results = {}
    print(f"\n=== 4. Ucenje + evaluacija {len(classifiers)} klasifikatorjev ===")
    for name, clf in classifiers.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        clf.fit(X_bal, y_bal)
        fit_time = time.time() - t0
        probs = clf.predict_proba(X_test_pca)
        probs = np.clip(probs, 1e-7, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        pred = np.argmax(probs, axis=1)
        oa = accuracy_score(y_test, pred)
        ll = log_loss(y_test, probs, labels=np.arange(NUM_CLASSES))
        ref = REF_TABLE1_SD.get(name)
        ref_str = f"  (clanek: {ref:.2f}%)" if ref is not None else ""
        print(f"  {name}: TEST OA={oa*100:.2f}%  ll={ll:.5f}  "
              f"cas={format_duration(fit_time)}{ref_str}")
        print_per_class_table(y_test, pred, probs, f"Per-class ({name}):")
        results[name] = (oa, ll, fit_time)
        write_results_report(f"modelA_baseline_{name}", oa, ll,
                            extra_note=METODOLOGIJA_OPOMBA,
                            total_duration_str=format_duration(fit_time))

    print("\n\n=== POVZETEK — primerjava s clanek Table 1 (SD stolpec) ===")
    print(f"  {'Klasifikator':<12}  {'nas OA':>8}  {'clanek OA':>10}  {'razlika':>9}  {'ll':>9}")
    for name in classifiers:
        oa, ll, _ = results[name]
        ref = REF_TABLE1_SD.get(name)
        diff = f"{(oa*100-ref):+.2f}pp" if ref is not None else "-"
        ref_str = f"{ref:.2f}%" if ref is not None else "-"
        print(f"  {name:<12}  {oa*100:>7.2f}%  {ref_str:>10}  {diff:>9}  {ll:>9.5f}")
    print(f"\n  SKUPAJ CAS TEKA: {format_duration(time.time() - run_start)}")


if __name__ == "__main__":
    main()
