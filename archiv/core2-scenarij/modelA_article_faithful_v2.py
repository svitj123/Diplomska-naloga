"""
Model A — Article-faithful spectral baseline
=============================================
Preprocessing pipeline (Section 2.3, Berisha et al. 2018):
  1. Rubberband (piece-wise linear) baseline correction
  2. Amide I normalization: vsak spekter delimo z njegovo vrednostjo pri ~1650 cm-1
  3. PCA (16 komponent)
  4. RBF SVM  (C=1.0, gamma=1/16, probability=True)

Metrika za diplomo: OA (overall accuracy) — isto kot v članku.
Log loss poročamo informativno (tekmovanje).

Komentar o razliki z rezultati iz članka:
  Clanek poroča OA na cross-slide validaciji (drug slide = drug pacient).
  Mi validiramo na drugem prostorskem delu istega slide-a.
  Pricakujemo višji OA (manjša distribucijska razlika), kar je metodološka razlika,
  ne napaka — to je treba razložiti v diplomi.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Konstante
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660

# Amide I pas (~1650 cm-1) — za normalizacijo kot v članku
AMIDE_I_TARGET_WN = 1650.0


# ---------------------------------------------------------------------------
# Nalaganje podatkov
# ---------------------------------------------------------------------------
def load_data(hdf5_path: str):
    """Naloži FTIR podatke iz HDF5 datoteke."""
    with h5py.File(hdf5_path, "r") as f:
        data = np.array(f["data"], dtype=np.float32)
        wns = np.array(f["wns"])
        tissue_mask = np.array(f["tissue_mask"])
        classes = np.array(f["classes"])
    return data, wns, tissue_mask, classes


def find_amide_i_index(wns: np.ndarray) -> int:
    """Poišči indeks v wns najbližji Amide I pasu (1650 cm-1)."""
    idx = int(np.argmin(np.abs(wns - AMIDE_I_TARGET_WN)))
    print(f"  Amide I: target={AMIDE_I_TARGET_WN:.1f} cm-1 | "
          f"actual={wns[idx]:.2f} cm-1 | index={idx}")
    return idx


# ---------------------------------------------------------------------------
# Preprocessing: korak 1 — rubberband baseline korekcija
# ---------------------------------------------------------------------------
def _rubberband_single(spectrum: np.ndarray) -> np.ndarray:
    """
    Piece-wise linear (rubber band) baseline korekcija za en spekter.

    Algoritem:
    - Poišče spodnjo konveksno ovojnico (lower convex hull) spektra.
    - Interpolira spodnjo ovojnico kot baseline.
    - Odšteje baseline od spektra.

    Fizikalni pomen: odstrani razpršeno ozadje (scattering baseline),
    ki ni del absorpcijskega signala tkiva.
    """
    n = len(spectrum)
    x = np.arange(n, dtype=np.float64)
    y = spectrum.astype(np.float64)

    # Andrew's monotone chain — spodnja konveksna ovojnica
    # Pogoj za odstranjanje: cross <= 0 (ne-levi zavoj)
    lower = []
    for i in range(n):
        p = (x[i], y[i])
        while len(lower) >= 2:
            o = lower[-2]
            a = lower[-1]
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)

    lower_x = np.array([p[0] for p in lower])
    lower_y = np.array([p[1] for p in lower])

    # Linearna interpolacija baseline na vse točke spektra
    baseline = np.interp(x, lower_x, lower_y)
    return (y - baseline).astype(np.float32)


def rubberband_baseline_correction(spectra: np.ndarray) -> np.ndarray:
    """Rubberband baseline korekcija za vse spektre (vrstica po vrstica)."""
    out = np.empty_like(spectra, dtype=np.float32)
    for i in range(len(spectra)):
        out[i] = _rubberband_single(spectra[i])
    return out


# ---------------------------------------------------------------------------
# Preprocessing: korak 2 — normalizacija z Amide I
# ---------------------------------------------------------------------------
def amide_i_normalize(spectra: np.ndarray, amide_i_idx: int,
                      eps: float = 1e-6) -> np.ndarray:
    """
    Normalizacija: vsak spekter delimo z njegovo vrednostjo pri Amide I (~1650 cm-1).

    Fizikalni pomen: odstranjuje variacije v intenziteti, ki so posledica
    debeline vzorca in drugih neabsorpcijskih faktorjev.

    Referenca: Berisha et al. (2018), Section 2.3, korak 2.
    """
    amide_vals = spectra[:, amide_i_idx].astype(np.float64)

    n_problematic = int(np.sum(amide_vals <= eps))
    if n_problematic > 0:
        print(f"  Opozorilo: {n_problematic} spektrov ima Amide I <= {eps} "
              f"(po baseline korekciji). Uporabimo eps={eps} za deljenje.")
    amide_vals_safe = np.where(amide_vals > eps, amide_vals, eps)

    normalized = spectra / amide_vals_safe[:, np.newaxis]
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Celoten preprocessing pipeline (zvest članku)
# ---------------------------------------------------------------------------
def preprocess_article_style(spectra: np.ndarray, amide_i_idx: int) -> np.ndarray:
    """
    Preprocessing pipeline iz članka:
      1. Rubberband baseline korekcija
      2. Amide I normalizacija

    Brez StandardScalerja — članek ga ne omenja.
    PCA se aplicira naknadno (v fit_model_a).
    """
    t0 = time.time()
    print(f"  [1/2] Rubberband baseline korekcija ({len(spectra)} spektrov)...")
    spectra_bc = rubberband_baseline_correction(spectra)
    print(f"        Končano v {time.time() - t0:.1f}s | "
          f"min={spectra_bc.min():.4f}, max={spectra_bc.max():.4f}")

    t1 = time.time()
    print(f"  [2/2] Amide I normalizacija (indeks={amide_i_idx})...")
    spectra_norm = amide_i_normalize(spectra_bc, amide_i_idx)
    print(f"        Končano v {time.time() - t1:.1f}s | "
          f"min={spectra_norm.min():.4f}, max={spectra_norm.max():.4f}")

    return spectra_norm


# ---------------------------------------------------------------------------
# Validacijski prostorski split
# ---------------------------------------------------------------------------
def make_prediction_crop_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1] = True
    return mask


def collect_component_info(labeled_components, usable_annotated_mask,
                           classes, min_size=20):
    """Zberi info o vsaki tkivni komponenti: velikost in prisotni razredi."""
    component_ids = np.unique(labeled_components[usable_annotated_mask])
    component_ids = component_ids[component_ids != 0]

    components = []
    for cid in component_ids:
        comp_mask = (labeled_components == cid) & usable_annotated_mask
        cnt = int(comp_mask.sum())
        if cnt < min_size:
            continue
        y_comp = classes[comp_mask].astype(np.int64)
        present_classes = set(np.unique(y_comp).tolist())
        components.append({
            "cid": int(cid),
            "size": cnt,
            "present_classes": present_classes,
        })

    # Večje komponente najprej
    components.sort(key=lambda d: d["size"], reverse=True)
    return components


def make_spatial_class_cover_split(
    tissue_mask: np.ndarray,
    classes: np.ndarray,
    prediction_crop_mask: np.ndarray,
    val_fraction: float = 0.2,
    min_component_size: int = 20,
):
    """
    Prostorski split po tkivnih komponentah.

    Zagotavlja, da validacijski set vsebuje vse razrede.
    Tekmovalni crop je izločen iz učenja in validacije.
    """
    usable_annotated_mask = (classes != -1) & (~prediction_crop_mask)

    labeled_components, num_components = ndimage.label(tissue_mask)
    print(f"  Tkivne komponente skupaj: {num_components}")

    components = collect_component_info(
        labeled_components, usable_annotated_mask, classes, min_size=min_component_size
    )

    total_usable = int(usable_annotated_mask.sum())
    target_val = int(val_fraction * total_usable)

    print(f"  Uporabnih anotiranih pikslov: {total_usable}")
    print(f"  Kandidatnih komponent (po min_size filtru): {len(components)}")
    print(f"  Ciljna velikost validacije: {target_val}")

    val_cids = []
    val_pixels = 0
    covered_classes = set()
    remaining = components.copy()

    # 1) Najprej pokrij vse razrede v validaciji
    while len(covered_classes) < NUM_CLASSES and remaining:
        best_idx, best_gain, best_size = None, -1, None
        for i, comp in enumerate(remaining):
            gain = len(comp["present_classes"] - covered_classes)
            if gain > best_gain or (gain == best_gain and gain > 0
                                    and best_size is not None
                                    and comp["size"] < best_size):
                best_gain = gain
                best_idx = i
                best_size = comp["size"]
        if best_idx is None or best_gain <= 0:
            break
        chosen = remaining.pop(best_idx)
        val_cids.append(chosen["cid"])
        val_pixels += chosen["size"]
        covered_classes |= chosen["present_classes"]

    # 2) Dopolni do ciljne velikosti
    toggle = True
    for comp in remaining:
        if val_pixels >= target_val:
            break
        if toggle:
            val_cids.append(comp["cid"])
            val_pixels += comp["size"]
        toggle = not toggle

    val_cids_set = set(val_cids)
    train_cids = {c["cid"] for c in components if c["cid"] not in val_cids_set}

    train_mask = np.isin(labeled_components, list(train_cids)) & usable_annotated_mask
    val_mask = np.isin(labeled_components, list(val_cids_set)) & usable_annotated_mask

    y_train = classes[train_mask].astype(np.int64)
    y_val = classes[val_mask].astype(np.int64)

    print(f"  Train pikslov: {train_mask.sum()} | "
          f"Val pikslov: {val_mask.sum()} "
          f"({val_mask.sum() / total_usable:.3f})")
    print(f"  Train razredi: {np.bincount(y_train, minlength=NUM_CLASSES).tolist()}")
    print(f"  Val razredi:   {np.bincount(y_val,   minlength=NUM_CLASSES).tolist()}")

    for split_name, y in [("Train", y_train), ("Val", y_val)]:
        missing = [c for c in range(NUM_CLASSES) if np.sum(y == c) == 0]
        if missing:
            print(f"  OPOZORILO: {split_name} manjka razrede: {missing}")

    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Balansiranje razredov
# ---------------------------------------------------------------------------
def oversample_to_max_class(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Oversample manjšinske razrede do velikosti največjega."""
    rng = np.random.default_rng(seed)
    class_indices = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    class_counts = [len(idx) for idx in class_indices]
    target_n = max(class_counts)

    print(f"  Razredi pred oversamplingom: {class_counts}")
    print(f"  Cilj na razred: {target_n}")

    sampled = []
    for c in range(NUM_CLASSES):
        idx = class_indices[c]
        if len(idx) == 0:
            continue
        sampled.append(rng.choice(idx, size=target_n, replace=True))

    idx_all = np.concatenate(sampled)
    rng.shuffle(idx_all)
    print(f"  Razredi po oversamplu: {np.bincount(y[idx_all], minlength=NUM_CLASSES).tolist()}")
    return X[idx_all], y[idx_all]


# ---------------------------------------------------------------------------
# Fitanje modela A
# ---------------------------------------------------------------------------
def fit_model_a(X_spec: np.ndarray, y: np.ndarray, amide_i_idx: int,
                n_components: int = 16, seed: int = 42):
    """
    Zgradi Model A pipeline:
      preprocessing → PCA(16) → oversampling → RBF SVM

    Vrne: (pca, svm, amide_i_idx) — scaler ni, ker ga clanek ne omenja.
    """
    print("  Preprocessing...")
    X_pp = preprocess_article_style(X_spec, amide_i_idx)

    print(f"  PCA na {n_components} komponent...")
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_pp)
    print(f"  PCA pojasnjena varianca: {pca.explained_variance_ratio_.sum():.5f} "
          f"({pca.explained_variance_ratio_.sum() * 100:.2f}%)")

    print("  Oversampling...")
    X_bal, y_bal = oversample_to_max_class(X_pca, y, seed=seed)

    gamma = 1.0 / n_components  # = 1/16, kot v članku (gamma = 1/n_features)
    print(f"  Učenje RBF SVM (C=1.0, gamma={gamma:.4f})...")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=1.0, gamma=gamma, probability=True, random_state=seed)
    svm.fit(X_bal, y_bal)
    print(f"  SVM naučen v {time.time() - t0:.1f}s")

    return pca, svm


# ---------------------------------------------------------------------------
# Evaluacija (OA kot primarna metrika, log loss informativno)
# ---------------------------------------------------------------------------
def evaluate_model_a(pca, svm, X_val_spec: np.ndarray, y_val: np.ndarray,
                     amide_i_idx: int):
    """Evaluacija modela na validacijski množici."""
    X_pp = preprocess_article_style(X_val_spec, amide_i_idx)
    X_pca = pca.transform(X_pp)

    probs = svm.predict_proba(X_pca)
    preds = np.argmax(probs, axis=1)

    oa = accuracy_score(y_val, preds)
    ll = log_loss(y_val, probs, labels=np.arange(NUM_CLASSES))

    print(f"\n  OA (overall accuracy): {oa * 100:.2f}%   ← primarna metrika (kot clanek)")
    print(f"  Log loss:              {ll:.5f}          ← informativno (tekmovanje)")
    print(f"\n  Primerjava s clankom:")
    print(f"    Clanek (SD): 56.41% OA  |  Clanek (HD): 76.28% OA")
    print(f"    Naš model:   {oa * 100:.2f}% OA  "
          f"(višje pricakovano — same-slide val, ne cross-slide)")

    print(f"\n  Natancnost po razredih:")
    for c in range(NUM_CLASSES):
        mask = (y_val == c)
        if mask.sum() == 0:
            print(f"    Razred {c}: N/A (ni vzorcev v validaciji)")
        else:
            acc_c = (preds[mask] == y_val[mask]).mean()
            print(f"    Razred {c}: {acc_c * 100:.2f}%  (n={mask.sum()})")

    cm = confusion_matrix(y_val, preds, labels=np.arange(NUM_CLASSES))
    print(f"\n  Matrika zmede (vrstice=prava, stolpci=napovedana):")
    print(cm)

    return oa, ll


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def make_submission(pca, svm, data: np.ndarray, tissue_mask: np.ndarray,
                    amide_i_idx: int, output_path: str,
                    train_class_counts: np.ndarray = None):
    """
    Ustvari .npy submission datoteko za tekmovalni crop.

    Ločimo tkivne in ne-tkivne piksle:
    - Tkivni piksli: normalen preprocessing + napoved SVM.
    - Ne-tkivni piksli (ozadje): dodelimo prior iz train seta.
      Ti piksli imajo po rubberband korekciji Amide I ≈ 0 in bi dali
      popačene feature vektorje. Ker so verjetno brez anotacije (-1),
      ne vplivajo na log loss, a je vseeno metodološko čisto, da jih
      ločimo od pravih tkivnih pikslov.
    """
    crop_h = PRED_R1 - PRED_R0  # 200
    crop_w = PRED_C1 - PRED_C0  # 300
    n_crop = crop_h * crop_w    # 60000

    crop_data = data[PRED_R0:PRED_R1, PRED_C0:PRED_C1]
    crop_tissue = tissue_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1].reshape(-1)

    n_tissue = int(crop_tissue.sum())
    n_bg = n_crop - n_tissue
    print(f"  Crop pikslov skupaj: {n_crop}")
    print(f"  Tkivni piksli:       {n_tissue} ({100*n_tissue/n_crop:.1f}%)")
    print(f"  Ozadje (ne-tkivo):   {n_bg} ({100*n_bg/n_crop:.1f}%) → dobijo prior")

    # Izračunaj class prior iz train seta (ali uniform, če ni podan)
    if train_class_counts is not None:
        prior = train_class_counts.astype(np.float32)
        prior = prior / prior.sum()
    else:
        prior = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
    print(f"  Prior za ozadje: {np.round(prior, 4).tolist()}")

    # Inicializiraj submission z priorom (velja za ozadje)
    submission_flat = np.tile(prior, (n_crop, 1)).astype(np.float32)

    # Napoveduj samo za tkivne piksle
    if n_tissue > 0:
        X_tissue = crop_data.reshape(-1, crop_data.shape[-1])[crop_tissue]
        print(f"\n  Preprocessing {n_tissue} tkivnih pikslov...")
        X_pp = preprocess_article_style(X_tissue, amide_i_idx)
        X_pca = pca.transform(X_pp)
        probs_tissue = svm.predict_proba(X_pca).astype(np.float32)
        submission_flat[crop_tissue] = probs_tissue

    submission = submission_flat.reshape(crop_h, crop_w, NUM_CLASSES)

    np.save(output_path, submission)
    print(f"\n  Submission shranjen: {output_path}")
    print(f"  Shape: {submission.shape} | dtype: {submission.dtype}")
    # Preveri vzorčni piksel iz tkiva in ozadja
    tissue_2d = crop_tissue.reshape(crop_h, crop_w)
    if tissue_2d.any():
        r, c = np.argwhere(tissue_2d)[0]
        print(f"  Vsota verjetnosti (tkivni piksel [{r},{c}]): "
              f"{submission[r, c].sum():.6f}")
    print(f"  Vsota verjetnosti (piksel [0,0], ozadje pricakovano): "
          f"{submission[0, 0].sum():.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model A — article-faithful PCA + RBF SVM (Berisha et al. 2018)"
    )
    parser.add_argument("--input",  default="image1-competition.hdf5")
    parser.add_argument("--output", default="modelA_faithful.npy")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-component-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Vhodna datoteka ni najdena: {input_path}")

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje podatkov ===")
    data, wns, tissue_mask, classes = load_data(str(input_path))
    h, w, _ = data.shape
    print(f"  data: {data.shape} | wns: {wns.shape} | "
          f"Anotiranih pikslov: {(classes != -1).sum()}")

    amide_i_idx = find_amide_i_index(wns)
    prediction_crop_mask = make_prediction_crop_mask(h, w)

    # ------------------------------------------------------------------
    print("\n=== 2. Prostorski split (class-coverage) ===")
    train_mask, val_mask = make_spatial_class_cover_split(
        tissue_mask=tissue_mask,
        classes=classes,
        prediction_crop_mask=prediction_crop_mask,
        val_fraction=args.val_fraction,
        min_component_size=args.min_component_size,
    )

    X_train = data[train_mask]
    y_train = classes[train_mask].astype(np.int64)
    X_val = data[val_mask]
    y_val = classes[val_mask].astype(np.int64)

    # ------------------------------------------------------------------
    print("\n=== 3. Ucenje modela na train splitu ===")
    pca, svm = fit_model_a(
        X_spec=X_train,
        y=y_train,
        amide_i_idx=amide_i_idx,
        n_components=args.pca_components,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    print("\n=== 4. Lokalna validacija (OA kot primarna metrika) ===")
    oa, ll = evaluate_model_a(pca, svm, X_val, y_val, amide_i_idx)

    # ------------------------------------------------------------------
    print("\n=== 5. Ucenje finalnega modela na vseh dovoljenih pikslih ===")
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    X_all = data[usable_mask]
    y_all = classes[usable_mask].astype(np.int64)
    print(f"  Skupaj pikslov za finalni model: {len(y_all)}")

    pca_final, svm_final = fit_model_a(
        X_spec=X_all,
        y=y_all,
        amide_i_idx=amide_i_idx,
        n_components=args.pca_components,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    print("\n=== 6. Ustvari submission ===")
    train_class_counts = np.bincount(y_all, minlength=NUM_CLASSES)
    make_submission(
        pca=pca_final,
        svm=svm_final,
        data=data,
        tissue_mask=tissue_mask,
        amide_i_idx=amide_i_idx,
        output_path=args.output,
        train_class_counts=train_class_counts,
    )

    # ------------------------------------------------------------------
    print("\n=== POVZETEK ===")
    print(f"  Validacijski OA:       {oa * 100:.2f}%")
    print(f"  Validacijski log loss: {ll:.5f}")
    print(f"  Submission:            {args.output}")
    print(f"  Primerjava (clanek):   SD=56.41%, HD=76.28% OA")


if __name__ == "__main__":
    main()
