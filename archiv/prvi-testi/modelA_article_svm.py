import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


NUM_CLASSES = 6
PRED_R0, PRED_R1 = 265, 465
PRED_C0, PRED_C1 = 360, 660


# Naloži FTIR podatke iz HDF5 datoteke
def load_data(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        data = np.array(f["data"])
        wns = np.array(f["wns"])
        tissue_mask = np.array(f["tissue_mask"])
        classes = np.array(f["classes"])
    return data, wns, tissue_mask, classes


# Ustvari masko za tekmovalni izsek, ki ga izločimo iz učenja
def make_prediction_crop_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1] = True
    return mask


# Naredi prostorsko validacijo po povezanih komponentah tkiva
def make_spatial_split(
    tissue_mask: np.ndarray,
    classes: np.ndarray,
    prediction_crop_mask: np.ndarray,
    val_fraction: float = 0.2,
):
    """
    Spatial split by connected tissue components.
    This is not identical to the paper, but is much more honest than random pixel split.
    """
    usable_annotated_mask = (classes != -1) & (~prediction_crop_mask)

    # Poišči povezane komponente v maski tkiva
    labeled_components, num_components = ndimage.label(tissue_mask)
    print(f"Connected tissue components: {num_components}")

    # Uporabi samo tiste komponente, kjer dejansko obstajajo anotirani piksli
    component_ids = np.unique(labeled_components[usable_annotated_mask])
    component_ids = component_ids[component_ids != 0]

    component_sizes = []
    for cid in component_ids:
        cnt = np.sum((labeled_components == cid) & usable_annotated_mask)
        if cnt > 0:
            component_sizes.append((cid, int(cnt)))

    # Največje komponente obravnavaj najprej
    component_sizes = sorted(component_sizes, key=lambda x: x[1], reverse=True)

    target_val = int(val_fraction * usable_annotated_mask.sum())

    val_component_ids = []
    train_component_ids = []

    # Enostaven "greedy" razrez na train/val po komponentah
    running = 0
    toggle = True
    for cid, cnt in component_sizes:
        if running < target_val and toggle:
            val_component_ids.append(cid)
            running += cnt
        else:
            train_component_ids.append(cid)
        toggle = not toggle

    train_mask = np.isin(labeled_components, train_component_ids) & usable_annotated_mask
    val_mask = np.isin(labeled_components, val_component_ids) & usable_annotated_mask

    print(f"Usable annotated pixels: {usable_annotated_mask.sum()}")
    print(f"Train annotated pixels:  {train_mask.sum()}")
    print(f"Val annotated pixels:    {val_mask.sum()}")
    if usable_annotated_mask.sum() > 0:
        print(f"Val ratio:              {val_mask.sum() / usable_annotated_mask.sum():.4f}")

    return train_mask, val_mask


# Oversampling do velikosti največjega razreda
def oversample_to_max_class(X: np.ndarray, y: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)

    class_indices = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    class_counts = [len(idx) for idx in class_indices]
    target_n = max(class_counts)

    print("Original class counts:", class_counts)
    print("Target per class after oversampling:", target_n)

    sampled_indices = []
    for c in range(NUM_CLASSES):
        idx = class_indices[c]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=target_n, replace=True)
        sampled_indices.append(chosen)

    sampled_indices = np.concatenate(sampled_indices)
    rng.shuffle(sampled_indices)

    X_bal = X[sampled_indices]
    y_bal = y[sampled_indices]

    print("Balanced class counts:", np.bincount(y_bal, minlength=NUM_CLASSES).tolist())
    return X_bal, y_bal


# Ujemanje s spectral baseline idejo iz članka: StandardScaler + PCA(16) + RBF SVM
def fit_article_style_svm(
    X_train_spec: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 16,
    seed: int = 42,
):
    # Standardizacija spektralnih kanalov
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_spec)

    # Zmanjšanje dimenzionalnosti na 16 PCA komponent
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)

    print(f"PCA explained variance sum: {pca.explained_variance_ratio_.sum():.5f}")

    # Balansiranje razredov pred učenjem SVM
    X_train_bal, y_train_bal = oversample_to_max_class(X_train_pca, y_train, seed=seed)

    # Nastavitve čim bližje članku: RBF kernel, C=1.0, gamma=1/n_features
    gamma = 1.0 / n_components
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma=gamma,
        probability=True,
        random_state=seed,
    )
    svm.fit(X_train_bal, y_train_bal)

    return scaler, pca, svm


# Evaluacija na lokalni validacijski množici
def evaluate_model(
    scaler: StandardScaler,
    pca: PCA,
    svm: SVC,
    X_val_spec: np.ndarray,
    y_val: np.ndarray,
):
    X_val_scaled = scaler.transform(X_val_spec)
    X_val_pca = pca.transform(X_val_scaled)

    val_probs = svm.predict_proba(X_val_pca)
    val_pred = np.argmax(val_probs, axis=1)

    val_ll = log_loss(y_val, val_probs, labels=np.arange(NUM_CLASSES))
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Validation log loss: {val_ll:.5f}")
    print(f"Validation accuracy: {val_acc:.5f}")
    print()
    print("Classification report:")
    print(classification_report(y_val, val_pred, digits=4))

    return val_ll, val_acc


# Končni model nauči na vseh dovoljenih anotiranih piksljih
def fit_final_model_on_all_usable(
    data: np.ndarray,
    classes: np.ndarray,
    prediction_crop_mask: np.ndarray,
    n_components: int = 16,
    seed: int = 42,
):
    usable_mask = (classes != -1) & (~prediction_crop_mask)

    X_all_spec = data[usable_mask]
    y_all = classes[usable_mask].astype(np.int64)

    print(f"Final training pixels (all usable): {len(y_all)}")

    # Fit standardizacije in PCA na vseh dovoljenih učnih podatkih
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X_all_spec)

    pca_all = PCA(n_components=n_components, random_state=seed)
    X_all_pca = pca_all.fit_transform(X_all_scaled)
    print(f"Final PCA explained variance sum: {pca_all.explained_variance_ratio_.sum():.5f}")

    # Balansiranje razredov tudi pri final modelu
    X_all_bal, y_all_bal = oversample_to_max_class(X_all_pca, y_all, seed=seed)

    gamma = 1.0 / n_components
    svm_all = SVC(
        kernel="rbf",
        C=1.0,
        gamma=gamma,
        probability=True,
        random_state=seed,
    )
    svm_all.fit(X_all_bal, y_all_bal)

    return scaler_all, pca_all, svm_all


# Ustvari submission za tekmovalni crop
def make_submission(
    data: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    svm: SVC,
    output_path: str,
):
    data_predict = data[PRED_R0:PRED_R1, PRED_C0:PRED_C1]  # (200, 300, B)

    # Model napoveduje za vsak piksel posebej
    X_pred = data_predict.reshape(-1, data_predict.shape[-1])
    X_pred_scaled = scaler.transform(X_pred)
    X_pred_pca = pca.transform(X_pred_scaled)

    pred_probs = svm.predict_proba(X_pred_pca)
    pred_array = pred_probs.reshape(200, 300, NUM_CLASSES).astype(np.float32)

    np.save(output_path, pred_array)
    print(f"Saved submission to: {output_path}")
    print(f"Submission shape:    {pred_array.shape}")
    print(f"Submission dtype:    {pred_array.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Model A: article-style spectral baseline (PCA + RBF SVM)")
    parser.add_argument(
        "--input",
        type=str,
        default="image1-competition.hdf5",
        help="Path to input HDF5 file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="modelA_article_svm.npy",
        help="Path to output .npy submission file",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=16,
        help="Number of PCA components (article-style default: 16)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Approximate validation fraction for spatial split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading data...")
    data, wns, tissue_mask, classes = load_data(str(input_path))
    h, w, b = data.shape

    print(f"data shape:        {data.shape}")
    print(f"wns shape:         {wns.shape}")
    print(f"tissue_mask shape: {tissue_mask.shape}")
    print(f"classes shape:     {classes.shape}")
    print(f"Annotated pixels:  {(classes != -1).sum()}")

    # Maska za tekmovalni crop, ki ga ne uporabimo za učenje
    prediction_crop_mask = make_prediction_crop_mask(h, w)

    print("\n=== Spatial validation split ===")
    train_mask, val_mask = make_spatial_split(
        tissue_mask=tissue_mask,
        classes=classes,
        prediction_crop_mask=prediction_crop_mask,
        val_fraction=args.val_fraction,
    )

    # Spektri za train in validation
    X_train = data[train_mask]
    y_train = classes[train_mask].astype(np.int64)
    X_val = data[val_mask]
    y_val = classes[val_mask].astype(np.int64)

    print("\n=== Fit article-style SVM on train split ===")
    scaler, pca, svm = fit_article_style_svm(
        X_train_spec=X_train,
        y_train=y_train,
        n_components=args.pca_components,
        seed=args.seed,
    )

    print("\n=== Local validation ===")
    evaluate_model(
        scaler=scaler,
        pca=pca,
        svm=svm,
        X_val_spec=X_val,
        y_val=y_val,
    )

    print("\n=== Fit final model on all usable labeled pixels ===")
    scaler_all, pca_all, svm_all = fit_final_model_on_all_usable(
        data=data,
        classes=classes,
        prediction_crop_mask=prediction_crop_mask,
        n_components=args.pca_components,
        seed=args.seed,
    )

    print("\n=== Create submission ===")
    make_submission(
        data=data,
        scaler=scaler_all,
        pca=pca_all,
        svm=svm_all,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()