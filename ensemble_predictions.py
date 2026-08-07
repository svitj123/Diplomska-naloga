"""
Ansambel ze obstojecih napovedi (BREZ ponovnega treniranja)
=============================================================
Zdruzi shranjene verjetnostne karte razlicnih modelov (povprecje verjetnosti)
in oceni OA + log-loss na istem TEST setu (22 crop-ov, 439,704 anotiranih
pikslov) kot posamezni modeli.

Ideja: modeli z razlicno arhitekturo (CNN vs transformer) ali razlicno
regularizacijo (z/brez label smoothing) delajo RAZLICNE napake, zato lahko
povprecje njihovih verjetnosti prekasa vsakega posebej -- posebej pri
log-lossu, kjer se prevec samozavestne posamicne napake medsebojno omilijo.

Formati vhodnih datotek:
  - .npz (modelC/modelD): kljuci crop{ii}_probs (H x W x 6 platno bbox-a) in
    crop{ii}_offset (r0, c0) -- verjetnosti se izlusci na anotiranih koordinatah
  - .npy (modelB): ze konkatenirane verjetnosti po anotiranih koordinatah (N x 6)
    v istem vrstnem redu kot sortiran glob test crop-ov

Zagon (na strezniku, kjer so .npz/.npy datoteke in FTIR-data):
  python3 ensemble_predictions.py
"""

import glob
import os

import h5py
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

NUM_CLASSES = 6
TEST_DIR = "FTIR-data/test_preprocessed_full"
BBOX_MARGIN = 12

# Posamezni modeli in njihovi samostojni rezultati (za primerjavo v tabeli)
MODELI = {
    "CNN widerT (73.11%)":        ("modelC_faithful_no24_lrn_smooth3_widerT.npz", 73.11, 1.02331),
    "CNN lrn+smooth3 (72.90%)":   ("modelC_faithful_no24_lrn_smooth3.npz",        72.90, 1.01135),
    "CNN no24 (72.16%)":          ("modelC_faithful_no24_v1.npz",                 72.16, 1.01644),
    "CNN lrn (72.02%)":           ("modelC_faithful_no24_lrn.npz",                72.02, 1.02797),
    "CNN labelsmooth (70.09%)":   ("modelC_faithful_no24_lrn_smooth3_labelsmooth.npz", 70.09, 0.98092),
    "CNN ls005 (69.70%)":         ("modelC_faithful_no24_lrn_smooth3_ls005.npz",  69.70, 0.99778),
    "CNN ls02 (69.87%)":          ("modelC_faithful_no24_lrn_smooth3_ls02.npz",   69.87, 1.03002),
    "ViT hybrid p15 (58.05%)":    ("modelD_hybrid_patch15.npz",                   58.05, 1.39631),
    "ViT plain (52.82%)":         ("modelD_transformer_final.npz",                52.82, 1.45889),
}

# Kombinacije za preizkus
KOMBINACIJE = [
    ("CNN top-2", ["CNN widerT (73.11%)", "CNN lrn+smooth3 (72.90%)"]),
    ("CNN top-3", ["CNN widerT (73.11%)", "CNN lrn+smooth3 (72.90%)", "CNN no24 (72.16%)"]),
    ("CNN top-4", ["CNN widerT (73.11%)", "CNN lrn+smooth3 (72.90%)", "CNN no24 (72.16%)",
                   "CNN lrn (72.02%)"]),
    ("CNN raznolika regularizacija (widerT+lrn_smooth3+labelsmooth)",
     ["CNN widerT (73.11%)", "CNN lrn+smooth3 (72.90%)", "CNN labelsmooth (70.09%)"]),
    ("CNN vseh 7", [k for k in MODELI if k.startswith("CNN")]),
    ("CNN widerT + ViT p15", ["CNN widerT (73.11%)", "ViT hybrid p15 (58.05%)"]),
    ("CNN top-3 + ViT p15", ["CNN widerT (73.11%)", "CNN lrn+smooth3 (72.90%)",
                             "CNN no24 (72.16%)", "ViT hybrid p15 (58.05%)"]),
    ("CNN vseh 7 + ViT p15", [k for k in MODELI if k.startswith("CNN")] + ["ViT hybrid p15 (58.05%)"]),
]


def get_annotated_bbox(classes, margin=BBOX_MARGIN):
    H, W = classes.shape
    coords = np.argwhere(classes != -1)
    r0 = max(0, int(coords[:, 0].min()) - margin)
    r1 = min(H, int(coords[:, 0].max()) + margin + 1)
    c0 = max(0, int(coords[:, 1].min()) - margin)
    c1 = min(W, int(coords[:, 1].max()) + margin + 1)
    return r0, r1, c0, c1


def load_ground_truth(test_dir=TEST_DIR):
    """Vrne (y_true, seznam (crop_idx, coords_v_izvornem_prostoru)) v kanonicnem
    vrstnem redu (sortiran glob, znotraj crop-a np.argwhere vrstni red) --
    ISTI vrstni red, kot ga uporabljajo model skripte."""
    paths = sorted(glob.glob(os.path.join(test_dir, "test_crop_*.hdf5")))
    y_parts, coord_parts = [], []
    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            classes_full = np.array(f["classes"])
        r0, r1, c0, c1 = get_annotated_bbox(classes_full)
        classes_bbox = classes_full[r0:r1, c0:c1]
        ann = classes_bbox != -1
        coords_bbox = np.argwhere(ann)          # koordinate ZNOTRAJ bbox-a
        y_parts.append(classes_bbox[ann].astype(np.int64))
        coord_parts.append((i, coords_bbox))
    return np.concatenate(y_parts), coord_parts, len(paths)


def load_model_probs(path, coord_parts):
    """Izlusci verjetnosti na anotiranih koordinatah, v kanonicnem vrstnem redu."""
    if path.endswith(".npy"):
        probs = np.load(path)                    # ze (N x 6) po koordinatah
        return probs.astype(np.float64)

    data = np.load(path)
    parts = []
    for crop_idx, coords_bbox in coord_parts:
        key = f"crop{crop_idx:02d}_probs"
        if key not in data:
            raise KeyError(f"{path}: manjka {key}")
        canvas = data[key]                        # (H_bbox, W_bbox, 6)
        parts.append(canvas[coords_bbox[:, 0], coords_bbox[:, 1]])
    return np.concatenate(parts, axis=0).astype(np.float64)


def evaluate(probs, y_true):
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = np.argmax(probs, axis=1)
    return accuracy_score(y_true, pred), log_loss(y_true, probs, labels=np.arange(NUM_CLASSES))


def per_class_table(y_true, probs, title):
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = np.argmax(probs, axis=1)
    print(f"\n  {title}")
    print(f"  {'Razred':>8}  {'N':>8}  {'OA':>8}  {'Log-loss':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for c in range(NUM_CLASSES):
        m = (y_true == c)
        if m.sum() == 0:
            continue
        oa_c = accuracy_score(y_true[m], pred[m])
        ll_c = log_loss(y_true[m], probs[m], labels=np.arange(NUM_CLASSES))
        print(f"  {c:>8}  {m.sum():>8,}  {oa_c*100:>7.2f}%  {ll_c:>10.5f}")
    oa, ll = evaluate(probs, y_true)
    print(f"  {'SKUPAJ':>8}  {len(y_true):>8,}  {oa*100:>7.2f}%  {ll:>10.5f}")


def main():
    print("=== 1. Nalaganje ground truth (TEST) ===")
    y_true, coord_parts, n_crops = load_ground_truth()
    print(f"  {len(y_true):,} anotiranih pikslov iz {n_crops} crop-ov")

    print("\n=== 2. Nalaganje napovedi posameznih modelov ===")
    probs_cache = {}
    for ime, (path, ref_oa, ref_ll) in MODELI.items():
        if not os.path.exists(path):
            print(f"  PRESKOCENO (ni datoteke): {ime} -> {path}")
            continue
        p = load_model_probs(path, coord_parts)
        if len(p) != len(y_true):
            print(f"  OPOZORILO: {ime} ima {len(p):,} vrstic, pricakovano {len(y_true):,} "
                  f"-- preskocen")
            continue
        probs_cache[ime] = p
        oa, ll = evaluate(p, y_true)
        flag = "" if abs(oa*100 - ref_oa) < 0.5 else "  <-- ODSTOPA od porocanega!"
        print(f"  {ime:<32} OA={oa*100:6.2f}%  ll={ll:.5f}  "
              f"(porocano: {ref_oa:.2f}%/{ref_ll:.5f}){flag}")

    print("\n=== 3. Ansambli (povprecje verjetnosti) ===")
    print(f"  {'Kombinacija':<62}  {'OA':>8}  {'log-loss':>9}")
    print(f"  {'-'*62}  {'-'*8}  {'-'*9}")
    rezultati = []
    for ime, clani in KOMBINACIJE:
        na_voljo = [c for c in clani if c in probs_cache]
        if len(na_voljo) < 2:
            continue
        avg = np.mean([probs_cache[c] for c in na_voljo], axis=0)
        oa, ll = evaluate(avg, y_true)
        rezultati.append((ime, oa, ll, avg))
        print(f"  {ime:<62}  {oa*100:>7.2f}%  {ll:>9.5f}")

    if rezultati:
        najboljsi_oa = max(rezultati, key=lambda r: r[1])
        najboljsi_ll = min(rezultati, key=lambda r: r[2])
        print(f"\n  Najboljsi ansambel po OA:       {najboljsi_oa[0]} "
              f"({najboljsi_oa[1]*100:.2f}%)")
        print(f"  Najboljsi ansambel po log-loss: {najboljsi_ll[0]} "
              f"({najboljsi_ll[2]:.5f})")
        per_class_table(y_true, najboljsi_oa[3], f"Per-class: {najboljsi_oa[0]}")
        if najboljsi_ll[0] != najboljsi_oa[0]:
            per_class_table(y_true, najboljsi_ll[3], f"Per-class: {najboljsi_ll[0]}")

    print("\n  Referenca clanka: SVM=56.41%, CNN-spektralno=62.52%, CNN-prostorsko=79.45%")


if __name__ == "__main__":
    main()
