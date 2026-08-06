"""
spatial_split.py
================
Prostorski train / val / test split za FTIR hiperspektralno klasifikacijo.

Razdeli anotirane piksle (zunaj competition cropa) v tri prostorsko ločene
množice po tkivnih komponentah (connected components). Pikselski split ni
ustrezen, ker so sosednji piksli koreliranji.

Algoritem
---------
Faza 1 — Pokritost razredov:
    Za test in val z greedy pristopom izberemo najmanjše komponente,
    ki zagotovijo prisotnost vseh 6 razredov v obeh splitih.
    (Najmanjše → čim manj "porabimo" za kritje razredov.)

Faza 2 — Polnjenje do ciljnih deležev:
    Preostale komponente razdelimo med val in test glede na to,
    kateri split je najbolj oddaljen od svojega cilja.
    Preostanek gre v train.

Zagotovila
----------
- Vsi razredi so prisotni v train, val in test (opozorilo če ni mogoče).
- Competition crop je izključen iz vseh treh splitov.
- Split je deterministicen (neodvisen od vrstnega reda v OS).

Primer uporabe
--------------
    from spatial_split import make_spatial_three_way_split, print_split_summary

    train_mask, val_mask, test_mask = make_spatial_three_way_split(
        tissue_mask=tissue_mask,
        classes=classes,
        prediction_crop_mask=prediction_crop_mask,
    )
    print_split_summary(train_mask, val_mask, test_mask, classes)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

NUM_CLASSES = 6


# ---------------------------------------------------------------------------
# Interne pomožne funkcije
# ---------------------------------------------------------------------------

def _collect_component_info(
    labeled_components: np.ndarray,
    usable_mask: np.ndarray,
    classes: np.ndarray,
    min_size: int = 20,
) -> list[dict]:
    """
    Za vsako tkivno komponento zberi ID, število anotiranih pikslov in
    prisotne razrede. Vrne seznam, sortiran naraščajoče po velikosti
    (manjše komponente najprej — za greedy class coverage v fazi 1).
    """
    component_ids = np.unique(labeled_components[usable_mask])
    component_ids = component_ids[component_ids != 0]

    components = []
    for cid in component_ids:
        mask = (labeled_components == cid) & usable_mask
        n = int(mask.sum())
        if n < min_size:
            continue
        y = classes[mask].astype(np.int64)
        present = set(np.unique(y).tolist())
        components.append({"cid": int(cid), "size": n, "present_classes": present})

    # Primarno: naraščajoče po velikosti, sekundarno: po cid (determinizem)
    components.sort(key=lambda d: (d["size"], d["cid"]))
    return components


def _greedy_cover_classes(
    remaining: list[dict],
    target_classes: set,
) -> tuple[list[dict], list[dict], set]:
    """
    Iz `remaining` greedy izberi najmanjše komponente, ki skupaj pokrijejo
    vse razrede v `target_classes`.

    Vrne:
        chosen    — komponente izbrane za pokritost
        new_remaining — preostale komponente (brez izbranih)
        covered   — razredi, ki jih chosen pokriva
    """
    covered: set = set()
    chosen: list[dict] = []
    pool = list(remaining)  # kopija, ki jo modificiramo

    while covered < target_classes and pool:
        # Poišči najmanjšo komponento, ki doda vsaj en nov razred
        best_idx = None
        for i, comp in enumerate(pool):
            if comp["present_classes"] - covered:
                best_idx = i
                break  # pool je sortiran naraščajoče → prvi je najmanjši

        if best_idx is None:
            break  # noben razred ne more biti dodan

        chosen.append(pool.pop(best_idx))
        covered |= chosen[-1]["present_classes"]

    return chosen, pool, covered


# ---------------------------------------------------------------------------
# Glavna funkcija
# ---------------------------------------------------------------------------

def make_spatial_three_way_split(
    tissue_mask: np.ndarray,
    classes: np.ndarray,
    prediction_crop_mask: np.ndarray,
    val_fraction: float = 0.20,
    test_fraction: float = 0.20,
    min_component_size: int = 20,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prostorski train / val / test split po tkivnih komponentah.

    Parametri
    ---------
    tissue_mask          : bool maska tkiva (H, W)
    classes              : anotacije (H, W), -1 = neanotirano
    prediction_crop_mask : bool maska competition cropa (izključena iz vseh splitov)
    val_fraction         : ciljni delež val (privzeto 0.20)
    test_fraction        : ciljni delež test (privzeto 0.20)
    min_component_size   : najmanjša dovoljena komponenta (privzeto 20 pikslov)
    verbose              : ali izpisati potek

    Vrne
    ----
    train_mask, val_mask, test_mask — bool maske oblike (H, W)

    Opomba o neodvisnosti
    ---------------------
    Z enim samim slidom noben split ni resnično neodvisen (isti pacient,
    isto tkivo). Prostorska ločitev po komponentah je boljša od naključnega
    pixel splita, ne pa enako neodvisna kot cross-slide validacija iz članka.
    To je treba eksplicitno navesti v diplomi.
    """
    assert val_fraction + test_fraction < 1.0, \
        "val_fraction + test_fraction mora biti < 1.0"

    train_fraction = 1.0 - val_fraction - test_fraction

    # --- Pripravi piksle ---
    usable_mask = (classes != -1) & (~prediction_crop_mask)
    total_usable = int(usable_mask.sum())

    labeled_components, n_components = ndimage.label(tissue_mask)

    if verbose:
        print(f"  Tkivnih komponent skupaj:          {n_components}")

    components = _collect_component_info(
        labeled_components, usable_mask, classes, min_component_size
    )

    if verbose:
        print(f"  Komponent po filtriranju:          {len(components)}")
        print(f"  Skupaj anotiranih pikslov:         {total_usable}")
        print(f"  Ciljni delezi:  train={train_fraction:.0%} | "
              f"val={val_fraction:.0%} | test={test_fraction:.0%}")
        print(f"  Ciljni piksli: train≈{int(train_fraction*total_usable)} | "
              f"val≈{int(val_fraction*total_usable)} | "
              f"test≈{int(test_fraction*total_usable)}")

    all_classes = set(range(NUM_CLASSES))
    remaining = list(components)  # sorted ascending

    # ------------------------------------------------------------------
    # Faza 1: Pokritost razredov — najprej TEST, nato VAL
    # (Test obdelamo prvi, ker ima manj komponent za izbiro)
    # ------------------------------------------------------------------
    if verbose:
        print("\n  [Faza 1] Pokritost razredov:")

    test_comps, remaining, test_covered = _greedy_cover_classes(remaining, all_classes)
    test_pixels = sum(c["size"] for c in test_comps)

    if verbose:
        missing_t = all_classes - test_covered
        print(f"    Test:  {len(test_comps)} komponent, {test_pixels} pikslov, "
              f"razredi={sorted(test_covered)}"
              + (f"  ⚠ manjka: {sorted(missing_t)}" if missing_t else ""))

    val_comps, remaining, val_covered = _greedy_cover_classes(remaining, all_classes)
    val_pixels = sum(c["size"] for c in val_comps)

    if verbose:
        missing_v = all_classes - val_covered
        print(f"    Val:   {len(val_comps)} komponent, {val_pixels} pikslov, "
              f"razredi={sorted(val_covered)}"
              + (f"  ⚠ manjka: {sorted(missing_v)}" if missing_v else ""))

    # ------------------------------------------------------------------
    # Faza 2: Polnjenje do ciljnih deležev
    # Preostale komponente sortiramo padajoče (večje najprej → učinkoviteje)
    # ------------------------------------------------------------------
    remaining.sort(key=lambda d: d["size"], reverse=True)

    target_val  = int(val_fraction  * total_usable)
    target_test = int(test_fraction * total_usable)

    if verbose:
        print(f"\n  [Faza 2] Polnjenje:")

    for comp in remaining:
        val_deficit  = target_val  - val_pixels
        test_deficit = target_test - test_pixels

        if val_deficit <= 0 and test_deficit <= 0:
            break  # oba splita sta dosegla cilj, preostanek → train

        # Dodaj tistemu splitu, ki je bolj oddaljen od cilja
        if test_deficit >= val_deficit:
            test_comps.append(comp)
            test_pixels += comp["size"]
        else:
            val_comps.append(comp)
            val_pixels += comp["size"]

    # ------------------------------------------------------------------
    # Sestavi maske
    # ------------------------------------------------------------------
    test_cids  = {c["cid"] for c in test_comps}
    val_cids   = {c["cid"] for c in val_comps}
    train_cids = {c["cid"] for c in components
                  if c["cid"] not in test_cids and c["cid"] not in val_cids}

    train_mask = np.isin(labeled_components, list(train_cids)) & usable_mask
    val_mask   = np.isin(labeled_components, list(val_cids))   & usable_mask
    test_mask  = np.isin(labeled_components, list(test_cids))  & usable_mask

    # ------------------------------------------------------------------
    # Validacija
    # ------------------------------------------------------------------
    y_train = classes[train_mask].astype(np.int64)
    y_val   = classes[val_mask].astype(np.int64)
    y_test  = classes[test_mask].astype(np.int64)

    for split_name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        missing = [c for c in range(NUM_CLASSES) if np.sum(y == c) == 0]
        if missing:
            print(f"  ⚠ OPOZORILO: {split_name} nima razredov {missing}!")

    assert not (train_mask & val_mask).any(),  "Prekrivanje train/val!"
    assert not (train_mask & test_mask).any(), "Prekrivanje train/test!"
    assert not (val_mask   & test_mask).any(), "Prekrivanje val/test!"

    if verbose:
        print(f"\n  Rezultat:")
        print(f"    Train: {train_mask.sum():>6} pikslov "
              f"({100*train_mask.sum()/total_usable:.1f}%)")
        print(f"    Val:   {val_mask.sum():>6} pikslov "
              f"({100*val_mask.sum()/total_usable:.1f}%)")
        print(f"    Test:  {test_mask.sum():>6} pikslov "
              f"({100*test_mask.sum()/total_usable:.1f}%)")

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Izpis poročila o splitu
# ---------------------------------------------------------------------------

def print_split_summary(
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    classes: np.ndarray,
) -> None:
    """Izpiše podrobno poročilo o porazdelitvi razredov v vsakem splitu."""
    total = train_mask.sum() + val_mask.sum() + test_mask.sum()
    y_train = classes[train_mask].astype(np.int64)
    y_val   = classes[val_mask].astype(np.int64)
    y_test  = classes[test_mask].astype(np.int64)

    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print(  "  ║              PORAZDELITEV RAZREDOV PO SPLITIH               ║")
    print(  "  ╠══════════╦════════════╦════════════╦════════════╦═══════════╣")
    print(  "  ║  Razred  ║   Train    ║    Val     ║    Test    ║  Skupaj   ║")
    print(  "  ╠══════════╬════════════╬════════════╬════════════╬═══════════╣")

    totals = [0, 0, 0, 0]
    for c in range(NUM_CLASSES):
        nt = int(np.sum(y_train == c))
        nv = int(np.sum(y_val   == c))
        ns = int(np.sum(y_test  == c))
        na = nt + nv + ns
        totals[0] += nt; totals[1] += nv; totals[2] += ns; totals[3] += na
        print(f"  ║  {c:^8} ║ {nt:>6} ({100*nt/na if na else 0:4.1f}%) "
              f"║ {nv:>6} ({100*nv/na if na else 0:4.1f}%) "
              f"║ {ns:>6} ({100*ns/na if na else 0:4.1f}%) "
              f"║ {na:>7}   ║")

    print(  "  ╠══════════╬════════════╬════════════╬════════════╬═══════════╣")
    nt, nv, ns, na = totals
    print(f"  ║  Skupaj  ║ {nt:>6} ({100*nt/na:.1f}%) "
          f"║ {nv:>6} ({100*nv/na:.1f}%) "
          f"║ {ns:>6} ({100*ns/na:.1f}%) "
          f"║ {na:>7}   ║")
    print(  "  ╚══════════╩════════════╩════════════╩════════════╩═══════════╝")


# ---------------------------------------------------------------------------
# Test pri direktnem zagonu
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import h5py
    from pathlib import Path

    hdf5_path = Path("image1-competition.hdf5")
    if not hdf5_path.exists():
        print(f"Datoteka {hdf5_path} ni najdena. Zaženi iz mape projekta.")
        raise SystemExit(1)

    print("Nalaganje podatkov...")
    with h5py.File(hdf5_path, "r") as f:
        tissue_mask = np.array(f["tissue_mask"])
        classes     = np.array(f["classes"])

    h, w = classes.shape
    PRED_R0, PRED_R1 = 265, 465
    PRED_C0, PRED_C1 = 360, 660
    crop_mask = np.zeros((h, w), dtype=bool)
    crop_mask[PRED_R0:PRED_R1, PRED_C0:PRED_C1] = True

    print("\n=== 3-way prostorski split (60 / 20 / 20) ===")
    train_mask, val_mask, test_mask = make_spatial_three_way_split(
        tissue_mask=tissue_mask,
        classes=classes,
        prediction_crop_mask=crop_mask,
        val_fraction=0.20,
        test_fraction=0.20,
        verbose=True,
    )

    print_split_summary(train_mask, val_mask, test_mask, classes)
