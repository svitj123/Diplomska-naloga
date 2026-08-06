from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image


# ============================================================
# NASTAVITVE
# ============================================================

DATA_DIR = Path("/home/sjesenk/mayerich-testni")
CLASS_DIR = DATA_DIR / "supervised-class"

CLASSES = [
    "coll",
    "epith",
    "fibro",
    "lymph",
    "myo",
    "necrosis",
]

# Trenutni testni crop, ki sva ga že izdelala.
CURRENT_ROW_START = 500
CURRENT_ROW_END = 1300
CURRENT_COL_START = 500
CURRENT_COL_END = 1700

# Pri primerjavi pregledamo možne crope enake velikosti.
# STRIDE = 100 pomeni gosto iskanje; 200 je hitrejši.
STRIDE = 100

OUTPUT_CSV = Path(
    "/home/sjesenk/local/test_crop_analysis.csv"
)


# ============================================================
# POMOŽNE FUNKCIJE
# ============================================================

def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Maska ne obstaja: {path}")

    with Image.open(path) as image:
        return np.asarray(image) > 0


def find_tissue_mask() -> Path:
    candidates = [
        CLASS_DIR / "tissue_mask.png",
        CLASS_DIR / "mask.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Ni najdene tissue maske. Pričakovano: "
        "'tissue_mask.png' ali 'mask.png' v "
        f"{CLASS_DIR}"
    )


def integral_image(mask: np.ndarray) -> np.ndarray:
    """
    Summed-area table z dodatno zgornjo vrstico in levim stolpcem.
    Omogoča hiter izračun vsote poljubnega pravokotnika.
    """
    integral = np.pad(
        mask.astype(np.int64),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    return integral.cumsum(axis=0).cumsum(axis=1)


def rectangle_sum(
    integral: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> int:
    return int(
        integral[row_end, col_end]
        - integral[row_start, col_end]
        - integral[row_end, col_start]
        + integral[row_start, col_start]
    )


def proportions(counts: np.ndarray) -> np.ndarray:
    total = int(counts.sum())

    if total == 0:
        return np.zeros_like(counts, dtype=np.float64)

    return counts.astype(np.float64) / total


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Jensen-Shannon divergenca med dvema porazdelitvama.
    0 pomeni popolnoma enako razredno porazdelitev.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum == 0 or q_sum == 0:
        return float("inf")

    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def kl_divergence(a, b):
        valid = a > 0
        return float(
            np.sum(a[valid] * np.log2(a[valid] / b[valid]))
        )

    return 0.5 * (
        kl_divergence(p, m)
        + kl_divergence(q, m)
    )


def percentile_rank(
    values,
    current_value,
    higher_is_better=True,
) -> float:
    values = np.asarray(values, dtype=np.float64)

    if higher_is_better:
        return 100.0 * np.mean(values <= current_value)

    return 100.0 * np.mean(values >= current_value)


def make_starts(
    total_size: int,
    crop_size: int,
    stride: int,
):
    starts = list(
        range(0, total_size - crop_size + 1, stride)
    )

    last_start = total_size - crop_size

    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    return sorted(set(starts))


# ============================================================
# GLAVNI POSTOPEK
# ============================================================

def main() -> None:
    tissue_path = find_tissue_mask()
    tissue_mask = read_mask(tissue_path)

    image_height, image_width = tissue_mask.shape

    crop_height = CURRENT_ROW_END - CURRENT_ROW_START
    crop_width = CURRENT_COL_END - CURRENT_COL_START

    if not (
        0 <= CURRENT_ROW_START < CURRENT_ROW_END <= image_height
        and 0 <= CURRENT_COL_START < CURRENT_COL_END <= image_width
    ):
        raise ValueError(
            "Trenutni crop je zunaj meja testne slike."
        )

    print("TESTNI PODATKI")
    print(f"  Mapa: {DATA_DIR}")
    print(f"  Tissue maska: {tissue_path.name}")
    print(f"  Celotna velikost: {image_height} × {image_width}")
    print(f"  Crop velikost: {crop_height} × {crop_width}")
    print(
        f"  Trenutni crop: rows "
        f"{CURRENT_ROW_START}:{CURRENT_ROW_END}, cols "
        f"{CURRENT_COL_START}:{CURRENT_COL_END}"
    )

    class_masks = []

    for class_name in CLASSES:
        mask = read_mask(
            CLASS_DIR / f"class_{class_name}.png"
        )

        if mask.shape != tissue_mask.shape:
            raise ValueError(
                f"Maska razreda {class_name} ima obliko "
                f"{mask.shape}, tissue maska pa "
                f"{tissue_mask.shape}."
            )

        class_masks.append(mask)

    # Enaka prioriteta kot pri izdelavi HDF5:
    # poznejši razred prepiše prejšnjega ob morebitnem prekrivanju.
    class_map = np.full(
        tissue_mask.shape,
        -1,
        dtype=np.int8,
    )

    for class_index, mask in enumerate(class_masks):
        class_map[mask] = class_index

    final_class_masks = [
        class_map == class_index
        for class_index in range(len(CLASSES))
    ]

    annotated_mask = class_map != -1
    valid_mask = tissue_mask | annotated_mask

    full_class_counts = np.asarray(
        [
            np.count_nonzero(mask)
            for mask in final_class_masks
        ],
        dtype=np.int64,
    )

    full_annotated = int(full_class_counts.sum())
    full_tissue = int(np.count_nonzero(tissue_mask))
    full_valid = int(np.count_nonzero(valid_mask))

    crop_slice = (
        slice(CURRENT_ROW_START, CURRENT_ROW_END),
        slice(CURRENT_COL_START, CURRENT_COL_END),
    )

    current_class_counts = np.asarray(
        [
            np.count_nonzero(mask[crop_slice])
            for mask in final_class_masks
        ],
        dtype=np.int64,
    )

    current_annotated = int(current_class_counts.sum())
    current_tissue = int(
        np.count_nonzero(tissue_mask[crop_slice])
    )
    current_valid = int(
        np.count_nonzero(valid_mask[crop_slice])
    )

    full_distribution = proportions(full_class_counts)
    current_distribution = proportions(current_class_counts)
    current_js = js_divergence(
        current_distribution,
        full_distribution,
    )

    crop_area = crop_height * crop_width
    current_annotation_density = (
        current_annotated / crop_area
    )
    current_tissue_density = current_tissue / crop_area
    current_valid_density = current_valid / crop_area

    print("\nCELOTEN TESTNI SET")
    print(f"  Tkivni piksli: {full_tissue:,}")
    print(f"  Veljavni piksli: {full_valid:,}")
    print(f"  Anotirani piksli: {full_annotated:,}")

    print("\nTRENUTNI CROP")
    print(f"  Tkivni piksli: {current_tissue:,}")
    print(f"  Veljavni piksli: {current_valid:,}")
    print(f"  Anotirani piksli: {current_annotated:,}")
    print(
        f"  Delež vseh testnih anotacij: "
        f"{100 * current_annotated / full_annotated:.2f}%"
    )
    print(
        f"  Gostota anotacij: "
        f"{100 * current_annotation_density:.2f}% površine"
    )
    print(
        f"  Gostota tkiva: "
        f"{100 * current_tissue_density:.2f}% površine"
    )
    print(
        f"  Jensen-Shannon divergenca do celotnega seta: "
        f"{current_js:.4f}"
    )
    print("  Razredi:")

    for index, class_name in enumerate(CLASSES):
        full_count = int(full_class_counts[index])
        crop_count = int(current_class_counts[index])

        coverage = (
            100 * crop_count / full_count
            if full_count > 0
            else 0
        )

        print(
            f"    {class_name:9s}: "
            f"{crop_count:8,} / {full_count:8,} "
            f"({coverage:6.2f}% razreda), "
            f"delež v cropu "
            f"{100 * current_distribution[index]:6.2f}% "
            f"vs. celota "
            f"{100 * full_distribution[index]:6.2f}%"
        )

    # --------------------------------------------------------
    # Primerjava z vsemi možnimi cropi enake velikosti
    # --------------------------------------------------------

    print(
        f"\nPrimerjam z vsemi cropi velikosti "
        f"{crop_height} × {crop_width} pri koraku {STRIDE} ..."
    )

    tissue_integral = integral_image(tissue_mask)
    valid_integral = integral_image(valid_mask)
    class_integrals = [
        integral_image(mask)
        for mask in final_class_masks
    ]

    row_starts = make_starts(
        image_height,
        crop_height,
        STRIDE,
    )
    col_starts = make_starts(
        image_width,
        crop_width,
        STRIDE,
    )

    results = []

    for row_start in row_starts:
        row_end = row_start + crop_height

        for col_start in col_starts:
            col_end = col_start + crop_width

            counts = np.asarray(
                [
                    rectangle_sum(
                        integral,
                        row_start,
                        row_end,
                        col_start,
                        col_end,
                    )
                    for integral in class_integrals
                ],
                dtype=np.int64,
            )

            annotated = int(counts.sum())

            if annotated == 0:
                continue

            tissue = rectangle_sum(
                tissue_integral,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            valid = rectangle_sum(
                valid_integral,
                row_start,
                row_end,
                col_start,
                col_end,
            )

            distribution = proportions(counts)
            js = js_divergence(
                distribution,
                full_distribution,
            )
            represented_classes = int(
                np.count_nonzero(counts)
            )
            minimum_class_count = int(
                counts[counts > 0].min()
            )

            results.append(
                {
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "annotated": annotated,
                    "tissue": tissue,
                    "valid": valid,
                    "annotation_density": annotated / crop_area,
                    "tissue_density": tissue / crop_area,
                    "represented_classes": represented_classes,
                    "minimum_present_class_count": minimum_class_count,
                    "js_divergence": js,
                    "class_counts": counts,
                }
            )

    if not results:
        raise RuntimeError(
            "Ni bilo najdenih nobenih anotiranih cropov."
        )

    annotation_values = [
        item["annotated"]
        for item in results
    ]
    js_values = [
        item["js_divergence"]
        for item in results
    ]
    represented_values = [
        item["represented_classes"]
        for item in results
    ]

    annotation_percentile = percentile_rank(
        annotation_values,
        current_annotated,
        higher_is_better=True,
    )
    representativeness_percentile = percentile_rank(
        js_values,
        current_js,
        higher_is_better=False,
    )
    class_coverage_percentile = percentile_rank(
        represented_values,
        np.count_nonzero(current_class_counts),
        higher_is_better=True,
    )

    print("\nUVRSTITEV TRENUTNEGA CROPA")
    print(
        f"  Število primerjanih anotiranih cropov: "
        f"{len(results):,}"
    )
    print(
        f"  Po številu anotacij: "
        f"{annotation_percentile:.1f}. percentil"
    )
    print(
        f"  Po podobnosti razredne porazdelitve celoti: "
        f"{representativeness_percentile:.1f}. percentil"
    )
    print(
        f"  Po številu zastopanih razredov: "
        f"{class_coverage_percentile:.1f}. percentil"
    )
    print(
        f"  Zastopani razredi: "
        f"{np.count_nonzero(current_class_counts)}/"
        f"{len(CLASSES)}"
    )

    # Kombinirana ocena ni znanstvena metrika, ampak praktičen pregled.
    score = (
        0.40 * annotation_percentile
        + 0.45 * representativeness_percentile
        + 0.15 * class_coverage_percentile
    )

    print(
        f"  Praktična kombinirana ocena: "
        f"{score:.1f}/100"
    )

    if (
        np.count_nonzero(current_class_counts) == len(CLASSES)
        and representativeness_percentile >= 60
        and annotation_percentile >= 50
    ):
        verdict = (
            "DOBER PILOTNI CROP: vsebuje vse razrede, "
            "ima nadpovprečno količino anotacij in je "
            "razmeroma podoben celotnemu testnemu setu."
        )
    elif (
        np.count_nonzero(current_class_counts) >= 5
        and annotation_percentile >= 40
    ):
        verdict = (
            "SPREJEMLJIV PILOTNI CROP: uporaben za razvoj, "
            "vendar ni dovolj za končno poročanje rezultatov."
        )
    else:
        verdict = (
            "ŠIBEK ALI PRISTRANSKI PILOTNI CROP: za razvoj "
            "je lahko uporaben, vendar razredno ali prostorsko "
            "slabo predstavlja celoten testni set."
        )

    print(f"\nSKLEP\n  {verdict}")

    # --------------------------------------------------------
    # Najbolj reprezentativni kandidati
    # --------------------------------------------------------

    best_representative = sorted(
        results,
        key=lambda item: (
            item["js_divergence"],
            -item["represented_classes"],
            -item["annotated"],
        ),
    )[:10]

    print("\n10 NAJBOLJ REPREZENTATIVNIH CROPOV")
    for rank, item in enumerate(best_representative, start=1):
        print(
            f"  {rank:2d}. rows "
            f"{item['row_start']}:{item['row_end']}, cols "
            f"{item['col_start']}:{item['col_end']} | "
            f"anotacije={item['annotated']:,}, "
            f"razredi={item['represented_classes']}/6, "
            f"JS={item['js_divergence']:.4f}"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "row_start",
            "row_end",
            "col_start",
            "col_end",
            "annotated",
            "tissue",
            "valid",
            "annotation_density",
            "tissue_density",
            "represented_classes",
            "minimum_present_class_count",
            "js_divergence",
        ] + [
            f"count_{class_name}"
            for class_name in CLASSES
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for item in sorted(
            results,
            key=lambda value: value["js_divergence"],
        ):
            row = {
                key: item[key]
                for key in fieldnames
                if key in item
            }

            for index, class_name in enumerate(CLASSES):
                row[f"count_{class_name}"] = int(
                    item["class_counts"][index]
                )

            writer.writerow(row)

    print(f"\nCSV rezultat: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
