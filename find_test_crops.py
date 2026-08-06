"""
Najdi vse ne-prazne (anotirane) crop-e na CELOTNEM test slajdu (mayerich-testni,
brc961-br1001, 2800x6800) — analogno find_train_crops.py za train slajd.

Trenutno je bil iz tega slajda izluscen SAMO EN pilotni crop (rows 500:1300,
cols 500:1700, rocno izbran zaradi dobre razredne uravnotezenosti) — to
pomeni da je bilo doslej izkoriscenih samo 86,993 od cca. 439,704 razpolozljivih
anotiranih pikslov (~20%). Ta skripta pripravi seznam VSEH ne-prekrivajocih se
800x1200 kock z anotacijami, da jih create_test_crops.py lahko vse izlusci in
predprocesira — enako kot je bilo narejeno za train (11 crop-ov).

Zagnati na strezniku (o1.biolab.si), kjer zivijo surovi podatki:
  python3 find_test_crops.py
"""

from pathlib import Path

import numpy as np
from PIL import Image


DATA_DIR = Path("/home/sjesenk/mayerich-testni")
CLASS_DIR = DATA_DIR / "supervised-class"

IMAGE_HEIGHT = 2800
IMAGE_WIDTH = 6800

CROP_HEIGHT = 800
CROP_WIDTH = 1200

# Isti vrstni red kot pri train (create_pilot_train_crops.py) — kasnejsi
# razred prepise prejsnjega ob prekrivanju.
CLASSES = [
    "coll",
    "epith",
    "fibro",
    "lymph",
    "myo",
    "necrosis",
]


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Maska ne obstaja: {path}")
    with Image.open(path) as image:
        return np.asarray(image) > 0


def make_starts(total_size: int, crop_size: int) -> list:
    """Ne-prekrivajoca mreza, zadnja kocka se lahko manjsa (ostanek)."""
    starts = list(range(0, total_size, crop_size))
    return starts


def main() -> None:
    tissue_mask = read_mask(CLASS_DIR / "tissue_mask.png")

    if tissue_mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            f"Nepricakovana oblika tissue maske: {tissue_mask.shape}, "
            f"pricakovano ({IMAGE_HEIGHT}, {IMAGE_WIDTH})."
        )

    class_masks_raw = {
        class_name: read_mask(CLASS_DIR / f"class_{class_name}.png")
        for class_name in CLASSES
    }
    for class_name, mask in class_masks_raw.items():
        if mask.shape != tissue_mask.shape:
            raise ValueError(
                f"Maska razreda {class_name} ima obliko {mask.shape}, "
                f"tissue maska pa {tissue_mask.shape}."
            )

    # Enaka prioriteta kot pri izdelavi HDF5 (create_pilot_train_crops.py):
    # kasnejsi razred v seznamu prepise prejsnjega ob prekrivanju.
    class_map = np.full(tissue_mask.shape, -1, dtype=np.int8)
    for class_index, class_name in enumerate(CLASSES):
        class_map[class_masks_raw[class_name]] = class_index

    row_starts = make_starts(IMAGE_HEIGHT, CROP_HEIGHT)
    col_starts = make_starts(IMAGE_WIDTH, CROP_WIDTH)

    selected_crops = []
    crop_index = 1

    for row_start in row_starts:
        row_end = min(row_start + CROP_HEIGHT, IMAGE_HEIGHT)
        for col_start in col_starts:
            col_end = min(col_start + CROP_WIDTH, IMAGE_WIDTH)

            region = class_map[row_start:row_end, col_start:col_end]
            annotated = region != -1
            total_annotated = int(annotated.sum())

            if total_annotated == 0:
                continue

            class_counts = {}
            for class_index, class_name in enumerate(CLASSES):
                class_counts[class_name] = int((region == class_index).sum())

            tissue_count = int(
                tissue_mask[row_start:row_end, col_start:col_end].sum()
            )

            selected_crops.append(
                {
                    "index": crop_index,
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "height": row_end - row_start,
                    "width": col_end - col_start,
                    "tissue_count": tissue_count,
                    "total_annotated": total_annotated,
                    "class_counts": class_counts,
                }
            )
            crop_index += 1

    print(
        f"Najdenih nepodvojenih test crop-ov z anotacijami: "
        f"{len(selected_crops)}"
    )
    print()

    for crop in selected_crops:
        print(
            f"CROP {crop['index']:02d}: "
            f"rows {crop['row_start']}:{crop['row_end']}, "
            f"cols {crop['col_start']}:{crop['col_end']}"
        )
        print(f"  velikost: {crop['height']} x {crop['width']}")
        print(f"  tissue: {crop['tissue_count']:,}")
        print(f"  anotacije skupaj: {crop['total_annotated']:,}")
        for class_name in CLASSES:
            count = crop["class_counts"][class_name]
            if count > 0:
                print(f"  {class_name}: {count:,}")
        print()

    print("POVZETEK PO RAZREDIH:")
    total_all = 0
    for class_name in CLASSES:
        total = sum(crop["class_counts"][class_name] for crop in selected_crops)
        total_all += total
        print(f"  {class_name}: {total:,}")
    print()
    print(f"Vsota vseh anotacij: {total_all:,}")

    # Primerjava z obstojecim pilotnim cropom (rows 500:1300, cols 500:1700)
    pilot_region = class_map[500:1300, 500:1700]
    pilot_annotated = int((pilot_region != -1).sum())
    print(f"\nZa primerjavo — obstojeci PILOTNI crop (500:1300, 500:1700): "
          f"{pilot_annotated:,} anotiranih")
    print(f"Nova mreza pokrije: {total_all:,} anotiranih "
          f"({100*total_all/max(pilot_annotated,1):.1f}x vec)")


if __name__ == "__main__":
    main()
