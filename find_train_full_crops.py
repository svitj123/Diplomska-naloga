"""
Najdi vse ne-prazne (anotirane) crop-e na CELOTNEM train slajdu (mayerich2-ucni,
br1003-br2085b, 3400x6800) — analogno find_test_crops.py za test slajd.

Obstojecih 11 train_crop_*.hdf5 pokrije samo 152,316 od 204,743 razpolozljivih
anotiranih pikslov na celem slajdu (74.4%). Prvih 8 (01,03,08,09,12,14,17,20)
je iz standardne 800x1200 mreze; zadnji trije (21,22,23) so bili dodani rocno,
niso poravnani na mrezo.

Ta skripta naredi standarden mrezni pregled (kot find_test_crops.py) cez CEL
slajd in izloci celice, ki se TOCNO ujemajo z ze obstojecimi 8 mreznimi crop-i
-- ostanek (vkljucno z zadnjimi 200 vrsticami, ki jih star komentar napacno
oznacuje kot "brez anotacij") je NOVA, se neizluscena vsebina.

Zagnati na strezniku (o1.biolab.si), kjer zivijo surovi podatki:
  python3 find_train_full_crops.py
"""

from pathlib import Path

import numpy as np
from PIL import Image


DATA_DIR = Path("/home/sjesenk/mayerich2-učni")
CLASS_DIR = DATA_DIR / "supervised-class"

IMAGE_HEIGHT = 3400
IMAGE_WIDTH = 6800

CROP_HEIGHT = 800
CROP_WIDTH = 1200

CLASSES = [
    "coll",
    "epith",
    "fibro",
    "lymph",
    "myo",
    "necrosis",
]

# Ze obstojeci mrezno-poravnani crop-i (iz create_pilot_train_crops.py) --
# celice, ki se TOCNO ujemajo s temi mejami, preskocimo.
EXISTING_GRID_CROPS = {
    (0, 800, 0, 1200),      # 01
    (0, 800, 2400, 3600),   # 03
    (800, 1600, 1200, 2400),  # 08
    (800, 1600, 2400, 3600),  # 09
    (800, 1600, 6000, 6800),  # 12
    (1600, 2400, 1200, 2400),  # 14
    (1600, 2400, 6000, 6800),  # 17
    (2400, 3200, 2400, 3600),  # 20
}
# Rocno dodani, ne-mrezni crop-i (21,22,23) -- obdrzimo za informacijo, se ne
# uporabljajo za deduplikacijo (prekrivanje z novo najdenimi celicami je
# sprejemljivo -- manjsa redundanca v train setu ni metodoloska tezava).
MANUAL_CROPS = {
    (1000, 1800, 0, 1200),    # 21
    (700, 1500, 4800, 6000),  # 22
    (2400, 3200, 350, 1550),  # 23
}


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Maska ne obstaja: {path}")
    with Image.open(path) as image:
        return np.asarray(image) > 0


def make_starts(total_size: int, crop_size: int) -> list:
    return list(range(0, total_size, crop_size))


def main() -> None:
    tissue_mask = read_mask(CLASS_DIR / "mask.png")

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

    class_map = np.full(tissue_mask.shape, -1, dtype=np.int8)
    for class_index, class_name in enumerate(CLASSES):
        class_map[class_masks_raw[class_name]] = class_index

    row_starts = make_starts(IMAGE_HEIGHT, CROP_HEIGHT)
    col_starts = make_starts(IMAGE_WIDTH, CROP_WIDTH)

    new_crops = []
    skipped_existing = []
    crop_index = 24  # nadaljujemo stevilcenje za 23

    for row_start in row_starts:
        row_end = min(row_start + CROP_HEIGHT, IMAGE_HEIGHT)
        for col_start in col_starts:
            col_end = min(col_start + CROP_WIDTH, IMAGE_WIDTH)

            key = (row_start, row_end, col_start, col_end)
            region = class_map[row_start:row_end, col_start:col_end]
            annotated = region != -1
            total_annotated = int(annotated.sum())

            if total_annotated == 0:
                continue

            if key in EXISTING_GRID_CROPS:
                skipped_existing.append((key, total_annotated))
                continue

            class_counts = {}
            for class_index, class_name in enumerate(CLASSES):
                class_counts[class_name] = int((region == class_index).sum())

            tissue_count = int(
                tissue_mask[row_start:row_end, col_start:col_end].sum()
            )

            new_crops.append(
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

    print(f"Preskocenih (ze obstojeci mrezni crop-i): {len(skipped_existing)}")
    for key, n in skipped_existing:
        print(f"  rows {key[0]}:{key[1]}, cols {key[2]}:{key[3]} -- {n:,} anotiranih (ze imamo)")

    print(f"\nNajdenih NOVIH crop-ov z anotacijami: {len(new_crops)}\n")

    for crop in new_crops:
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

    print("POVZETEK PO RAZREDIH (samo NOVI crop-i):")
    total_all = 0
    for class_name in CLASSES:
        total = sum(crop["class_counts"][class_name] for crop in new_crops)
        total_all += total
        print(f"  {class_name}: {total:,}")
    print()
    print(f"Vsota vseh NOVIH anotacij: {total_all:,}")
    print(f"(Obstojecih 11 crop-ov: 152,316; skupaj po dodatku: {152316+total_all:,})")


if __name__ == "__main__":
    main()
