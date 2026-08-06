from pathlib import Path

import numpy as np
from PIL import Image


DATA_DIR = Path("/home/sjesenk/mayerich2-učni")
CLASS_DIR = DATA_DIR / "supervised-class"

CLASSES = [
    "coll",
    "epith",
    "fibro",
    "lymph",
    "myo",
    "necrosis",
]


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def bounding_box(mask: np.ndarray):
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return None

    return {
        "row_min": int(rows.min()),
        "row_max": int(rows.max()) + 1,
        "col_min": int(cols.min()),
        "col_max": int(cols.max()) + 1,
        "count": int(mask.sum()),
    }


def main():
    tissue_mask = read_mask(CLASS_DIR / "mask.png")

    print("Oblika tissue mask:", tissue_mask.shape)
    print("Tkivni piksli:", int(tissue_mask.sum()))
    print()

    combined = np.zeros_like(tissue_mask, dtype=bool)

    for class_name in CLASSES:
        mask = read_mask(
            CLASS_DIR / f"class_{class_name}.png"
        )

        combined |= mask

        box = bounding_box(mask)

        if box is None:
            print(f"{class_name}: ni anotacij")
            continue

        outside_tissue = int(
            np.count_nonzero(mask & ~tissue_mask)
        )

        print(f"{class_name}:")
        print(f"  count: {box['count']}")
        print(
            f"  rows: {box['row_min']}:{box['row_max']}"
        )
        print(
            f"  cols: {box['col_min']}:{box['col_max']}"
        )
        print(
            f"  zunaj tissue_mask: {outside_tissue}"
        )
        print()

    all_box = bounding_box(combined)

    print("Skupni bounding box vseh 6 razredov:")
    print(
        f"  rows: {all_box['row_min']}:{all_box['row_max']}"
    )
    print(
        f"  cols: {all_box['col_min']}:{all_box['col_max']}"
    )
    print(
        f"  anotirani piksli: {all_box['count']}"
    )

    height = all_box["row_max"] - all_box["row_min"]
    width = all_box["col_max"] - all_box["col_min"]

    print(f"  velikost: {height} × {width}")


if __name__ == "__main__":
    main()