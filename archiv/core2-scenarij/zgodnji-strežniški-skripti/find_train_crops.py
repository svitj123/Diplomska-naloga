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


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Maska ne obstaja: {path}"
        )

    with Image.open(path) as image:
        return np.asarray(image) > 0


# Mreža brez prekrivanja.
# Zadnjih 200 vrstic ne vsebuje anotacij.
ROW_STARTS = [
    0,
    800,
    1600,
    2400,
]

# Zadnji crop je širok 800 pikslov: 6000:6800.
COL_STARTS = [
    0,
    1200,
    2400,
    3600,
    4800,
    6000,
]


def main() -> None:
    tissue_mask = read_mask(
        CLASS_DIR / "mask.png"
    )

    if tissue_mask.shape != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ):
        raise ValueError(
            f"Nepričakovana oblika tissue maske: "
            f"{tissue_mask.shape}, pričakovano "
            f"({IMAGE_HEIGHT}, {IMAGE_WIDTH})."
        )

    class_masks = {
        class_name: read_mask(
            CLASS_DIR / f"class_{class_name}.png"
        )
        for class_name in CLASSES
    }

    for class_name, class_mask in class_masks.items():
        if class_mask.shape != tissue_mask.shape:
            raise ValueError(
                f"Maska razreda {class_name} ima obliko "
                f"{class_mask.shape}, tissue maska pa "
                f"{tissue_mask.shape}."
            )

    selected_crops = []
    crop_index = 1

    for row_start in ROW_STARTS:
        row_end = min(
            row_start + CROP_HEIGHT,
            IMAGE_HEIGHT,
        )

        for col_start in COL_STARTS:
            col_end = min(
                col_start + CROP_WIDTH,
                IMAGE_WIDTH,
            )

            class_counts = {}
            total_annotated = 0

            for class_name in CLASSES:
                count = int(
                    np.count_nonzero(
                        class_masks[class_name][
                            row_start:row_end,
                            col_start:col_end,
                        ]
                    )
                )

                class_counts[class_name] = count
                total_annotated += count

            if total_annotated == 0:
                continue

            tissue_count = int(
                np.count_nonzero(
                    tissue_mask[
                        row_start:row_end,
                        col_start:col_end,
                    ]
                )
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
        f"Najdenih nepodvojenih učnih cropov "
        f"z anotacijami: {len(selected_crops)}"
    )
    print()

    for crop in selected_crops:
        print(
            f"CROP {crop['index']:02d}: "
            f"rows {crop['row_start']}:{crop['row_end']}, "
            f"cols {crop['col_start']}:{crop['col_end']}"
        )

        print(
            f"  velikost: "
            f"{crop['height']} × {crop['width']}"
        )

        print(
            f"  tissue: "
            f"{crop['tissue_count']:,}"
        )

        print(
            f"  anotacije skupaj: "
            f"{crop['total_annotated']:,}"
        )

        for class_name in CLASSES:
            count = crop["class_counts"][class_name]

            if count > 0:
                print(
                    f"  {class_name}: {count:,}"
                )

        print()

    print("POVZETEK PO RAZREDIH:")

    total_all = 0

    for class_name in CLASSES:
        total = sum(
            crop["class_counts"][class_name]
            for crop in selected_crops
        )

        total_all += total

        print(
            f"  {class_name}: {total:,}"
        )

    print()
    print(
        f"Vsota vseh anotacij: {total_all:,}"
    )


if __name__ == "__main__":
    main()