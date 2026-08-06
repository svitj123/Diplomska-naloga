from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from spectral.io import envi


HEADER_FILE = Path("/home/sjesenk/mayerich/brc961-br1001.hdr")
CLASS_DIR = Path("/home/sjesenk/mayerich/supervised-class")
OUTPUT_FILE = Path("/home/sjesenk/local/test_crop_full_spectrum.hdf5")

CLASSES = ["coll", "epith", "fibro", "lymph", "myo", "necrosis"]

ROW_START = 750
ROW_END = 950
COL_START = 1050
COL_END = 1250


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def main() -> None:
    print("Odpiram ENVI podatke ...")
    image = envi.open(str(HEADER_FILE))

    print("Celotna oblika:", image.shape)

    wavelengths = np.asarray(
        [float(value) for value in image.metadata["wavelength"]],
        dtype=np.float32,
    )

    print("Število kanalov:", len(wavelengths))
    print("Prvo valovno število:", wavelengths[0])
    print("Zadnje valovno število:", wavelengths[-1])

    data_memmap = image.open_memmap(interleave="bip")

    crop = np.array(
        data_memmap[
            ROW_START:ROW_END,
            COL_START:COL_END,
            :,
        ],
        dtype=np.float32,
        copy=True,
    )

    tissue_mask = read_mask(
        CLASS_DIR / "tissue_mask.png"
    )[ROW_START:ROW_END, COL_START:COL_END]

    class_map = np.full(tissue_mask.shape, -1, dtype=np.int8)

    for class_index, class_name in enumerate(CLASSES):
        mask_path = CLASS_DIR / f"class_{class_name}.png"
        class_mask = read_mask(mask_path)[
            ROW_START:ROW_END,
            COL_START:COL_END,
        ]
        class_map[class_mask] = class_index

        print(
            f"{class_name}: "
            f"{np.count_nonzero(class_map == class_index)}"
        )

    crop[~tissue_mask] = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(OUTPUT_FILE, "w") as output:
        output.create_dataset(
            "data",
            data=crop,
            dtype="float32",
            compression="lzf",
            chunks=(50, 50, crop.shape[2]),
        )
        output.create_dataset(
            "wns",
            data=wavelengths,
            dtype="float32",
        )
        output.create_dataset(
            "tissue_mask",
            data=tissue_mask,
            dtype="bool",
            compression="lzf",
        )
        output.create_dataset(
            "classes",
            data=class_map,
            dtype="int8",
            compression="lzf",
        )

    print("Končano.")
    print("Izhod:", OUTPUT_FILE)
    print("Oblika podatkov:", crop.shape)


if __name__ == "__main__":
    main()