from pathlib import Path

import h5py
import numpy as np


INPUT_DIR = Path("/home/sjesenk/local/train_crops_pilot_raw")

EXPECTED_FILES = [
    "train_crop_01.hdf5",
    "train_crop_03.hdf5",
    "train_crop_08.hdf5",
    "train_crop_09.hdf5",
    "train_crop_12.hdf5",
    "train_crop_14.hdf5",
    "train_crop_17.hdf5",
    "train_crop_20.hdf5",
]

CLASS_NAMES = [
    "coll",
    "epith",
    "fibro",
    "lymph",
    "myo",
    "necrosis",
]

EXPECTED_DATA_SHAPE = (800, 1200, 813)
EXPECTED_MAP_SHAPE = (800, 1200)
SCAN_ROWS = 25


def scan_data(dataset, valid_mask):
    nonfinite = 0
    nonzero_outside = 0

    for row_start in range(0, dataset.shape[0], SCAN_ROWS):
        row_end = min(row_start + SCAN_ROWS, dataset.shape[0])

        block = dataset[row_start:row_end, :, :]
        block_valid = valid_mask[row_start:row_end, :]

        nonfinite += int(
            block.size - np.count_nonzero(np.isfinite(block))
        )

        outside = ~block_valid
        if np.any(outside):
            outside_spectra = block[outside]
            nonzero_outside += int(
                np.count_nonzero(
                    np.any(outside_spectra != 0, axis=1)
                )
            )

    return nonfinite, nonzero_outside


def main():
    print("PREVERJANJE PILOTNIH UČNIH HDF5 DATOTEK")
    print(f"Mapa: {INPUT_DIR}\n")

    missing = [
        name for name in EXPECTED_FILES
        if not (INPUT_DIR / name).exists()
    ]

    if missing:
        print("Manjkajoče datoteke:")
        for name in missing:
            print(f"  - {name}")
        raise SystemExit(1)

    reference_wns = None
    total_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    total_annotated = 0
    all_ok = True

    for filename in EXPECTED_FILES:
        path = INPUT_DIR / filename

        print("=" * 70)
        print(filename)

        with h5py.File(path, "r") as h5:
            required = {"data", "wns", "tissue_mask", "classes"}
            missing_datasets = required - set(h5.keys())

            if missing_datasets:
                print(
                    f"  NAPAKA: manjkajo dataseti "
                    f"{sorted(missing_datasets)}"
                )
                all_ok = False
                continue

            data = h5["data"]
            wns = h5["wns"][:]
            tissue_mask = h5["tissue_mask"][:]
            classes = h5["classes"][:]

            print(f"  data: {data.shape}, {data.dtype}")
            print(f"  wns: {wns.shape}, {wns.dtype}")
            print(f"  tissue_mask: {tissue_mask.shape}")
            print(f"  classes: {classes.shape}")

            if data.shape != EXPECTED_DATA_SHAPE:
                print("  NAPAKA: napačna oblika data.")
                all_ok = False

            if tissue_mask.shape != EXPECTED_MAP_SHAPE:
                print("  NAPAKA: napačna oblika tissue_mask.")
                all_ok = False

            if classes.shape != EXPECTED_MAP_SHAPE:
                print("  NAPAKA: napačna oblika classes.")
                all_ok = False

            if reference_wns is None:
                reference_wns = wns.copy()
            elif not np.array_equal(reference_wns, wns):
                print("  NAPAKA: wns se med cropi ne ujema.")
                all_ok = False

            valid_labels = np.isin(
                classes,
                np.array([-1, 0, 1, 2, 3, 4, 5], dtype=np.int8),
            )
            if not np.all(valid_labels):
                print("  NAPAKA: najdene neveljavne oznake.")
                all_ok = False

            class_counts = np.array(
                [
                    np.count_nonzero(classes == index)
                    for index in range(len(CLASS_NAMES))
                ],
                dtype=np.int64,
            )

            annotated = int(np.count_nonzero(classes != -1))
            valid_mask = tissue_mask | (classes != -1)
            outside_annotations = int(
                np.count_nonzero(
                    (classes != -1) & (~tissue_mask)
                )
            )

            print(f"  Anotirani piksli: {annotated:,}")
            print(
                f"  Anotirani zunaj tissue_mask: "
                f"{outside_annotations:,}"
            )

            for index, class_name in enumerate(CLASS_NAMES):
                print(
                    f"    {class_name:9s}: "
                    f"{class_counts[index]:,}"
                )

            print("  Pregledujem NaN/Inf in podatke zunaj valid_mask ...")
            nonfinite, nonzero_outside = scan_data(
                data,
                valid_mask,
            )

            print(f"  NaN/Inf vrednosti: {nonfinite:,}")
            print(
                f"  Nen ničelni spektri zunaj valid_mask: "
                f"{nonzero_outside:,}"
            )

            if nonfinite != 0 or nonzero_outside != 0:
                all_ok = False

            total_counts += class_counts
            total_annotated += annotated

        print()

    print("=" * 70)
    print("SKUPNI POVZETEK")
    print(f"  Anotirani piksli: {total_annotated:,}")

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name:9s}: "
            f"{total_counts[index]:,}"
        )

    if reference_wns is not None:
        print(
            f"  Spektralni razpon: "
            f"{reference_wns[0]:.1f}–"
            f"{reference_wns[-1]:.1f} cm^-1"
        )
        print(
            f"  Spektralni korak: "
            f"{reference_wns[1] - reference_wns[0]:.1f} cm^-1"
        )

    if all_ok:
        print("\nREZULTAT: VSE DATOTEKE SO PRAVILNE.")
    else:
        print("\nREZULTAT: NAJDENE SO BILE NAPAKE.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
