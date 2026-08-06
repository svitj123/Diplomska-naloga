from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
from spectral.io import envi


# ============================================================
# NASTAVITVE
# ============================================================

DATA_DIR = Path("/home/sjesenk/mayerich2-učni")
HEADER_FILE = DATA_DIR / "br1003-br2085b.hdr"
CLASS_DIR = DATA_DIR / "supervised-class"

OUTPUT_DIR = Path("/home/sjesenk/local/train_crops_pilot_raw")

CLASSES = [
    "coll",
    "epith",
    "fibro",
    "lymph",
    "myo",
    "necrosis",
]

CROP_HEIGHT = 800
CROP_WIDTH = 1200

SPECTRAL_STEP = 2
CHUNK_ROWS = 25
HDF5_CHUNK_ROWS = 25
HDF5_CHUNK_COLS = 50

# Pilotni nepodvajajoči se učni cropi.
# Izbor skupaj pokrije vseh šest razredov in različne dele slajda.
# Desna cropa sta široka 800 px in bosta dopolnjena do 1200 px.
CROPS = [
    (1, 0, 800, 0, 1200),
    (3, 0, 800, 2400, 3600),
    (8, 800, 1600, 1200, 2400),
    (9, 800, 1600, 2400, 3600),
    (12, 800, 1600, 6000, 6800),
    (14, 1600, 2400, 1200, 2400),
    (17, 1600, 2400, 6000, 6800),
    (20, 2400, 3200, 2400, 3600),
]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} h {minutes} min {seconds} s"
    if minutes:
        return f"{minutes} min {seconds} s"
    return f"{seconds} s"


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Maska ne obstaja: {path}")

    with Image.open(path) as image:
        return np.asarray(image) > 0


def print_progress(
    crop_index: int,
    processed_rows: int,
    total_rows: int,
    start_time: float,
) -> None:
    elapsed = time.perf_counter() - start_time
    fraction = processed_rows / total_rows
    percent = 100 * fraction
    speed = processed_rows / elapsed if elapsed > 0 else 0
    eta = (
        (total_rows - processed_rows) / speed
        if speed > 0
        else 0
    )

    bar_width = 30
    filled = int(bar_width * fraction)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(
        f"\rCROP {crop_index:02d} [{bar}] "
        f"{percent:6.2f}% | "
        f"{processed_rows:4d}/{total_rows} vrstic | "
        f"pretečeno: {format_duration(elapsed)} | "
        f"ETA: {format_duration(eta)}",
        end="",
        flush=True,
    )


def build_class_map(
    class_masks,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
):
    actual_height = row_end - row_start
    actual_width = col_end - col_start

    class_map = np.full(
        (CROP_HEIGHT, CROP_WIDTH),
        -1,
        dtype=np.int8,
    )

    class_counts = {}

    for class_index, class_name in enumerate(CLASSES):
        class_mask = class_masks[class_name][
            row_start:row_end,
            col_start:col_end,
        ]

        class_map[
            :actual_height,
            :actual_width,
        ][class_mask] = class_index

        class_counts[class_name] = int(
            np.count_nonzero(class_mask)
        )

    return class_map, class_counts


def main() -> None:
    total_start = time.perf_counter()

    if not HEADER_FILE.exists():
        raise FileNotFoundError(
            f"ENVI .hdr datoteka ne obstaja:\n{HEADER_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Odpiram ENVI podatke ...", flush=True)
    image = envi.open(str(HEADER_FILE))
    print(f"Celotna oblika: {image.shape}")

    full_height, full_width, full_band_count = image.shape

    wavelengths_full = np.asarray(
        [float(value) for value in image.metadata["wavelength"]],
        dtype=np.float32,
    )

    if len(wavelengths_full) != full_band_count:
        raise ValueError(
            "Število valovnih števil se ne ujema "
            "s številom kanalov."
        )

    wavelengths = wavelengths_full[::SPECTRAL_STEP]
    output_band_count = len(wavelengths)

    amide_index = int(np.abs(wavelengths - 1650).argmin())
    amide_wavenumber = float(wavelengths[amide_index])

    print(
        f"Spektralni kanali: {output_band_count}, "
        f"razpon: {wavelengths[0]:.1f}–"
        f"{wavelengths[-1]:.1f} cm^-1"
    )
    print(
        f"Najbližji Amide I kanal: "
        f"{amide_wavenumber:.1f} cm^-1"
    )

    print("\nNalagam maske ...", flush=True)
    full_tissue_mask = read_mask(CLASS_DIR / "mask.png")

    if full_tissue_mask.shape != (full_height, full_width):
        raise ValueError(
            f"Tissue maska ima obliko "
            f"{full_tissue_mask.shape}, podatki pa "
            f"({full_height}, {full_width})."
        )

    class_masks = {
        class_name: read_mask(
            CLASS_DIR / f"class_{class_name}.png"
        )
        for class_name in CLASSES
    }

    for class_name, mask in class_masks.items():
        if mask.shape != full_tissue_mask.shape:
            raise ValueError(
                f"Maska {class_name} ima napačno obliko: "
                f"{mask.shape}"
            )

    print("Odpiram podatkovni memmap ...", flush=True)
    data_memmap = image.open_memmap(interleave="bip")

    completed = 0

    for (
        crop_index,
        row_start,
        row_end,
        col_start,
        col_end,
    ) in CROPS:
        if not (
            0 <= row_start < row_end <= full_height
            and 0 <= col_start < col_end <= full_width
        ):
            raise ValueError(
                f"Neveljaven CROP {crop_index:02d}: "
                f"rows {row_start}:{row_end}, "
                f"cols {col_start}:{col_end}"
            )

        actual_height = row_end - row_start
        actual_width = col_end - col_start

        output_file = (
            OUTPUT_DIR / f"train_crop_{crop_index:02d}.hdf5"
        )

        # Omogoča nadaljevanje po prekinitvi.
        if output_file.exists():
            print(
                f"\nCROP {crop_index:02d} že obstaja, "
                f"preskakujem: {output_file}"
            )
            completed += 1
            continue

        print(
            f"\nCROP {crop_index:02d}: "
            f"rows {row_start}:{row_end}, "
            f"cols {col_start}:{col_end}"
        )
        print(
            f"  dejanska velikost: "
            f"{actual_height} × {actual_width}"
        )
        print(
            f"  HDF5 velikost: "
            f"{CROP_HEIGHT} × {CROP_WIDTH} × "
            f"{output_band_count}"
        )

        tissue_mask = np.zeros(
            (CROP_HEIGHT, CROP_WIDTH),
            dtype=bool,
        )
        tissue_mask[
            :actual_height,
            :actual_width,
        ] = full_tissue_mask[
            row_start:row_end,
            col_start:col_end,
        ]

        class_map, class_counts = build_class_map(
            class_masks,
            row_start,
            row_end,
            col_start,
            col_end,
        )

        valid_mask = tissue_mask | (class_map != -1)

        tissue_count = int(np.count_nonzero(tissue_mask))
        annotated_count = int(np.count_nonzero(class_map != -1))
        outside_count = int(
            np.count_nonzero(
                (class_map != -1) & (~tissue_mask)
            )
        )

        for class_name in CLASSES:
            print(
                f"  {class_name}: "
                f"{class_counts[class_name]:,}"
            )

        print(f"  Tkivni piksli: {tissue_count:,}")
        print(f"  Anotirani piksli: {annotated_count:,}")

        if outside_count:
            print(
                f"  Anotirani zunaj tissue_mask: "
                f"{outside_count:,} "
                f"(ohranjeni z valid_mask)"
            )

        temp_file = output_file.with_suffix(".hdf5.tmp")
        if temp_file.exists():
            temp_file.unlink()

        crop_start = time.perf_counter()
        amide_values_parts = []

        try:
            with h5py.File(temp_file, "w") as output:
                data_dataset = output.create_dataset(
                    "data",
                    shape=(
                        CROP_HEIGHT,
                        CROP_WIDTH,
                        output_band_count,
                    ),
                    dtype="float32",
                    chunks=(
                        HDF5_CHUNK_ROWS,
                        HDF5_CHUNK_COLS,
                        output_band_count,
                    ),
                    compression="lzf",
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

                output.attrs["source_header"] = str(HEADER_FILE)
                output.attrs["crop_index"] = crop_index
                output.attrs["row_start"] = row_start
                output.attrs["row_end"] = row_end
                output.attrs["col_start"] = col_start
                output.attrs["col_end"] = col_end
                output.attrs["actual_height"] = actual_height
                output.attrs["actual_width"] = actual_width
                output.attrs["padded_height"] = CROP_HEIGHT
                output.attrs["padded_width"] = CROP_WIDTH
                output.attrs["spectral_step"] = SPECTRAL_STEP
                output.attrs["amide_i_wavenumber"] = (
                    amide_wavenumber
                )
                output.attrs["valid_mask_definition"] = (
                    "tissue_mask OR classes != -1"
                )
                output.attrs[
                    "annotated_outside_tissue_mask"
                ] = outside_count

                for local_row_start in range(
                    0,
                    actual_height,
                    CHUNK_ROWS,
                ):
                    local_row_end = min(
                        local_row_start + CHUNK_ROWS,
                        actual_height,
                    )

                    source_row_start = (
                        row_start + local_row_start
                    )
                    source_row_end = (
                        row_start + local_row_end
                    )

                    block = np.array(
                        data_memmap[
                            source_row_start:source_row_end,
                            col_start:col_end,
                            ::SPECTRAL_STEP,
                        ],
                        dtype=np.float32,
                        copy=True,
                    )

                    expected_shape = (
                        local_row_end - local_row_start,
                        actual_width,
                        output_band_count,
                    )

                    if block.shape != expected_shape:
                        raise ValueError(
                            f"CROP {crop_index:02d}: "
                            f"blok ima obliko {block.shape}, "
                            f"pričakovano {expected_shape}."
                        )

                    block_valid_mask = valid_mask[
                        local_row_start:local_row_end,
                        :actual_width,
                    ]

                    block_amide = block[
                        :,
                        :,
                        amide_index,
                    ][block_valid_mask]

                    if block_amide.size:
                        amide_values_parts.append(
                            block_amide.copy()
                        )

                    block[~block_valid_mask] = 0

                    data_dataset[
                        local_row_start:local_row_end,
                        :actual_width,
                        :,
                    ] = block

                    print_progress(
                        crop_index=crop_index,
                        processed_rows=local_row_end,
                        total_rows=actual_height,
                        start_time=crop_start,
                    )

            # Datoteko preimenujemo šele po uspešnem koncu.
            temp_file.replace(output_file)

        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

        print()

        if amide_values_parts:
            amide_values = np.concatenate(
                amide_values_parts
            )
            finite_values = amide_values[
                np.isfinite(amide_values)
            ]

            if finite_values.size:
                print(
                    f"  Amide I mediana: "
                    f"{np.median(finite_values):.6f}"
                )
                print(
                    f"  Amide I povprečje: "
                    f"{np.mean(finite_values):.6f}"
                )

        size_gib = output_file.stat().st_size / 1024**3
        elapsed = time.perf_counter() - crop_start

        print(f"  Zapisano: {output_file}")
        print(f"  Velikost: {size_gib:.2f} GiB")
        print(f"  Čas: {format_duration(elapsed)}")

        completed += 1

    total_elapsed = time.perf_counter() - total_start

    print("\nVSI CROPI KONČANI.")
    print(f"Končanih datotek: {completed}/{len(CROPS)}")
    print(f"Izhodna mapa: {OUTPUT_DIR}")
    print(f"Skupni čas: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
