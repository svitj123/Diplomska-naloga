from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
from spectral.io import envi


# ============================================================
# NASTAVITVE — UČNI SKLOP, DESNI DEL
# ============================================================

DATA_DIR = Path("/home/sjesenk/mayerich2-učni")

HEADER_FILE = DATA_DIR / "br1003-br2085b.hdr"
CLASS_DIR = DATA_DIR / "supervised-class"

OUTPUT_FILE = Path(
    "/home/sjesenk/local/train_right_expanded.hdf5"
)

CLASSES = [
    "coll",
    "epith",
    "fibro",
    "lymph",
    "myo",
    "necrosis",
]

# Desni del celotnega učnega sklopa.
ROW_START = 0
ROW_END = 3400

COL_START = 3200
COL_END = 6800

SPECTRAL_STEP = 2
CHUNK_ROWS = 25
HDF5_CHUNK_ROWS = 25
HDF5_CHUNK_COLS = 50


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours} h {minutes} min {seconds} s"
    if minutes > 0:
        return f"{minutes} min {seconds} s"
    return f"{seconds} s"


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Maska ne obstaja: {path}")

    with Image.open(path) as image:
        return np.asarray(image) > 0


def print_progress(
    processed_rows: int,
    total_rows: int,
    start_time: float,
) -> None:
    elapsed = time.perf_counter() - start_time
    fraction = processed_rows / total_rows
    percent = fraction * 100
    rows_per_second = processed_rows / elapsed if elapsed > 0 else 0
    remaining_rows = total_rows - processed_rows
    eta_seconds = (
        remaining_rows / rows_per_second
        if rows_per_second > 0
        else 0
    )

    bar_width = 30
    filled = int(bar_width * fraction)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(
        f"\r[{bar}] "
        f"{percent:6.2f}% | "
        f"{processed_rows:4d}/{total_rows} vrstic | "
        f"pretečeno: {format_duration(elapsed)} | "
        f"ETA: {format_duration(eta_seconds)}",
        end="",
        flush=True,
    )


def main() -> None:
    total_start_time = time.perf_counter()

    if not HEADER_FILE.exists():
        raise FileNotFoundError(
            f"ENVI .hdr datoteka ne obstaja:\n{HEADER_FILE}"
        )

    print("Odpiram ENVI podatke ...", flush=True)

    image = envi.open(str(HEADER_FILE))
    print(f"Celotna oblika surovih podatkov: {image.shape}")

    full_height, full_width, full_band_count = image.shape

    if not (0 <= ROW_START < ROW_END <= full_height):
        raise ValueError(
            f"Neveljaven vrstični izrez: "
            f"{ROW_START}:{ROW_END}, višina je {full_height}."
        )

    if not (0 <= COL_START < COL_END <= full_width):
        raise ValueError(
            f"Neveljaven stolpčni izrez: "
            f"{COL_START}:{COL_END}, širina je {full_width}."
        )

    wavelengths_full = np.asarray(
        [float(value) for value in image.metadata["wavelength"]],
        dtype=np.float32,
    )

    if len(wavelengths_full) != full_band_count:
        raise ValueError(
            "Število valovnih števil se ne ujema "
            "s številom spektralnih kanalov."
        )

    wavelengths = wavelengths_full[::SPECTRAL_STEP]

    output_height = ROW_END - ROW_START
    output_width = COL_END - COL_START
    output_band_count = len(wavelengths)

    print("\nNastavitve izhoda:")
    print(f"  Prostorski izrez: {output_height} × {output_width}")
    print(f"  Spektralni kanali: {output_band_count}")
    print(
        f"  Spektralni razpon: "
        f"{wavelengths[0]:.1f}–{wavelengths[-1]:.1f} cm^-1"
    )

    if len(wavelengths) > 1:
        print(
            f"  Korak valovnih števil: "
            f"{wavelengths[1] - wavelengths[0]:.1f} cm^-1"
        )

    uncompressed_gib = (
        output_height
        * output_width
        * output_band_count
        * np.dtype(np.float32).itemsize
        / 1024**3
    )
    print(
        f"  Nestisnjena velikost data: "
        f"približno {uncompressed_gib:.2f} GiB"
    )

    print("\nNalagam masko tkiva in razrede ...", flush=True)

    full_tissue_mask = read_mask(CLASS_DIR / "mask.png")
    tissue_mask = full_tissue_mask[
        ROW_START:ROW_END,
        COL_START:COL_END,
    ]

    if tissue_mask.shape != (output_height, output_width):
        raise ValueError(
            f"Nepričakovana oblika tissue_mask: "
            f"{tissue_mask.shape}"
        )

    class_map = np.full(tissue_mask.shape, -1, dtype=np.int8)

    for class_index, class_name in enumerate(CLASSES):
        full_class_mask = read_mask(
            CLASS_DIR / f"class_{class_name}.png"
        )
        class_mask = full_class_mask[
            ROW_START:ROW_END,
            COL_START:COL_END,
        ]

        if class_mask.shape != tissue_mask.shape:
            raise ValueError(
                f"Maska razreda {class_name} ima napačno "
                f"obliko: {class_mask.shape}"
            )

        class_map[class_mask] = class_index
        print(
            f"  {class_name}: "
            f"{np.count_nonzero(class_map == class_index):,}"
        )

    tissue_count = int(np.count_nonzero(tissue_mask))
    annotated_count = int(np.count_nonzero(class_map != -1))
    invalid_annotations = int(
        np.count_nonzero((class_map != -1) & (~tissue_mask))
    )

    print(f"  Tkivni piksli: {tissue_count:,}")
    print(f"  Anotirani piksli: {annotated_count:,}")

    if invalid_annotations:
        print(
            "OPOZORILO: "
            f"{invalid_annotations:,} anotiranih pikslov "
            "ni znotraj tissue_mask."
        )
        print(
            "Ti piksli bodo vseeno ohranjeni z valid_mask."
        )

    valid_mask = tissue_mask | (class_map != -1)

    print(
        f"  Veljavni piksli za zapis: "
        f"{np.count_nonzero(valid_mask):,}"
    )

    print("\nOdpiram podatkovni memmap ...", flush=True)
    data_memmap = image.open_memmap(interleave="bip")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(
            f"\nIzhodna datoteka že obstaja in bo "
            f"prepisana:\n{OUTPUT_FILE}"
        )
        OUTPUT_FILE.unlink()

    amide_index = int(np.abs(wavelengths - 1650).argmin())
    amide_wavenumber = float(wavelengths[amide_index])

    print(
        f"\nNajbližji kanal Amide I: "
        f"{amide_wavenumber:.1f} cm^-1"
    )

    amide_values_parts = []

    print("\nZačenjam branje in zapisovanje po blokih:")
    progress_start_time = time.perf_counter()

    with h5py.File(OUTPUT_FILE, "w") as output:
        data_dataset = output.create_dataset(
            "data",
            shape=(
                output_height,
                output_width,
                output_band_count,
            ),
            dtype="float32",
            chunks=(
                min(HDF5_CHUNK_ROWS, output_height),
                min(HDF5_CHUNK_COLS, output_width),
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
        output.attrs["row_start"] = ROW_START
        output.attrs["row_end"] = ROW_END
        output.attrs["col_start"] = COL_START
        output.attrs["col_end"] = COL_END
        output.attrs["spectral_step"] = SPECTRAL_STEP
        output.attrs["amide_i_wavenumber"] = amide_wavenumber
        output.attrs["valid_mask_definition"] = (
            "tissue_mask OR classes != -1"
        )
        output.attrs["annotated_outside_tissue_mask"] = (
            invalid_annotations
        )

        for local_row_start in range(
            0,
            output_height,
            CHUNK_ROWS,
        ):
            local_row_end = min(
                local_row_start + CHUNK_ROWS,
                output_height,
            )

            source_row_start = ROW_START + local_row_start
            source_row_end = ROW_START + local_row_end

            block = np.array(
                data_memmap[
                    source_row_start:source_row_end,
                    COL_START:COL_END,
                    ::SPECTRAL_STEP,
                ],
                dtype=np.float32,
                copy=True,
            )

            expected_shape = (
                local_row_end - local_row_start,
                output_width,
                output_band_count,
            )

            if block.shape != expected_shape:
                raise ValueError(
                    f"Blok ima napačno obliko "
                    f"{block.shape}, pričakovano "
                    f"{expected_shape}."
                )

            block_valid_mask = valid_mask[
                local_row_start:local_row_end,
                :,
            ]

            block_amide = block[
                :,
                :,
                amide_index,
            ][block_valid_mask]

            if block_amide.size:
                amide_values_parts.append(block_amide.copy())

            block[~block_valid_mask] = 0

            data_dataset[
                local_row_start:local_row_end,
                :,
                :,
            ] = block

            print_progress(
                processed_rows=local_row_end,
                total_rows=output_height,
                start_time=progress_start_time,
            )

    print()

    if amide_values_parts:
        amide_values = np.concatenate(amide_values_parts)
        finite_amide_values = amide_values[
            np.isfinite(amide_values)
        ]

        if finite_amide_values.size:
            print("\nAmide I statistika pred normalizacijo:")
            print(f"  kanal: {amide_wavenumber:.1f} cm^-1")
            print(
                f"  min: "
                f"{np.min(finite_amide_values):.6f}"
            )
            print(
                f"  mediana: "
                f"{np.median(finite_amide_values):.6f}"
            )
            print(
                f"  povprečje: "
                f"{np.mean(finite_amide_values):.6f}"
            )
            print(
                f"  max: "
                f"{np.max(finite_amide_values):.6f}"
            )

    total_elapsed = time.perf_counter() - total_start_time
    file_size_gib = OUTPUT_FILE.stat().st_size / 1024**3

    print("\nKončano.")
    print(f"Izhodna datoteka: {OUTPUT_FILE}")
    print(
        f"Oblika data: "
        f"({output_height}, {output_width}, {output_band_count})"
    )
    print(f"Velikost datoteke: {file_size_gib:.2f} GiB")
    print(f"Skupni čas: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
