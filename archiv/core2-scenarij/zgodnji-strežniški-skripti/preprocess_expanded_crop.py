from pathlib import Path
import time

import h5py
import numpy as np


# ============================================================
# NASTAVITVE
# ============================================================

INPUT_FILE = Path(
    "/home/sjesenk/local/test_expanded_crop.hdf5"
)

OUTPUT_FILE = Path(
    "/home/sjesenk/local/test_expanded_crop_preprocessed.hdf5"
)

AMIDE_I_TARGET_WN = 1650.0

# Koliko vrstic slike se obdela naenkrat.
# 10 vrstic pomeni največ približno 10 × 1200 spektrov v enem bloku.
ROW_CHUNK = 10

# Enaka vrednost kot v modelA_v4.py in modelB_v3.py.
EPS = 1e-6


# ============================================================
# RUBBER-BAND BASELINE — ENAKA LOGIKA KOT V MODELIH
# ============================================================

def rubberband_single(spectrum: np.ndarray) -> np.ndarray:
    """
    Rubber-band baseline correction z uporabo spodnje konveksne
    ovojnice spektra in linearne interpolacije med njenimi točkami.
    """
    n = len(spectrum)
    x = np.arange(n, dtype=np.float64)
    y = spectrum.astype(np.float64, copy=False)

    lower = []

    for i in range(n):
        point = (x[i], y[i])

        while len(lower) >= 2:
            origin = lower[-2]
            previous = lower[-1]

            cross = (
                (previous[0] - origin[0])
                * (point[1] - origin[1])
                - (previous[1] - origin[1])
                * (point[0] - origin[0])
            )

            if cross <= 0:
                lower.pop()
            else:
                break

        lower.append(point)

    lower_x = np.asarray(
        [point[0] for point in lower],
        dtype=np.float64,
    )
    lower_y = np.asarray(
        [point[1] for point in lower],
        dtype=np.float64,
    )

    baseline = np.interp(x, lower_x, lower_y)

    return (y - baseline).astype(np.float32)


def rubberband_baseline_correction(
    spectra: np.ndarray,
) -> np.ndarray:
    """
    Izvede baseline correction za matriko oblike:
    (število_spektrov, število_kanalov).
    """
    output = np.empty_like(spectra, dtype=np.float32)

    for index in range(len(spectra)):
        output[index] = rubberband_single(spectra[index])

    return output


def amide_i_normalize(
    spectra: np.ndarray,
    amide_i_index: int,
    eps: float = EPS,
):
    """
    Vsak spekter deli z njegovo baseline-korigirano vrednostjo
    pri Amide I.
    """
    amide_values = spectra[:, amide_i_index].astype(
        np.float64
    )

    invalid_mask = (
        (~np.isfinite(amide_values))
        | (amide_values <= eps)
    )

    safe_values = np.where(
        invalid_mask,
        eps,
        amide_values,
    )

    normalized = (
        spectra / safe_values[:, np.newaxis]
    ).astype(np.float32)

    return normalized, invalid_mask


# ============================================================
# IZPIS NAPREDKA
# ============================================================

def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} h {minutes} min {seconds} s"

    if minutes:
        return f"{minutes} min {seconds} s"

    return f"{seconds} s"


def print_progress(
    completed_rows: int,
    total_rows: int,
    start_time: float,
    processed_spectra: int,
) -> None:
    elapsed = time.perf_counter() - start_time
    fraction = completed_rows / total_rows
    percentage = 100 * fraction

    if completed_rows > 0:
        estimated_total = elapsed / fraction
        eta = estimated_total - elapsed
    else:
        eta = 0

    width = 30
    filled = int(width * fraction)
    bar = "█" * filled + "░" * (width - filled)

    rate = (
        processed_spectra / elapsed
        if elapsed > 0
        else 0
    )

    print(
        f"\r[{bar}] "
        f"{percentage:6.2f}% | "
        f"{completed_rows}/{total_rows} vrstic | "
        f"{processed_spectra:,} spektrov | "
        f"{rate:,.0f} spektrov/s | "
        f"ETA: {format_duration(eta)}",
        end="",
        flush=True,
    )


# ============================================================
# GLAVNI POSTOPEK
# ============================================================

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Vhodna datoteka ne obstaja: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Izhod že obstaja: {OUTPUT_FILE}\n"
            "Preimenuj ali izbriši ga pred ponovnim zagonom."
        )

    total_start = time.perf_counter()

    with h5py.File(INPUT_FILE, "r") as source:
        required = {
            "data",
            "wns",
            "classes",
            "tissue_mask",
        }

        missing = required - set(source.keys())

        if missing:
            raise KeyError(
                f"V HDF5 manjkajo dataseti: {sorted(missing)}"
            )

        source_data = source["data"]
        wns = np.asarray(
            source["wns"],
            dtype=np.float32,
        )
        classes = np.asarray(
            source["classes"],
            dtype=np.int8,
        )
        tissue_mask = np.asarray(
            source["tissue_mask"],
            dtype=bool,
        )

        height, width, channel_count = source_data.shape

        if classes.shape != (height, width):
            raise ValueError(
                "Oblika classes se ne ujema z data."
            )

        if tissue_mask.shape != (height, width):
            raise ValueError(
                "Oblika tissue_mask se ne ujema z data."
            )

        if len(wns) != channel_count:
            raise ValueError(
                "Število wns se ne ujema s kanali data."
            )

        amide_i_index = int(
            np.abs(wns - AMIDE_I_TARGET_WN).argmin()
        )
        amide_i_wavenumber = float(
            wns[amide_i_index]
        )

        print("Vhod:", INPUT_FILE)
        print("Izhod:", OUTPUT_FILE)
        print("Oblika data:", source_data.shape)
        print(
            "Tkivni piksli:",
            f"{np.count_nonzero(tissue_mask):,}",
        )
        print(
            "Amide I:",
            f"{amide_i_wavenumber:.1f} cm^-1",
            f"(indeks {amide_i_index})",
        )
        print(
            "Predobdelava: rubber-band baseline "
            "+ Amide I normalizacija"
        )

        with h5py.File(OUTPUT_FILE, "w") as target:
            output_data = target.create_dataset(
                "data",
                shape=source_data.shape,
                dtype="float32",
                chunks=source_data.chunks
                or (
                    min(ROW_CHUNK, height),
                    min(50, width),
                    channel_count,
                ),
                compression="lzf",
            )

            target.create_dataset(
                "wns",
                data=wns,
                dtype="float32",
            )

            target.create_dataset(
                "classes",
                data=classes,
                dtype="int8",
                compression="lzf",
            )

            target.create_dataset(
                "tissue_mask",
                data=tissue_mask,
                dtype="bool",
                compression="lzf",
            )

            # Kopiramo obstoječe atribute.
            for key, value in source.attrs.items():
                target.attrs[key] = value

            # Dodamo podatke o predobdelavi.
            target.attrs["baseline_correction"] = (
                "lower-convex-hull rubber-band"
            )
            target.attrs["amide_i_normalized"] = True
            target.attrs["amide_i_target_wavenumber"] = (
                AMIDE_I_TARGET_WN
            )
            target.attrs["amide_i_actual_wavenumber"] = (
                amide_i_wavenumber
            )
            target.attrs["preprocessing_source"] = (
                "modelA_v4.py / modelB_v3.py"
            )

            process_start = time.perf_counter()

            processed_spectra = 0
            invalid_amide_total = 0

            normalized_amide_min = np.inf
            normalized_amide_max = -np.inf
            normalized_amide_sum = 0.0
            normalized_amide_count = 0

            for row_start in range(
                0,
                height,
                ROW_CHUNK,
            ):
                row_end = min(
                    row_start + ROW_CHUNK,
                    height,
                )

                block = np.asarray(
                    source_data[
                        row_start:row_end,
                        :,
                        :,
                    ],
                    dtype=np.float32,
                )

                block_mask = tissue_mask[
                    row_start:row_end,
                    :,
                ]

                output_block = np.zeros_like(
                    block,
                    dtype=np.float32,
                )

                # Obdelujemo samo tkivne piksle.
                spectra = block[block_mask]

                if spectra.size:
                    corrected = (
                        rubberband_baseline_correction(
                            spectra
                        )
                    )

                    normalized, invalid_mask = (
                        amide_i_normalize(
                            corrected,
                            amide_i_index,
                        )
                    )

                    invalid_count = int(
                        np.count_nonzero(invalid_mask)
                    )
                    invalid_amide_total += invalid_count

                    # Neveljavne spektre nastavimo na 0.
                    # Tako se izognemo ekstremnim vrednostim zaradi
                    # deljenja z EPS.
                    if invalid_count:
                        normalized[invalid_mask] = 0

                    output_block[block_mask] = normalized

                    valid_normalized = normalized[
                        ~invalid_mask,
                        amide_i_index,
                    ]

                    if valid_normalized.size:
                        normalized_amide_min = min(
                            normalized_amide_min,
                            float(valid_normalized.min()),
                        )
                        normalized_amide_max = max(
                            normalized_amide_max,
                            float(valid_normalized.max()),
                        )
                        normalized_amide_sum += float(
                            valid_normalized.sum(
                                dtype=np.float64
                            )
                        )
                        normalized_amide_count += (
                            valid_normalized.size
                        )

                    processed_spectra += len(spectra)

                output_data[
                    row_start:row_end,
                    :,
                    :,
                ] = output_block

                print_progress(
                    completed_rows=row_end,
                    total_rows=height,
                    start_time=process_start,
                    processed_spectra=processed_spectra,
                )

            print()

            target.attrs[
                "invalid_amide_i_spectra"
            ] = invalid_amide_total

    elapsed = time.perf_counter() - total_start
    output_size_gib = (
        OUTPUT_FILE.stat().st_size / 1024**3
    )

    print("\nKončano.")
    print("Izhod:", OUTPUT_FILE)
    print(
        "Velikost:",
        f"{output_size_gib:.2f} GiB",
    )
    print(
        "Neveljavni Amide I spektri:",
        f"{invalid_amide_total:,}",
    )

    if normalized_amide_count:
        normalized_amide_mean = (
            normalized_amide_sum
            / normalized_amide_count
        )

        print("\nAmide I po normalizaciji:")
        print(
            "  min:",
            f"{normalized_amide_min:.8f}",
        )
        print(
            "  povprečje:",
            f"{normalized_amide_mean:.8f}",
        )
        print(
            "  max:",
            f"{normalized_amide_max:.8f}",
        )

    print(
        "Skupni čas:",
        format_duration(elapsed),
    )


if __name__ == "__main__":
    main()