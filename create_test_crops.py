"""
Izlusci in predprocesira VSE anotirane crop-e s celotnega test slajda
(mayerich-testni, brc961-br1001) -- seznam crop-ov iz find_test_crops.py.

Zdruzuje dva koraka, ki sta bila za train locena (create_pilot_train_crops.py
+ preprocess_pilot_train_crops.py) v enega: bere neposredno iz ENVI memmap-a
po kosih (CHUNK_ROWS vrstic naenkrat) in TAKOJ aplicira rubber-band baseline
correction + Amide I normalizacijo na tkivne piksle, brez vmesnega surovega
zapisa na disk (prihrani ~68GB vmesnega prostora za 22 crop-ov).

Izhodna shema je IDENTICNA train_crop_*.hdf5 (classes/data/tissue_mask/wns),
da modelC_crossSlide_*.py skripte lahko berejo test_crop_*.hdf5 na enak nacin.

Omogoca nadaljevanje po prekinitvi (preskoci ze obstojece izhodne datoteke).

VZPOREDNA OBDELAVA: rubber-band baseline correction (ozko grlo, cist Python
po pikslu) je vzporedena cez vsa razpolozljiva CPU jedra z ProcessPoolExecutor
-- rubberband_single je klican na vsak anotiran/tkivni spekter neodvisno,
zato je "embarrassingly parallel". CHUNK_ROWS je povecan (10->100), da je
vsak paket dela dovolj velik za smiseln multiprocessing overhead.

Zagnati na strezniku (o1.biolab.si), kjer zivijo surovi podatki:
  python3 create_test_crops.py
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from spectral.io import envi

N_WORKERS = os.cpu_count() or 4

DATA_DIR = Path("/home/sjesenk/mayerich-testni")
HEADER_FILE = DATA_DIR / "brc961-br1001.hdr"
CLASS_DIR = DATA_DIR / "supervised-class"

OUTPUT_DIR = Path("/home/sjesenk/diploma/FTIR-data/test_preprocessed_full")

CLASSES = ["coll", "epith", "fibro", "lymph", "myo", "necrosis"]

CROP_HEIGHT = 800
CROP_WIDTH = 1200

# Iz find_test_crops.py — vsi ne-prazni crop-i na celotnem 2800x6800 slajdu.
CROPS = [
    (1,  0,    800,  0,    1200),
    (2,  0,    800,  1200, 2400),
    (3,  0,    800,  2400, 3600),
    (4,  0,    800,  3600, 4800),
    (5,  0,    800,  4800, 6000),
    (6,  0,    800,  6000, 6800),
    (7,  800,  1600, 0,    1200),
    (8,  800,  1600, 1200, 2400),
    (9,  800,  1600, 2400, 3600),
    (10, 800,  1600, 3600, 4800),
    (11, 800,  1600, 4800, 6000),
    (12, 800,  1600, 6000, 6800),
    (13, 1600, 2400, 0,    1200),
    (14, 1600, 2400, 1200, 2400),
    (15, 1600, 2400, 2400, 3600),
    (16, 1600, 2400, 3600, 4800),
    (17, 1600, 2400, 4800, 6000),
    (18, 2400, 2800, 0,    1200),
    (19, 2400, 2800, 1200, 2400),
    (20, 2400, 2800, 2400, 3600),
    (21, 2400, 2800, 3600, 4800),
    (22, 2400, 2800, 4800, 6000),
]

SPECTRAL_STEP = 2
CHUNK_ROWS = 100
HDF5_CHUNK_ROWS = 25
HDF5_CHUNK_COLS = 50
AMIDE_I_TARGET_WN = 1650.0
EPS = 1e-6


def read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Maska ne obstaja: {path}")
    with Image.open(path) as image:
        return np.asarray(image) > 0


def rubberband_single(spectrum: np.ndarray) -> np.ndarray:
    n = len(spectrum)
    x = np.arange(n, dtype=np.float64)
    y = spectrum.astype(np.float64, copy=False)
    lower = []
    for i in range(n):
        point = (x[i], y[i])
        while len(lower) >= 2:
            origin, previous = lower[-2], lower[-1]
            cross = ((previous[0] - origin[0]) * (point[1] - origin[1])
                     - (previous[1] - origin[1]) * (point[0] - origin[0]))
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(point)
    lower_x = np.asarray([p[0] for p in lower], dtype=np.float64)
    lower_y = np.asarray([p[1] for p in lower], dtype=np.float64)
    baseline = np.interp(x, lower_x, lower_y)
    return (y - baseline).astype(np.float32)


def rubberband_baseline_correction(spectra: np.ndarray) -> np.ndarray:
    """Zaporedna razlicica -- obdrzana za majhne kose / referenco."""
    output = np.empty_like(spectra, dtype=np.float32)
    for index in range(len(spectra)):
        output[index] = rubberband_single(spectra[index])
    return output


def rubberband_baseline_correction_parallel(spectra: np.ndarray, executor) -> np.ndarray:
    """Vzporedna razlicica -- rubberband_single je neodvisen na vsak spekter
    (embarrassingly parallel), zato ga porazdelimo cez vsa CPU jedra."""
    n = len(spectra)
    if n == 0:
        return np.empty_like(spectra, dtype=np.float32)
    chunksize = max(1, n // (N_WORKERS * 4))
    results = list(executor.map(rubberband_single, spectra, chunksize=chunksize))
    return np.stack(results).astype(np.float32)


def amide_i_normalize(spectra, amide_i_index, eps=EPS):
    amide_values = spectra[:, amide_i_index].astype(np.float64)
    invalid_mask = (~np.isfinite(amide_values)) | (amide_values <= eps)
    safe_values = np.where(invalid_mask, eps, amide_values)
    normalized = (spectra / safe_values[:, np.newaxis]).astype(np.float32)
    return normalized, invalid_mask


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h {m}min {s}s"
    if m: return f"{m}min {s}s"
    return f"{s}s"


def main() -> None:
    if not HEADER_FILE.exists():
        raise FileNotFoundError(f"ENVI .hdr ne obstaja: {HEADER_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Odpiram ENVI podatke ...", flush=True)
    image = envi.open(str(HEADER_FILE))
    full_height, full_width, full_band_count = image.shape
    print(f"Celotna oblika: {image.shape}")

    wavelengths_full = np.asarray(
        [float(v) for v in image.metadata["wavelength"]], dtype=np.float32)
    wavelengths = wavelengths_full[::SPECTRAL_STEP]
    output_band_count = len(wavelengths)
    amide_index = int(np.abs(wavelengths - AMIDE_I_TARGET_WN).argmin())
    amide_wn = float(wavelengths[amide_index])
    print(f"Spektralnih kanalov: {output_band_count}, "
          f"Amide I: {amide_wn:.1f} cm^-1 (idx={amide_index})")

    print("Nalagam maske ...", flush=True)
    full_tissue_mask = read_mask(CLASS_DIR / "tissue_mask.png")
    if full_tissue_mask.shape != (full_height, full_width):
        raise ValueError(f"tissue_mask oblika {full_tissue_mask.shape} != "
                         f"({full_height},{full_width})")
    class_masks = {c: read_mask(CLASS_DIR / f"class_{c}.png") for c in CLASSES}
    for c, m in class_masks.items():
        if m.shape != full_tissue_mask.shape:
            raise ValueError(f"Maska {c} ima napacno obliko: {m.shape}")

    print("Odpiram podatkovni memmap ...", flush=True)
    data_memmap = image.open_memmap(interleave="bip")

    print(f"Odpiram ProcessPoolExecutor ({N_WORKERS} delavcev) ...", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        _run_all_crops(image, data_memmap, wavelengths, output_band_count,
                       amide_index, amide_wn, full_tissue_mask, class_masks,
                       executor)


def _run_all_crops(image, data_memmap, wavelengths, output_band_count,
                   amide_index, amide_wn, full_tissue_mask, class_masks,
                   executor):
    total_start = time.perf_counter()

    for crop_index, row_start, row_end, col_start, col_end in CROPS:
        output_file = OUTPUT_DIR / f"test_crop_{crop_index:02d}.hdf5"
        if output_file.exists():
            print(f"\nCROP {crop_index:02d} ze obstaja, preskakujem.")
            continue

        actual_height = row_end - row_start
        actual_width = col_end - col_start
        print(f"\n=== CROP {crop_index:02d}: rows {row_start}:{row_end}, "
              f"cols {col_start}:{col_end} ({actual_height}x{actual_width}) ===")

        tissue_mask = np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=bool)
        tissue_mask[:actual_height, :actual_width] = \
            full_tissue_mask[row_start:row_end, col_start:col_end]

        class_map = np.full((CROP_HEIGHT, CROP_WIDTH), -1, dtype=np.int8)
        for class_index, class_name in enumerate(CLASSES):
            m = class_masks[class_name][row_start:row_end, col_start:col_end]
            class_map[:actual_height, :actual_width][m] = class_index

        valid_mask = tissue_mask | (class_map != -1)
        annotated_count = int(np.count_nonzero(class_map != -1))
        tissue_count = int(np.count_nonzero(tissue_mask))
        print(f"  Tkivnih: {tissue_count:,}  Anotiranih: {annotated_count:,}")

        temp_file = output_file.with_suffix(".hdf5.tmp")
        if temp_file.exists():
            temp_file.unlink()

        crop_start = time.perf_counter()
        invalid_amide_total = 0
        processed_pixels = 0

        with h5py.File(temp_file, "w") as out:
            out_data = out.create_dataset(
                "data", shape=(CROP_HEIGHT, CROP_WIDTH, output_band_count),
                dtype="float32",
                chunks=(HDF5_CHUNK_ROWS, HDF5_CHUNK_COLS, output_band_count),
                compression="lzf")
            out.create_dataset("wns", data=wavelengths, dtype="float32")
            out.create_dataset("tissue_mask", data=tissue_mask, dtype="bool",
                              compression="lzf")
            out.create_dataset("classes", data=class_map, dtype="int8",
                              compression="lzf")
            out.attrs["source_header"] = str(HEADER_FILE)
            out.attrs["crop_index"] = crop_index
            out.attrs["row_start"] = row_start
            out.attrs["row_end"] = row_end
            out.attrs["col_start"] = col_start
            out.attrs["col_end"] = col_end
            out.attrs["actual_height"] = actual_height
            out.attrs["actual_width"] = actual_width
            out.attrs["padded_height"] = CROP_HEIGHT
            out.attrs["padded_width"] = CROP_WIDTH
            out.attrs["spectral_step"] = SPECTRAL_STEP
            out.attrs["baseline_correction"] = "lower-convex-hull rubber-band"
            out.attrs["amide_i_normalized"] = True
            out.attrs["amide_i_target_wavenumber"] = AMIDE_I_TARGET_WN
            out.attrs["amide_i_actual_wavenumber"] = amide_wn
            out.attrs["valid_mask_definition"] = "tissue_mask OR classes != -1"
            out.attrs["preprocessing_source"] = "create_test_crops.py (zdruzen ekstrakcija+preprocesiranje)"

            for local_r0 in range(0, actual_height, CHUNK_ROWS):
                local_r1 = min(local_r0 + CHUNK_ROWS, actual_height)
                src_r0, src_r1 = row_start + local_r0, row_start + local_r1

                block_raw = np.array(
                    data_memmap[src_r0:src_r1, col_start:col_end, ::SPECTRAL_STEP],
                    dtype=np.float32, copy=True)

                block_valid = valid_mask[local_r0:local_r1, :actual_width]
                out_block = np.zeros(
                    (local_r1 - local_r0, CROP_WIDTH, output_band_count),
                    dtype=np.float32)

                spectra = block_raw[block_valid]
                if spectra.size:
                    corrected = rubberband_baseline_correction_parallel(spectra, executor)
                    normalized, invalid_mask = amide_i_normalize(corrected, amide_index)
                    n_invalid = int(np.count_nonzero(invalid_mask))
                    invalid_amide_total += n_invalid
                    if n_invalid:
                        normalized[invalid_mask] = 0
                    out_block[:, :actual_width, :][block_valid] = normalized
                    processed_pixels += len(spectra)

                out_data[local_r0:local_r1, :, :] = out_block

                elapsed = time.perf_counter() - crop_start
                frac = local_r1 / actual_height
                eta = elapsed / frac - elapsed if frac > 0 else 0
                print(f"\r  [{'#'*int(30*frac):30s}] {100*frac:5.1f}%  "
                      f"{processed_pixels:,} pikslov  "
                      f"pretečeno={format_duration(elapsed)}  "
                      f"ETA={format_duration(eta)}", end="", flush=True)

            out.attrs["invalid_amide_i_spectra"] = invalid_amide_total

        print()
        temp_file.rename(output_file)
        print(f"  Shranjeno: {output_file}  ({time.perf_counter()-crop_start:.1f}s)")

    print(f"\nVSI CROP-I KONCANI v {format_duration(time.perf_counter()-total_start)}")


if __name__ == "__main__":
    main()
