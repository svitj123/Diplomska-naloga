# Najboljši model doslej — modelC_crossSlide_faithful, LRN + smooth-scale=3 (2026-08-02, v3)

## Rezultat

| | OA | log-loss |
|---|---|---|
| Inner-val (zdruzen) | 78.32% | 0.51268 |
| **TEST (koncni, 22 crop-ov, 439,704 pikslov)** | **72.90%** | **1.01135** |
| Prejsnji najboljsi (v2, brez LRN/smooth) | 72.16% | 1.01644 |
| Clanek CNN (referenca) | 79.45% ± 1.25 | — |

Per-class (TEST):

| Razred | N | OA | Log-loss | (v2 brez LRN/smooth) |
|---|---|---|---|---|
| R0 (collagen) | 81,546 | 90.46% | 0.558 | 92.95% |
| R1 (epithelium) | 85,984 | 68.60% | 1.270 | 71.75% |
| R2 (fibro) | 39,201 | 8.15% | 2.405 | 7.42% |
| R3 (lymph) | 17,545 | 28.05% | 1.699 | 28.10% |
| R4 (myo) | 161,974 | **81.02%** | 0.858 | 76.51% |
| R5 (necrosis) | 53,454 | 90.60% | 0.503 | 89.87% |

## Kaj je bilo dodano

- **LRN** (Local Response Normalization) po Conv32 in po drugem Conv64, pred MaxPool
  — clanek Fig. 3 kaze, da SD arhitektura izpusti BN, a OBDRZI LRN. Doslej nisva
  tega implementirala (Task #2 na seznamu). `--use-lrn` CLI flag.
- **`--extra-smooth-scale 3`** — dodatni uniform_filter-glajeni PCA kanali
  (scale=3) kot dodatne vhodne znacilke, konkatenirane na osnovni patch.
  Predhodno testirano na starih 11-crop podatkih (pomagalo R3 brez skode R4),
  zdaj ponovljeno na novih 22-crop (brez crop_24) podatkih.

## Kljucna ugotovitev

R4 (myo) se je izboljsal (76.51%->81.02%, nov najboljsi), R0/R1 sta rahlo
padla (majhen kompromis), **R2/R3 pa se NISTA premaknila** (7.42%->8.15%,
28.10%->28.05% -- v mejah suma). To je zdaj ze CETRTI neodvisen poskus
(vec train podatkov, weight-soften, LRN, smooth-scale), ki R2/R3 NI uspel
premakniti iz cone ~7-8%/~28%. Mocna indikacija, da gre za temeljno
prekrivanje v PCA16 feature prostoru med coll-fibro-myo (in lymph-myo)
kontinuumom vezivnega tkiva, ne za pomanjkanje podatkov/utezi/regularizacije.

## Odlocitev (2026-08-02)

Uporabnik se je odlocil ZAKLJUCITI CNN razvojno smer pri tem rezultatu
(72.90%/1.01135) -- nadaljnji fokus gre na transformer vejo naloge.

## Skripta

`modelC_crossSlide_faithful_BEST_72.90pct.py` (zamrznjena kopija v korenu
repozitorija).

## Natancen ukaz

```bash
cd ~/diploma
export PATH=$HOME/.local/bin:$PATH
python3 -u modelC_crossSlide_faithful.py \
  --train-dir FTIR-data/train_preprocessed_no24 \
  --test-dir FTIR-data/test_preprocessed_full \
  --cache-dir FTIR-data/_cache \
  --n-ensemble 12 --calib-ensemble 12 \
  --balance-strategy weights --per-class-temperature \
  --use-lrn --extra-smooth-scale 3 \
  --num-workers 4 \
  --output modelC_faithful_no24_lrn_smooth3.npy \
  2>&1 | tee -a cnn_lrn_smooth3_run.log
```

## Podatki

- Train: 22 crop-ov iz `FTIR-data/train_preprocessed_no24/` (206,431 pikslov)
- Test: 22 crop-ov iz `FTIR-data/test_preprocessed_full/` (439,704 pikslov)

## Shranjene datoteke v tej mapi

- `modelC_faithful_no24_lrn_smooth3.npz` — napovedne verjetnostne karte
- `cnn_lrn_smooth3_run.log` — celoten log teka
