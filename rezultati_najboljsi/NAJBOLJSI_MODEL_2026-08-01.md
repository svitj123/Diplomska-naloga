# Najboljši model doslej — modelC_crossSlide_faithful (2026-08-01)

## Rezultat

| | OA | log-loss |
|---|---|---|
| Inner-val (zdruzen) | 77.91% | 0.55092 |
| **TEST (koncni, 22 crop-ov, 439,704 pikslov)** | **70.82%** | **0.99422** |
| Clanek CNN (referenca) | 79.45% ± 1.25 | — |
| Clanek SVM (referenca) | 56.41% | — |

Per-class (TEST):

| Razred | N | OA | Log-loss |
|---|---|---|---|
| R0 (collagen) | 81,546 | 94.00% | 0.321 |
| R1 (epithelium) | 85,984 | 77.01% | 1.031 |
| R2 (fibro) | 39,201 | 7.84% | 2.434 |
| R3 (lymph) | 17,545 | 30.82% | 1.506 |
| R4 (myo/myofibroblasti) | 161,974 | 68.77% | 1.095 |
| R5 (necrosis) | 53,454 | 91.01% | 0.433 |

Temperature (per-class): [0.988, 1.222, 10.000, 10.000, 1.152, 0.553]
Sigma: 0.0

## Skripta

`modelC_crossSlide_faithful_BEST_70.82pct.py` (v tej mapi je samo dokumentacija;
zamrznjena kopija skripte je v korenu repozitorija) — identicna
`modelC_crossSlide_faithful.py` v stanju na dan 2026-08-01, PRED zacetkom
eksperimentov z vecjim patch-em (#1 naslednji korak: R2/R3 izboljsava).

## Natancen ukaz, ki je dal ta rezultat

Pognano na o1.biolab.si (TITAN X GPU), v tmux seji `cnn_final`:

```bash
cd ~/diploma
export PATH=$HOME/.local/bin:$PATH
python3 -u modelC_crossSlide_faithful.py \
  --train-dir FTIR-data/train_preprocessed \
  --test-dir FTIR-data/test_preprocessed_full \
  --cache-dir FTIR-data/_cache \
  --n-ensemble 12 --calib-ensemble 12 \
  --balance-strategy weights \
  --per-class-temperature \
  --num-workers 4 \
  --output modelC_faithful_full_final.npy \
  2>&1 | tee -a cnn_final_run.log
```

(Opomba: ker je bil podan `--test-dir`, se je izhod dejansko shranil kot
`modelC_faithful_full_final.npz`, ne `.npy` — glej kodo v `main()`.)

## Podatki

- **Train**: 11 crop-ov iz `FTIR-data/train_preprocessed/` (br1003-br2085b, 152,316 pikslov)
- **Test**: 22 crop-ov iz `FTIR-data/test_preprocessed_full/` (brc961-br1001,
  439,704 pikslov — CEL test slajd, ne le star pilotni crop 86,993 pikslov)

## Konfiguracija arhitekture (privzeto v skripti na ta dan)

- SingleStreamCNN: Conv(32,3x3) -> MaxPool -> Conv(64,3x3) -> Conv(64,3x3) -> MaxPool
  -> FC(128) -> Softmax(6)
- Brez BatchNorm, brez LRN (LRN popravek je se vedno na TODO listi)
- Softplus aktivacija, dropout 0.5, init N(0,0.02)
- Patch 17x17x16 (PCA), BREZ extra-smooth-scale (privzeto izklopljeno)
- Adam optimizer, fiksnih 8 epoh, batch_size 128
- Balance strategy: weights (soften=0.5)
- Per-class temperature scaling
- Brez rotate-inner-val (en kandidat: train_crop_08)

## Shranjene datoteke v tej mapi

- `modelC_faithful_full_final.npz` — napovedne verjetnostne karte za vseh 22 test crop-ov
- Ta datoteka (dokumentacija)

## Zakaj shranjeno

Preden gremo v naslednji korak (vecji patch za izboljsavo R2/R3), da imamo
zanesljivo referencno tocko za primerjavo in moznost obnovitve, ce naslednji
poskusi ne bi uspeli.
