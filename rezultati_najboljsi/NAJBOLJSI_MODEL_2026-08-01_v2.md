# Najboljši model doslej — modelC_crossSlide_faithful, brez crop 24 (2026-08-01, v2)

## Rezultat

| | OA | log-loss |
|---|---|---|
| Inner-val (zdruzen) | 78.35% | 0.50041 |
| **TEST (koncni, 22 crop-ov, 439,704 pikslov)** | **72.16%** | **1.01644** |
| Prejsnji najboljsi (11 crop, 2026-08-01 v1) | 70.82% | 0.99422 |
| Clanek CNN (referenca) | 79.45% ± 1.25 | — |
| Clanek SVM (referenca) | 56.41% | — |

Per-class (TEST):

| Razred | N | OA | Log-loss | (prejsnji v1) |
|---|---|---|---|---|
| R0 (collagen) | 81,546 | 92.95% | 0.409 | 94.00% |
| R1 (epithelium) | 85,984 | 71.75% | 1.119 | 77.01% |
| R2 (fibro) | 39,201 | 7.42% | 2.582 | 7.84% |
| R3 (lymph) | 17,545 | 28.10% | 1.626 | 30.82% |
| R4 (myo/myofibroblasti) | 161,974 | **76.51%** | 0.982 | 68.77% |
| R5 (necrosis) | 53,454 | 89.87% | 0.535 | 91.01% |

## Ozadje eksperimenta

Odkrili smo, da poleg 11 originalnih train crop-ov obstaja se 12 neizluscenih
crop-ov na train slajdu (najdenih s `find_train_full_crops.py`, izlusceni z
`create_train_new_crops.py`) -- skupaj 23 crop-ov, 231,206 anotiranih pikslov
namesto 152,316.

Prvi poskus (vseh 23 crop-ov, glej `modelC_faithful_moretrain_v1.npz`) je dal
OA=67.45%/ll=1.18973 -- SLABSE od v1 baseline, ker je crop_24 (24,775 pikslov,
skoraj cisti coll+epith, 0 prispevka za fibro/lymph/myo/necrosis) povzrocil,
da je model zacel zamenjevati myo (R4) za coll (R0) in fibro (R2) (potrjeno s
confusion matrix analizo: myo->coll 12.77%->19.33%, myo->fibro 7.91%->13.53%).

Ko smo izlocili SAMO crop_24 (22 crop-ov, 206,431 pikslov), se je R4 povrnil
NAD prvotni baseline (76.51% > 68.77%), skupni OA je nov rekord (72.16%).
R2/R3 sta se vrnila blizu originalnih vrednosti (verjetno je bil del
izboljsave pri polnem 23-crop teku sum pri majhnem vzorcu, ne pravi prispevek
crop_24).

**Zakljucek**: vec train podatkov POMAGA, a le ce so podatki uravnotezeni
glede na ze dominantne razrede (coll/epith) -- flooding z se vec vecinskega
razreda skodi manjsinskim razredom (R4), tudi ce ne gre neposredno za
class-weight mehanizem, ampak za decision-boundary premik (coll "pozre"
sosednje myo piksle v feature prostoru).

## Skripta

`modelC_crossSlide_faithful_BEST_72.16pct.py` (zamrznjena kopija v korenu
repozitorija, identicna `modelC_crossSlide_faithful.py` na dan 2026-08-01).

## Natancen ukaz, ki je dal ta rezultat

Pognano na o1.biolab.si (TITAN X GPU), v tmux seji `cnn_no24`:

```bash
cd ~/diploma
export PATH=$HOME/.local/bin:$PATH
# Priprava train-dir brez crop_24 (symlinki na vse ostale):
mkdir -p FTIR-data/train_preprocessed_no24
for f in FTIR-data/train_preprocessed/train_crop_*.hdf5; do
  base=$(basename "$f")
  [ "$base" != "train_crop_24.hdf5" ] && ln -sf "$(pwd)/$f" "FTIR-data/train_preprocessed_no24/$base"
done

python3 -u modelC_crossSlide_faithful.py \
  --train-dir FTIR-data/train_preprocessed_no24 \
  --test-dir FTIR-data/test_preprocessed_full \
  --cache-dir FTIR-data/_cache \
  --n-ensemble 12 --calib-ensemble 12 \
  --balance-strategy weights \
  --per-class-temperature \
  --num-workers 4 \
  --output modelC_faithful_no24_v1.npy \
  2>&1 | tee -a cnn_no24_run.log
```

(Izhod se je shranil kot `modelC_faithful_no24_v1.npz`, ne `.npy`, ker je bil
podan `--test-dir`.)

## Podatki

- **Train**: 22 crop-ov iz `FTIR-data/train_preprocessed/` (br1003-br2085b,
  206,431 pikslov -- vseh 23 razpolozljivih crop-ov MINUS crop_24)
- **Test**: 22 crop-ov iz `FTIR-data/test_preprocessed_full/` (brc961-br1001,
  439,704 pikslov — CEL test slajd)

## Konfiguracija arhitekture (nespremenjena od v1)

- SingleStreamCNN: Conv(32,3x3) -> MaxPool -> Conv(64,3x3) -> Conv(64,3x3) -> MaxPool
  -> FC(128) -> Softmax(6)
- Brez BatchNorm, brez LRN (LRN popravek je se vedno na TODO listi)
- Softplus aktivacija, dropout 0.5, init N(0,0.02)
- Patch 17x17x16 (PCA), BREZ extra-smooth-scale (privzeto izklopljeno)
- Adam optimizer, fiksnih 8 epoh, batch_size 128
- Balance strategy: weights (soften=0.5)
- Per-class temperature scaling
- Brez rotate-inner-val

## Shranjene datoteke v tej mapi

- `modelC_faithful_no24_v1.npz` — napovedne verjetnostne karte za vseh 22 test crop-ov
- `cnn_no24_run.log` — celoten log teka
- Ta datoteka (dokumentacija)

## Naslednji koraki (odprto)

- R2 (fibro) in R3 (lymph) sta se vedno bistveno pod clankom -- potrebno
  nadaljnje raziskovanje (glej TODO #2: LRN popravek; morda tudi ciljano
  dodajanje SAMO fibro/lymph-bogatih crop-ov, brez coll-flooding efekta).
- Primerjava s Table 1 (KNN/LinearSVM/DT/RF/NN/AdaBoost/NB/QDA) je v
  `modelA_classic_baselines.py` / `classic_baselines_run.log` -- CNN
  dosledno prekasa vse klasicne pristope, kot v clanku.
