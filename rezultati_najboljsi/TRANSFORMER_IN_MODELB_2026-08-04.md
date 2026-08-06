# Transformer (najboljši doslej) in modelB (spektralni CNN, baseline)

## Transformer — hibridni CNN-ViT, patch=15 (najboljši od štirih poskusov)

| | Inner-val OA | TEST OA | log-loss |
|---|---|---|---|
| Osnovni ViT (192/8/6, patch=33) | 78.82% | 52.82% | 1.45889 |
| Večja kapaciteta (256/10/8, patch=33) | 78.35% | 52.08% | 1.50821 |
| Hibridni CNN-ViT (192/8/6+stem, patch=33) | 79.43% | 48.92% | 1.59373 |
| **Hibridni CNN-ViT, ozje receptivno polje (192/8/6+stem, patch=15)** | 77.14% | **58.05%** | **1.39631** |

Ozje receptivno polje (patch 15x15 namesto 33x33) je zmanjšalo vrzel med
inner-val in TEST (30.5pp -> 19.1pp) in dalo najboljši transformer rezultat
doslej — potrjuje hipotezo, da širok kontekst ViT-a lovi slajd-specifične
vzorce namesto splošnih tkivnih znacilnosti. Se vedno precej pod CNN-jem
(72.90%).

Skripta: `modelD_transformer_hybrid.py --patch-size 15 --token-size 3`
Shranjeno: `modelD_hybrid_patch15.npz`, `transformer_hybrid_patch15_run.log`

## modelB — spektralni 1D CNN (brez PCA, brez prostorske informacije)

Baseline rezultat (pred izboljšavami weights/label-smoothing, ki so v teku):
TEST OA=50.45%, ll=1.44460. Članek (Table 8, SD): SVM=56.41%,
CNN-spektralno=62.52%, CNN-prostorsko=79.45%.

Per-class: R4 (myofibroblasti) kolabira na 3.83% brez prostorske informacije
(vs. 81.02% pri CNN s prostorskim kontekstom) — cista ilustracija, zakaj
prostorska informacija pomaga prav pri tem razredu.

Skripta: `modelB_crossSlide.py` (privzeto zdaj `--balance-strategy weights`)
Shranjeno: `modelB_crossSlide_final.npy`, `modelB_crossSlide_run.log`

Izboljšana verzija (`--balance-strategy weights --label-smoothing 0.1`) je
bila v teku ob pisanju te dokumentacije — glej `rezultati_report.txt` za
najnovejši vnos.
