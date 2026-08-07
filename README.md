# Diplomska naloga — FTIR klasifikacija tkiva

Replikacija in nadgradnja članka Berisha et al. (2018), "Deep learning for
FTIR histology" — klasifikacija 6 histoloških razredov tkiva (kolagen,
epitelij, fibroblasti, limfociti, miofibroblasti, nekroza) iz FTIR
hiperspektralnih slik, na pravem cross-slide train/test razdelku iz članka
(train: BR1003, BR2085b; test: BR961, BR1001 — fizično ločeni tkivni rezini).

> **Za pisanje diplome**: celoten pregled (podatki, metodologija, vsi rezultati,
> ugotovitve, metodološka opozorila, ponovljivost) je v
> [`DOKUMENTACIJA_ZA_DIPLOMO.md`](DOKUMENTACIJA_ZA_DIPLOMO.md).

## Struktura repozitorija

**Trenutni pipeline (cross-slide train/test split, glavno delo):**
- `modelA_crossSlide.py` — RBF SVM (PCA16), člankuzvest baseline
- `modelA_classic_baselines.py` — 8 dodatnih klasičnih klasifikatorjev (Table 1 replikacija)
- `modelB_crossSlide.py` — spektralni 1D CNN (brez PCA, brez prostorske informacije)
- `modelC_crossSlide_v1.py`, `modelC_crossSlide_v2.py` — zgodnje/vmesne CNN različice (dual-stream, ablacije)
- `modelC_crossSlide_faithful.py` — glavni, člankuzvest prostorski CNN (+ zamrznjene `_BEST_*.py` kopije najboljših rezultatov)
- `modelD_transformer_crossSlide.py`, `modelD_transformer_hybrid.py` — Vision Transformer nadgradnja (presega članek)
- `create_test_crops.py`, `create_train_new_crops.py`, `find_test_crops.py`, `find_train_full_crops.py` — priprava podatkov
- `rezultati_report.txt` — kronološki dnevnik VSEH eksperimentov (OA/log-loss/konfiguracija)
- `rezultati_najboljsi/` — kurirana dokumentacija najboljših modelov (natančen ukaz, rezultati, razlaga)

**`archiv/`** — zgodnejše, presežene faze dela (ohranjeno zaradi sledljivosti):
- `archiv/prvi-testi/` — čisto prvi poskusi
- `archiv/core2-scenarij/` — stari leave-one-core-out validacijski scenarij (pred prehodom na pravi cross-slide split)

**Podatki**: `FTIR-data/` (predprocesirani train/test crop-i) ni v repozitoriju
(prevelik, ~11GB, avtorski podatki članka) — glej `zapiski.md` za izvor podatkov.

## Namestitev

1. **Klonirajte repozitorij**

   ```bash
   git clone <URL_DO_REPO> Diplomska-naloga
   cd Diplomska-naloga
   ```

2. **Ustvarite in aktivirajte virtualno okolje**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Namestite odvisnosti**

   ```bash
   python -m ensurepip --upgrade
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Zaženite glavni model** (potrebuje `FTIR-data/`, glej zgoraj)

   ```bash
   python modelC_crossSlide_faithful.py \
     --train-dir FTIR-data/train_preprocessed_no24 \
     --test-dir FTIR-data/test_preprocessed_full \
     --use-lrn --extra-smooth-scale 3
   ```

## Opomba

Če ukaz `pip` ne deluje pravilno ali kaže na napačno Python okolje, vedno uporabite:

```bash
python -m pip
```

## Reševanje težav

**`pip: command not found` (pyenv okolja)**

Če pyenv prestreže ukaz `pip`, uporabite:
```bash
python -m pip install -r requirements.txt
```

**`ModuleNotFoundError` kljub nameščenemu modulu**

Če sklearn ali drug paket javlja manjkajoč modul, čeprav je nameščen, je virtualno okolje verjetno poškodovano. Rešitev — popolna ponovna namestitev:
```bash
deactivate
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
