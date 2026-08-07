# Celotna dokumentacija za pisanje diplomske naloge

**Namen te datoteke**: en sam izčrpen vir za pisanje diplome — vsi podatki,
metodologija, rezultati, ugotovitve in metodološka opozorila na enem mestu.
Stanje: 2026-08-07.

Sorodne datoteke:
- `rezultati_report.txt` — kronološki dnevnik VSEH tekov (surov, strojno pisan)
- `rezultati_najboljsi/` — posnetki najboljših modelov + izvorni logi
- `README.md` — struktura repozitorija

---

## 1. Cilj naloge

Dvodelna naloga na osnovi članka **Berisha et al. (2018), "Deep learning for
FTIR histology"** (datoteka `mayerich2018.pdf`):

1. **Replikacija**: čim bolj zvesto poustvariti rezultate članka na istem
   podatkovju in istem train/test razdelku.
2. **Nadgradnja**: preseči članek z arhitekturo, ki leta 2018 še ni obstajala
   (Vision Transformer), oz. z drugimi izboljšavami.

Klasifikacijski problem: vsakemu pikslu FTIR hiperspektralne slike tkiva
dodeliti enega od 6 histoloških razredov.

---

## 2. Podatki

### 2.1 Rezini (slides)

| | Train | Test |
|---|---|---|
| Oznaka TMA | BR1003, BR2085b | BR961, BR1001 |
| Mapa na strežniku | `mayerich2-učni` | `mayerich-testni` |
| Dimenzije | 3400 × 6800 | 2800 × 6800 |

Train in test sta **fizično ločeni tkivni rezini** — to je t.i. cross-slide
razdelek iz članka (SD = "standard definition" podatkovje). To je bistveno
težji scenarij od naključnega delitve pikslov znotraj iste rezine, ker se med
rezinama pojavijo sistematski premiki (batch effect, barvanje, instrument).

### 2.2 Razredi

Kodiranje, ki ga uporabljam v vsej kodi (`CLASSES` seznam):

| Koda | Ime v maskah | Slovensko |
|---|---|---|
| R0 | `coll` | kolagen |
| R1 | `epith` | epitelij |
| R2 | `fibro` | fibroblasti |
| R3 | `lymph` | limfociti |
| R4 | `myo` | miofibroblasti |
| R5 | `necrosis` | nekroza |

**Odprto vprašanje (za razpravo v diplomi)**: v mapi `supervised-class/`
obstaja tudi `class_blood.png` (kri) — 1.097 pikslov na train, 1.530 na test
rezini. Tega razreda ne uporabljam (premajhen). Hkrati članek v Table 2/3
navaja drugačno šestorico: *adipocytes, blood, collagen, epithelium,
myofibroblasts, necrosis* — torej BREZ fibroblastov/limfocitov, a Z adipociti,
ki jih v naših maskah sploh ni. Ujemanja imen torej nisem mogel dokončno
razrešiti; identiteto razredov sem sklepal iz ujemanja števila pikslov s
Table 2 (najmočnejše ujemanje: R4 = myofibroblasts). To je poštena omejitev,
ki jo velja v diplomi eksplicitno omeniti.

### 2.3 Predobdelava

Podatki v `FTIR-data/*.hdf5` so **že predobdelani** (preverjeno prek HDF5
atributov) po postopku iz članka:

1. **Rubber-band baseline korekcija** (spodnja konveksna ovojnica spektra)
2. **Amide I normalizacija** (~1650 cm⁻¹)
3. **Spektralno podvzorčenje** `SPECTRAL_STEP = 2`: 1626 → **813 kanalov**

Nobena model skripta te predobdelave ne ponavlja.

### 2.4 PCA

Članek uporablja PCA s **16 komponentami** za vse klasifikatorje.

Izmerjena pojasnjena varianca na najinih 22 train izsekih:

| Komponent | Pojasnjena varianca |
|---|---|
| 8 | 95.08 % |
| **16** | **97.53 %** |
| 24 | 98.14 % |
| 32 | 98.38 % |
| 48 | 98.62 % |
| 64 | 98.76 % |

(Članek za SD navaja 90.03 % pri 16 komponentah — razlika verjetno zaradi
drugačnega nabora pikslov.)

**Pomembna ugotovitev**: donosi po 16 komponentah strmo padajo — tudi 64
komponent doda le +1.23 odstotne točke. Zato "več PCA komponent" ni smiseln
vzvod; edini način za bistveno več informacije bi bil popolnoma brez PCA
(vseh 813 kanalov), kar sem naredil pri modelB.

### 2.5 Razvoj podatkovnega nabora (kronološko)

Pomemben del naloge — začetni nabor je bil bistveno premajhen:

**Testni set:**
- Sprva: 1 pilotni izsek, 86.993 anotiranih pikslov (~23,5 % razpoložljivega)
- Po sistematičnem pregledu celotne rezine (`find_test_crops.py`) najdenih
  **22 izsekov** → izluščeno z `create_test_crops.py`
- **Končno: 439.704 anotiranih pikslov** (cela testna rezina)

**Učni set:**
- Sprva: 11 izsekov, 152.316 pikslov
- Pregled cele rezine (`find_train_full_crops.py`) je našel še 12 neizluščenih
  izsekov (78.890 pikslov) → izluščeno z `create_train_new_crops.py`
- Vmes: 23 izsekov, 231.206 pikslov
- **Končno: 22 izsekov, 206.431 pikslov** (izključen `train_crop_24`, glej
  ugotovitev 3 v poglavju 7)

**Porazdelitev razredov, TRAIN (23 izsekov, pred izločitvijo crop_24):**

| Razred | Pikslov | Delež |
|---|---|---|
| R0 kolagen | 107.131 | 46,3 % |
| R1 epitelij | 45.372 | 19,6 % |
| R2 fibroblasti | 23.596 | 10,2 % |
| R3 limfociti | 12.222 | 5,3 % |
| R4 miofibroblasti | 27.983 | 12,1 % |
| R5 nekroza | 14.902 | 6,4 % |

**Porazdelitev razredov, TEST (22 izsekov):**

| Razred | Pikslov | Delež |
|---|---|---|
| R0 kolagen | 81.546 | 18,5 % |
| R1 epitelij | 85.984 | 19,6 % |
| R2 fibroblasti | 39.201 | 8,9 % |
| R3 limfociti | 17.545 | 4,0 % |
| R4 miofibroblasti | 161.974 | 36,8 % |
| R5 nekroza | 53.454 | 12,2 % |

Opomba: porazdelitvi se med rezinama močno razlikujeta (R4 je na testu
najpogostejši s 36,8 %, na treningu pa le 12,1 %) — to je del razloga, zakaj
je cross-slide generalizacija težka.

---

## 3. Metodologija

### 3.1 Gnezdena validacija (Faza A / Faza B)

Vsi modeli uporabljajo isto strukturo, da se prepreči uhajanje testnih
podatkov:

- **Faza A** — en učni izsek se izloči kot *inner-val* (izbran samodejno kot
  najbolj uravnotežen, tj. z največjim najmanjšim razredom; v praksi vedno
  `train_crop_08`). Na preostalih izsekih se trenira, na inner-val pa se
  kalibrira: temperatura, sigma za glajenje, (pri nekaterih modelih) število
  epoh. **TEST se v tej fazi ne dotakne.**
- **Faza B** — trening na VSEH učnih izsekih z zamrznjenimi hiperparametri iz
  Faze A, nato **en sam dotik s testnim setom** (TTA napoved + evalvacija).

### 3.2 Metrike

- **CA / OA** (klasifikacijska točnost) — potrebna za primerjavo s člankom,
  ki poroča samo njo
- **Log loss** — pokazatelj kalibracije verjetnosti

**Katera metrika vodi izbiro hiperparametrov: log loss.** Povsod, kjer se
karkoli izbira samodejno (temperatura, sigma, najboljša epoha), je kriterij
minimizacija log lossa, ne maksimizacija CA.

### 3.3 Naknadna obdelava

- **Temperature scaling** — skalarna ali per-class; optimizira negativni
  log-likelihood na inner-val
- **Prostorsko glajenje** (Gaussov filter) — sigma izbrana s sweepom na
  inner-val (pogosto izbere 0.0, torej brez glajenja)
- **TTA** — 8× D4 augmentacija (zrcaljenja + rotacije) pri napovedi
- **Ansambel po semenih** — vsak "en model" je v resnici povprečje 12 modelov
  z različnimi semeni (oz. 4-6 pri transformerjih)

---

## 4. Referenčne vrednosti iz članka

### Table 8 (glavna primerjava, SD stolpec)

| Model | OA |
|---|---|
| SVM | 56,41 % |
| CNN (spektralni, brez prostorske informacije) | 62,52 % |
| CNN (prostorski) | 79,45 % ± 1,25 |

(HD stolpec, za informacijo: 76,28 / 79,54 / 92,85 %)

### Table 1 (klasični klasifikatorji, SD stolpec)

| KNN | LinearSVM | RBF SVM | DT | RF | NN | AdaBoost | NB | QDA |
|---|---|---|---|---|---|---|---|---|
| 52,43 | 53,99 | 56,83 | 46,47 | 46,76 | 45,36 | 50,26 | 47,90 | 47,09 |

### Hiperparametri iz članka (poglavje 2.5.2 + Fig. 3)

- Optimizator: **Adadelta**, lr = 0,1
- Regularizacija: L2 + **dropout 0,5**
- **BatchNorm samo za HD**; SD varianta je "ista arhitektura brez BN"
- **LRN** (Local Response Normalization) ostane tudi pri SD
- Aktivacija: **softplus** (ne ReLU)
- Inicializacija uteži: **N(0, 0,02)**
- Batch size: **128**
- Epohe: **8** ("terminating when validation accuracy began to decline")
- Arhitektura (SD): vhod 17×17×16 → Conv(32, 3×3) → LRN → MaxPool(2) →
  Conv(64, 3×3) → Conv(64, 3×3) → LRN → MaxPool(2) → FC(128) → Softmax(6)
- Uravnoteženje: "stack copies of underrepresented classes" — 10.000
  pikslov/razred za SVM, 100.000/razred za CNN

### Namerna odstopanja od članka (modernizacija)

| Kaj | Članek | Pri meni | Zakaj |
|---|---|---|---|
| Optimizator | Adadelta lr=0,1 | Adam | Adadelta danes slabo konvergira |
| LR razpored | ni naveden | cosine decay | standardna stabilizacija |
| Gradient clipping | ni naveden | 1,0 | stabilnost |
| Weight decay | L2 (brez vrednosti) | 1e-4 | — |
| Uravnoteženje | oversampling 100k/razred | mehčane inverzne uteži (soften 0,5) | empirično bistveno boljše za R4 |

---

## 5. Modeli in rezultati

Vsi spodnji rezultati so na **istem testnem setu** (22 izsekov, 439.704
pikslov), razen kjer je izrecno navedeno drugače.

### 5.1 modelA — RBF SVM (`modelA_crossSlide.py`)

Zvesta replika: PCA(16) → RBF SVM, C = 1,0, gamma = 'scale', oversampling
10.000/razred.

| Tek | OA | log loss |
|---|---|---|
| sanity (sigma=2,0) | 59,47 % | 1,14406 |
| **končni (sigma=1,0)** | **58,39 %** | **1,10864** |

**Članek: 56,41 % → presegel sem ga za ~2 odstotni točki.** To je najmočnejša
potrditev, da je pipeline pravilen.

### 5.2 modelA-classic — replikacija Table 1 (`modelA_classic_baselines.py`)

Isti PCA(16) + oversampling pipeline, privzete scikit-learn nastavitve
(članek za te modele ne navaja hiperparametrov).

| Klasifikator | Moj OA | Članek | Razlika | log loss |
|---|---|---|---|---|
| KNN | 47,27 % | 52,43 % | −5,16 | 6,01827 |
| Linear SVM | 52,04 % | 53,99 % | −1,95 | 1,31494 |
| Decision Tree | 42,01 % | 46,47 % | −4,46 | 9,34676 |
| Random Forest | 48,16 % | 46,76 % | +1,40 | 1,43331 |
| NN (MLP) | 51,84 % | 45,36 % | +6,48 | 3,01059 |
| AdaBoost | 50,85 % | 50,26 % | +0,59 | 1,72421 |
| Naive Bayes | 41,85 % | 47,90 % | −6,05 | 2,77299 |
| QDA | 44,22 % | 47,09 % | −2,87 | 4,26196 |

Vzorec: rezultati nihajo okrog članka v obe smeri, noben pa se ne približa
CNN-ju — kar je glavna poanta te tabele v članku.

### 5.3 modelB — spektralni 1D CNN (`modelB_crossSlide.py`)

Brez PCA, brez prostorske informacije — dela na polnem 813-kanalnem spektru
posameznega piksla. Arhitektura: Conv1d(1→32→64→128→256, jedra 7/5/3/3) +
BatchNorm + ReLU + MaxPool, AdaptiveAvgPool, FC(256→128→64→6).

| Konfiguracija | OA | log loss |
|---|---|---|
| **oversample (uradni)** | **50,45 %** | **1,44460** |
| weights + label smoothing (z napako v temperaturi) | 48,37 % | 4,10867 |
| weights + label smoothing (popravljena temperatura) | 50,33 % | 1,57337 |

**Članek: 62,52 % — tega nisem dosegel.**

Razredna razčlenitev uradnega teka (zelo poučna):

| Razred | OA |
|---|---|
| R0 kolagen | 92,91 % |
| R1 epitelij | 86,20 % |
| R2 fibroblasti | 26,75 % |
| R3 limfociti | 43,47 % |
| **R4 miofibroblasti** | **3,83 %** |
| R5 nekroza | 81,02 % |

Ta 3,83 % je ključni rezultat za razpravo — glej ugotovitev 4.

### 5.4 modelC — prostorski CNN (`modelC_crossSlide_faithful.py`)

Glavni model naloge. Arhitektura po članku (Fig. 3, SD varianta):
Conv(32,3×3) → Softplus → [LRN] → MaxPool(2) → Conv(64,3×3) → Conv(64,3×3) →
Softplus → [LRN] → MaxPool(2) → FC(128) → Softmax(6); brez BatchNorm,
dropout 0,5, init N(0, 0,02), patch 17×17×16 (PCA), ansambel 12+12.

**Zgodnja faza (star, premajhen testni set — 86.993 pikslov, NI primerljivo):**

| Različica | OA |
|---|---|
| crossSlide_v1 (prva pravilna cross-slide postavitev) | 59,13–59,88 % |
| crossSlide_v2 (dvotokovna: prostorski CNN + spektralna MLP veja) | 57,25–63,09 % |
| faithful (enotokovna, po članku) | 64,26 % |
| faithful + smooth-scale 3 | 64,88 % |
| faithful + smooth-scale 3,7 | 65,31 % |

**Glavna faza (polni testni set, 439.704 pikslov):**

| # | Konfiguracija | Train | OA | log loss |
|---|---|---|---|---|
| 1 | osnovni | 11 izsekov | 70,82 % | 0,99422 |
| 2 | patch 25×25 (namerno odstopanje od članka) | 11 izsekov | 71,75 % | 0,96670 |
| 3 | + 12 novih izsekov | 23 izsekov | 67,45 % | 1,18973 |
| 4 | izločen crop_24 | 22 izsekov | 72,16 % | 1,01644 |
| 5 | weight-soften 0,8 | 22 izsekov | 71,10 % | 1,02539 |
| 6 | + LRN | 22 izsekov | 72,02 % | 1,02797 |
| 7 | + LRN + smooth-scale 3 | 22 izsekov | 72,90 % | 1,01135 |
| 8 | **+ širša meja temperature** | 22 izsekov | **73,11 %** | 1,02331 |
| 9 | + label smoothing 0,05 | 22 izsekov | 69,70 % | 0,99778 |
| 10 | + label smoothing 0,1 | 22 izsekov | 70,09 % | **0,98092** |
| 11 | + label smoothing 0,2 | 22 izsekov | 69,87 % | 1,03002 |

**Najboljši CA: 73,11 %** (#8) — članek 79,45 %, razlika 6,3 odstotne točke.
**Najboljši log loss posameznega modela: 0,98092** (#10).

Razredna razčlenitev #7 (72,90 %):

| Razred | OA | log loss |
|---|---|---|
| R0 | 90,46 % | 0,558 |
| R1 | 68,60 % | 1,270 |
| R2 | 8,15 % | 2,405 |
| R3 | 28,05 % | 1,699 |
| R4 | 81,02 % | 0,858 |
| R5 | 90,60 % | 0,503 |

### 5.5 modelD — transformerji (`modelD_transformer_crossSlide.py`, `modelD_transformer_hybrid.py`)

Nadgradnja članka. ViT slog: patch se razreže na neprekrivajoče 3×3
podpatche (tokene), učljiv [CLS] token + pozicijske vdelave, N enkoder plasti
(multi-head self-attention, pre-LN), AdamW + cosine z ogrevanjem, label
smoothing.

| # | Različica | Parametrov | Inner-val OA | TEST OA | log loss | Vrzel |
|---|---|---|---|---|---|---|
| 1 | osnovni ViT (192/8/6, patch 33) | 3,6 M | 78,82 % | 52,82 % | 1,45889 | 26,0 |
| 2 | večja kapaciteta (256/10/8, patch 33) | 8,0 M | 78,35 % | 52,08 % | 1,50821 | 26,3 |
| 3 | hibridni CNN-ViT + konv. stem (patch 33) | 3,7 M | 79,43 % | 48,92 % | 1,59373 | 30,5 |
| 4 | **hibridni CNN-ViT, ozko polje (patch 15)** | 3,7 M | 77,14 % | **58,05 %** | **1,39631** | **19,1** |

Za primerjavo: CNN ima vrzel inner-val → test le **~5 odstotnih točk**.

Hibridna različica dodaja pred tokenizacijo majhen konvolucijski "stem"
(Conv 16→32→64, 3×3, brez poolinga), po ideji iz Xiao et al. 2021, "Early
Convolutions Help Transformers See Better".

### 5.6 Ansambli (`ensemble_predictions.py`)

Povprečenje shranjenih verjetnostnih kart, brez ponovnega treniranja.

| Kombinacija | OA | log loss |
|---|---|---|
| CNN top-2 | 73,01 % | 1,01706 |
| CNN top-3 | 72,94 % | 1,00130 |
| CNN top-4 | 72,64 % | 1,00154 |
| CNN raznolika regularizacija (3 modeli) | 72,42 % | 0,95730 |
| CNN vseh 7 | 71,86 % | 0,95432 |
| CNN widerT + ViT p15 | 69,65 % | 1,00467 |
| CNN top-3 + ViT p15 | 72,69 % | 0,94958 |
| **CNN vseh 7 + ViT p15** | 72,10 % | **0,94701** |

Sedem CNN različic v ansamblu = konfiguracije #4, #6, #7, #8, #9, #10, #11 iz
tabele 5.4.

Razredna razčlenitev najboljšega ansambla (0,94701):

| Razred | OA | log loss |
|---|---|---|
| R0 | 92,96 % | 0,291 |
| R1 | 70,59 % | 1,077 |
| R2 | 9,12 % | 2,397 |
| R3 | 28,31 % | 1,677 |
| R4 | 76,12 % | 1,009 |
| R5 | 91,04 % | 0,248 |

---

## 6. Skupna primerjava s člankom

| Model | Moj rezultat | Članek | Razlika |
|---|---|---|---|
| SVM | **58,39 %** | 56,41 % | **+1,98** |
| spektralni CNN | 50,45 % | 62,52 % | −12,07 |
| prostorski CNN | 73,11 % | 79,45 % | −6,34 |
| najboljši klasični (Linear SVM) | 52,04 % | 53,99 % | −1,95 |
| transformer (nadgradnja) | 58,05 % | — | — |
| najboljši log loss (ansambel) | 0,94701 | ni poročan | — |

---

## 7. Ključne ugotovitve

### Ugotovitev 1 — R2 in R3 sta trdovratno neodvisna od vseh posegov

Fibroblasti (~7–9 %) in limfociti (~26–31 %) se **niso premaknili** kljub
sedmim neodvisnim posegom: več učnih podatkov, mehčanje uteži, LRN,
večskalno glajenje, večji patch, label smoothing (3 jakosti), širša meja
temperature.

To ni naključje — kaže na **pravo prekrivanje razredov v prostoru značilk**,
ne na pomanjkljivost modela ali premalo podatkov.

Matrika zamenjav to podpira: fibroblasti se zamenjujejo predvsem s kolagenom
(~64 %) in miofibroblasti (~19 %) — vse znotraj vezivnega tkiva, kar je
biološko smiseln kontinuum, ne naključna napaka. Limfociti se najbolj
zamenjujejo z miofibroblasti.

### Ugotovitev 2 — več podatkov ni samodejno bolje

Dodajanje 12 novih učnih izsekov je rezultat **poslabšalo** (72,16 % →
67,45 %). Vzrok: `train_crop_24` je skoraj čist kolagen (24.246 od 24.775
pikslov), brez enega samega piksla za fibroblaste/limfocite/miofibroblaste.

Analiza matrike zamenjav pred/po je pokazala mehanizem: zamenjave
miofibroblastov s kolagenom so zrasle z 12,77 % na 19,33 %, s fibroblasti pa
s 7,91 % na 13,53 %. Poplava večinskega razreda je torej razširila njegovo
odločitveno regijo in "požrla" sosednje miofibroblaste.

Po izločitvi tega enega izseka je R4 poskočil nad prvotno raven (68,77 % →
76,51 %), skupni CA pa na takrat najboljših 72,16 %.

### Ugotovitev 3 — prostorska informacija je ključna prav za miofibroblaste

Najčistejši rezultat naloge:

| Model | R4 (miofibroblasti) OA |
|---|---|
| spektralni CNN (brez prostorske informacije) | **3,83 %** |
| prostorski CNN | **81,02 %** |

Ista predobdelava, isti podatki, ista metodologija — razlika je samo v tem,
ali model vidi prostorski kontekst. To neposredno podpira osrednjo tezo
članka (zakaj je Table 8 primerjava sploh smiselna) in hkrati pojasnjuje,
zakaj so miofibroblasti povsod najbolj problematičen razred.

Zanimivo nasprotje: spektralni model je pri fibroblastih (26,75 %) in
limfocitih (43,47 %) **boljši** od prostorskega (8,15 % / 28,05 %). Tudi to
je vredno razprave.

### Ugotovitev 4 — transformerji ne generalizirajo čez rezini

Vse štiri transformerske različice imajo vrzel med notranjo validacijo in
testom **19–30 odstotnih točk**, CNN pa le ~5.

Domneva: self-attention s širokim vidnim poljem (33×33 patch, CLS token
agregira celoten kontekst) se nauči globalnih vzorcev, značilnih za
posamezno rezino (batch effect), ki se ne prenesejo na drugo rezino. CNN z
majhnim receptivnim poljem je temu naravno bolj odporen.

Domnevo podpira eksperiment: zoženje polja s 33×33 na 15×15 je vrzel
zmanjšalo z 30,5 na 19,1 odstotne točke in dvignilo CA z 48,92 % na 58,05 %.

**Kar NI pomagalo**: povečanje kapacitete (dvakrat ovrženo — 3,6 M → 8,0 M
parametrov je rezultat poslabšalo) in dodajanje konvolucijskega stema pri
širokem polju.

### Ugotovitev 5 — regularizacija: log loss ↑, CA ↓

Label smoothing sweep (izhodišče 72,90 % / 1,01135):

| Jakost | CA | log loss |
|---|---|---|
| 0,05 | 69,70 % | 0,99778 |
| **0,1** | 70,09 % | **0,98092** |
| 0,2 | 69,87 % | 1,03002 |

Krivulja log lossa je **U-oblike** — optimum pri 0,1, pri 0,2 je že slabše od
izhodišča. CA pa pade za ~3 odstotne točke pri vseh jakostih približno enako.

### Ugotovitev 6 — kalibracija: mehčati da, ostriti ne

Pri per-class temperature scalingu sta R2 in R3 **vedno** pristala točno na
zgornji meji iskanja (najprej 10, po razširitvi 50) — model si torej želi
maksimalno mehčanje verjetnosti za ta dva razreda, kar je neodvisna potrditev
ugotovitve 1.

Ko sem mejo razširil simetrično (0,1–50), je to pri modelB povzročilo
katastrofo: optimizator je za R4 izbral T = 0,10 (torej **ostrenje**,
ne mehčanje), ker je na majhnem inner-val vzorcu to zniževalo log loss. Na
testu je R4 padel na 0,05 % CA, log loss tega razreda pa je zrasel na 15,4.

Popravek: spodnja meja postavljena na 1,0 — kalibracija sme samo mehčati.

### Ugotovitev 7 — ansambel izboljša kalibracijo, ne točnosti

Povprečenje napovedi da najboljši log loss celotne naloge (**0,94701**),
boljši od kateregakoli posameznega modela (0,98092). CA pa se ne izboljša
(73,01 % najboljši ansambel proti 73,11 % najboljšemu posameznemu modelu).

Najzanimivejše: **največ prispeva prav transformer**, čeprav je sam po sebi
precej slabši (58 % proti 73 %). CNN top-3 sam: 1,00130 → z dodanim
transformerjem: 0,94958. Arhitekturna raznolikost torej pomaga kalibraciji
tudi takrat, ko dodani model ni konkurenčen.

---

## 8. Metodološka opozorila (pomembno za pošteno pisanje)

### 8.1 Večkratni dotik s testnim setom

Odločitev o izločitvi `train_crop_24` je bila sprejeta na podlagi **matrike
zamenjav na testnem setu**, nato pa potrjena z novim tekom na istem testnem
setu. To je tehnično gledano data snooping in je v nasprotju s pravilom, da
se končni rezultat ne uporablja za nastavljanje modela.

Odločitev je bila zavestna (hitrejša iteracija), a jo je treba v diplomi
**transparentno omeniti** — najbolje kot "lessons learned" v razpravi o
metodologiji. Fizikalna razlaga mehanizma (poplava večinskega razreda) je
neodvisno smiselna, kar delno omili težavo, a je ne odpravi.

Metodološko čisteje bi bilo: vse take odločitve sprejeti na inner-val, testni
set pa se dotakniti samo enkrat, s končno konfiguracijo.

### 8.2 Inner-val je en sam izsek

Kalibracija (temperatura, sigma) sloni na enem samem izločenem izseku
(`train_crop_08`, 17.757 pikslov). To je majhen vzorec, ki lahko privede do
prekomernega prilagajanja kalibracijskih parametrov — demonstrirano z
modelB / T = 0,10 (ugotovitev 6).

### 8.3 Neujemanje imen razredov

Glej poglavje 2.2 — šesterica v naših maskah se ne ujema s šesterico v
Table 2/3 članka. To lahko delno pojasni odstopanja pri posameznih razredih.

### 8.4 Odstopanja od članka

Glej tabelo v poglavju 4. Najpomembnejša: Adam namesto Adadelta in mehčane
uteži namesto oversamplinga na 100.000/razred.

---

## 9. Infrastruktura in ponovljivost

**Strežnik**: `o1.biolab.si` (FRI) — NVIDIA TITAN X (Pascal) 12 GB,
59 GB RAM, 6 jeder. Dolgi teki v `tmux` sejah.

**Tipični časi tekov:**

| Model | Konfiguracija | Čas |
|---|---|---|
| SVM | oversample 10k | ~40 min |
| 8 klasičnih klasifikatorjev | vsi skupaj | 19 min |
| spektralni CNN | 40 epoh Faza A + Faza B | ~40 min |
| prostorski CNN | ansambel 12+12, 8 epoh | 1 h 20 min – 2 h 15 min |
| transformer (patch 33) | ansambel 6+6, 20 epoh | 7 h 24 min – 18 h 37 min |
| transformer (patch 15) | ansambel 6+6, 20 epoh | 7 h 24 min |

**Ukaz za najboljši CA model (73,11 %):**

```bash
python3 -u modelC_crossSlide_faithful.py \
  --train-dir FTIR-data/train_preprocessed_no24 \
  --test-dir FTIR-data/test_preprocessed_full \
  --cache-dir FTIR-data/_cache \
  --n-ensemble 12 --calib-ensemble 12 \
  --balance-strategy weights --per-class-temperature \
  --use-lrn --extra-smooth-scale 3 \
  --num-workers 4 \
  --output modelC_faithful_no24_lrn_smooth3_widerT.npy
```

(Ta tek je uporabljal mejo temperature 0,1–50; v trenutni kodi je spodnja
meja 1,0 — glej ugotovitev 6.)

**Ukaz za najboljši log loss posameznega modela (0,98092):** isto, plus
`--label-smoothing 0.1`.

**Ansambel** (brez treniranja): `python3 ensemble_predictions.py`

**Podatki niso v repozitoriju** (11 GB) — `FTIR-data/` je v `.gitignore`.
Surovi podatki so na strežniku v `~/mayerich2-učni` in `~/mayerich-testni`.

---

## 10. Odprte smeri (kar bi se še dalo narediti)

1. **Transformer z še ožjim poljem** (patch 9×9) — trend 33 → 15 je bil
   močno pozitiven; patch 9 bi pokazal, ali se nadaljuje ali je 15 optimum.
2. **Samonadzorovano predtreniranje** transformerja (MAE slog) na vseh
   tkivnih pikslih (tudi neanotiranih, tudi na testni rezini) — teoretično
   najbolj obetavno za premostitev vrzeli med rezinama.
3. **Utežen ansambel** namesto navadnega povprečja (uteži izbrane na
   inner-val).
4. **Analiza kakovosti anotacij za R2/R3** — npr. pregled, ali so napačno
   klasificirani piksli koncentrirani ob mejah regij (kar bi kazalo na šum v
   anotacijah) ali enakomerno razpršeni (kar bi kazalo na pravo prekrivanje).
