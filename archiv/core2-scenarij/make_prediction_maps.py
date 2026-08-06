"""
make_prediction_maps.py
=======================
Dve vizualizaciji napovedi na tekmovalnem cropu (data[265:465, 360:660]):

  FIG 1 (zrcali članek Fig. 8):
    (a) Amide I absorbanca (~1648 cm-1)  — "anatomska" referenca
    (b) v5 napovedani razredi (barvno)   — kemijska/klasifikacijska karta
    (c) v5 zanesljivost (max verjetnost) — kje je model (ne)gotov

  FIG 2 (evolucija modelov):
    LR (multilogreg) → modelC_best (v1) → v2 → v5
    Pokaže prehod iz "soli in popra" (LR, brez prostorske info) v
    prostorsko koherentne regije (v5, prostorsko+spektralno+glajenje).

Crop nima pravih oznak (skriti test) → prikaz je KAKOVOSTEN, ne kvantitativen.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

R0, R1, C0, C1 = 265, 465, 360, 660
AMIDE_IDX = 162  # ~1648 cm-1
NUM_CLASSES = 6

# 6 razlocnih barv za razrede (Razred 0..5 = 6 histoloskih tipov iz clanka)
CLASS_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B3", "#937860"]
CLASS_LABELS = [f"Razred {i}" for i in range(NUM_CLASSES)]
cmap = ListedColormap(CLASS_COLORS)


def load_crop_context():
    with h5py.File("image1-competition.hdf5", "r") as f:
        data   = np.array(f["data"][R0:R1, C0:C1, AMIDE_IDX], dtype=np.float32)
        tissue = np.array(f["tissue_mask"][R0:R1, C0:C1], dtype=bool)
    return data, tissue


def argmax_map(npy_path, tissue):
    """Vrne (class_map masked, confidence masked)."""
    probs = np.load(npy_path)                 # (200, 300, 6)
    cls   = np.argmax(probs, axis=-1).astype(float)
    conf  = probs.max(axis=-1)
    cls_m  = np.ma.masked_where(~tissue, cls)
    conf_m = np.ma.masked_where(~tissue, conf)
    return cls_m, conf_m


def main():
    amide, tissue = load_crop_context()

    # ---------- FIG 1: Amide I | v5 razredi | v5 zanesljivost ----------
    cls5, conf5 = argmax_map("modelC_best_v5.npy", tissue)
    amide_m = np.ma.masked_where(~tissue, amide)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150)

    im0 = axs[0].imshow(amide_m, cmap="gray")
    axs[0].set_title("(a) Amide I absorbanca (~1648 cm⁻¹)", fontsize=12)
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    axs[1].imshow(cls5, cmap=cmap, vmin=-0.5, vmax=NUM_CLASSES - 0.5,
                  interpolation="nearest")
    axs[1].set_title("(b) v5 napovedani razredi", fontsize=12)
    axs[1].legend(handles=[Patch(color=CLASS_COLORS[i], label=CLASS_LABELS[i])
                           for i in range(NUM_CLASSES)],
                  loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9,
                  frameon=False)

    im2 = axs[2].imshow(conf5, cmap="viridis", vmin=0.0, vmax=1.0)
    axs[2].set_title("(c) v5 zanesljivost (max verjetnost)", fontsize=12)
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Tekmovalni crop (200×300): napoved modela v5 — zrcali članek Fig. 8",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("fig_prediction_v5.png", dpi=150, bbox_inches="tight")
    print("✓ fig_prediction_v5.png")

    # ---------- FIG 2: evolucija modelov ----------
    models = [
        ("multilogreg.npy", "LR (samo spekter)\nFinal=0.545"),
        ("modelC_best.npy", "v1 dual-stream\nFinal=0.459"),
        ("modelC_best_v2.npy", "v2 +neighbourhood\nFinal=0.409"),
        ("modelC_best_v5.npy", "v5 flagship\nFinal=0.395"),
    ]
    fig2, axs2 = plt.subplots(1, len(models), figsize=(5 * len(models), 5.6), dpi=150)
    for ax, (path, title) in zip(axs2, models):
        cls_m, _ = argmax_map(path, tissue)
        ax.imshow(cls_m, cmap=cmap, vmin=-0.5, vmax=NUM_CLASSES - 0.5,
                  interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig2.legend(handles=[Patch(color=CLASS_COLORS[i], label=CLASS_LABELS[i])
                         for i in range(NUM_CLASSES)],
                loc="lower center", ncol=NUM_CLASSES, fontsize=10, frameon=False,
                bbox_to_anchor=(0.5, -0.04))
    fig2.suptitle("Evolucija napovedi: iz 'soli in popra' (LR) v prostorsko koherentne regije (v5)",
                  fontsize=13, y=1.02)
    fig2.tight_layout()
    fig2.savefig("fig_model_evolution.png", dpi=150, bbox_inches="tight")
    print("✓ fig_model_evolution.png")

    # Kvantitativna opomba: delez "spremenjenih" pikslov LR->v5
    lr_cls, _  = argmax_map("multilogreg.npy", tissue)
    v5_cls, _  = argmax_map("modelC_best_v5.npy", tissue)
    diff = (lr_cls != v5_cls) & (~lr_cls.mask)
    print(f"\n  LR vs v5: {100*diff.sum()/tissue.sum():.1f}% tkivnih pikslov dobi drugačno oznako")
    print(f"  Povprečna zanesljivost v5 (tkivo): {conf5.mean():.3f}")


if __name__ == "__main__":
    main()
