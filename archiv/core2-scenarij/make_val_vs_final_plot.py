"""
make_val_vs_final_plot.py
=========================
Izriše korelacijo med lokalno validacijo (VAL log loss) in tekmovalnim
rezultatom (Final). Dokaz, da lokalna validacija napove generalizacijo —
torej da razvoj NI bil naključno oddajanje na lestvico.

Ključ: modeli z DOKAZANIM glajenjem (σ=1.5) ležijo na premici — VAL log loss
popolnoma napove vrstni red po Final. v4 je edina izjema (σ avto-tuniran na 0),
in to je razložljivo: lokalna VAL ne more oceniti prostorskega glajenja,
ker so val piksli razpršeni, ne sklenjeni kot tekmovalni crop.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# (label, VAL log loss, Final, σ uporabljen)
DATA = [
    ("v1",  0.25168, 0.45890, 1.5),
    ("v2",  0.23217, 0.40937, 1.5),
    ("v3",  0.25498, 0.46571, 1.5),
    ("v4",  0.23467, 0.46712, 0.0),   # izjema — σ avto-tune okvara
    ("v5",  0.23094, 0.39533, 1.5),
]

proven = [d for d in DATA if d[3] == 1.5]
outlier = [d for d in DATA if d[3] != 1.5]

px = np.array([d[1] for d in proven])
py = np.array([d[2] for d in proven])

# Pearsonov koeficient za σ=1.5 modele
r = np.corrcoef(px, py)[0, 1]

# Linearna regresija skozi σ=1.5 točke
coef = np.polyfit(px, py, 1)
xs = np.linspace(px.min() - 0.003, px.max() + 0.003, 100)
ys = np.polyval(coef, xs)

fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=150)

# Trend
ax.plot(xs, ys, "--", color="#888", lw=1.5, zorder=1,
        label=f"Trend (σ=1.5):  r = {r:.3f}")

# σ=1.5 točke
ax.scatter(px, py, s=140, color="#2E75B6", zorder=3,
           edgecolors="white", linewidths=1.5, label="Modeli s σ=1.5 (dokazano)")

# Izjema v4
for lab, vx, vy, _ in outlier:
    ax.scatter([vx], [vy], s=170, color="#C0392B", marker="X", zorder=3,
               edgecolors="white", linewidths=1.5,
               label="v4: σ avto-tuniran → 0 (post-proc. okvara)")

# Oznake točk
for lab, vx, vy, sig in DATA:
    dy = 0.006 if lab not in ("v2",) else -0.012
    dx = 0.0005
    ax.annotate(lab, (vx, vy), xytext=(vx + dx, vy + dy),
                fontsize=12, fontweight="bold",
                color="#C0392B" if sig == 0 else "#1F4E79")

# Označi zmagovalca tekmovanja kot referenco
ax.axhline(0.40031, color="#27AE60", lw=1.2, ls=":", zorder=0)
ax.text(0.2565, 0.40031 + 0.002, "zmagovalec tekmovanja (0.40031)",
        color="#1E8449", fontsize=9, ha="right")

ax.set_xlabel("Lokalna validacija — VAL log loss  (nižje = boljše)", fontsize=12)
ax.set_ylabel("Tekmovanje — Final  (nižje = boljše)", fontsize=12)
ax.set_title("Lokalna validacija napove tekmovalni rezultat\n"
             "(razvoj voden z validacijo, ne z lestvico)", fontsize=13, pad=12)
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

# Lepši robovi
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
out = "val_vs_final.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"✓ shranjeno: {out}")
print(f"  Pearson r (σ=1.5 modeli) = {r:.4f}")
print(f"  Med modeli s σ=1.5 VAL log loss POPOLNOMA rangira Final:")
for lab, vx, vy, sig in sorted(proven, key=lambda d: d[1]):
    print(f"    {lab}: VAL={vx:.5f} → Final={vy:.5f}")
