"""Visual del hilo del sabado: Ben Johnson contra Brian Flores.

Dos PNG separados, uno por tuit del hilo (19-sep-2026: juntos iban apretados):
  1) blitz_flores_*: % de blitz de Minnesota por temporada frente a la liga
  2) epa_johnson_vs_flores_*: EPA por dropback de Johnson en los 6 partidos,
     con el % de blitz que recibio en cada uno

Los datos se recalculan aqui (PBP + FTN); nada sale de fuentes ajenas.
    python lab/johnson_vs_flores_png.py --season 2026 --week 2
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pbp_loader import cargar_pbp, cargar_ftn, salida, season_cli, week_cli, sello

BG, CARD, FG, GRID, ACCENT = "#0f1115", "#151924", "#EDEDED", "#2a2f3a", "#2d6cdf"
MIN_MORADO, CHI_NARANJA, DET_AZUL = "#4F2683", "#C83803", "#0076B6"
LOGOS_DIR = "logos"

SEASON = season_cli() or 2026
WEEK = week_cli() or 2
PARTIDOS = {2023: "DET", 2024: "DET", 2025: "CHI", 2026: "CHI"}


def dropbacks(season):
    df, _ = cargar_pbp(season, avisar=False)
    db = df[(df["qb_dropback"] == 1) & df["epa"].notna()
            & (df["play_type"] != "no_play")].copy()
    ftn, _ = cargar_ftn(season)
    db = db.merge(ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers"]],
                  left_on=["game_id", "play_id"],
                  right_on=["nflverse_game_id", "nflverse_play_id"], how="inner")
    db["blitz"] = db["n_blitzers"] > 0
    return db


blitz_min, blitz_liga, juegos = {}, {}, []
for season, equipo in PARTIDOS.items():
    db = dropbacks(season)
    por_def = db.groupby("defteam")["blitz"].mean() * 100
    blitz_min[season] = por_def["MIN"]
    blitz_liga[season] = por_def.mean()
    g = db[(db["posteam"] == equipo) & (db["defteam"] == "MIN")]
    for gid, x in g.groupby("game_id"):
        juegos.append({"gid": gid, "season": season, "off": equipo,
                       "epa": x["epa"].mean(), "blitz": x["blitz"].mean() * 100,
                       "n": len(x)})
J = pd.DataFrame(juegos).sort_values("gid").reset_index(drop=True)
ETIQUETAS = ["2023 sem. 16", "2023 sem. 18", "2024 sem. 7", "2024 sem. 18",
             "2025 sem. 1", "2025 sem. 11"]
J["etq"] = ETIQUETAS[:len(J)]

def lienzo(alto, eje_y="y"):
    fig, ax = plt.subplots(figsize=(9.2, alto))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors="#9aa0aa", labelsize=10)
    ax.grid(axis=eje_y, color=GRID, lw=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def pie(fig, ax):
    fig.text(0.01, 0.012, f"Fuente: nflverse PBP + FTN  ·  {sello(SEASON)}",
             color="#555555", fontsize=8.5, fontstyle="italic", ha="left")
    ax.text(0.99, 0.01, "@CuartayDato", transform=ax.transAxes, ha="right",
            va="bottom", color="#888888", fontsize=9, alpha=0.8, fontstyle="italic")


fig1, ax1 = lienzo(6.2)

# ── Panel 1: el blitz de Flores contra la media de la liga ──────────────────
temps = list(PARTIDOS)
x = np.arange(len(temps))
ax1.bar(x - 0.19, [blitz_min[t] for t in temps], 0.38, color=MIN_MORADO,
        label="Minnesota (Flores)")
ax1.bar(x + 0.19, [blitz_liga[t] for t in temps], 0.38, color="#3a4150",
        label="media de la NFL")
for i, t in enumerate(temps):
    ax1.text(i - 0.19, blitz_min[t] + 1.5, f"{blitz_min[t]:.0f}%", ha="center",
             color=FG, fontsize=11, fontweight="bold")
    ax1.text(i + 0.19, blitz_liga[t] + 1.5, f"{blitz_liga[t]:.0f}%", ha="center",
             color="#9aa0aa", fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels([str(t) if t < 2026 else "2026 (sem. 1)" for t in temps],
                    color=FG, fontsize=11)
ax1.set_ylim(0, max(blitz_min.values()) + 14)
ax1.set_ylabel("% de dropbacks con blitz", color=FG, fontsize=11)
ax1.set_title("% de dropbacks rivales con blitz · charting de FTN", color="#9aa0aa",
              fontsize=11, loc="left", pad=10)
fig1.suptitle("Flores blitzea como nadie", color=FG, fontsize=19,
              fontweight="bold", x=0.055, ha="left", y=0.975)
leg = ax1.legend(loc="upper left", frameon=False, fontsize=10)
for t in leg.get_texts():
    t.set_color("#c9ced8")

pie(fig1, ax1)
fig1.tight_layout(rect=[0, 0.02, 1, 0.93])
out1 = salida(f"blitz_flores_{SEASON}.png", SEASON, WEEK)
plt.savefig(out1, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig1)

# ── PNG 2: como le fue al ataque de Johnson en cada partido ────────────────
fig2, ax2 = lienzo(7.0, eje_y="x")
colores = [DET_AZUL if r.off == "DET" else CHI_NARANJA for r in J.itertuples()]
y = np.arange(len(J))[::-1]
ax2.barh(y, J["epa"], 0.62, color=colores)
ax2.axvline(0, color="#6b7280", lw=1)
for yi, r in zip(y, J.itertuples()):
    dx = 0.012 if r.epa >= 0 else -0.012
    ax2.text(r.epa + dx, yi, f"{r.epa:+.2f}", va="center",
             ha="left" if r.epa >= 0 else "right", color=FG, fontsize=11,
             fontweight="bold")
    ax2.text(-0.155, yi, f"blitz {r.blitz:.0f}%", va="center", ha="left",
             color="#9aa0aa", fontsize=9.5)
ax2.set_yticks(y)
ax2.set_yticklabels(J["etq"], color=FG, fontsize=10.5)
ax2.set_xlim(-0.17, 0.66)
ax2.set_xlabel("EPA por dropback", color=FG, fontsize=11)
# Sin titular: el tuit ya cuenta la historia (pedido por Luis, 19-sep-2026)
ax2.set_title("EPA por dropback del ataque de Ben Johnson ante Flores, partido a partido",
              color="#9aa0aa", fontsize=11.5, loc="left", pad=10)

# Logos del equipo de Johnson en cada partido
for yi, r in zip(y, J.itertuples()):
    path = os.path.join(LOGOS_DIR, f"{r.off}.png")
    if not os.path.exists(path):
        continue
    img = plt.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        ys, xs = np.where(img[:, :, 3] > 0.02)
        if len(ys):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    z = 0.040 * 500.0 / max((h * w) ** 0.5, 1.0)
    ax2.add_artist(AnnotationBbox(OffsetImage(img, zoom=z, resample=True),
                                  (0.615, yi), frameon=False))

pie(fig2, ax2)
fig2.tight_layout(rect=[0, 0.02, 1, 0.99])
out2 = salida(f"epa_johnson_vs_flores_{SEASON}.png", SEASON, WEEK)
plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"Guardado: {out1}")
print(f"Guardado: {out2}")
print(J.round(3).to_string(index=False))
print({t: (round(blitz_min[t], 1), round(blitz_liga[t], 1)) for t in temps})
