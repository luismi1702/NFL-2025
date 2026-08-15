"""
evolucion_contender.py
Compara un equipo en 2 temporadas contra los 12 requisitos de
contenders_tracker.py: que arreglaron (o rompieron) de un año a otro.
Reutiliza contenders_tracker.compute() — mismos umbrales y definiciones.
NFL 2025
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sys.stdout.reconfigure(encoding="utf-8")

import contenders_tracker as ct

BG    = "#0f1115"
CARD  = "#151924"
FG    = "#EDEDED"
GRID  = "#2a2f3a"
GREEN = "#06d6a0"
RED   = "#d84a4a"
DPI   = 170
LOGOS_DIR = "logos"


def load_logo(team, base_zoom=0.075):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        z = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * base_zoom:
            z = 900.0 * base_zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


def fmt_val(metric, val):
    if pd.isna(val):
        return "N/D"
    if metric in ("def_third_conv", "drive_score_rate", "success_rate_def"):
        return f"{val*100:.1f}%"
    if metric in ("def_epa", "ypa", "yards_play_def"):
        return f"{val:+.2f}" if metric == "def_epa" else f"{val:.2f}"
    return f"{val:.0f}"


# ── INPUT ──────────────────────────────────────────────────────────────────────
team    = input("Equipo (siglas, ej: SEA): ").strip().upper()
season_a = int(input("Temporada A (la más antigua, ej: 2024): ").strip())
season_b = int(input("Temporada B (la más reciente, ej: 2025): ").strip())

# ── DATOS ──────────────────────────────────────────────────────────────────────
sch = ct.load_schedules()

def snapshot(season):
    reg = sch[(sch["season"] == season) & (sch["game_type"] == "REG") & sch["home_score"].notna()]
    if reg.empty:
        raise SystemExit(f"Sin datos para la temporada {season}.")
    max_week = int(reg["week"].max())
    df, _ = ct.compute(sch, season, max_week)
    row = df[df["team"] == team]
    if row.empty:
        raise SystemExit(f"Equipo '{team}' no encontrado en {season}.")
    return row.iloc[0]

row_a = snapshot(season_a)
row_b = snapshot(season_b)
n_a, n_b = int(row_a["n_ok"]), int(row_b["n_ok"])

metrics = list(ct.THRESHOLDS.keys())
filas = []
for m in metrics:
    cfg = ct.THRESHOLDS[m]
    ok_a, ok_b = row_a.get(f"ok_{m}"), row_b.get(f"ok_{m}")
    ok_a = bool(ok_a) if pd.notna(ok_a) else None
    ok_b = bool(ok_b) if pd.notna(ok_b) else None
    filas.append(dict(
        label=cfg["label"], ok_a=ok_a, ok_b=ok_b,
        val_a=fmt_val(m, row_a.get(f"val_{m}")),
        val_b=fmt_val(m, row_b.get(f"val_{m}")),
        arreglado=(ok_a is False and ok_b is True),
        roto=(ok_a is True and ok_b is False),
    ))

n_arreglados = sum(f["arreglado"] for f in filas)
n_rotos      = sum(f["roto"] for f in filas)

print(f"\n{team}: {season_a} {n_a}/12  ->  {season_b} {n_b}/12")
print(f"Arreglados: {n_arreglados}  ·  Rotos: {n_rotos}")

# ── DIBUJO ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10.5), facecolor=BG)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Cabecera
ax.add_patch(plt.Rectangle((0, 88), 100, 12, color=CARD, zorder=0))
logo = load_logo(team, base_zoom=0.09)
if logo is not None:
    ab = AnnotationBbox(logo, (8, 94), frameon=False, zorder=3)
    ax.add_artist(ab)
ax.text(50, 97.0, f"{team}  ·  ¿QUÉ CAMBIÓ?", ha="center", va="center",
        fontsize=17, fontweight="bold", color=FG, zorder=2)
ax.text(50, 93.0, f"Los 12 requisitos de campeón · {season_a} vs {season_b}",
        ha="center", va="center", fontsize=9.5, color="#888888",
        fontstyle="italic", zorder=2)

col_score = GREEN if n_b == 12 else FG
ax.text(23, 89.8, f"{season_a}", ha="center", va="center", fontsize=9,
        color="#9aa3b5", zorder=2)
ax.text(23, 88.6, f"{n_a}/12", ha="center", va="center", fontsize=13,
        fontweight="bold", color=FG, zorder=2)
ax.annotate("", xy=(58, 89.2), xytext=(38, 89.2),
            arrowprops=dict(arrowstyle="-|>", color="#8fa1c0", lw=2), zorder=2)
ax.text(77, 89.8, f"{season_b}", ha="center", va="center", fontsize=9,
        color="#9aa3b5", zorder=2)
ax.text(77, 88.6, f"{n_b}/12", ha="center", va="center", fontsize=13,
        fontweight="bold", color=col_score, zorder=2)

# Tabla
x_lbl, x_a, x_b, x_val = 2.5, 60.0, 76.0, 88.0
y = 83.5
row_h = 6.35
for f in filas:
    ax.add_patch(plt.Rectangle((1.5, y - row_h/2 + 0.3), 97, row_h - 0.6,
                               color=CARD, zorder=0, alpha=0.6))
    if f["arreglado"]:
        ax.add_patch(plt.Rectangle((1.5, y - row_h/2 + 0.3), 3, row_h - 0.6,
                                   color=GREEN, zorder=1))
    elif f["roto"]:
        ax.add_patch(plt.Rectangle((1.5, y - row_h/2 + 0.3), 3, row_h - 0.6,
                                   color=RED, zorder=1))

    ax.text(x_lbl + 3, y + 0.9, f["label"], ha="left", va="center",
            fontsize=8.7, fontweight="bold", color=FG, zorder=2)
    etiqueta = ("arreglado ✓" if f["arreglado"] else
               "roto ✗" if f["roto"] else
               "se mantiene")
    col_etq = GREEN if f["arreglado"] else (RED if f["roto"] else "#666666")
    ax.text(x_lbl + 3, y - 1.4, etiqueta, ha="left", va="center",
            fontsize=6.8, color=col_etq, zorder=2, fontstyle="italic")

    for x, ok, val in [(x_a, f["ok_a"], f["val_a"]), (x_b, f["ok_b"], f["val_b"])]:
        col = GREEN if ok else (RED if ok is False else "#555555")
        sym = "✓" if ok else ("✗" if ok is False else "–")
        ax.text(x, y + 0.9, sym, ha="center", va="center", fontsize=11,
                fontweight="bold", color=col, zorder=2)
        ax.text(x, y - 1.4, val, ha="center", va="center", fontsize=7,
                color="#9aa3b5", zorder=2)

    y -= row_h

fig.text(0.985, 0.012, "@CuartayDato", ha="right", va="bottom", fontsize=9,
         color="#888888", alpha=0.85, fontstyle="italic")
fig.text(0.012, 0.012, "Fuente: nflverse-data  ·  Ritmo prorrateado a 17 partidos",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")

out = f"evolucion_{team}_{season_a}_{season_b}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
