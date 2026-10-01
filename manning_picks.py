"""
manning_picks.py — los picks de Manning Bot de una jornada en un PNG (01-oct-2026).

Lee lo que ya ha escrito el bot, sin volver a predecir nada: asi el numero
del PNG y el del post son el mismo por construccion.

  picks   salidas/{año}/w{N-1}/textos/bot_picks.txt (lo deja el batch del
          martes en la carpeta de la ultima jornada jugada)
  orden   salidas/{año}/w{N}/textos/previas_numeros_{año}_w{N}.txt, que va
          por kickoff (TNF primero); si aun no existe, el orden del bot

Una fila por partido: visitante a la izquierda, local a la derecha y la barra
partida con la probabilidad de cada uno; el pick en naranja (el color de los
LED del bot) y el otro lado apagado. Sin titulo (regla de la casa): la
mascota de portada va pequeña abajo, con el sello y la marca de agua.

    python manning_picks.py --week 4
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pbp_loader import salida, sello, season_cli, week_cli, temporada_actual, proxima_semana

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli() or temporada_actual()
WEEK   = week_cli() or proxima_semana()
BG, CARD, FG, GRID = "#0f1115", "#151924", "#EDEDED", "#2a2f3a"
LED    = "#ff8a1f"            # el naranja de Manning Bot
APAGADO = "#3a4050"
RAIZ   = os.path.dirname(os.path.abspath(__file__))
LOGOS_DIR = os.path.join(RAIZ, "logos")
MASCOTA = os.path.join(RAIZ, "marca", "manning_bot", "manning_bot_portada.png")
DIAS = {"Thursday": "JUE", "Friday": "VIE", "Saturday": "SÁB", "Sunday": "DOM",
        "Monday": "LUN", "Tuesday": "MAR", "Wednesday": "MIÉ"}


def carpeta(w):
    return os.path.join(RAIZ, "salidas", str(SEASON), f"w{int(w):02d}", "textos")


def load_logo(team, zoom=0.030, alpha=1.0):
    """ranking_wrs.load_logo: recorte por tinta real y zoom por su area."""
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    img = plt.imread(path).copy()
    if img.ndim == 3 and img.shape[2] == 4:
        ys, xs = np.where(img[:, :, 3] > 0.02)
        if len(ys):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img[:, :, 3] *= alpha
    h, w = img.shape[:2]
    z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
    if w * z > 900.0 * zoom:
        z = 900.0 * zoom / w
    return OffsetImage(img, zoom=z, resample=True)


# ── DATOS ──────────────────────────────────────────────────────────────────────
ruta_picks = os.path.join(carpeta(WEEK - 1), "bot_picks.txt")
if not os.path.exists(ruta_picks):
    sys.exit(f"No esta {ruta_picks}: el bot de la semana {WEEK} sale en el batch del martes")
texto = open(ruta_picks, encoding="utf-8").read()
m = re.search(r"MANNING BOT .*?Semana (\d+)", texto)
if not m or int(m.group(1)) != WEEK:
    sys.exit(f"{ruta_picks} no es de la semana {WEEK}")
picks = {}
for away, home, pick, p_home, p_away in re.findall(
        r"^\s+([A-Z]{2,3}) @ ([A-Z]{2,3})\s+([A-Z]{2,3})\s+([\d.]+)%\s+([\d.]+)%", texto, re.M):
    picks[(away, home)] = dict(pick=pick, p_home=float(p_home), p_away=float(p_away), dia="")
print(f"Picks de la semana {WEEK}: {len(picks)} partidos ({os.path.relpath(ruta_picks, RAIZ)})")

orden = list(picks)
ruta_orden = os.path.join(carpeta(WEEK), f"previas_numeros_{SEASON}_w{WEEK:02d}.txt")
if os.path.exists(ruta_orden):
    kick = re.findall(r"^== \d+ ([A-Z]{2,3}) @ ([A-Z]{2,3}) — (\w+)", open(ruta_orden, encoding="utf-8").read(), re.M)
    por_kick = [(a, h) for a, h, _ in kick if (a, h) in picks]
    for a, h, d in kick:
        if (a, h) in picks:
            picks[(a, h)]["dia"] = DIAS.get(d, "")
    orden = por_kick + [k for k in orden if k not in por_kick]
else:
    print("  (sin previas_numeros: orden del bot, sin dia)")

# ── FIGURA ────────────────────────────────────────────────────────────────────
n = len(orden)
fig = plt.figure(figsize=(9, 0.62 * n + 2.2))
fig.patch.set_facecolor(BG)
alto_mascota = 1.55 / (0.62 * n + 2.2)          # fraccion de la figura
ax = fig.add_axes([0.03, alto_mascota + 0.02, 0.94, 1 - alto_mascota - 0.03])
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(-0.6, n - 0.4)
ax.axis("off")

X0, X1 = 32, 70                                  # la barra de probabilidad
for i, (a, h) in enumerate(orden):
    y = n - 1 - i
    r = picks[(a, h)]
    gana_local = r["pick"] == h
    ax.add_patch(plt.Rectangle((0.5, y - 0.44), 99, 0.88, color=CARD, zorder=0, lw=0))
    ax.text(4.5, y, r["dia"], color="#888888", fontsize=9, ha="center", va="center",
            fontweight="bold")
    # visitante
    lo = load_logo(a, 0.026, 1.0 if not gana_local else 0.35)
    if lo:
        ax.add_artist(AnnotationBbox(lo, (13, y), frameon=False))
    ax.text(21, y, a, color=FG if not gana_local else "#6b7280", fontsize=12,
            fontweight="bold" if not gana_local else "normal", ha="left", va="center")
    # local
    lo = load_logo(h, 0.026, 1.0 if gana_local else 0.35)
    if lo:
        ax.add_artist(AnnotationBbox(lo, (91, y), frameon=False))
    ax.text(83, y, h, color=FG if gana_local else "#6b7280", fontsize=12,
            fontweight="bold" if gana_local else "normal", ha="right", va="center")
    # barra: el visitante desde la izquierda, el local desde la derecha
    corte = X0 + (X1 - X0) * r["p_away"] / 100
    ax.add_patch(plt.Rectangle((X0, y - 0.22), corte - X0, 0.44,
                               color=APAGADO if gana_local else LED, lw=0, zorder=2))
    ax.add_patch(plt.Rectangle((corte, y - 0.22), X1 - corte, 0.44,
                               color=LED if gana_local else APAGADO, lw=0, zorder=2))
    ax.plot([corte, corte], [y - 0.3, y + 0.3], color=BG, lw=2.5, zorder=3)
    # Un tramo de menos del 15% no tiene sitio para su numero: va por fuera
    fuera_a, fuera_h = r["p_away"] < 15, r["p_home"] < 15
    ax.text(X0 - 0.8 if fuera_a else X0 + 0.8, y, f"{r['p_away']:.1f}%",
            color="#888888" if fuera_a else (BG if not gana_local else FG),
            fontsize=9.5, fontweight="bold", ha="right" if fuera_a else "left",
            va="center", zorder=4)
    ax.text(X1 + 0.8 if fuera_h else X1 - 0.8, y, f"{r['p_home']:.1f}%",
            color="#888888" if fuera_h else (BG if gana_local else FG),
            fontsize=9.5, fontweight="bold", ha="left" if fuera_h else "right",
            va="center", zorder=4)

# Leyenda minima: que lado es cual
ax.text(X0, n - 0.42, "VISITANTE", color="#888888", fontsize=7.5, ha="left", va="bottom")
ax.text(X1, n - 0.42, "LOCAL", color="#888888", fontsize=7.5, ha="right", va="bottom")

# Mascota abajo, pequeña, y el pie
if os.path.exists(MASCOTA):
    axm = fig.add_axes([0.40, 0.005, 0.20, alto_mascota])
    axm.imshow(plt.imread(MASCOTA))
    axm.axis("off")
fig.text(0.03, 0.012, sello(SEASON) + f" · picks de la semana {WEEK}", color="#888888",
         fontsize=8, ha="left", va="bottom")
fig.text(0.97, 0.012, "@CuartayDato", color="#888888", fontsize=9, alpha=0.8,
         fontstyle="italic", ha="right", va="bottom")

out = salida(f"manning_picks_{SEASON}.png", SEASON, week=WEEK)
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Guardado: {out}")
