"""
under_center.py
Uso de formacion BAJO CENTRO (under center) por equipo y por liga.

Nace de un hallazgo de sep-2026: en la semana 1 de 2026 la liga paso del 29%
al 42% de jugadas bajo centro, el mayor salto de los ultimos cinco anos, y el
EPA de esas jugadas paso de negativo a positivo. Los ataques buscan looks mas
ventajosos contra defensas pesadas.

Es una metrica de CONTEO (columna `shotgun` del PBP), no de modelo: se replica
exactamente, sin depender de charting de pago.

Uso:
  python under_center.py                    # semana con datos mas reciente
  python under_center.py --season 2026 --week 1
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pbp_loader import (cargar_pbp, salida, season_cli, week_cli, sello,
                        ultima_semana)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli()
BG     = "#0f1115"
CARD   = "#151924"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
ACCENT = "#2d6cdf"
VERDE  = "#06d6a0"
ROJO   = "#d84a4a"
LOGOS_DIR = "logos"
DPI    = 200
N_ANOS = 5          # temporadas de la serie historica

COLS = ["season", "week", "season_type", "posteam", "play_type", "epa",
        "shotgun", "qb_kneel", "qb_spike"]


# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, zoom=0.030):
    """Logo normalizado por el AREA DE TINTA real (ver CLAUDE.md).

    Hay escudos con mucho margen transparente (NYJ es un wordmark 3768x1186
    dentro de un lienzo 4096x4096): sin recortar salen aplastados.
    """
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
        z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * zoom:
            z = 900.0 * zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


def jugadas(df, week=None):
    """Jugadas de ataque validas: pase o carrera, con EPA y con formacion.

    Fuera arrodillados y spikes, que no son decisiones de formacion y en la
    semana 1 de 2026 pesaban lo suyo en los equipos que cerraron partidos.
    """
    o = df[df["play_type"].isin(["pass", "run"]) &
           df["epa"].notna() & df["shotgun"].notna()].copy()
    for c in ("qb_kneel", "qb_spike"):
        if c in o.columns:
            o = o[o[c] != 1]
    if week is not None:
        o = o[o["week"] == week]
    return o


def por_equipo(o):
    """DataFrame por equipo: % bajo centro, EPA bajo centro y jugadas."""
    tot = o.groupby("posteam").size()
    uc  = o[o["shotgun"] == 0]
    n_uc = uc.groupby("posteam").size().reindex(tot.index).fillna(0)
    return pd.DataFrame({
        "uc":     n_uc / tot * 100,
        "epa_uc": uc.groupby("posteam")["epa"].mean().reindex(tot.index),
        "n":      tot,
    })


def liga(o):
    """(% bajo centro, EPA bajo centro, EPA del resto) de toda la liga."""
    uc = o[o["shotgun"] == 0]
    return (len(uc) / len(o) * 100 if len(o) else np.nan,
            uc["epa"].mean(),
            o[o["shotgun"] == 1]["epa"].mean())


# ── DATOS ──────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON, columns=COLS, avisar=False)
week = week_cli() or ultima_semana(SEASON) or int(df["week"].max())

act = jugadas(df, week)
if act.empty:
    raise SystemExit(f"Sin jugadas para la semana {week} de {SEASON}.")

# Norma del equipo: su temporada anterior COMPLETA (el punto de partida real)
prev, _ = cargar_pbp(SEASON - 1, columns=COLS, avisar=False)
base = jugadas(prev)

t_act  = por_equipo(act)
t_base = por_equipo(base)

tabla = pd.DataFrame({
    "uc_act":  t_act["uc"],
    "uc_base": t_base["uc"].reindex(t_act.index),
    "epa_uc":  t_act["epa_uc"],
    "n":       t_act["n"],
}).dropna(subset=["uc_act"])
tabla["cambio"] = tabla["uc_act"] - tabla["uc_base"]
tabla = tabla.sort_values("cambio", ascending=False)

# Serie de liga: la MISMA semana de cada temporada, para comparar peras con peras
serie = []
for s in range(SEASON - N_ANOS + 1, SEASON + 1):
    try:
        d_s = df if s == SEASON else cargar_pbp(s, columns=COLS, avisar=False)[0]
        o_s = jugadas(d_s, week)
        if len(o_s) == 0:
            continue
        pct, epa_uc, epa_no = liga(o_s)
        serie.append(dict(season=s, pct=pct, epa_uc=epa_uc, epa_no=epa_no,
                          n=len(o_s[o_s["shotgun"] == 0])))
    except Exception as e:
        print(f"  Aviso: sin serie para {s} ({type(e).__name__})")
serie = pd.DataFrame(serie)

# ── CONSOLA ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}\n  BAJO CENTRO — semana {week} de {SEASON}\n{'='*70}")
if not serie.empty:
    for _, r in serie.iterrows():
        print(f"  {int(r.season)}: {r.pct:5.1f}% bajo centro ({int(r.n):3d} jugadas) "
              f"| EPA bajo centro {r.epa_uc:+.3f} | resto {r.epa_no:+.3f}")
print("\n  Mayores subidas respecto a su temporada anterior:")
for eq, r in tabla.head(6).iterrows():
    print(f"    {eq:4} {r.uc_act:5.1f}% (era {r.uc_base:5.1f}%) "
          f"{r.cambio:+5.1f} pp | EPA bajo centro {r.epa_uc:+.3f}")

# ── PNG ────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 15), facecolor=BG)
gs  = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.5], hspace=0.13,
                       left=0.085, right=0.965, top=0.935, bottom=0.045)

fig.text(0.5, 0.972, "LA NFL SE PONE BAJO CENTRO",
         ha="center", va="center", fontsize=21, fontweight="bold", color=FG)
fig.text(0.5, 0.950,
         f"Jugadas sin shotgun · semana {week} de cada temporada · "
         f"cambio de cada equipo respecto a su {SEASON - 1} completo",
         ha="center", va="center", fontsize=10.5, color="#9aa3b5",
         fontstyle="italic")

# ── Panel 1: la liga, temporada a temporada ───────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(CARD)
for s in ax1.spines.values():
    s.set_visible(False)
ax1.tick_params(colors="#9aa3b5", labelsize=10)
ax1.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.5)
ax1.set_axisbelow(True)

if not serie.empty:
    xs = np.arange(len(serie))
    cols = [ACCENT] * (len(serie) - 1) + [VERDE]
    ax1.bar(xs, serie["pct"], color=cols, width=0.6, zorder=3)
    for x, (_, r) in zip(xs, serie.iterrows()):
        ax1.text(x, r.pct + 1.0, f"{r.pct:.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=FG, zorder=4)
        ax1.text(x, 2.0, f"EPA {r.epa_uc:+.3f}", ha="center", va="bottom",
                 fontsize=8.5, color="#0f1115" if r.season == SEASON else "#cfd6e4",
                 zorder=4)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(int(s)) for s in serie["season"]], fontsize=11)
    ax1.set_ylim(0, max(serie["pct"]) * 1.25)
    ax1.set_ylabel("% de jugadas bajo centro", color="#9aa3b5", fontsize=10)

# ── Panel 2: equipo a equipo ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(CARD)
for s in ax2.spines.values():
    s.set_visible(False)
ax2.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.5)
ax2.set_axisbelow(True)
ax2.tick_params(colors="#9aa3b5", labelsize=9)

ys = np.arange(len(tabla))[::-1]
for y, (eq, r) in zip(ys, tabla.iterrows()):
    col = VERDE if r.cambio >= 0 else ROJO
    ax2.barh(y, r.cambio, color=col, height=0.62, zorder=3, alpha=0.92)
    logo = load_logo(eq, zoom=0.026)
    if logo is not None:
        ax2.add_artist(AnnotationBbox(logo, (0, y), frameon=False, zorder=5,
                                      xycoords=("data", "data")))
    lado = 1 if r.cambio >= 0 else -1
    ax2.text(r.cambio + lado * 0.7, y,
             f"{r.uc_act:.0f}%  ({r.cambio:+.1f} pp)",
             ha="left" if lado > 0 else "right", va="center",
             fontsize=8.5, fontweight="bold", color=col, zorder=4)

ax2.axvline(0, color="#6b7280", linewidth=1.2, zorder=4)
ax2.set_yticks([])
ax2.set_ylim(-1, len(tabla))
# Margen asimetrico: las etiquetas cuelgan del extremo de cada barra, asi que
# el lado con mas recorrido necesita mas aire que el otro
ax2.set_xlim(min(tabla["cambio"].min(), 0) - 9.0,
             max(tabla["cambio"].max(), 0) + 11.0)
ax2.set_xlabel(f"Cambio en puntos porcentuales respecto a su temporada {SEASON - 1}",
               color="#9aa3b5", fontsize=10)
ax2.set_title("Quién ha cambiado y cuánto", color=FG, fontsize=12,
              fontweight="bold", pad=10, loc="left")

fig.text(0.085, 0.012,
         f"Fuente: nflverse-data  ·  {sello(SEASON)}  ·  "
         f"sin arrodillados ni spikes  ·  bajo centro = jugada sin shotgun",
         ha="left", va="bottom", fontsize=8, color="#555555", fontstyle="italic")
fig.text(0.965, 0.012, "@CuartayDato", ha="right", va="bottom",
         fontsize=9, color="#888888", alpha=0.8, fontstyle="italic")

out = salida(f"under_center_{SEASON}.png", SEASON, week)
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nGuardado: {out}")
