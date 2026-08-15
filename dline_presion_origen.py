"""
dline_presion_origen.py
Espejo defensivo de oline_presion_origen.py: no que presion SUFRE cada ataque,
sino cual GENERA cada defensa y desde donde.

- Sin equipo (Enter): heatmap 32 equipos — tasa de presion + reparto por origen
  (interior, exterior, blitz LB, blitz DB).
- Con equipo (ej: DEN): perfil del equipo contra la media de la liga, con los
  generadores de presion del equipo como dato secundario.

Nota de datos (importante):
  Las presiones son REALES, no un proxy: `prss` de pfr_advstats incluye hurries,
  que es lo que faltaba en 2025 para poder hacer este script (ver decisiones).

  La clasificacion interior/exterior NO se puede hacer con la posicion de PFR ni
  con la de nflverse: ambas fallan en los edge rushers modernos — Micah Parsons
  es "DL" para PFR y "LB" para nflverse, y con esas etiquetas Green Bay salia
  como una defensa que presiona por dentro cuando es justo lo contrario. La
  unica fuente fiable es `depth_chart_position` del roster, que le da OLB.
  Es la misma fuente que ya usa oline_presion_origen.

  Aun asi el depth chart no distingue el DE de un 4-3 (edge) del de un 3-4
  (juega por dentro), asi que se afina con el peso: DE de 280 lb o mas cuenta
  como interior. Ver PESO_INTERIOR.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pbp_loader import (cargar_pbp, cargar_pfr, cargar_rosters, salida, sello,
                        season_cli)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON    = season_cli()
BG        = "#0f1115"
CARD      = "#151924"
FG        = "#EDEDED"
GRID      = "#2a2f3a"
DPI       = 170
LOGOS_DIR = "logos"

ORIGENES = ["INT", "EXT", "LB", "DB"]
ORIGEN_LABELS = {
    "INT": "Interior\nDT · NT",
    "EXT": "Exterior\nDE · OLB",
    "LB":  "Blitz LB\nILB · MLB",
    "DB":  "Blitz DB\nCB · S",
}
ORIGEN_COLOR = {
    "INT": "#2d6cdf",   # azul   — trenches por dentro
    "EXT": "#06d6a0",   # verde  — perimetro
    "LB":  "#ffd166",   # ambar  — blitz de linebacker
    "DB":  "#d84a4a",   # rojo   — blitz de secundario
}

# depth_chart_position (roster) → origen; el position generico es el fallback
MAPA_DEPTH = {
    "DT": "INT", "NT": "INT",
    "DE": "EXT", "OLB": "EXT", "EDGE": "EXT", "RUSH": "EXT",
    "JACK": "EXT", "LEO": "EXT",
    "LB": "LB", "ILB": "LB", "MLB": "LB", "WLB": "LB", "SLB": "LB",
    "MIKE": "LB", "WILL": "LB", "SAM": "LB",
    "CB": "DB", "DB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "NB": "DB",
}
MAPA_POS = {
    "DT": "INT", "NT": "INT",
    "DE": "EXT", "OLB": "EXT", "EDGE": "EXT",
    "LB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
}


# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.038):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        # Recorta margenes transparentes: algunos archivos traen mucho aire
        # (NYJ: tinta 3768x1186 en lienzo 4096x4096) y sin recorte salen enanos
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        # Normaliza por el area de tinta real; los wordmarks apaisados
        # pueden ensancharse hasta 1.8x para compensar su poca altura
        z = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * base_zoom:
            z = 900.0 * base_zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


# "DE" no significa lo mismo en un 4-3 que en un 3-4: en el segundo juega por
# dentro. El depth chart no distingue los dos, pero el peso si — y el corte
# cae donde lo pone el propio futbol. Comprobado sobre los 60 DE con >=8
# presiones en 2025: por debajo son edges sin discusion (Will Anderson 243,
# Leonard Floyd 240) y por encima son interiores puros (Leonard Williams 302,
# Derrick Brown 318, Jonathan Allen 300). Reclasifica 14 de 60.
PESO_INTERIOR = 280


def clasificar(depth, pos, peso=None):
    d = depth.upper() if isinstance(depth, str) else ""
    if d == "DE" and peso and float(peso) >= PESO_INTERIOR:
        return "INT"
    if d in MAPA_DEPTH:
        return MAPA_DEPTH[d]
    if isinstance(pos, str) and pos.upper() in MAPA_POS:
        return MAPA_POS[pos.upper()]
    return None


def clave(n):
    """Cruce entre fuentes: PFR escribe 'Myles Garrett' y el roster igual, pero
    los sufijos y los puntos varian. Inicial + apellido es estable en ambas."""
    n = str(n).lower().replace(".", " ").replace("'", "").replace("-", " ")
    p = [x for x in n.split() if x not in ("jr", "sr", "ii", "iii", "iv", "v")]
    if not p:
        return ""
    return p[0] if len(p) == 1 else p[0][0] + " " + p[-1]


# ── INPUT ──────────────────────────────────────────────────────────────────────
team_input = input("Equipo (siglas, ej: DEN — Enter = heatmap 32 equipos): ").strip().upper()

# ── DATOS ──────────────────────────────────────────────────────────────────────
# Dropbacks encajados por cada defensa: es el denominador de la tasa de presion
df, SEASON = cargar_pbp(SEASON, columns=["defteam", "qb_dropback"])
df["qb_dropback"] = pd.to_numeric(df["qb_dropback"], errors="coerce")
dropbacks = df[(df["qb_dropback"] == 1) & df["defteam"].notna()] \
    .groupby("defteam").size()
print(f"PBP {SEASON}: {int(dropbacks.sum()):,} dropbacks")

# Presiones reales por jugador (incluyen hurries)
pfr, _ = cargar_pfr("def", SEASON)
pfr = pfr[pfr["tm"] != "3TM"].copy()          # fila agregada de PFR, no un equipo
pfr["prss"] = pd.to_numeric(pfr["prss"], errors="coerce").fillna(0)
pfr = pfr[pfr["prss"] > 0]

# Clasificacion por depth chart (la unica fiable para edge vs interior)
ros, _ = cargar_rosters(SEASON)
mapa = {}
for _, r in ros.iterrows():
    cl = clasificar(r.get("depth_chart_position"), r.get("position"),
                    r.get("weight"))
    if cl:
        mapa.setdefault(clave(r.get("full_name")), cl)

pfr["origen"] = pfr["player"].map(clave).map(mapa)
sin = pfr[pfr["origen"].isna()]["prss"].sum() / pfr["prss"].sum() * 100
print(f"Presiones clasificadas: {100 - sin:.0f}%  (sin clasificar {sin:.0f}%)")

pres = pfr.dropna(subset=["origen"])

# ── TABLA POR EQUIPO ──────────────────────────────────────────────────────────
tabla = pres.pivot_table(index="tm", columns="origen", values="prss",
                         aggfunc="sum").reindex(columns=ORIGENES).fillna(0)
tabla["total"]     = tabla[ORIGENES].sum(axis=1)
tabla["dropbacks"] = dropbacks
tabla = tabla.dropna(subset=["dropbacks"])
tabla["press_pct"] = tabla["total"] / tabla["dropbacks"] * 100
for o in ORIGENES:
    tabla[f"pct_{o}"] = tabla[o] / tabla["total"] * 100
tabla = tabla.sort_values("press_pct", ascending=False)   # mas presion arriba

media_liga = {o: tabla[f"pct_{o}"].mean() for o in ORIGENES}

print()
print(f"{'Def':<5}{'Press%':>8}" + "".join(f"{o:>9}" for o in ORIGENES))
for tm, r in tabla.iterrows():
    print(f"{tm:<5}{r['press_pct']:>7.1f}%" +
          "".join(f"{r[f'pct_{o}']:>8.0f}%" for o in ORIGENES))


# ── HEATMAP 32 EQUIPOS ────────────────────────────────────────────────────────
def draw_heatmap():
    teams  = tabla.index.tolist()
    n      = len(teams)
    cols   = ["PRESS"] + ORIGENES
    labels = {"PRESS": "Tasa de presión\n(presiones/dropback)"} | ORIGEN_LABELS
    logo_w = 1.2
    fig_w  = logo_w + len(cols) * 1.5 + 1.4
    fig_h  = max(8, n * 0.52 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.995, bottom=0.008)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(-logo_w, len(cols))
    ax.set_ylim(-1, n + 2.0)

    # Verde = MAS presion (al reves que en la version ofensiva: aqui presionar
    # es bueno). En las columnas de origen el color no valora, solo compara.
    cmaps  = {"PRESS": plt.cm.RdYlGn}
    normas = {}
    for c in cols:
        vals = tabla["press_pct"] if c == "PRESS" else tabla[f"pct_{c}"]
        normas[c] = Normalize(vmin=np.percentile(vals, 5),
                              vmax=np.percentile(vals, 95))
        cmaps.setdefault(c, plt.cm.Blues)

    for i, tm in enumerate(teams):
        y = n - i - 1
        for j, c in enumerate(cols):
            val = tabla.loc[tm, "press_pct"] if c == "PRESS" else tabla.loc[tm, f"pct_{c}"]
            bgc = cmaps[c](normas[c](val))
            ax.add_patch(plt.Rectangle((j, y), 1, 1, color=bgc,
                                       linewidth=0.4, edgecolor=BG, zorder=1))
            lum = 0.299 * bgc[0] + 0.587 * bgc[1] + 0.114 * bgc[2]
            txt = "#0a0e13" if lum > 0.45 else FG
            if c == "PRESS":
                ax.text(j + 0.5, y + 0.5, f"{val:.1f}%", ha="center", va="center",
                        color=txt, fontsize=8.5, fontweight="bold", zorder=2)
            else:
                ax.text(j + 0.5, y + 0.60, f"{val:.0f}%", ha="center", va="center",
                        color=txt, fontsize=8, fontweight="bold", zorder=2)
                ax.text(j + 0.5, y + 0.25, f"{tabla.loc[tm, c]:.0f}", ha="center",
                        va="center", color=txt, fontsize=6.5, zorder=2)

        img = load_logo(tm, base_zoom=0.036)
        if img is not None:
            ax.add_artist(AnnotationBbox(img, (-logo_w / 2, y + 0.5),
                                         frameon=False, zorder=3))
        else:
            ax.text(-logo_w / 2, y + 0.5, tm, ha="center", va="center",
                    color=FG, fontsize=7.5, fontweight="bold")

    for j, c in enumerate(cols):
        ax.text(j + 0.5, n + 0.42, labels[c], ha="center", va="center",
                color=FG, fontsize=8, fontweight="bold", linespacing=1.3)
    ax.axhline(n, color=GRID, linewidth=0.8, zorder=3)

    fig.text(0.5, 0.99, f"¿Desde dónde presiona cada defensa? | NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.975,
             "Ordenado por tasa de presión (mejor defensa arriba)  ·  "
             "Origen = % de las presiones del equipo según el puesto del rusher  ·  "
             "el número pequeño son presiones",
             ha="center", va="top", fontsize=8.5, color="#888888", fontstyle="italic")
    fig.text(0.01, 0.005,
             f"Fuente: nflverse-data · Pro Football Reference (presiones reales, con hurries)  |  {sello(SEASON)}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.99, 0.005, "@CuartayDato", ha="right", va="bottom", fontsize=9,
             color="#888888", alpha=0.85, fontstyle="italic")

    out = salida(f"dline_presion_origen_{SEASON}.png", SEASON)
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGuardado: {out}")


# ── PERFIL DE UN EQUIPO ───────────────────────────────────────────────────────
def draw_equipo(team):
    if team not in tabla.index:
        raise SystemExit(f"\n  Equipo '{team}' sin datos. Prueba con las siglas "
                         f"(DEN, SF, KC...).\n")
    r    = tabla.loc[team]
    rank = list(tabla.index).index(team) + 1

    fig = plt.figure(figsize=(13.5, 7.6), dpi=DPI, facecolor=BG)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.16,
                           left=0.055, right=0.965, top=0.80, bottom=0.09)

    # ── Izquierda: reparto por origen contra la media de la liga ──────────
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG)
    y = np.arange(len(ORIGENES))[::-1]
    for i, o in enumerate(ORIGENES):
        yy = y[i]
        ax.barh(yy + 0.17, r[f"pct_{o}"], height=0.34,
                color=ORIGEN_COLOR[o], zorder=3)
        ax.barh(yy - 0.19, media_liga[o], height=0.26,
                color="#39404f", zorder=2)
        ax.text(r[f"pct_{o}"] + 1.2, yy + 0.17, f"{r[f'pct_{o}']:.0f}%",
                va="center", ha="left", color=FG, fontsize=11, fontweight="bold")
        ax.text(media_liga[o] + 1.2, yy - 0.19, f"liga {media_liga[o]:.0f}%",
                va="center", ha="left", color="#8A93A6", fontsize=8.5)
        ax.text(-2.0, yy, ORIGEN_LABELS[o].replace("\n", "  "), va="center",
                ha="right", color=FG, fontsize=10.5, linespacing=1.2)

    ax.set_yticks([])
    ax.set_xlim(0, max(max(tabla[[f"pct_{o}" for o in ORIGENES]].max()), 60) * 1.25)
    ax.set_ylim(-0.7, len(ORIGENES) - 0.3)
    ax.set_xlabel("% de las presiones del equipo", color="#8A93A6", fontsize=9.5)
    ax.tick_params(colors="#8A93A6", labelsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", linestyle="--", alpha=0.18, color=GRID, zorder=0)
    ax.set_title("Reparto de la presión", color=FG, fontsize=12.5,
                 fontweight="bold", pad=12, loc="left")

    # ── Derecha: quien la genera ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG)
    ax2.axis("off")
    top = pres[pres["tm"] == team].nlargest(9, "prss")
    ax2.set_title("Quién la genera", color=FG, fontsize=12.5,
                  fontweight="bold", pad=12, loc="left")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(len(top) + 0.2, -1.0)
    maxp = float(top["prss"].max()) if len(top) else 1.0

    for i, (_, p) in enumerate(top.iterrows()):
        o = p["origen"]
        ax2.barh(i, p["prss"] / maxp * 5.4, left=4.3, height=0.56,
                 color=ORIGEN_COLOR[o], alpha=0.85, zorder=2)
        ax2.text(4.15, i, str(p["player"]), va="center", ha="right",
                 color=FG, fontsize=10)
        ax2.text(4.3 + p["prss"] / maxp * 5.4 + 0.12, i, f"{p['prss']:.0f}",
                 va="center", ha="left", color=FG, fontsize=9.5,
                 fontweight="bold", fontfamily="monospace")
        ax2.text(0.05, i, ORIGEN_LABELS[o].split("\n")[0], va="center",
                 ha="left", color=ORIGEN_COLOR[o], fontsize=8.5)
    ax2.text(4.3, -0.75, "presiones en la temporada", color="#8A93A6",
             fontsize=8.5, va="center", style="italic")

    # ── Cabecera ───────────────────────────────────────────────────────────
    img = load_logo(team, base_zoom=0.10)
    if img is not None:
        fig.add_artist(AnnotationBbox(img, (0.075, 0.905), frameon=False,
                                      xycoords="figure fraction"))
    fig.text(0.135, 0.925, f"¿Desde dónde presiona {team}?", ha="left", va="center",
             fontsize=21, fontweight="bold", color=FG)
    fig.text(0.135, 0.868,
             f"{r['total']:.0f} presiones en {int(r['dropbacks'])} dropbacks  ·  "
             f"tasa {r['press_pct']:.1f}%  ·  {rank}º de la liga",
             ha="left", va="center", fontsize=11.5, color="#8A93A6")

    fig.text(0.012, 0.018,
             f"Fuente: nflverse-data · Pro Football Reference (presiones reales, con hurries)  ·  "
             f"puesto según depth chart (DE ≥{PESO_INTERIOR} lb = interior)  ·  {sello(SEASON)}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.988, 0.018, "@CuartayDato", ha="right", va="bottom", fontsize=9,
             color="#888888", alpha=0.85, fontstyle="italic")

    out = salida(f"dline_presion_origen_{team}_{SEASON}.png", SEASON)
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGuardado: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if team_input:
    draw_equipo(team_input)
else:
    draw_heatmap()
