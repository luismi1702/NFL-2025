"""
dline_presion_origen.py
Espejo defensivo de oline_presion_origen.py: no que presion SUFRE cada ataque,
sino cual GENERA cada defensa y desde donde.

- Sin equipo (Enter): heatmap 32 equipos — tasa de presion + reparto por origen
  (interior, exterior, blitz LB, blitz DB).
- Con equipo (ej: DEN): ficha 4:5 con el pass rush win rate de ESPN, las
  presiones por 100 pases, el origen contra la media y sus tres que mas
  presionan con su win rate de ESPN si estan en el top 20 (03-oct-2026).

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

from pbp_loader import (cargar_pbp, cargar_pfr, cargar_rosters, cargar_web,
                        DatosNoDisponibles, salida, sello, season_cli)
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)

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
# TASA por 100 dropbacks, no reparto: comparar la composicion contra la de la
# liga engaña. SF salia "60% exterior vs 49% de la liga" (parece que presiona
# mucho por fuera) cuando su tasa exterior es 10.1 contra 11.4, o sea MENOS —
# ese 60% solo decia que su poca presion se concentra ahi. Con tasas, las
# cuatro columnas suman la de presion total.
for o in ORIGENES:
    tabla[f"pct_{o}"] = tabla[o] / tabla["dropbacks"] * 100
tabla = tabla.sort_values("press_pct", ascending=False)   # mas presion arriba

media_liga = {o: tabla[f"pct_{o}"].mean() for o in ORIGENES}

print()
print(f"{'Def':<5}{'TOTAL':>8}" + "".join(f"{o:>9}" for o in ORIGENES))
for tm, r in tabla.iterrows():
    print(f"{tm:<5}{r['press_pct']:>8.1f}" +
          "".join(f"{r[f'pct_{o}']:>9.1f}" for o in ORIGENES))


# ── WIN RATES DE ESPN (copiados a mano a datos_web/, 03-oct-2026) ─────────────
# PRWR = % de duelos en los que el rusher gana a su bloqueo antes de 2,5 s.
# Por equipo estan los 32; por jugador, solo el top 20 de edges y de DTs.
try:
    WR_EQ, SEM_WR = cargar_web("espn_win_rates_equipos", SEASON)
    WR_EQ = WR_EQ.set_index("team")
    WR_JUG, _ = cargar_web("espn_win_rates_jugadores", SEASON)
except DatosNoDisponibles as e:
    print(f"  Aviso: sin win rates de ESPN ({e})")
    WR_EQ, WR_JUG, SEM_WR = None, None, None

press_rk = tabla["press_pct"].rank(ascending=False, method="min").astype(int)  # 1 = mas presion
CORTO = {"INT": "por dentro", "EXT": "por fuera", "LB": "con blitz de LB", "DB": "con blitz de DB"}
MINI = {"INT": "dentro", "EXT": "fuera", "LB": "blitz LB", "DB": "blitz DB"}
ETIQ = {"INT": "Por dentro (DT, NT)", "EXT": "Por fuera (DE, OLB)", "LB": "Blitz de LB",
        "DB": "Blitz de DB (CB, S)"}
SELLO_WR = (f"NFL {SEASON} · win rates de ESPN hasta sem. {SEM_WR} · presiones de PFR"
            if SEM_WR else f"{sello(SEASON)} · presiones de PFR")


def wr_jugador(nombre, team):
    """(win rate, puesto, grupo) del pass rush de ESPN si esta en su top 20."""
    if WR_JUG is None:
        return None
    j = WR_JUG[(WR_JUG.metrica == "pass_rush") & (WR_JUG.team == team)]
    j = j[j.jugador.map(clave) == clave(nombre)]
    if j.empty:
        return None
    j = j.iloc[0]
    return int(j.win_rate), int(j.rk), "edge" if j.grupo == "EDGE" else "DT"


def titular_equipo(team):
    T = nombre_equipo(team)
    if WR_EQ is not None and team in WR_EQ.index:
        pr, rk = int(WR_EQ.loc[team, "prwr"]), int(WR_EQ.loc[team, "prwr_rk"])
        if rk <= 5:
            return f"{T} {v(T, 'gana', 'ganan')} el {pr}% de sus duelos de pass rush, {rk}º de la NFL."
        if rk >= 28:
            return f"{T} solo {v(T, 'gana', 'ganan')} el {pr}% de sus duelos de pass rush, {rk}º de la NFL."
    r = tabla.loc[team]
    ratio = {o: r[f"pct_{o}"] / media_liga[o] for o in ORIGENES if media_liga[o] > 0}
    o = max(ratio, key=ratio.get)
    if ratio[o] >= 1.4 and r[o] >= 5:
        return (f"{T} {v(T, 'presiona', 'presionan')} sobre todo {CORTO[o]}: "
                f"{num(r[f'pct_{o}'], 1, signo=False)} por cada 100 pases "
                f"(media, {num(media_liga[o], 1, signo=False)}).")
    return None


def draw_equipo(team):
    if team not in tabla.index:
        raise SystemExit(f"\n  Equipo '{team}' sin datos. Prueba con las siglas (DEN, SF, KC...).\n")
    r = tabla.loc[team]
    dely = pres[pres["tm"] == team]
    n_eq = len(tabla)
    fig = lienzo()
    antetitulo(fig, f"Pass rush · {team}")
    tit = titular_equipo(team)
    y_top = (titular(fig, tit[0].upper() + tit[1:], y=0.925) - 0.03) if tit else 0.90

    yc = y_top - 0.045
    poner_logo(fig, team, L + 0.035, yc, zoom=0.022)
    fig.text(L + 0.08, yc, team, family=TXT, weight="bold", size=10, color=color_equipo(team),
             va="center")
    fig.text(L + 0.15, yc, "·  PRESIÓN SOBRE EL QB", family=TXT, weight="semibold", size=8.5,
             color=GRIS1, va="center")

    yb = yc - 0.06
    if WR_EQ is not None and team in WR_EQ.index:
        pr, rk = int(WR_EQ.loc[team, "prwr"]), int(WR_EQ.loc[team, "prwr_rk"])
        etiqueta(fig, "Duelos ganados en pass rush (ESPN)", L + 0.01, yb)
        t = numero(fig, f"{pr}", L + 0.005, yb - 0.065, size=44)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        fig.text(bb.x1 / fig.bbox.width + 0.004, yb - 0.08, "%", family=DISP, weight="semibold",
                 size=19, color=FG, va="center")
        fig.text(L + 0.01, yb - 0.13, ordinal(rk), family=TXT, weight="bold", size=9,
                 color=color_rango(rk), va="center")
        nota(fig, f"Contra la carrera: {int(WR_EQ.loc[team, 'rswr'])}% ({int(WR_EQ.loc[team, 'rswr_rk'])}º)",
             L + 0.01, yb - 0.158)
        pista32(fig, L + 0.01, yb - 0.195, 0.36, rk)
    prk = int(press_rk[team])
    etiqueta(fig, "Presiones por 100 pases (PFR)", 0.54, yb)
    numero(fig, num(r["press_pct"], 1, signo=False), 0.535, yb - 0.065, size=44)
    fig.text(0.54, yb - 0.13, ordinal(prk, n_eq), family=TXT, weight="bold", size=9,
             color=color_rango(prk, n_eq), va="center")
    nota(fig, f"{int(r['total'])} presiones en {int(r['dropbacks'])} pases", 0.54, yb - 0.158)
    pista32(fig, 0.54, yb - 0.195, 0.36, prk, n=n_eq)
    linea_v(fig, 0.5, yb - 0.21, yb + 0.015)

    ym = yb - 0.245
    linea_h(fig, ym + 0.025)
    etiqueta(fig, "Desde dónde llegan sus presiones (por 100 pases)", L + 0.01, ym)
    tope = max(max(r[f"pct_{o}"] for o in ORIGENES), max(media_liga.values())) * 1.15 or 1
    x0b, wb = L + 0.30, 0.42
    yy = ym - 0.04
    for o in ORIGENES:
        val, med = r[f"pct_{o}"], media_liga[o]
        fig.text(L + 0.01, yy, ETIQ[o], family=TXT, size=8.3, color=FG, va="center")
        fig.add_artist(Rectangle((x0b, yy - 0.008), wb * val / tope, 0.016,
                                 transform=fig.transFigure,
                                 color=color_equipo(team) if val > med else NEUTRO, lw=0))
        fig.add_artist(Rectangle((x0b + wb * med / tope - 0.001, yy - 0.013), 0.002, 0.026,
                                 transform=fig.transFigure, color=FG, lw=0))
        fig.text(x0b + wb + 0.015, yy, f"{num(val, 1, signo=False)}  ({int(r[o])})", family=TXT,
                 weight="semibold", size=8.3, color=FG, va="center")
        yy -= 0.033
    nota(fig, "Raya blanca = media NFL · en color, por encima de la media · (n) = presiones",
         L + 0.01, yy + 0.008)

    yl = yy - 0.035
    linea_h(fig, yl + 0.02)
    etiqueta(fig, "Los que más presionan", L + 0.01, yl)
    fig.text(R - 0.01, yl, "WIN RATE ESPN", family=TXT, weight="medium", size=6.5, color=GRIS1,
             va="center", ha="right")
    yl -= 0.035
    for _, p in dely.nlargest(3, "prss").iterrows():
        fig.text(L + 0.01, yl, str(p["player"]), family=TXT, weight="semibold", size=9, color=FG,
                 va="center")
        fig.text(L + 0.42, yl, f"{int(p['prss'])} presiones", family=TXT, size=8.5, color=GRIS1,
                 va="center")
        w = wr_jugador(p["player"], team)
        fig.text(R - 0.01, yl, f"{w[0]}% · {w[1]}º {w[2]}" if w else "fuera del top 20",
                 family=TXT, size=8.5, color=GRIS1 if w else GRIS2, va="center", ha="right")
        yl -= 0.03

    tarjeta(fig, yl - 0.005, y_top)
    pie(fig, f"Duelo ganado: gana su bloqueo antes de 2,5 s (ESPN) · DE de {PESO_INTERIOR}+ lb = interior",
        SELLO_WR)
    guardar(fig, salida(f"dline_presion_origen_{team}_{SEASON}.png", SEASON))


def draw_liga():
    if WR_EQ is not None:
        orden = WR_EQ.reindex(tabla.index).sort_values("prwr_rk").index.tolist()
    else:
        orden = tabla.sort_values("press_pct", ascending=False).index.tolist()
    n = len(orden)
    fig = lienzo()
    antetitulo(fig, "Defensas · pass rush")
    if WR_EQ is not None:
        mejor, peor = orden[0], orden[-1]
        nm, np_ = nombre_equipo(mejor), nombre_equipo(peor)
        tit = (f"{nm} {v(nm, 'gana', 'ganan')} el {int(WR_EQ.loc[mejor, 'prwr'])}% de sus duelos "
               f"de pass rush; {np_}, el {int(WR_EQ.loc[peor, 'prwr'])}%.")
        y_top = titular(fig, tit[0].upper() + tit[1:], y=0.925) - 0.03
    else:
        y_top = 0.90
    por_col = (n + 1) // 2
    alto = (y_top - 0.15) / (por_col + 1)
    ancho = (R - L) / 2
    for c in range(2):
        x0 = L + c * ancho + (0.012 if c else 0)
        yh = y_top - 0.03
        for xx, txt in ((0.20, "GANA"), (0.29, "PRESIÓN"), (0.405, "ORIGEN")):
            fig.text(x0 + xx, yh, txt, family=TXT, weight="medium", size=6.3, color=GRIS1,
                     ha="right", va="center")
    for i, tm in enumerate(orden):
        c, f = divmod(i, por_col)
        x0 = L + c * ancho + (0.012 if c else 0)
        yy = y_top - 0.03 - (f + 1) * alto
        rk = int(WR_EQ.loc[tm, "prwr_rk"]) if WR_EQ is not None else i + 1
        fig.text(x0 + 0.025, yy, f"{rk}", family=TXT, weight="bold", size=7.6,
                 color=color_rango(rk, n), ha="right", va="center")
        poner_logo(fig, tm, x0 + 0.05, yy, zoom=0.0095)
        fig.text(x0 + 0.072, yy, tm, family=TXT, weight="bold", size=7.4, color=FG, va="center")
        if WR_EQ is not None:
            fig.text(x0 + 0.20, yy, f"{int(WR_EQ.loc[tm, 'prwr'])}%", family=TXT, weight="semibold",
                     size=7.6, color=FG, ha="right", va="center")
        fig.text(x0 + 0.29, yy, num(tabla.loc[tm, "press_pct"], 1, signo=False), family=TXT,
                 size=7.4, color=GRIS1, ha="right", va="center")
        rr = tabla.loc[tm]
        o = max(ORIGENES, key=lambda k: rr[f"pct_{k}"] / media_liga[k] if media_liga[k] else 0)
        fig.text(x0 + 0.405, yy, MINI[o], family=TXT, size=6.8, color=GRIS2, ha="right",
                 va="center")
    y_fin = y_top - 0.03 - por_col * alto - alto / 2 - 0.01
    linea_v(fig, L + ancho + 0.002, y_fin + 0.015, y_top - 0.015)
    tarjeta(fig, y_fin, y_top)
    pie(fig, "Gana: duelos de pass rush ganados (ESPN) · Presión por 100 pases (PFR) · "
             "Origen: el más por encima de la media", SELLO_WR)
    guardar(fig, salida(f"dline_presion_origen_{SEASON}.png", SEASON))


# ── MAIN ──────────────────────────────────────────────────────────────────────
if team_input:
    draw_equipo(team_input)
else:
    draw_liga()
