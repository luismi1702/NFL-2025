"""
oline_presion_origen.py
Por dónde le llega la presión a cada línea ofensiva.
- Sin equipo (Enter): heatmap 32 equipos — presión sufrida por 100 dropbacks
  + origen de la presión atribuida: interior (DT/NT), exterior (DE/OLB),
  blitz LB, blitz DB.
- Con equipo (ej: SF): ficha 4:5 con el win rate de ESPN, la presión sufrida,
  el origen de los sacks/golpes contra la media y sus linieros en el top 20 de
  ESPN (03-oct-2026: el diagrama de campo necesitaba la participación de
  nflverse para poner nombre a cada puesto, y en temporada no existe).

Nota de datos: nflverse no dice qué liniero fue batido (eso es PFF de pago).
El proxy honesto es el ORIGEN de la presión: quién la generó — interior =
guards/center batidos; exterior = tackles batidos. La atribución solo existe
en sacks y QB hits (los hurries no traen autor en los datos públicos), asi que
las columnas de origen NO suman la de presión total.
La presión total viene de PFR (times_pressured, con hurries) y no de
was_pressure de participación: ese dataset no tiene cron en nflverse y se
congela durante la temporada.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.path import Path as MPath
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pbp_loader import (cargar_pbp, cargar_participation, cargar_stats,
                        cargar_rosters, cargar_pfr, cargar_web, DatosNoDisponibles,
                        salida, season_cli, sello)
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)

sys.stdout.reconfigure(encoding="utf-8")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON    = season_cli()   # None = auto-detectar última temporada
BG        = "#0f1115"
FG        = "#EDEDED"
GRID      = "#2a2f3a"
DPI       = 170
LOGOS_DIR = "logos"

COL_MAL  = "#d84a4a"   # rojo  — cede más presión que la media
COL_BIEN = "#06d6a0"   # verde — cede menos que la media
COL_QB   = "#E5C070"   # dorado — QB

ORIGENES = ["INT", "EXT", "LB", "DB"]
ORIGEN_LABELS = {
    "INT": "Interior\nDT · NT",
    "EXT": "Exterior\nDE · OLB",
    "LB":  "Blitz LB\nILB · MLB",
    "DB":  "Blitz DB\nCB · S",
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
        # Recorta margenes transparentes y normaliza por la tinta real
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


# "DE" no significa lo mismo en un 4-3 que en un 3-4: en el segundo juega por
# dentro. Misma regla que dline_presion_origen — los dos scripts clasifican a
# los mismos defensores y tienen que coincidir, o el "exterior" de uno no es el
# del otro. Ver PESO_INTERIOR allí para la validación del corte.
PESO_INTERIOR = 280


def clasificar(depth, pos, peso=None):
    d = depth.strip().upper() if isinstance(depth, str) else ""
    if d == "DE" and peso and float(peso) >= PESO_INTERIOR:
        return "INT"
    if d in MAPA_DEPTH:
        return MAPA_DEPTH[d]
    if isinstance(pos, str) and pos.strip().upper() in MAPA_POS:
        return MAPA_POS[pos.strip().upper()]
    return None


# ── INPUT ──────────────────────────────────────────────────────────────────────
team_input = input("Equipo (siglas, ej: SF — Enter = heatmap 32 equipos): ").strip().upper()

# ── DATOS ──────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)
for c in ["qb_dropback", "sack", "qb_hit"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

# Presión real FTN
try:
    part, _ = cargar_participation(SEASON)
    part = part[["nflverse_game_id", "play_id", "was_pressure",
                 "offense_players"]].rename(
        columns={"nflverse_game_id": "game_id"})
    part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")
    df["play_id"]   = pd.to_numeric(df["play_id"],   errors="coerce")
    df = df.merge(part, on=["game_id", "play_id"], how="left")
    df["was_pressure"] = pd.to_numeric(df["was_pressure"], errors="coerce")
except Exception as e:
    print(f"  Aviso: FTN no disponible ({e})")
    df["was_pressure"] = np.nan
    df["offense_players"] = np.nan

# Posiciones: roster (depth_chart_position granular) + fallback player_stats
ros, _ = cargar_rosters(SEASON)
id2clase = {}
for _, r in ros.iterrows():
    gid = r.get("gsis_id")
    if pd.isna(gid):
        continue
    cl = clasificar(r.get("depth_chart_position"), r.get("position"),
                    r.get("weight"))
    if cl:
        id2clase[gid] = cl

st, _ = cargar_stats(SEASON)
for _, r in st.drop_duplicates("player_id").iterrows():
    pid = r.get("player_id")
    if pid in id2clase or pd.isna(pid):
        continue
    cl = clasificar(None, r.get("position"))
    if cl:
        id2clase[pid] = cl
print(f"Defensores clasificados: {len(id2clase):,}")

# ── EVENTOS DE PRESIÓN ATRIBUIDOS ─────────────────────────────────────────────
drops = df[(df["qb_dropback"] == 1) & df["posteam"].notna()].copy()

CRED_COLS = [("sack_player_id", "sack_player_name"),
             ("half_sack_1_player_id", "half_sack_1_player_name"),
             ("half_sack_2_player_id", "half_sack_2_player_name"),
             ("qb_hit_1_player_id", "qb_hit_1_player_name"),
             ("qb_hit_2_player_id", "qb_hit_2_player_name")]

eventos = []   # (posteam, clase, player_name)
for id_col, nm_col in CRED_COLS:
    if id_col not in drops.columns:
        continue
    sub = drops[drops[id_col].notna()][["posteam", id_col, nm_col]]
    for _, r in sub.iterrows():
        cl = id2clase.get(r[id_col])
        if cl:
            eventos.append((r["posteam"], cl, r[nm_col]))
ev = pd.DataFrame(eventos, columns=["posteam", "clase", "player"])
print(f"Presiones atribuidas (sacks + QB hits): {len(ev):,}")

# ── TABLA POR EQUIPO ──────────────────────────────────────────────────────────
equipos = sorted(drops["posteam"].dropna().unique())
tabla = pd.DataFrame(index=equipos)
tabla["dropbacks"] = drops.groupby("posteam").size()

# Presión total sufrida desde PFR (times_pressured por QB, incluye hurries).
# Antes salía de was_pressure de participación, que es el dataset SIN cron de
# nflverse: se congela durante la temporada. Además el informe de equipo ya usa
# PFR para este mismo KPI, y daban números distintos para el mismo concepto.
try:
    _pfr, _ = cargar_pfr("pass", SEASON)
    _pfr = _pfr[_pfr["team"] != "3TM"].copy()
    _pfr["times_pressured"] = pd.to_numeric(_pfr["times_pressured"],
                                            errors="coerce").fillna(0)
    tabla["press_pct"] = (_pfr.groupby("team")["times_pressured"].sum()
                          / tabla["dropbacks"] * 100)
    FUENTE_PRESS = "PFR (con hurries)"
except Exception as e:
    print(f"  Aviso: presión total desde FTN ({type(e).__name__})")
    tabla["press_pct"] = drops.groupby("posteam")["was_pressure"].mean() * 100
    FUENTE_PRESS = "FTN"
tabla["sacks"]     = drops.groupby("posteam")["sack"].sum().astype(int)
tabla["hits"]      = drops.groupby("posteam")["qb_hit"].sum().astype(int)
for o in ORIGENES:
    cnt = ev[ev["clase"] == o].groupby("posteam").size()
    tabla[f"n_{o}"]   = cnt.reindex(equipos).fillna(0).astype(int)
    tabla[f"pct_{o}"] = tabla[f"n_{o}"] / tabla["dropbacks"] * 100
# Orden por presión FTN; si FTN aún no ha publicado (inicio de temporada),
# ordena por la suma de presión atribuida para que el heatmap siga teniendo sentido
_orden = tabla["press_pct"].fillna(
    tabla[[f"pct_{o}" for o in ORIGENES]].sum(axis=1))
tabla = tabla.loc[_orden.sort_values().index]

media_liga = {o: tabla[f"pct_{o}"].mean() for o in ORIGENES}

# Consola
print(f"\n{'='*78}")
print(f"  Origen de la presión cedida | NFL {SEASON}  (por 100 dropbacks; menos = mejor OL)")
print(f"{'='*78}")
print(f"{'Off':<5}{'Press%':>8}" + "".join(f"{o:>9}" for o in ORIGENES))
for tm in tabla.index:
    r = tabla.loc[tm]
    p = f"{r['press_pct']:.1f}" if pd.notna(r["press_pct"]) else "N/D"
    print(f"{tm:<5}{p:>8}" + "".join(f"{r[f'pct_{o}']:>9.1f}" for o in ORIGENES))


# ── WIN RATES DE ESPN (copiados a mano a datos_web/, 03-oct-2026) ─────────────
# PBWR = % de duelos de pase en los que el bloqueador aguanta 2,5 s o mas.
# Por equipo estan los 32; por jugador, solo el top 20 de OT y de IOL (sin lado).
try:
    WR_EQ, SEM_WR = cargar_web("espn_win_rates_equipos", SEASON)
    WR_EQ = WR_EQ.set_index("team")
    WR_JUG, _ = cargar_web("espn_win_rates_jugadores", SEASON)
except DatosNoDisponibles as e:
    print(f"  Aviso: sin win rates de ESPN ({e})")
    WR_EQ, WR_JUG, SEM_WR = None, None, None

press_rk = tabla["press_pct"].rank(method="min").astype("Int64")    # 1 = la que menos sufre
CORTO = {"INT": "por dentro", "EXT": "por fuera", "LB": "con blitz de LB", "DB": "con blitz de DB"}
MINI = {"INT": "dentro", "EXT": "fuera", "LB": "blitz LB", "DB": "blitz DB"}
ETIQ = {"INT": "Por dentro (DT, NT)", "EXT": "Por fuera (DE, OLB)", "LB": "Blitz de LB",
        "DB": "Blitz de DB (CB, S)"}
SELLO_WR = (f"NFL {SEASON} · win rates de ESPN hasta sem. {SEM_WR} · presión de PFR y nflverse"
            if SEM_WR else f"{sello(SEASON)} · presión de PFR y nflverse")


def titular_equipo(team):
    """Hechos que el script calcula: el puesto en PBWR si es extremo; si no, el
    origen de la presion que mas se aleja de la media de la liga."""
    T = nombre_equipo(team)
    if WR_EQ is not None and team in WR_EQ.index:
        pb, rk = int(WR_EQ.loc[team, "pbwr"]), int(WR_EQ.loc[team, "pbwr_rk"])
        if rk <= 5:
            return f"{T} {v(T, 'gana', 'ganan')} el {pb}% de sus duelos en protección, {rk}º de la NFL."
        if rk >= 28:
            return f"{T} solo {v(T, 'gana', 'ganan')} el {pb}% de sus duelos en protección, {rk}º de la NFL."
    r = tabla.loc[team]
    ratio = {o: r[f"pct_{o}"] / media_liga[o] for o in ORIGENES if media_liga[o] > 0}
    o = max(ratio, key=ratio.get)
    if ratio[o] >= 1.4 and r[f"n_{o}"] >= 3:
        return (f"A {T} la presión le llega sobre todo {CORTO[o]}: "
                f"{num(r[f'pct_{o}'], 1, signo=False)} por cada 100 pases "
                f"(media, {num(media_liga[o], 1, signo=False)}).")
    return None


def draw_equipo(team):
    r = tabla.loc[team]
    fig = lienzo()
    antetitulo(fig, f"Línea ofensiva · {team}")
    tit = titular_equipo(team)
    y = (titular(fig, tit[0].upper() + tit[1:], y=0.925) - 0.03) if tit else 0.90
    y_top = y

    # Cabecera
    yc = y - 0.045
    poner_logo(fig, team, L + 0.035, yc, zoom=0.022)
    fig.text(L + 0.08, yc, team, family=TXT, weight="bold", size=10, color=color_equipo(team),
             va="center")
    fig.text(L + 0.15, yc, "·  PROTECCIÓN DEL PASE", family=TXT, weight="semibold", size=8.5,
             color=GRIS1, va="center")

    # Dos numeros: win rate de ESPN y presion sufrida (PFR)
    yb = yc - 0.06
    n_eq = len(tabla)
    if WR_EQ is not None and team in WR_EQ.index:
        pb, rk = int(WR_EQ.loc[team, "pbwr"]), int(WR_EQ.loc[team, "pbwr_rk"])
        etiqueta(fig, "Duelos ganados en pase (ESPN)", L + 0.01, yb)
        t = numero(fig, f"{pb}", L + 0.005, yb - 0.065, size=44)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        fig.text(bb.x1 / fig.bbox.width + 0.004, yb - 0.08, "%", family=DISP, weight="semibold",
                 size=19, color=FG, va="center")
        fig.text(L + 0.01, yb - 0.13, ordinal(rk), family=TXT, weight="bold", size=9,
                 color=color_rango(rk), va="center")
        nota(fig, f"En carrera: {int(WR_EQ.loc[team, 'rbwr'])}% ({int(WR_EQ.loc[team, 'rbwr_rk'])}º)",
             L + 0.01, yb - 0.158)
        pista32(fig, L + 0.01, yb - 0.195, 0.36, rk)
    if pd.notna(r["press_pct"]):
        prk = int(press_rk[team])
        etiqueta(fig, "Presión sufrida por 100 pases", 0.54, yb)
        numero(fig, num(r["press_pct"], 1, signo=False), 0.535, yb - 0.065, size=44)
        fig.text(0.54, yb - 0.13, ordinal(prk, n_eq), family=TXT, weight="bold", size=9,
                 color=color_rango(prk, n_eq), va="center")
        nota(fig, f"{int(r['hits'])} golpes al QB, {int(r['sacks'])} de ellos sacks, en {int(r['dropbacks'])} pases",
             0.54, yb - 0.158)
        pista32(fig, 0.54, yb - 0.195, 0.36, prk, n=n_eq)
    linea_v(fig, 0.5, yb - 0.21, yb + 0.015)

    # Por donde llega: cuatro barras contra la media de la liga
    ym = yb - 0.245
    linea_h(fig, ym + 0.025)
    etiqueta(fig, "Por dónde llegan los sacks y los golpes (por 100 pases)", L + 0.01, ym)
    tope = max(max(r[f"pct_{o}"] for o in ORIGENES), max(media_liga.values())) * 1.15 or 1
    x0b, wb = L + 0.30, 0.42
    yy = ym - 0.04
    for o in ORIGENES:
        val, med, n = r[f"pct_{o}"], media_liga[o], int(r[f"n_{o}"])
        fig.text(L + 0.01, yy, ETIQ[o], family=TXT, size=8.3, color=FG, va="center")
        fig.add_artist(Rectangle((x0b, yy - 0.008), wb * val / tope, 0.016,
                                 transform=fig.transFigure,
                                 color=color_equipo(team) if val > med else NEUTRO, lw=0))
        fig.add_artist(Rectangle((x0b + wb * med / tope - 0.001, yy - 0.013), 0.002, 0.026,
                                 transform=fig.transFigure, color=FG, lw=0))
        fig.text(x0b + wb + 0.015, yy, f"{num(val, 1, signo=False)}  ({n})", family=TXT,
                 weight="semibold", size=8.3, color=FG, va="center")
        yy -= 0.033
    nota(fig, "Raya blanca = media NFL · en color, por encima de la media · (n) = jugadas", L + 0.01, yy + 0.008)

    # Sus linieros en el top 20 de ESPN
    yl = yy - 0.035
    linea_h(fig, yl + 0.02)
    etiqueta(fig, "Sus linieros en el top 20 de ESPN en protección", L + 0.01, yl)
    yl -= 0.035
    suyos = (WR_JUG[(WR_JUG.metrica == "pass_block") & (WR_JUG.team == team)]
             if WR_JUG is not None else pd.DataFrame())
    if suyos.empty:
        nota(fig, "Ninguno entre los 20 mejores tackles ni los 20 mejores interiores",
             L + 0.01, yl)
        yl -= 0.03
    for _, j in suyos.sort_values("win_rate", ascending=False).iterrows():
        pos = "tackle" if j.grupo == "OT" else "interior"
        fig.text(L + 0.01, yl, j.jugador, family=TXT, weight="semibold", size=9, color=FG,
                 va="center")
        fig.text(R - 0.01, yl, f"{int(j.win_rate)}% · {int(j.rk)}º {pos}", family=TXT, size=8.5,
                 color=GRIS1, va="center", ha="right")
        yl -= 0.03

    tarjeta(fig, yl - 0.005, y_top)
    pie(fig, "Duelo ganado: el bloqueador aguanta 2,5 s (ESPN) · origen por el puesto del rusher",
        SELLO_WR)
    guardar(fig, salida(f"oline_presion_origen_{team}_{SEASON}.png", SEASON))


def draw_liga():
    """Las 32 lineas ordenadas por PBWR de ESPN (o por presion si no hay ESPN)."""
    if WR_EQ is not None:
        orden = WR_EQ.reindex(tabla.index).sort_values("pbwr_rk").index.tolist()
    else:
        orden = tabla.sort_values("press_pct").index.tolist()
    n = len(orden)
    fig = lienzo()
    antetitulo(fig, "Líneas ofensivas · protección del pase")
    if WR_EQ is not None:
        mejor, peor = orden[0], orden[-1]
        nm, np_ = nombre_equipo(mejor), nombre_equipo(peor)
        tit = (f"{nm} {v(nm, 'gana', 'ganan')} el {int(WR_EQ.loc[mejor, 'pbwr'])}% de sus duelos "
               f"en protección; {np_}, el {int(WR_EQ.loc[peor, 'pbwr'])}%.")
        y_top = titular(fig, tit[0].upper() + tit[1:], y=0.925) - 0.03
    else:
        y_top = 0.90
    por_col = (n + 1) // 2
    alto = (y_top - 0.15) / (por_col + 1)
    ancho = (R - L) / 2
    for c in range(2):
        x0 = L + c * ancho + (0.012 if c else 0)
        yh = y_top - 0.03
        fig.text(x0 + 0.20, yh, "GANA", family=TXT, weight="medium", size=6.3, color=GRIS1,
                 ha="right", va="center")
        fig.text(x0 + 0.29, yh, "PRESIÓN", family=TXT, weight="medium", size=6.3, color=GRIS1,
                 ha="right", va="center")
        fig.text(x0 + 0.405, yh, "ORIGEN", family=TXT, weight="medium", size=6.3,
                 color=GRIS1, ha="right", va="center")
    for i, tm in enumerate(orden):
        c, f = divmod(i, por_col)
        x0 = L + c * ancho + (0.012 if c else 0)
        yy = y_top - 0.03 - (f + 1) * alto
        rk = int(WR_EQ.loc[tm, "pbwr_rk"]) if WR_EQ is not None else i + 1
        fig.text(x0 + 0.025, yy, f"{rk}", family=TXT, weight="bold", size=7.6,
                 color=color_rango(rk, n), ha="right", va="center")
        poner_logo(fig, tm, x0 + 0.05, yy, zoom=0.0095)
        fig.text(x0 + 0.072, yy, tm, family=TXT, weight="bold", size=7.4, color=FG, va="center")
        if WR_EQ is not None:
            fig.text(x0 + 0.20, yy, f"{int(WR_EQ.loc[tm, 'pbwr'])}%", family=TXT, weight="semibold",
                     size=7.6, color=FG, ha="right", va="center")
        p = tabla.loc[tm, "press_pct"]
        fig.text(x0 + 0.29, yy, num(p, 1, signo=False) if pd.notna(p) else "–", family=TXT,
                 size=7.4, color=GRIS1, ha="right", va="center")
        rr = tabla.loc[tm]
        o = max(ORIGENES, key=lambda k: rr[f"pct_{k}"] / media_liga[k] if media_liga[k] else 0)
        fig.text(x0 + 0.405, yy, MINI[o], family=TXT, size=6.8, color=GRIS2, ha="right",
                 va="center")
    y_fin = y_top - 0.03 - por_col * alto - alto / 2 - 0.01
    linea_v(fig, L + ancho + 0.002, y_fin + 0.015, y_top - 0.015)
    tarjeta(fig, y_fin, y_top)
    pie(fig, "Gana: duelos de pase ganados (ESPN) · Presión por 100 pases (PFR) · "
             "Origen: el más por encima de la media", SELLO_WR)
    guardar(fig, salida(f"oline_presion_origen_{SEASON}.png", SEASON))


# ── MAIN ──────────────────────────────────────────────────────────────────────
if team_input:
    if team_input not in tabla.index:
        raise SystemExit(f"Equipo '{team_input}' no encontrado. "
                         f"Disponibles: {', '.join(tabla.index)}")
    draw_equipo(team_input)
else:
    draw_liga()
