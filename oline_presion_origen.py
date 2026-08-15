"""
oline_presion_origen.py
Por dónde le llega la presión a cada línea ofensiva.
- Sin equipo (Enter): heatmap 32 equipos — presión total (FTN) + origen de la
  presión atribuida: interior (DT/NT), exterior (DE/OLB), blitz LB, blitz DB.
- Con equipo (ej: SF): diagrama de campo estilo run_gap con las flechas de
  presión convergiendo sobre el QB.

Nota de datos: nflverse no dice qué liniero fue batido (eso es PFF de pago).
El proxy honesto es el ORIGEN de la presión: quién la generó — interior =
guards/center batidos; exterior = tackles batidos. La atribución solo existe
en sacks y QB hits (los hurries no traen autor en los datos públicos); la
presión total sí es real (was_pressure de FTN).
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
from pbp_loader import cargar_pbp, cargar_participation, cargar_stats, salida, season_cli, sello

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


def clasificar(depth, pos):
    if isinstance(depth, str) and depth.strip().upper() in MAPA_DEPTH:
        return MAPA_DEPTH[depth.strip().upper()]
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
import nflreadpy as nfl
ros = nfl.load_rosters(SEASON).to_pandas()
id2clase = {}
for _, r in ros.iterrows():
    gid = r.get("gsis_id")
    if pd.isna(gid):
        continue
    cl = clasificar(r.get("depth_chart_position"), r.get("position"))
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
tabla["press_pct"] = drops.groupby("posteam")["was_pressure"].mean() * 100
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
print(f"  Origen de la presión cedida | NFL {SEASON}  (% de dropbacks; menos = mejor OL)")
print(f"{'='*78}")
print(f"{'Off':<5}{'Press%':>8}" + "".join(f"{o:>9}" for o in ORIGENES))
for tm in tabla.index:
    r = tabla.loc[tm]
    p = f"{r['press_pct']:.1f}" if pd.notna(r["press_pct"]) else "N/D"
    print(f"{tm:<5}{p:>8}" + "".join(f"{r[f'pct_{o}']:>8.1f}%" for o in ORIGENES))


# ── HEATMAP 32 EQUIPOS ────────────────────────────────────────────────────────
def draw_heatmap():
    teams   = tabla.index.tolist()
    n_teams = len(teams)
    cols    = ["PRESS"] + ORIGENES
    col_labels = {"PRESS": "Presión total\n(FTN)"} | ORIGEN_LABELS
    n_cols  = len(cols)
    logo_w  = 1.2
    fig_w   = logo_w + n_cols * 1.5 + 1.4
    fig_h   = max(8, n_teams * 0.52 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.995, bottom=0.008)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(-logo_w, n_cols)
    ax.set_ylim(-1, n_teams + 2.0)

    # Normalización por columna (percentiles 5-95); verde = poca presión cedida
    cmap = plt.cm.RdYlGn_r
    normas = {}
    for c in cols:
        vals = tabla["press_pct"] if c == "PRESS" else tabla[f"pct_{c}"]
        vals = vals.dropna()
        normas[c] = Normalize(vmin=np.percentile(vals, 5),
                              vmax=np.percentile(vals, 95))

    for row_i, tm in enumerate(teams):
        y = n_teams - row_i - 1
        for col_j, c in enumerate(cols):
            if c == "PRESS":
                val, n = tabla.loc[tm, "press_pct"], None
            else:
                val, n = tabla.loc[tm, f"pct_{c}"], tabla.loc[tm, f"n_{c}"]
            if pd.isna(val):
                ax.add_patch(plt.Rectangle((col_j, y), 1, 1, color="#1e2430",
                                           linewidth=0.4, edgecolor=BG))
                ax.text(col_j + 0.5, y + 0.5, "—", ha="center", va="center",
                        color="#444444", fontsize=10)
                continue
            bgc = cmap(normas[c](val))
            ax.add_patch(plt.Rectangle((col_j, y), 1, 1, color=bgc,
                                       linewidth=0.4, edgecolor=BG, zorder=1))
            _r, _g, _b = bgc[0], bgc[1], bgc[2]
            txt = "#0a0e13" if (0.299*_r + 0.587*_g + 0.114*_b) > 0.45 else FG
            if n is None:
                ax.text(col_j + 0.5, y + 0.5, f"{val:.1f}%", ha="center",
                        va="center", color=txt, fontsize=8.5,
                        fontweight="bold", zorder=2)
            else:
                ax.text(col_j + 0.5, y + 0.60, f"{val:.1f}%", ha="center",
                        va="center", color=txt, fontsize=8,
                        fontweight="bold", zorder=2)
                ax.text(col_j + 0.5, y + 0.25, f"n={int(n)}", ha="center",
                        va="center", color=txt, fontsize=6.5, zorder=2)

    for row_i, tm in enumerate(teams):
        y   = n_teams - row_i - 1
        img = load_logo(tm, base_zoom=0.036)
        if img is not None:
            ab = AnnotationBbox(img, (-logo_w / 2, y + 0.5), frameon=False,
                                zorder=3, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        else:
            ax.text(-logo_w / 2, y + 0.5, tm, ha="center", va="center",
                    color=FG, fontsize=7.5, fontweight="bold")

    for col_j, c in enumerate(cols):
        ax.text(col_j + 0.5, n_teams + 0.42, col_labels[c], ha="center",
                va="center", color=FG, fontsize=8, fontweight="bold",
                linespacing=1.3)
    ax.axhline(n_teams, color=GRID, linewidth=0.8, zorder=3)

    fig.text(0.5, 0.99, f"¿Por dónde cede presión cada línea ofensiva? | NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.975,
             "Ordenado por presión total FTN (mejor OL arriba)  ·  Origen = % de dropbacks con sack/QB hit de ese tipo de rusher  ·  "
             "Interior ≈ guards/center batidos, Exterior ≈ tackles",
             ha="center", va="top", fontsize=8.5, color="#888888", fontstyle="italic")
    fig.text(0.01, 0.005,
             f"Fuente: nflverse-data (FTN + atribución de sacks/QB hits)  |  {sello(SEASON)}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.99, 0.005, "@CuartayDato", ha="right", va="bottom", fontsize=9,
             color="#888888", alpha=0.85, fontstyle="italic")

    out = salida(f"oline_presion_origen_{SEASON}.png", SEASON)
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")


# ── OL POR PUESTO: presión del equipo con/sin cada titular ────────────────────
SLOTS = ["LT", "LG", "C", "RG", "RT"]
MIN_SIN = 40   # dropbacks mínimos sin el titular para enseñar el split


def ol_por_puesto(team):
    """{slot: dict(name, jersey, con, n_con, sin, n_sin)} usando el depth chart
    (titular por puesto) y la alineación real de cada dropback (participation).
    El depth chart es un snapshot: se valida al titular por snaps reales y, si
    el listado no jugó, se usa el suplente del mismo puesto que más jugó."""
    sub = drops[(drops["posteam"] == team) & drops["offense_players"].notna()]
    out = {}
    try:
        dc = nfl.load_depth_charts([SEASON]).to_pandas()
        dc = dc[(dc["team"] == team) & (dc["pos_abb"].isin(SLOTS))]
    except Exception:
        dc = pd.DataFrame(columns=["pos_abb", "pos_rank", "gsis_id", "player_name"])
    dorsal = {r["gsis_id"]: r["jersey_number"]
              for _, r in ros[ros["team"] == team].iterrows()
              if pd.notna(r.get("gsis_id")) and pd.notna(r.get("jersey_number"))}
    for slot in SLOTS:
        cands = dc[dc["pos_abb"] == slot].sort_values("pos_rank")
        mejor = None
        for _, r in cands.iterrows():
            gid = r.get("gsis_id")
            if pd.isna(gid):
                continue
            en = sub["offense_players"].str.contains(gid, regex=False)
            n_con = int(en.sum())
            if mejor is None or n_con > mejor["n_con"]:
                mejor = dict(name=r["player_name"], gid=gid, n_con=n_con,
                             con=sub.loc[en, "was_pressure"].mean() * 100,
                             n_sin=int((~en).sum()),
                             sin=sub.loc[~en, "was_pressure"].mean() * 100)
        if mejor is None or mejor["n_con"] < 50:
            out[slot] = None
        else:
            mejor["jersey"] = dorsal.get(mejor["gid"])
            out[slot] = mejor
    return out


# ── DIAGRAMA DE CAMPO (un equipo) ─────────────────────────────────────────────
OL_POS = {"LT": (-3.2, 2.5), "LG": (-1.6, 2.5), "C": (0.0, 2.5),
          "RG": (1.6, 2.5), "RT": (3.2, 2.5)}
OL_R  = 0.40
QB_XY = (0.0, 0.55)
QB_R  = 0.42

# Puntos de origen de las flechas de presión (arco defensivo)
ARR_XY = {
    "DB":  (-3.6, 4.9),
    "LB":  (-1.5, 5.1),
    "INT": ( 0.9, 5.1),
    "EXT": ( 3.9, 3.4),   # rodea al tackle; representa AMBOS lados
}
# Ángulo de llegada al QB de cada flecha (grados; 90 = desde arriba)
ANG_LLEGADA = {"DB": 152, "LB": 113, "INT": 72, "EXT": 8}


def draw_diagrama(team):
    r = tabla.loc[team]

    fig, ax = plt.subplots(figsize=(11, 7.8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-0.6, 6.6)

    qx, qy = QB_XY

    # Flechas de presión: origen defensivo → QB (Bézier), grosor ∝ % cedido,
    # color rojo si cede más que la media de la liga, verde si menos
    for o in ORIGENES:
        px, py = ARR_XY[o]
        pct    = r[f"pct_{o}"]
        n      = int(r[f"n_{o}"])
        col    = COL_MAL if pct >= media_liga[o] else COL_BIEN
        lw     = 2.0 + 6.0 * min(pct / 6.0, 1.0)

        ctrl_x = qx + (px - qx) * 0.80
        ctrl_y = 1.35
        ang = np.radians(ANG_LLEGADA[o])
        ux, uy = np.cos(ang), np.sin(ang)
        tipo = np.array([qx, qy]) + np.array([ux, uy]) * (QB_R + 0.28)
        verts = [(px, py), (ctrl_x, ctrl_y), tuple(tipo)]
        patch = mpatches.PathPatch(
            MPath(verts, [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
            facecolor="none", edgecolor=col, linewidth=lw, zorder=2,
            capstyle="round", alpha=0.9)
        ax.add_patch(patch)

        # Punta de flecha apuntando al QB
        perp = np.array([-uy, ux])
        tip  = np.array(tipo) - np.array([ux, uy]) * 0.05
        base = tip + np.array([ux, uy]) * 0.30
        ax.add_patch(plt.Polygon([tip, base + perp * 0.15, base - perp * 0.15],
                                 color=col, zorder=7))

        # Etiqueta en el origen
        nombre = ORIGEN_LABELS[o].split("\n")[0]
        extra  = "  (izq+dcha)" if o == "EXT" else ""
        ax.text(px, py + 0.52, f"{nombre}{extra}", ha="center", va="bottom",
                color=FG, fontsize=9.5, fontweight="bold", zorder=10)
        ax.text(px, py + 0.16, f"{pct:.1f}%  ·  n={n}  ·  liga {media_liga[o]:.1f}%",
                ha="center", va="bottom", color=col, fontsize=8, zorder=10)

    # Círculos OL: titular de cada puesto (dorsal + apellido, sin métricas —
    # la presión cedida por liniero concreto no existe en datos públicos)
    puestos = ol_por_puesto(team)
    halo = [pe.withStroke(linewidth=2.4, foreground=BG)]
    for pos, (cx, cy) in OL_POS.items():
        info = puestos.get(pos)
        ax.add_patch(plt.Circle((cx, cy), OL_R, facecolor="#1c2535",
                                edgecolor=GRID, linewidth=1.4, zorder=5))
        if info and info.get("jersey") is not None and pd.notna(info["jersey"]):
            ax.text(cx, cy + 0.11, pos, ha="center", va="center", color=FG,
                    fontsize=9, fontweight="bold", zorder=6)
            ax.text(cx, cy - 0.12, f"#{int(info['jersey'])}", ha="center",
                    va="center", color="#aaaaaa", fontsize=7, zorder=6)
        else:
            ax.text(cx, cy, pos, ha="center", va="center", color=FG,
                    fontsize=10, fontweight="bold", zorder=6)
        if info:
            apellido = info["name"].split()[-1][:12]
            ax.text(cx, cy - OL_R - 0.14, apellido, ha="center", va="top",
                    color=FG, fontsize=8, fontweight="bold", zorder=10,
                    path_effects=halo)

    # QB
    ax.add_patch(plt.Circle(QB_XY, QB_R, color=COL_QB, zorder=8))
    ax.text(qx, qy, "QB", ha="center", va="center", color=BG, fontsize=12,
            fontweight="bold", zorder=9)

    logo = load_logo(team, base_zoom=0.075)
    if logo:
        ab = AnnotationBbox(logo, (qx - 1.8, qy), frameon=False, zorder=3,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

    # Barra de stats
    press = f"{r['press_pct']:.1f}%" if pd.notna(r["press_pct"]) else "N/D"
    STATS = [(f"{int(r['dropbacks'])}", "DROPBACKS"),
             (press,                    "PRESIÓN (FTN)"),
             (f"{int(r['sacks'])}",     "SACKS"),
             (f"{int(r['hits'])}",      "QB HITS")]
    xs = np.linspace(-3.3, 3.3, len(STATS))
    sy = 6.05
    for i, (val, lbl) in enumerate(STATS):
        ax.text(xs[i], sy + 0.18, val, ha="center", va="bottom", color=FG,
                fontsize=11, fontweight="bold", zorder=5)
        ax.text(xs[i], sy + 0.02, lbl, ha="center", va="top",
                color="#888888", fontsize=8, zorder=5)
        if i < len(STATS) - 1:
            xsep = (xs[i] + xs[i + 1]) / 2
            ax.plot([xsep, xsep], [sy - 0.12, sy + 0.45], color=GRID,
                    linewidth=0.8, zorder=3)

    # Top rushers que castigaron a esta OL
    top = (ev[ev["posteam"] == team].groupby("player").size()
           .sort_values(ascending=False).head(3))
    if len(top):
        linea = "   ·   ".join(f"{nm} ×{ct}" for nm, ct in top.items())
        ax.text(0, -0.42, f"Más te castigaron:  {linea}", ha="center",
                va="center", color="#9aa3b5", fontsize=8.5, zorder=5)

    fig.text(0.5, 0.97, f"¿Por dónde le llega la presión? — {team} | NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.945,
             "Flechas: % de dropbacks con sack/QB hit según el origen del rusher (rojo = más que la media NFL)  ·  "
             "Interior ≈ guards/center batidos, Exterior ≈ tackles (sin lado en los datos públicos)",
             ha="center", va="top", fontsize=8, color="#888888", fontstyle="italic")
    fig.text(0.01, 0.01,
             f"Fuente: nflverse-data (FTN + atribución de sacks/QB hits)  |  {sello(SEASON)}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.99, 0.01, "@CuartayDato", ha="right", va="bottom", fontsize=9,
             color="#888888", alpha=0.85, fontstyle="italic")

    out = salida(f"oline_presion_origen_{team}_{SEASON}.png", SEASON)
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if team_input:
    if team_input not in tabla.index:
        raise SystemExit(f"Equipo '{team_input}' no encontrado. "
                         f"Disponibles: {', '.join(tabla.index)}")
    draw_diagrama(team_input)
else:
    draw_heatmap()
